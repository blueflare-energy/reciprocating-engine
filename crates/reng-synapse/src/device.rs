//! A held Gaudi2 device context: acquire once, run many recipes, release on
//! drop. Acquiring and releasing the device around every single op races on
//! this stack (a fresh acquire that lands before the previous release has
//! settled reads back zeros), so all ops of a computation share one `Device`.

use crate::ffi::*;
use crate::{bf16_to_f32, f32_to_bf16};
use core::ffi::c_void;
use reng_core::{Error, Result};
use std::ffi::CString;

macro_rules! syn {
    ($call:expr) => {{
        let st = unsafe { $call };
        if st != SYN_SUCCESS {
            return Err(Error::Other(format!(
                concat!(stringify!($call), " -> synStatus {}"),
                st
            )));
        }
    }};
}

/// An acquired Gaudi2 device. Holds the SynapseAI process init for its
/// lifetime; drop releases the device and tears down the process. Each op runs
/// as its own recipe launch. Reliable dimensions are at least 128: a smaller
/// matmul hits a readback coherency race on this stack (a fast kernel's HBM
/// writeback is not yet coherent when the readback DMA fires). Zero-padding up
/// to 128 does NOT help - the padded kernel runs fast and races the same way.
pub struct Device {
    id: synDeviceId,
}

impl Device {
    /// Acquire the first available Gaudi2 device.
    ///
    /// # Errors
    ///
    /// Returns an error if init or acquire fails.
    pub fn acquire() -> Result<Self> {
        syn!(synInitialize());
        let mut id: synDeviceId = 0;
        syn!(synDeviceAcquireByDeviceType(&mut id, SYN_DEVICE_GAUDI2));
        Ok(Self { id })
    }

    fn tensor(
        &self,
        graph: synGraphHandle,
        name: &CString,
        fcd: u64,
        outer: u64,
    ) -> Result<synTensor> {
        let mut t: synTensor = core::ptr::null_mut();
        syn!(synTensorHandleCreate(
            &mut t,
            graph,
            SYN_TENSOR_DATA,
            name.as_ptr()
        ));
        let mut sec: synSectionHandle = core::ptr::null_mut();
        syn!(synSectionCreate(&mut sec, 0, graph));
        syn!(synSectionSetPersistent(sec, true));
        syn!(synTensorAssignToSection(t, sec, 0));
        let mut geo = synTensorGeometry {
            sizes: [0; HABANA_DIM_MAX],
            dims: 2,
        };
        geo.sizes[0] = fcd;
        geo.sizes[1] = outer;
        syn!(synTensorSetGeometry(t, &geo, SYN_GEOMETRY_SIZES));
        syn!(synTensorSetDeviceDataType(t, SYN_TYPE_BF16));
        Ok(t)
    }

    /// `C = A @ B`, `a` row-major `[m,k]`, `b` row-major `[k,n]`, result
    /// row-major `[m,n]` as f32. For reliable results each of `m`, `k`, `n`
    /// should be at least 128 (see the [`Device`] docs).
    ///
    /// # Errors
    ///
    /// Returns an error if any SynapseAI call fails.
    ///
    /// # Panics
    ///
    /// Panics if `a.len() != m*k` or `b.len() != k*n`.
    pub fn gemm(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        let (mm, kk, nn) = (m as u64, k as u64, n as u64);
        let mut graph: synGraphHandle = core::ptr::null_mut();
        syn!(synGraphCreate(&mut graph, SYN_DEVICE_GAUDI2));

        let (n_a, n_b, n_c) = (
            CString::new("A").unwrap(),
            CString::new("B").unwrap(),
            CString::new("C").unwrap(),
        );
        let t_a = self.tensor(graph, &n_a, kk, mm)?;
        let t_b = self.tensor(graph, &n_b, nn, kk)?;
        let t_c = self.tensor(graph, &n_c, nn, mm)?;

        let params = synGEMMParams {
            transpose_a: false,
            transpose_b: false,
        };
        let guid = CString::new("gemm").unwrap();
        let inputs = [t_a, t_b];
        let outputs = [t_c];
        syn!(synNodeCreate(
            graph,
            inputs.as_ptr(),
            outputs.as_ptr(),
            2,
            1,
            (&raw const params).cast::<c_void>(),
            core::mem::size_of::<synGEMMParams>() as u32,
            guid.as_ptr(),
            CString::new("mm").unwrap().as_ptr(),
            core::ptr::null(),
            core::ptr::null(),
        ));

        let mut recipe: synRecipeHandle = core::ptr::null_mut();
        syn!(synGraphCompile(
            &mut recipe,
            graph,
            CString::new("gemm").unwrap().as_ptr(),
            core::ptr::null()
        ));

        let names = [n_a.as_ptr(), n_b.as_ptr(), n_c.as_ptr()];
        let mut ids: [u64; 3] = [0; 3];
        syn!(synTensorRetrieveIds(
            recipe,
            names.as_ptr(),
            ids.as_mut_ptr(),
            3
        ));

        let a_bytes = (m * k * 2) as u64;
        let b_bytes = (k * n * 2) as u64;
        let c_bytes = (m * n * 2) as u64;
        let (mut da, mut db, mut dc) = (0u64, 0u64, 0u64);
        syn!(synDeviceMalloc(self.id, a_bytes, 0, 0, &mut da));
        syn!(synDeviceMalloc(self.id, b_bytes, 0, 0, &mut db));
        syn!(synDeviceMalloc(self.id, c_bytes, 0, 0, &mut dc));
        let mut ws = 0u64;
        syn!(synWorkspaceGetSize(&mut ws, recipe));
        let mut dws = 0u64;
        if ws > 0 {
            syn!(synDeviceMalloc(self.id, ws, 0, 0, &mut dws));
        }

        let (mut ha, mut hb, mut hc): (*mut c_void, *mut c_void, *mut c_void) = (
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
        syn!(synHostMalloc(self.id, a_bytes, 0, &mut ha));
        syn!(synHostMalloc(self.id, b_bytes, 0, &mut hb));
        syn!(synHostMalloc(self.id, c_bytes, 0, &mut hc));

        // SAFETY: ha holds m*k and hb holds k*n bf16 elements.
        unsafe {
            let pa = ha.cast::<u16>();
            for (i, &x) in a.iter().enumerate() {
                *pa.add(i) = f32_to_bf16(x);
            }
            let pb = hb.cast::<u16>();
            for (i, &x) in b.iter().enumerate() {
                *pb.add(i) = f32_to_bf16(x);
            }
        }

        let mut stream: synStreamHandle = core::ptr::null_mut();
        syn!(synStreamCreateGeneric(&mut stream, self.id, 0));
        syn!(synMemCopyAsync(
            stream,
            ha as u64,
            a_bytes,
            da,
            SYN_HOST_TO_DRAM
        ));
        syn!(synMemCopyAsync(
            stream,
            hb as u64,
            b_bytes,
            db,
            SYN_HOST_TO_DRAM
        ));
        syn!(synStreamSynchronize(stream));

        let infos = [
            launch_info(&n_a, da, ids[0], kk, mm),
            launch_info(&n_b, db, ids[1], nn, kk),
            launch_info(&n_c, dc, ids[2], nn, mm),
        ];
        syn!(synLaunch(stream, infos.as_ptr(), 3, dws, recipe, 0));
        syn!(synStreamSynchronize(stream));
        syn!(synMemCopyAsync(
            stream,
            dc,
            c_bytes,
            hc as u64,
            SYN_DRAM_TO_HOST
        ));
        syn!(synStreamSynchronize(stream));

        let mut out = vec![0.0f32; m * n];
        // SAFETY: hc holds m*n bf16 elements just copied back.
        unsafe {
            let pc = hc.cast::<u16>();
            for (i, o) in out.iter_mut().enumerate() {
                *o = bf16_to_f32(*pc.add(i));
            }
        }

        unsafe {
            for h in [ha, hb, hc] {
                synHostFree(self.id, h, 0);
            }
            for d in [da, db, dc] {
                synDeviceFree(self.id, d, 0);
            }
            if dws != 0 {
                synDeviceFree(self.id, dws, 0);
            }
            synStreamDestroy(stream);
            synRecipeDestroy(recipe);
            synGraphDestroy(graph);
        }
        Ok(out)
    }

    /// Row-wise softmax of a `rows x cols` bf16 matrix (softmax over `cols`),
    /// via `softmax_fwd_bf16`. Returns f32.
    ///
    /// # Errors
    ///
    /// Returns an error if any SynapseAI call fails.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != rows*cols`.
    pub fn softmax(&self, input: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
        assert_eq!(input.len(), rows * cols);
        let (c, r) = (cols as u64, rows as u64);
        let mut graph: synGraphHandle = core::ptr::null_mut();
        syn!(synGraphCreate(&mut graph, SYN_DEVICE_GAUDI2));

        let name_in = CString::new("IN").unwrap();
        let name_out = CString::new("OUT").unwrap();
        let tin = self.tensor(graph, &name_in, c, r)?;
        let tout = self.tensor(graph, &name_out, c, r)?;

        let params = synSoftmaxParams { dim: 0 };
        let inputs = [tin];
        let outputs = [tout];
        let guid = CString::new("softmax_fwd_bf16").unwrap();
        syn!(synNodeCreate(
            graph,
            inputs.as_ptr(),
            outputs.as_ptr(),
            1,
            1,
            (&raw const params).cast::<c_void>(),
            core::mem::size_of::<synSoftmaxParams>() as u32,
            guid.as_ptr(),
            CString::new("sm").unwrap().as_ptr(),
            core::ptr::null(),
            core::ptr::null(),
        ));

        let mut recipe: synRecipeHandle = core::ptr::null_mut();
        syn!(synGraphCompile(
            &mut recipe,
            graph,
            CString::new("sm").unwrap().as_ptr(),
            core::ptr::null()
        ));

        let name_ptrs = [name_in.as_ptr(), name_out.as_ptr()];
        let mut ids: [u64; 2] = [0; 2];
        syn!(synTensorRetrieveIds(
            recipe,
            name_ptrs.as_ptr(),
            ids.as_mut_ptr(),
            2
        ));

        let bytes = (rows * cols * 2) as u64;
        let (mut dev_in, mut dev_out) = (0u64, 0u64);
        syn!(synDeviceMalloc(self.id, bytes, 0, 0, &mut dev_in));
        syn!(synDeviceMalloc(self.id, bytes, 0, 0, &mut dev_out));
        let mut ws = 0u64;
        syn!(synWorkspaceGetSize(&mut ws, recipe));
        let mut dws = 0u64;
        if ws > 0 {
            syn!(synDeviceMalloc(self.id, ws, 0, 0, &mut dws));
        }

        let (mut host_in, mut host_out): (*mut c_void, *mut c_void) =
            (core::ptr::null_mut(), core::ptr::null_mut());
        syn!(synHostMalloc(self.id, bytes, 0, &mut host_in));
        syn!(synHostMalloc(self.id, bytes, 0, &mut host_out));

        // SAFETY: host_in holds rows*cols bf16 elements.
        unsafe {
            let hi = host_in.cast::<u16>();
            for (i, &v) in input.iter().enumerate() {
                *hi.add(i) = f32_to_bf16(v);
            }
        }

        let mut stream: synStreamHandle = core::ptr::null_mut();
        syn!(synStreamCreateGeneric(&mut stream, self.id, 0));
        syn!(synMemCopyAsync(
            stream,
            host_in as u64,
            bytes,
            dev_in,
            SYN_HOST_TO_DRAM
        ));
        syn!(synStreamSynchronize(stream));

        let infos = [
            launch_info(&name_in, dev_in, ids[0], c, r),
            launch_info(&name_out, dev_out, ids[1], c, r),
        ];
        syn!(synLaunch(stream, infos.as_ptr(), 2, dws, recipe, 0));
        syn!(synStreamSynchronize(stream));
        syn!(synMemCopyAsync(
            stream,
            dev_out,
            bytes,
            host_out as u64,
            SYN_DRAM_TO_HOST
        ));
        syn!(synStreamSynchronize(stream));

        let mut out = vec![0.0f32; rows * cols];
        // SAFETY: host_out holds rows*cols bf16 elements just copied back.
        unsafe {
            let ho = host_out.cast::<u16>();
            for (i, o) in out.iter_mut().enumerate() {
                *o = bf16_to_f32(*ho.add(i));
            }
        }

        unsafe {
            synHostFree(self.id, host_in, 0);
            synHostFree(self.id, host_out, 0);
            synDeviceFree(self.id, dev_in, 0);
            synDeviceFree(self.id, dev_out, 0);
            if dws != 0 {
                synDeviceFree(self.id, dws, 0);
            }
            synStreamDestroy(stream);
            synRecipeDestroy(recipe);
            synGraphDestroy(graph);
        }
        Ok(out)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            synDeviceRelease(self.id);
            synDestroy();
        }
    }
}

fn launch_info(name: &CString, addr: u64, id: u64, fcd: u64, outer: u64) -> synLaunchTensorInfo {
    let mut ti = synLaunchTensorInfo {
        tensor_name: name.as_ptr(),
        tensor_address: addr,
        tensor_type: SYN_TENSOR_DATA,
        tensor_size: [0; HABANA_DIM_MAX],
        tensor_id: id,
    };
    ti.tensor_size[0] = fcd;
    ti.tensor_size[1] = outer;
    ti
}
