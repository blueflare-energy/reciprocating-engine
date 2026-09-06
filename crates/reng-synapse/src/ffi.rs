//! Raw FFI declarations for the Intel Gaudi SynapseAI graph C API (subset).
//!
//! Types and signatures mirror `synapse_api.h` / `synapse_common_types.h` from
//! the Intel Gaudi 1.19.0 stack; only the entry points needed for a matmul are
//! declared. `synGEMMParams` is C++-only upstream, so it is redeclared here as
//! a `repr(C)` struct with the same two-byte layout. The HCCL collectives
//! (`hccl.h`, checked against the 1.24.1 headers) follow at the end.
#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code
)]

use core::ffi::{c_char, c_int, c_uint, c_void};

pub type synStatus = c_int;
pub type synDeviceId = u32;
pub type synGraphHandle = *mut c_void;
pub type synRecipeHandle = *mut c_void;
pub type synStreamHandle = *mut c_void;
pub type synSectionHandle = *mut c_void;
pub type synTensor = *mut c_void;
pub type synEventHandle = *mut c_void;
pub type synNodeId = u64;

pub const SYN_SUCCESS: synStatus = 0;
pub const SYN_DEVICE_GAUDI2: c_int = 4;
pub const SYN_TYPE_BF16: c_int = 1 << 1;
pub const SYN_TYPE_F32: c_int = 1 << 2; // syn_type_single
pub const SYN_TYPE_INT32: c_int = 1 << 4; // syn_type_int32
/// `syn_type_fp8_143` (8192): 1 sign, 4 exponent, 3 mantissa bits, the
/// vendor's `hf8`.
pub const SYN_TYPE_FP8_143: c_int = 1 << 13;
/// `syn_type_fp8_152` (16384): 1 sign, 5 exponent, 2 mantissa bits, the
/// vendor's `f8`.
pub const SYN_TYPE_FP8_152: c_int = 1 << 14;
pub const SYN_TENSOR_DATA: c_int = 0; // DATA_TENSOR
pub const SYN_GEOMETRY_SIZES: c_int = 1; // synGeometryMaxSizes
pub const SYN_HOST_TO_DRAM: c_int = 0;
pub const SYN_DRAM_TO_HOST: c_int = 1;
pub const SYN_DRAM_TO_DRAM: c_int = 2;
/// synStatus for a second synInitialize in one process.
pub const SYN_OBJECT_ALREADY_INITIALIZED: synStatus = 5;
pub const HABANA_DIM_MAX: usize = 25;

/// `synQuantizationProperty`: which quantization record
/// `synTensorSetQuantizationData` is given. Only the floating-point one is
/// used here; the MME reads the exponent bias out of it and ignores the
/// integer records for an fp8 tensor.
pub const SYN_QUANT_DYNAMIC_RANGE: c_int = 0;
pub const SYN_QUANT_METADATA: c_int = 1;
pub const SYN_FP_QUANT_METADATA: c_int = 2;
pub const SYN_QUANT_FLAGS: c_int = 3;
pub const SYN_QUANT_PC_DYNAMIC_RANGE: c_int = 4;

/// `synFpQuantParam { double scale; unsigned expBias; }`, 16 bytes with its
/// trailing padding. On Gaudi2 the plain `gemm` path honours `expBias`
/// (one of 3, 7, 11, 15 for E4M3) and ignores `scale`.
#[repr(C)]
pub struct synFpQuantParam {
    pub scale: f64,
    pub expBias: c_uint,
}

/// `synFpQuantMetadata { synDataType dataType; synFpQuantParam*
/// fpQuantParams; unsigned numFpQuantParams; }`, 24 bytes with its
/// padding. `numFpQuantParams` above 1 asks for per-channel quantization,
/// which the plain `gemm` path ignores.
#[repr(C)]
pub struct synFpQuantMetadata {
    pub dataType: c_int,
    pub fpQuantParams: *const synFpQuantParam,
    pub numFpQuantParams: c_uint,
}

#[repr(C)]
pub struct synTensorGeometry {
    pub sizes: [u64; HABANA_DIM_MAX],
    pub dims: u32,
}

#[repr(C)]
pub struct synGEMMParams {
    pub transpose_a: bool,
    pub transpose_b: bool,
}

/// ns_Softmax::Params { int dim; } — the axis (FCD-first) to softmax over.
#[repr(C)]
pub struct synSoftmaxParams {
    pub dim: c_int,
}

#[repr(C)]
pub struct synLaunchTensorInfo {
    pub tensor_name: *const c_char,
    pub tensor_address: u64,
    pub tensor_type: c_int,
    pub tensor_size: [u64; HABANA_DIM_MAX],
    pub tensor_id: u64,
}

unsafe extern "C" {
    pub fn synInitialize() -> synStatus;
    pub fn synDestroy() -> synStatus;
    pub fn synDeviceAcquireByDeviceType(
        pDeviceId: *mut synDeviceId,
        deviceType: c_int,
    ) -> synStatus;
    pub fn synDeviceAcquireByModuleId(pDeviceId: *mut synDeviceId, moduleId: u32) -> synStatus;
    pub fn synDeviceRelease(deviceId: synDeviceId) -> synStatus;
    pub fn synDeviceMalloc(
        deviceId: synDeviceId,
        size: u64,
        reqAddr: u64,
        flags: u32,
        buffer: *mut u64,
    ) -> synStatus;
    pub fn synDeviceFree(deviceId: synDeviceId, buffer: u64, flags: u32) -> synStatus;
    pub fn synHostMalloc(
        deviceId: synDeviceId,
        size: u64,
        flags: u32,
        buffer: *mut *mut c_void,
    ) -> synStatus;
    pub fn synHostFree(deviceId: synDeviceId, buffer: *mut c_void, flags: u32) -> synStatus;
    pub fn synStreamCreateGeneric(
        pStreamHandle: *mut synStreamHandle,
        deviceId: synDeviceId,
        flags: u32,
    ) -> synStatus;
    pub fn synStreamDestroy(streamHandle: synStreamHandle) -> synStatus;
    pub fn synStreamSynchronize(streamHandle: synStreamHandle) -> synStatus;
    pub fn synDeviceSynchronize(deviceId: synDeviceId) -> synStatus;
    pub fn synEventCreate(
        pEventHandle: *mut synEventHandle,
        deviceId: synDeviceId,
        flags: u32,
    ) -> synStatus;
    pub fn synEventDestroy(eventHandle: synEventHandle) -> synStatus;
    pub fn synEventRecord(eventHandle: synEventHandle, streamHandle: synStreamHandle) -> synStatus;
    pub fn synStreamWaitEvent(
        streamHandle: synStreamHandle,
        eventHandle: synEventHandle,
        flags: u32,
    ) -> synStatus;
    pub fn synEventSynchronize(eventHandle: synEventHandle) -> synStatus;
    pub fn synEventQuery(eventHandle: synEventHandle) -> synStatus;
    pub fn synMemsetD32Async(
        pDeviceMem: u64,
        value: u32,
        numOfElements: usize,
        streamHandle: synStreamHandle,
    ) -> synStatus;
    pub fn synTensorSetExternal(tensor: synTensor, isExternal: bool) -> synStatus;
    pub fn synEventMapTensor(
        eventHandles: *mut synEventHandle,
        numOfEvents: usize,
        launchTensorsInfo: *const synLaunchTensorInfo,
        recipeHandle: synRecipeHandle,
    ) -> synStatus;
    pub fn synLaunchWithExternalEvents(
        streamHandle: synStreamHandle,
        launchTensorsInfoExt: *const synLaunchTensorInfo,
        numberOfTensors: u32,
        pWorkspace: u64,
        pRecipeHandle: synRecipeHandle,
        eventHandleList: *mut synEventHandle,
        numberOfEvents: u32,
        flags: u32,
    ) -> synStatus;
    pub fn synMemCopyAsync(
        streamHandle: synStreamHandle,
        src: u64,
        size: u64,
        dst: u64,
        direction: c_int,
    ) -> synStatus;
    pub fn synMemCopyAsyncMultiple(
        streamHandle: synStreamHandle,
        src: *const u64,
        size: *const u64,
        dst: *const u64,
        direction: c_int,
        numCopies: u64,
    ) -> synStatus;
    pub fn synGraphCreate(pGraphHandle: *mut synGraphHandle, deviceType: c_int) -> synStatus;
    pub fn synGraphDestroy(graphHandle: synGraphHandle) -> synStatus;
    pub fn synRecipeDestroy(recipeHandle: synRecipeHandle) -> synStatus;
    pub fn synRecipeSerialize(
        recipeHandle: synRecipeHandle,
        recipeFileName: *const c_char,
    ) -> synStatus;
    pub fn synRecipeDeSerialize(
        pRecipeHandle: *mut synRecipeHandle,
        recipeFileName: *const c_char,
    ) -> synStatus;
    pub fn synDriverGetVersion(pDriverVersion: *mut c_char, len: c_int) -> synStatus;
    pub fn synTensorGetName(tensor: synTensor, size: u64, name: *mut c_char) -> synStatus;
    pub fn synSectionDestroy(sectionHandle: synSectionHandle) -> synStatus;
    pub fn synGraphCompile(
        pRecipeHandle: *mut synRecipeHandle,
        graphHandle: synGraphHandle,
        pRecipeName: *const c_char,
        pBuildLog: *const c_char,
    ) -> synStatus;
    pub fn synSectionCreate(
        sectionHandle: *mut synSectionHandle,
        sectionDescriptor: u64,
        graph: synGraphHandle,
    ) -> synStatus;
    pub fn synSectionSetPersistent(
        sectionHandle: synSectionHandle,
        sectionIsPersistent: bool,
    ) -> synStatus;
    pub fn synTensorHandleCreate(
        tensor: *mut synTensor,
        graph: synGraphHandle,
        tensorType: c_int,
        tensorName: *const c_char,
    ) -> synStatus;
    pub fn synTensorAssignToSection(
        tensor: synTensor,
        section: synSectionHandle,
        byteOffset: u64,
    ) -> synStatus;
    pub fn synTensorSetGeometry(
        tensor: synTensor,
        geometry: *const synTensorGeometry,
        geometryType: c_int,
    ) -> synStatus;
    pub fn synTensorSetDeviceDataType(tensor: synTensor, deviceDataType: c_int) -> synStatus;
    pub fn synTensorSetQuantizationData(
        tensor: synTensor,
        prop: c_int,
        propVal: *mut c_void,
        propSize: u64,
    ) -> synStatus;
    pub fn synNodeCreate(
        graphHandle: synGraphHandle,
        pInputsTensorList: *const synTensor,
        pOutputsTensorList: *const synTensor,
        numberInputs: u32,
        numberOutputs: u32,
        pUserParams: *const c_void,
        paramsSize: u32,
        pGuid: *const c_char,
        pName: *const c_char,
        inputLayouts: *const *const c_char,
        outputLayouts: *const *const c_char,
    ) -> synStatus;
    pub fn synNodeCreateWithId(
        graphHandle: synGraphHandle,
        pInputsTensorList: *const synTensor,
        pOutputsTensorList: *const synTensor,
        numberInputs: u32,
        numberOutputs: u32,
        pUserParams: *const c_void,
        paramsSize: u32,
        pGuid: *const c_char,
        pName: *const c_char,
        nodeUniqueId: *mut synNodeId,
        inputLayouts: *const *const c_char,
        outputLayouts: *const *const c_char,
    ) -> synStatus;
    pub fn synNodeDependencySet(
        graphHandle: synGraphHandle,
        pBlockingNodesIdList: *const synNodeId,
        pBlockedNodesIdList: *const synNodeId,
        numberblocking: u32,
        numberblocked: u32,
    ) -> synStatus;
    pub fn synTensorRetrieveIds(
        pRecipeHandle: synRecipeHandle,
        tensorNames: *const *const c_char,
        tensorIds: *mut u64,
        numOfTensors: u32,
    ) -> synStatus;
    pub fn synWorkspaceGetSize(
        pWorkspaceSize: *mut u64,
        recipeHandle: synRecipeHandle,
    ) -> synStatus;
    pub fn synLaunch(
        streamHandle: synStreamHandle,
        launchTensorsInfo: *const synLaunchTensorInfo,
        numberOfTensors: u32,
        pWorkspace: u64,
        pRecipeHandle: synRecipeHandle,
        flags: u32,
    ) -> synStatus;
    pub fn synDeviceGetCount(pCount: *mut u32) -> synStatus;
    pub fn synDeviceGetModuleIDs(pDeviceModuleIds: *mut u32, size: *mut u32) -> synStatus;
    pub fn synEventElapsedTime(
        pNanoSeconds: *mut u64,
        eventHandleStart: synEventHandle,
        eventHandleEnd: synEventHandle,
    ) -> synStatus;
}

unsafe extern "C" {
    /// libc `_exit`: terminate without atexit handlers or destructors.
    pub fn _exit(status: c_int) -> !;
    /// libc `signal`: install `handler` for signal `sig`. Returns the
    /// previous handler (or `SIG_ERR`), which the engine does not need.
    pub fn signal(sig: c_int, handler: extern "C" fn(c_int)) -> usize;
    /// libc `raise`: send `sig` to this process (used to check the
    /// coordinator's handler).
    pub fn raise(sig: c_int) -> c_int;
}

/// `SIGINT`, the terminal's interrupt (Linux `signal.h`).
pub const SIGINT: c_int = 2;
/// `SIGTERM`, the default kill (Linux `signal.h`).
pub const SIGTERM: c_int = 15;

/// `synEventCreate` flag: the event carries a device timestamp, so a pair
/// of them gives `synEventElapsedTime` (`synapse_api_types.h`:
/// `enum eventCreateFlags {EVENT_COLLECT_TIME = 1}`).
pub const EVENT_COLLECT_TIME: u32 = 1;

// HCCL (`hccl.h` / `hccl_types.h`, HCCL 2.6.4 in SynapseAI 1.24.1). The
// symbols are exported by libSynapse.so (no libhccl.so on the stack) and
// forwarded to libhcl.so; `stream_handle` is a `synStreamHandle`; sendbuff and
// recvbuff are device addresses from `synDeviceMalloc`.
pub type hcclComm_t = *mut c_void;
pub type hcclResult_t = c_int;
pub type hcclDataType_t = c_int;
pub type hcclRedOp_t = c_int;
pub const HCCL_UNIQUE_ID_MAX_BYTES: usize = 1024;

/// `hcclUniqueId`: 1032 bytes, passed BY VALUE to `hcclCommInitRank` (a
/// memory-class aggregate under the SysV ABI: copied onto the stack).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct hcclUniqueId {
    pub internal: [u8; HCCL_UNIQUE_ID_MAX_BYTES],
    pub length: usize,
}

pub const hcclSuccess: hcclResult_t = 0;
pub const hcclPortDown: hcclResult_t = 16;
pub const hcclSum: hcclRedOp_t = 0;
pub const hcclMin: hcclRedOp_t = 2;
pub const hcclMax: hcclRedOp_t = 3;
pub const hcclInt32: hcclDataType_t = 2;
pub const hcclFloat32: hcclDataType_t = 7;
pub const hcclBfloat16: hcclDataType_t = 9;

unsafe extern "C" {
    pub fn hcclGetVersion(version: *mut c_int) -> hcclResult_t;
    pub fn hcclGetUniqueId(uniqueId: *mut hcclUniqueId) -> hcclResult_t;
    pub fn hcclCommInitRank(
        comm: *mut hcclComm_t,
        nranks: c_int,
        commId: hcclUniqueId,
        rank: c_int,
    ) -> hcclResult_t;
    pub fn hcclCommFinalize(comm: hcclComm_t) -> hcclResult_t;
    pub fn hcclCommDestroy(comm: hcclComm_t) -> hcclResult_t;
    pub fn hcclCommAbort(comm: hcclComm_t) -> hcclResult_t;
    pub fn hcclGetErrorString(result: hcclResult_t) -> *const c_char;
    pub fn hcclGetLastErrorMessage() -> *const c_char;
    pub fn hcclCommGetAsyncError(comm: hcclComm_t, asyncError: *mut hcclResult_t) -> hcclResult_t;
    pub fn hcclCommGetAsyncErrorMessage(comm: hcclComm_t) -> *const c_char;
    pub fn hcclCommCount(comm: hcclComm_t, count: *mut c_int) -> hcclResult_t;
    pub fn hcclCommSynDevice(comm: hcclComm_t, device: *mut c_int) -> hcclResult_t;
    pub fn hcclCommUserRank(comm: hcclComm_t, rank: *mut c_int) -> hcclResult_t;
    pub fn hcclAllReduce(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: hcclDataType_t,
        reduceOp: hcclRedOp_t,
        comm: hcclComm_t,
        stream_handle: synStreamHandle,
    ) -> hcclResult_t;
    pub fn hcclAllGather(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        sendcount: usize,
        datatype: hcclDataType_t,
        comm: hcclComm_t,
        stream_handle: synStreamHandle,
    ) -> hcclResult_t;
    pub fn hcclReduceScatter(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        recvcount: usize,
        datatype: hcclDataType_t,
        reduceOp: hcclRedOp_t,
        comm: hcclComm_t,
        stream_handle: synStreamHandle,
    ) -> hcclResult_t;
    pub fn hcclBroadcast(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: hcclDataType_t,
        root: c_int,
        comm: hcclComm_t,
        stream_handle: synStreamHandle,
    ) -> hcclResult_t;
    pub fn hcclReduce(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: hcclDataType_t,
        reduceOp: hcclRedOp_t,
        root: c_int,
        comm: hcclComm_t,
        stream_handle: synStreamHandle,
    ) -> hcclResult_t;
    pub fn hcclSend(
        sendbuff: *const c_void,
        count: usize,
        datatype: hcclDataType_t,
        peer: c_int,
        comm: hcclComm_t,
        stream: synStreamHandle,
    ) -> hcclResult_t;
    pub fn hcclRecv(
        recvbuff: *mut c_void,
        count: usize,
        datatype: hcclDataType_t,
        peer: c_int,
        comm: hcclComm_t,
        stream: synStreamHandle,
    ) -> hcclResult_t;
    pub fn hcclBarrier(comm: hcclComm_t, stream_handle: synStreamHandle) -> hcclResult_t;
    pub fn hcclGroupStart() -> hcclResult_t;
    pub fn hcclGroupEnd() -> hcclResult_t;
}
