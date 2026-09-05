//! Raw FFI declarations for the Intel Gaudi SynapseAI graph C API (subset).
//!
//! Types and signatures mirror `synapse_api.h` / `synapse_common_types.h` from
//! the Intel Gaudi 1.19.0 stack; only the entry points needed for a matmul are
//! declared. `synGEMMParams` is C++-only upstream, so it is redeclared here as
//! a `repr(C)` struct with the same two-byte layout.
#![allow(non_camel_case_types, non_snake_case, dead_code)]

use core::ffi::{c_char, c_int, c_void};

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
pub const SYN_TENSOR_DATA: c_int = 0; // DATA_TENSOR
pub const SYN_GEOMETRY_SIZES: c_int = 1; // synGeometryMaxSizes
pub const SYN_HOST_TO_DRAM: c_int = 0;
pub const SYN_DRAM_TO_HOST: c_int = 1;
pub const SYN_DRAM_TO_DRAM: c_int = 2;
/// synStatus for a second synInitialize in one process.
pub const SYN_OBJECT_ALREADY_INITIALIZED: synStatus = 5;
pub const HABANA_DIM_MAX: usize = 25;

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
    pub fn synGraphCreate(pGraphHandle: *mut synGraphHandle, deviceType: c_int) -> synStatus;
    pub fn synGraphDestroy(graphHandle: synGraphHandle) -> synStatus;
    pub fn synRecipeDestroy(recipeHandle: synRecipeHandle) -> synStatus;
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
}
