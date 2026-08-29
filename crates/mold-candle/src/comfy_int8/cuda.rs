// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// Mold adaptation: native Candle custom-op dispatch, retained U8 checkpoint
// storage, descriptor lifetime management, and dimension/device plan caching.

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::cublas::sys::cublasOperation_t;
use candle::cuda_backend::cudarc::cublaslt::{result as lt, sys};
use candle::cuda_backend::cudarc::driver::{
    DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PushKernelArg,
};
use candle::cuda_backend::{CudaDType, DeviceId, WrapErr};
use candle::{CpuStorage, CudaStorage, CustomOp3, DType, Layout, Result, Shape, WithDType};
use half::{bf16, f16};
use std::ffi::c_void;
use std::{cell::RefCell, collections::HashMap};

use super::NATIVE_INT8_CUBLAS_WORKSPACE_BYTES;

#[rustfmt::skip]
mod kernels {
    include!(concat!(env!("OUT_DIR"), "/comfy_int8_cuda.rs"));
}

const THREADS: u32 = 256;

struct LtPlan {
    _device: candle::CudaDevice,
    handle: sys::cublasLtHandle_t,
    operation: sys::cublasLtMatmulDesc_t,
    weight: sys::cublasLtMatrixLayout_t,
    activation: sys::cublasLtMatrixLayout_t,
    output: sys::cublasLtMatrixLayout_t,
    preference: sys::cublasLtMatmulPreference_t,
    algorithm: sys::cublasLtMatmulAlgo_t,
}

impl Drop for LtPlan {
    fn drop(&mut self) {
        // SAFETY: Every descriptor is created once by `new` and is owned by
        // this plan until drop. Cleanup failures cannot be recovered here.
        unsafe {
            let _ = lt::destroy_matmul_pref(self.preference);
            let _ = lt::destroy_matrix_layout(self.output);
            let _ = lt::destroy_matrix_layout(self.activation);
            let _ = lt::destroy_matrix_layout(self.weight);
            let _ = lt::destroy_matmul_desc(self.operation);
            let _ = lt::destroy_handle(self.handle);
        }
    }
}

impl LtPlan {
    fn new(device: candle::CudaDevice, rows: usize, columns: usize, inner: usize) -> Result<Self> {
        fn err(context: &str, error: impl std::fmt::Debug) -> candle::Error {
            candle::Error::Msg(format!("Comfy INT8 {context}: {error:?}"))
        }

        let handle = lt::create_handle().map_err(|error| err("cuBLASLt handle", error))?;
        let operation = match lt::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
            sys::cudaDataType_t::CUDA_R_32I,
        ) {
            Ok(value) => value,
            Err(error) => {
                // SAFETY: `handle` was created immediately above.
                unsafe {
                    let _ = lt::destroy_handle(handle);
                }
                return Err(err("cuBLASLt operation descriptor", error));
            }
        };

        let mut weight = std::ptr::null_mut();
        let mut activation = std::ptr::null_mut();
        let mut output = std::ptr::null_mut();
        let mut preference = std::ptr::null_mut();
        let result = (|| -> Result<sys::cublasLtMatmulAlgo_t> {
            let transa = cublasOperation_t::CUBLAS_OP_T;
            let transb = cublasOperation_t::CUBLAS_OP_N;
            // SAFETY: `operation` is live and the attribute pointers refer to
            // correctly typed stack values for the duration of each call.
            unsafe {
                lt::set_matmul_desc_attribute(
                    operation,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                    (&transa as *const cublasOperation_t).cast(),
                    std::mem::size_of_val(&transa),
                )
                .map_err(|error| err("cuBLASLt transpose A", error))?;
                lt::set_matmul_desc_attribute(
                    operation,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                    (&transb as *const cublasOperation_t).cast(),
                    std::mem::size_of_val(&transb),
                )
                .map_err(|error| err("cuBLASLt transpose B", error))?;
            }
            weight = lt::create_matrix_layout(
                sys::cudaDataType_t::CUDA_R_8I,
                inner as u64,
                columns as u64,
                inner as i64,
            )
            .map_err(|error| err("cuBLASLt weight layout", error))?;
            activation = lt::create_matrix_layout(
                sys::cudaDataType_t::CUDA_R_8I,
                inner as u64,
                rows as u64,
                inner as i64,
            )
            .map_err(|error| err("cuBLASLt activation layout", error))?;
            output = lt::create_matrix_layout(
                sys::cudaDataType_t::CUDA_R_32I,
                columns as u64,
                rows as u64,
                columns as i64,
            )
            .map_err(|error| err("cuBLASLt output layout", error))?;
            preference =
                lt::create_matmul_pref().map_err(|error| err("cuBLASLt preference", error))?;
            let workspace_bytes = NATIVE_INT8_CUBLAS_WORKSPACE_BYTES;
            // SAFETY: `preference` is live and the pointer is valid for this call.
            unsafe {
                lt::set_matmul_pref_attribute(
                    preference,
                    sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    (&workspace_bytes as *const usize).cast(),
                    std::mem::size_of_val(&workspace_bytes),
                )
                .map_err(|error| err("cuBLASLt workspace preference", error))?;
                let heuristic = lt::get_matmul_algo_heuristic(
                    handle, operation, weight, activation, output, output, preference,
                )
                .map_err(|error| err("cuBLASLt INT8 heuristic", error))?;
                Ok(heuristic.algo)
            }
        })();
        let algorithm = match result {
            Ok(value) => value,
            Err(error) => {
                // SAFETY: Only non-null descriptors were successfully created.
                unsafe {
                    if !preference.is_null() {
                        let _ = lt::destroy_matmul_pref(preference);
                    }
                    if !output.is_null() {
                        let _ = lt::destroy_matrix_layout(output);
                    }
                    if !activation.is_null() {
                        let _ = lt::destroy_matrix_layout(activation);
                    }
                    if !weight.is_null() {
                        let _ = lt::destroy_matrix_layout(weight);
                    }
                    let _ = lt::destroy_matmul_desc(operation);
                    let _ = lt::destroy_handle(handle);
                }
                return Err(error);
            }
        };
        Ok(Self {
            _device: device,
            handle,
            operation,
            weight,
            activation,
            output,
            preference,
            algorithm,
        })
    }
}

thread_local! {
    static LT_PLANS: RefCell<HashMap<(DeviceId, usize, usize, usize), LtPlan>> =
        RefCell::new(HashMap::new());
}

fn with_lt_plan<T>(
    device: &candle::CudaDevice,
    rows: usize,
    columns: usize,
    inner: usize,
    f: impl FnOnce(&LtPlan) -> Result<T>,
) -> Result<T> {
    LT_PLANS.with(|plans| {
        let mut plans = plans.borrow_mut();
        let plan = match plans.entry((device.id(), rows, columns, inner)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(LtPlan::new(device.clone(), rows, columns, inner)?)
            }
        };
        f(plan)
    })
}

#[derive(Debug)]
struct NativeInt8Linear {
    rows: usize,
    columns: usize,
    inner: usize,
    output_dtype: DType,
}

impl NativeInt8Linear {
    fn cuda_fwd_t<I, O>(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        weight: &CudaStorage,
        weight_layout: &Layout,
        scales: &CudaStorage,
        scale_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)>
    where
        I: CudaDType + DeviceRepr + WithDType,
        O: CudaDType + DeviceRepr + WithDType,
    {
        let contiguous = |layout: &Layout, label: &str| -> Result<(usize, usize)> {
            layout
                .contiguous_offsets()
                .ok_or_else(|| candle::Error::Msg(format!("Comfy INT8 {label} must be contiguous")))
        };
        let (input_start, input_end) = contiguous(input_layout, "INT8 activation")?;
        let (weight_start, weight_end) = contiguous(weight_layout, "INT8 weight")?;
        let (scale_start, scale_end) = contiguous(scale_layout, "INT8 scales")?;
        if input_layout.shape().dims() != [self.rows, self.inner]
            || weight_layout.shape().dims() != [self.columns, self.inner]
            || !matches!(scale_layout.shape().dims(), [columns, 1] if *columns == self.columns)
        {
            candle::bail!("Comfy native INT8 CUDA operand shape mismatch")
        }

        let device = input.device();
        let input = input.as_cuda_slice::<I>()?.slice(input_start..input_end);
        let weight = weight
            .as_cuda_slice::<u8>()?
            .slice(weight_start..weight_end);
        let scales = scales.as_cuda_slice::<f32>()?.slice(scale_start..scale_end);
        let stream = device.cuda_stream();
        // SAFETY: Every allocation is fully written by the queued kernel/GEMM
        // before it is read or wrapped as the returned Candle storage.
        let quantized = unsafe { stream.alloc::<i8>(self.rows * self.inner) }.w()?;
        let input_scales = unsafe { stream.alloc::<f32>(self.rows) }.w()?;
        let mut accumulator = unsafe { stream.alloc::<i32>(self.rows * self.columns) }.w()?;
        let mut workspace =
            unsafe { stream.alloc::<u8>(NATIVE_INT8_CUBLAS_WORKSPACE_BYTES) }.w()?;
        let output = unsafe { stream.alloc::<O>(self.rows * self.columns) }.w()?;

        let quantize_name = match I::DTYPE {
            DType::F32 => "h3_quantize_int8_rowwise_f32",
            DType::F16 => "h3_quantize_int8_rowwise_f16",
            DType::BF16 => "h3_quantize_int8_rowwise_bf16",
            dtype => {
                candle::bail!("Comfy native INT8 CUDA input dtype {dtype:?} is unsupported")
            }
        };
        let quantize = device.get_or_load_custom_func(
            quantize_name,
            "h3-int8-linear",
            kernels::INT8_LINEAR,
        )?;
        let mut builder = quantize.builder();
        builder.arg(&input);
        builder.arg(&quantized);
        builder.arg(&input_scales);
        candle::builder_arg!(builder, self.inner as i32);
        let cfg = LaunchConfig {
            grid_dim: (self.rows as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: Kernel arguments and launch geometry match the compiled signature.
        unsafe { builder.launch(cfg) }.w()?;

        let (activation_ptr, activation_read) = quantized.device_ptr(&stream);
        let (weight_ptr, weight_read) = weight.device_ptr(&stream);
        let (accumulator_ptr, accumulator_write) = accumulator.device_ptr_mut(&stream);
        let (workspace_ptr, workspace_write) = workspace.device_ptr_mut(&stream);
        let alpha: i32 = 1;
        let beta: i32 = 0;
        with_lt_plan(device, self.rows, self.columns, self.inner, |plan| {
            // SAFETY: The plan descriptors describe these exact contiguous
            // row-major allocations and all pointers remain live until the
            // queued operation.
            unsafe {
                lt::matmul(
                    plan.handle,
                    plan.operation,
                    (&alpha as *const i32).cast(),
                    (&beta as *const i32).cast(),
                    weight_ptr as *const c_void,
                    plan.weight,
                    activation_ptr as *const c_void,
                    plan.activation,
                    accumulator_ptr as *const c_void,
                    plan.output,
                    accumulator_ptr as *mut c_void,
                    plan.output,
                    &plan.algorithm,
                    workspace_ptr as *mut c_void,
                    NATIVE_INT8_CUBLAS_WORKSPACE_BYTES,
                    stream.cu_stream().cast(),
                )
                .map_err(|error| candle::Error::Msg(format!("Comfy INT8 cuBLASLt GEMM: {error:?}")))
            }
        })?;
        drop(workspace_write);
        drop(accumulator_write);
        drop(weight_read);
        drop(activation_read);

        let dequantize_name = match O::DTYPE {
            DType::F32 => "h3_dequantize_int8_linear_f32",
            DType::F16 => "h3_dequantize_int8_linear_f16",
            DType::BF16 => "h3_dequantize_int8_linear_bf16",
            dtype => {
                candle::bail!("Comfy native INT8 CUDA output dtype {dtype:?} is unsupported")
            }
        };
        let dequantize = device.get_or_load_custom_func(
            dequantize_name,
            "h3-int8-linear",
            kernels::INT8_LINEAR,
        )?;
        let mut builder = dequantize.builder();
        builder.arg(&accumulator);
        builder.arg(&input_scales);
        builder.arg(&scales);
        builder.arg(&output);
        candle::builder_arg!(
            builder,
            (self.rows * self.columns) as i64,
            self.columns as i32
        );
        let cfg = LaunchConfig::for_num_elems((self.rows * self.columns) as u32);
        // SAFETY: Kernel arguments and launch geometry match the compiled signature.
        unsafe { builder.launch(cfg) }.w()?;

        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            (self.rows, self.columns).into(),
        ))
    }
}

impl CustomOp3 for NativeInt8Linear {
    fn name(&self) -> &'static str {
        "comfy-native-int8-linear"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("Comfy native INT8 linear is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        weight: &CudaStorage,
        weight_layout: &Layout,
        scales: &CudaStorage,
        scale_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        macro_rules! output_dispatch {
            ($input:ty) => {
                match self.output_dtype {
                    DType::F32 => self.cuda_fwd_t::<$input, f32>(
                        input,
                        input_layout,
                        weight,
                        weight_layout,
                        scales,
                        scale_layout,
                    ),
                    DType::F16 => self.cuda_fwd_t::<$input, f16>(
                        input,
                        input_layout,
                        weight,
                        weight_layout,
                        scales,
                        scale_layout,
                    ),
                    DType::BF16 => self.cuda_fwd_t::<$input, bf16>(
                        input,
                        input_layout,
                        weight,
                        weight_layout,
                        scales,
                        scale_layout,
                    ),
                    dtype => candle::bail!(
                        "Comfy native INT8 CUDA output dtype {dtype:?} is unsupported"
                    ),
                }
            };
        }
        match input.dtype() {
            DType::F32 => output_dispatch!(f32),
            DType::F16 => output_dispatch!(f16),
            DType::BF16 => output_dispatch!(bf16),
            dtype => {
                candle::bail!("Comfy native INT8 CUDA input dtype {dtype:?} is unsupported")
            }
        }
    }
}

pub fn native_int8_linear(
    input: &candle::Tensor,
    weight: &candle::Tensor,
    scales: &candle::Tensor,
    output_dtype: DType,
) -> Result<candle::Tensor> {
    let (rows, inner) = input.dims2()?;
    let (columns, weight_inner) = weight.dims2()?;
    validate_native_int8_dimensions(rows, columns, inner, weight_inner)?;
    let activation_elements = rows
        .checked_mul(inner)
        .ok_or_else(|| candle::Error::Msg("Comfy native INT8 activation size overflows".into()))?;
    let weight_elements = columns
        .checked_mul(inner)
        .ok_or_else(|| candle::Error::Msg("Comfy native INT8 weight size overflows".into()))?;
    let output_elements = rows
        .checked_mul(columns)
        .ok_or_else(|| candle::Error::Msg("Comfy native INT8 output size overflows".into()))?;
    u32::try_from(output_elements).map_err(|_| {
        candle::Error::Msg("Comfy native INT8 output element count exceeds u32".into())
    })?;
    if input.elem_count() != activation_elements || weight.elem_count() != weight_elements {
        candle::bail!("Comfy native INT8 CUDA operand element count mismatch")
    }
    input.apply_op3_no_bwd(
        weight,
        scales,
        &NativeInt8Linear {
            rows,
            columns,
            inner,
            output_dtype,
        },
    )
}

fn validate_native_int8_dimensions(
    rows: usize,
    columns: usize,
    inner: usize,
    weight_inner: usize,
) -> Result<()> {
    if rows == 0 || columns == 0 || inner == 0 {
        candle::bail!("Comfy native INT8 CUDA dimensions must be positive")
    }
    if inner != weight_inner {
        candle::bail!("Comfy native INT8 CUDA inner dimension mismatch")
    }
    if !inner.is_multiple_of(4) || !columns.is_multiple_of(4) {
        candle::bail!("Comfy native INT8 CUDA dimensions must be multiples of four")
    }
    i32::try_from(rows)
        .map_err(|_| candle::Error::Msg("Comfy native INT8 row count exceeds i32".into()))?;
    i32::try_from(columns)
        .map_err(|_| candle::Error::Msg("Comfy native INT8 output width exceeds i32".into()))?;
    i32::try_from(inner)
        .map_err(|_| candle::Error::Msg("Comfy native INT8 inner width exceeds i32".into()))?;
    const MAX_SIGNED_INT8_PRODUCT: usize = 128 * 128;
    if inner > i32::MAX as usize / MAX_SIGNED_INT8_PRODUCT {
        candle::bail!("Comfy native INT8 dot product can overflow i32")
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_native_int8_dimensions;

    #[test]
    fn native_int8_dimensions_reject_index_and_accumulator_overflow() {
        validate_native_int8_dimensions(3, 8, 256, 256).unwrap();
        assert!(validate_native_int8_dimensions(i32::MAX as usize + 1, 8, 256, 256).is_err());
        assert!(validate_native_int8_dimensions(3, i32::MAX as usize + 1, 256, 256).is_err());
        assert!(validate_native_int8_dimensions(3, 8, 131_072, 131_072).is_err());
    }
}
