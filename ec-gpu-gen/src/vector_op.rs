use ec_gpu::GpuName;
use ff::PrimeField;
use rust_gpu_tools::{program_closures, Program};
use crate::error::EcResult;
#[cfg(feature = "cuda")]
use rust_gpu_tools::cuda;
#[cfg(feature = "opencl")]
use rust_gpu_tools::opencl;

/// Provide a cuda/opencl GPU script source to builder
/// Subtitute the field with real one
pub fn source<F: GpuName>() -> String {
    format!(include_str!("cl/vector.cl"), field = F::name())
}

/// Kernel name
pub fn kernel_name<F: GpuName>() -> String {
    format!("{}_pointwise_add", F::name())
}

const LOCAL_WORK_SIZE: usize = 256;
#[cfg(feature = "cuda")]
/// Single add
pub fn addv_single_op<F, KA>(program: &Program, elem: KA, n: usize)-> EcResult<Vec<F>> 
    where F: PrimeField + GpuName, KA: cuda::KernelArgument 
{ 
    let local_work_size = LOCAL_WORK_SIZE;
    let global_work_size = (n as f64 / local_work_size as f64).ceil() as usize;
    let kernal_name = kernel_name::<F>();

    let closures = program_closures!(|program, _args:()| -> EcResult<Vec<F>> {
        let kernal = program.create_kernel(&kernal_name, global_work_size, local_work_size).unwrap();
        let mut acc = F::ZERO;
        let vec = (0..n).map(|_| {
            acc = acc + F::ONE;
            acc
        }).collect::<Vec<F>>();
        let gpu_buffer = program.create_buffer_from_slice(vec.as_slice()).unwrap();

        kernal
            .arg(&elem)
            .arg(&gpu_buffer)
            .arg(&(n as u32))
            .run()
            .unwrap();

        let mut res = vec![F::ZERO; n];
        program.read_into_buffer(&gpu_buffer, &mut res).unwrap();
        Ok(res.to_vec())
    });

    let res = program.run(closures, ()).unwrap();
    Ok(res)
}