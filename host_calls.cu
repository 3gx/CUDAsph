#include <cutil.h>

#include "radixsort/radixsort.cuh"
#include "dev_solve_range.cu"
#include "dev_find_ngb.cu"
#include "dev_density_loop.cu"
#include "dev_compute_pressure.cu"
#include "dev_hydro_dots.cu"
#include "dev_add_ngb_acc.cu"
#include "sph_grav_kernel.cu"

double get_time();

extern "C"
{

  void initCUDA() {   
//    CUT_DEVICE_INIT();
  }
  int cuda_get_device() {
    int device;
    CUDA_SAFE_CALL(cudaGetDevice(&device));
    CUT_CHECK_ERROR("failed to get CUDA device!\n");
    return device;
  }
  void cuda_set_device0(int device_id) {
    if (device_id > 0) { 
      cudaSetDevice(device_id - 1);
      CUDA_SAFE_CALL(cudaSetDevice(device_id - 1));
      CUT_CHECK_ERROR("failed to set CUDA device!\n");
    }
  }

  void cuda_set_device(int device_id) {
//     if (device_id > 0) { 
//       cudaSetDevice(device_id - 1);
//       CUDA_SAFE_CALL(cudaSetDevice(device_id - 1));
//       CUT_CHECK_ERROR("failed to set CUDA device!\n");
//     }
  }
  
  void allocateCUDAarray(void** pos, int n, int device_id) {
    cuda_set_device(device_id);
    CUDA_SAFE_CALL(cudaMalloc((void**)pos, n));
    CUT_CHECK_ERROR("cudaMalloc failed!\n");
  }
  void deleteCUDAarray(void* pos, int device_id) {
    cuda_set_device(device_id);
    CUDA_SAFE_CALL(cudaFree((void*)pos));
    CUT_CHECK_ERROR("cudaFree failed!\n");
  }
  void copyArrayToDevice(void* device, const void* host, int n, int device_id) {
    cuda_set_device(device_id);
    CUDA_SAFE_CALL(cudaMemcpy(device, host, n, cudaMemcpyHostToDevice));
    CUT_CHECK_ERROR("cudaMemcpy (host->device) failed!\n");
  }
  void copyArrayFromDevice(void* host, const void* device, int n, int device_id) {   
    cuda_set_device(device_id);
    CUDA_SAFE_CALL(cudaMemcpy(host, device, n, cudaMemcpyDeviceToHost));
    CUT_CHECK_ERROR("cudaMemcpy (device->host) failed!\n");
  }
  inline void threadSync() { cudaThreadSynchronize(); }

  float host_solve_range(int n_bodies, int n_nodes,
			 float rel_err,
			 KeyValuePair *bodies_hash,
			 int    *bodies_map,
			 float4 *bodies_pos,
			 float4 *bodies_vel,
 			 float4 *kd_tree,
			 int    *n_ngb,
			 int    device_id) {
    cuda_set_device(device_id);

    

    double t1 = get_time();
    double t_begin = t1;
    
    int p = 128;
    dim3 threads(p,1,1);
    dim3 grid(n_bodies/p, 1, 1);
    
    /* set up a constant for relative error */
    CUDA_SAFE_CALL(cudaMemcpyToSymbol("rel_err", &rel_err, 
                                      sizeof(float), 0, 
                                      cudaMemcpyHostToDevice));
   
    CUDA_SAFE_CALL(cudaMemcpyToSymbol("n_nodes", &n_nodes, 
                                      sizeof(int), 0, 
                                      cudaMemcpyHostToDevice));
    
    /* assigned kd_tree to texture */
    kd_tree_tex.addressMode[0] = cudaAddressModeWrap;
    kd_tree_tex.addressMode[1] = cudaAddressModeWrap;
    kd_tree_tex.filterMode     = cudaFilterModePoint;
    kd_tree_tex.normalized     = false;
    CUDA_SAFE_CALL(cudaBindTexture(0, kd_tree_tex, kd_tree, (n_bodies+1)*sizeof(float4)));

    /* assigned kd_tree to texture */
    bodies_map_tex.addressMode[0] = cudaAddressModeWrap;
    bodies_map_tex.addressMode[1] = cudaAddressModeWrap;
    bodies_map_tex.filterMode     = cudaFilterModePoint;
    bodies_map_tex.normalized     = false;
    CUDA_SAFE_CALL(cudaBindTexture(0, bodies_map_tex, bodies_map, n_bodies*sizeof(int)));

    t1 = get_time();
    fprintf(stderr, "          solve_range ... ");
    dev_solve_range      <<<grid, threads>>> (bodies_pos, n_ngb, bodies_hash);
//     dev_get_ngb      <<<grid, threads>>> (bodies_pos, n_ngb);
    threadSync();
    fprintf(stderr, " done in %f sec\n", get_time() - t1);

    /* assigned bodies_pos to texture */
    bodies_pos_tex.addressMode[0] = cudaAddressModeWrap;
    bodies_pos_tex.addressMode[1] = cudaAddressModeWrap;
    bodies_pos_tex.filterMode     = cudaFilterModePoint;
    bodies_pos_tex.normalized     = false;
    CUDA_SAFE_CALL(cudaBindTexture(0, bodies_pos_tex, bodies_pos, n_bodies*sizeof(float4)));

    /* assigned bodies_vel to texture */
    bodies_vel_tex.addressMode[0] = cudaAddressModeWrap;
    bodies_vel_tex.addressMode[1] = cudaAddressModeWrap;
    bodies_vel_tex.filterMode     = cudaFilterModePoint;
    bodies_vel_tex.normalized     = false;
    CUDA_SAFE_CALL(cudaBindTexture(0, bodies_vel_tex, bodies_vel, n_bodies*sizeof(float4)));

    /* assigned n_ngb to texture */
    n_ngb_tex.addressMode[0] = cudaAddressModeWrap;
    n_ngb_tex.addressMode[1] = cudaAddressModeWrap;
    n_ngb_tex.filterMode     = cudaFilterModePoint;
    n_ngb_tex.normalized     = false;
    CUDA_SAFE_CALL(cudaBindTexture(0, n_ngb_tex, n_ngb, n_bodies*sizeof(int)));
    
    CUT_CHECK_ERROR("Kernel execution failed!"); 
    
    
    double dt_sph = get_time() - t_begin;
    return dt_sph;
  }
  
  float host_sph_accelerations(int n_bodies, int ngb_tot,
			       float av_alpha, 
			       float av_beta,
			       KeyValuePair *bodies_hash,
			       KeyValuePair *bodies_hash_extra,
			       int *bodies_map,
			       int *ngb_list,
			       int *ngb_offset,
			       float4 *hydro_data,
			       float4 *dots_data,
			       float4 *bodies_dots,
			       float4 *hydro_data2,
			       float *ngb_acc,
			       int device_id) {
    cuda_set_device(device_id);


    int p = 128;
    dim3 threads(p,1,1);
    dim3 grid(n_bodies/p, 1, 1);

    CUDA_SAFE_CALL(cudaMemcpyToSymbol("av_alpha", &av_alpha, 
                                      sizeof(float), 0, 
                                      cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol("av_beta", &av_beta, 
                                      sizeof(float), 0, 
                                      cudaMemcpyHostToDevice));

    double t1 = get_time();
    double t_begin = t1;
    double dt_sph = 0;

    /* assigned ngb_offset to texture */
    ngb_offset_tex.addressMode[0] = cudaAddressModeWrap;
    ngb_offset_tex.addressMode[1] = cudaAddressModeWrap;
    ngb_offset_tex.filterMode     = cudaFilterModePoint;
    ngb_offset_tex.normalized     = false;
    CUDA_SAFE_CALL(cudaBindTexture(0, ngb_offset_tex, ngb_offset, n_bodies*sizeof(int)));

    fprintf(stderr, "          find_ngb ... ");
    dev_find_ngb      <<<grid, threads>>> (ngb_list, bodies_hash);
    threadSync();
    dt_sph += get_time() - t1;
    fprintf(stderr, " done in %f sec\n", get_time() - t1);
    CUT_CHECK_ERROR("Kernel execution failed!"); 
    
    /* apply radix sort */
    t1 = get_time(); 
    fprintf(stderr, "          radix_sort ... ");
//     RadixSort(bodies_hash, bodies_hash_extra, n_bodies, 32);
    threadSync();
    dt_sph += get_time() - t1;
    fprintf(stderr, " done in %f sec\n", get_time() - t1);
    CUT_CHECK_ERROR("Kernel execution failed!"); 


    /* assigned ngb_list to texture */
    ngb_list_tex.addressMode[0] = cudaAddressModeWrap;
    ngb_list_tex.addressMode[1] = cudaAddressModeWrap;
    ngb_list_tex.filterMode     = cudaFilterModePoint;
    ngb_list_tex.normalized     = false;
    CUDA_SAFE_CALL(cudaBindTexture(0, ngb_list_tex, ngb_list, ngb_tot*sizeof(int)));
   
    t1 = get_time();
    fprintf(stderr, "          density_loop ... ");
    dev_density_loop     <<<grid, threads>>> (hydro_data, dots_data, bodies_hash);
    threadSync();
    dt_sph += get_time() - t1;
    fprintf(stderr, " done in %f sec\n", get_time() - t1);
    CUT_CHECK_ERROR("Kernel execution failed!"); 
    
    t1 = get_time();
    fprintf(stderr, "          compute_pressure ... ");
    dev_compute_pressure <<<grid, threads>>> (hydro_data);
    threadSync();
    dt_sph += get_time() - t1;
    fprintf(stderr, " done in %f sec\n", get_time() - t1);
    CUT_CHECK_ERROR("Kernel execution failed!"); 

    /* assigned hydro_data & dots_data to texture */
    hydro_data_tex.addressMode[0] = cudaAddressModeWrap;
    hydro_data_tex.addressMode[1] = cudaAddressModeWrap;
    hydro_data_tex.filterMode     = cudaFilterModePoint;
    hydro_data_tex.normalized     = false;
    CUDA_SAFE_CALL(cudaBindTexture(0, hydro_data_tex, hydro_data, n_bodies*sizeof(float4)));

    dots_data_tex.addressMode[0] = cudaAddressModeWrap;
    dots_data_tex.addressMode[1] = cudaAddressModeWrap;
    dots_data_tex.filterMode     = cudaFilterModePoint;
    dots_data_tex.normalized     = false;
    CUDA_SAFE_CALL(cudaBindTexture(0, dots_data_tex, dots_data, n_bodies*sizeof(float4)));
    

    t1 = get_time();
    fprintf(stderr, "          hydro_dots_loop ... ");
    dev_hydro_dots       <<<grid, threads>>> (bodies_dots, hydro_data2, ngb_acc, bodies_hash);
    threadSync();
    dt_sph += get_time() - t1;
    fprintf(stderr, " done in %f sec\n", get_time() - t1);
    CUT_CHECK_ERROR("Kernel execution failed!"); 
  
    /* assigned ngb_acc to texture */
    ngb_acc_tex.addressMode[0] = cudaAddressModeWrap;
    ngb_acc_tex.addressMode[1] = cudaAddressModeWrap;
    ngb_acc_tex.filterMode     = cudaFilterModePoint;
    ngb_acc_tex.normalized     = false;
    CUDA_SAFE_CALL(cudaBindTexture(0, ngb_acc_tex, ngb_acc, ngb_tot*sizeof(float)));

//     t1 = get_time();
//     fprintf(stderr, "          add_ngb_acc ... ");
// //     dev_add_ngb_acc       <<<grid, threads>>> (bodies_dots);
//     threadSync();
//     dt_sph += get_time() - t1;
//     fprintf(stderr, " done in %f sec\n", get_time() - t1);
//     CUT_CHECK_ERROR("Kernel execution failed!"); 
 

    /* unbind textures */

    CUDA_SAFE_CALL(cudaUnbindTexture(ngb_acc_tex));
    CUDA_SAFE_CALL(cudaUnbindTexture(dots_data_tex));
    CUDA_SAFE_CALL(cudaUnbindTexture(hydro_data_tex));
    CUDA_SAFE_CALL(cudaUnbindTexture(ngb_list_tex));
    CUDA_SAFE_CALL(cudaUnbindTexture(ngb_offset_tex));
    CUDA_SAFE_CALL(cudaUnbindTexture(n_ngb_tex));
    CUDA_SAFE_CALL(cudaUnbindTexture(bodies_pos_tex));
    CUDA_SAFE_CALL(cudaUnbindTexture(bodies_vel_tex));
    CUDA_SAFE_CALL(cudaUnbindTexture(kd_tree_tex));
    CUDA_SAFE_CALL(cudaUnbindTexture(bodies_map_tex));

    dt_sph = get_time() - t_begin;
    return dt_sph;
  }




  void zallocateNBodyArrays(float** pos, int numBodies, int device_id) {
    // 4 floats each for alignment reasons
    cuda_set_device(device_id);
    unsigned int memSize = sizeof( float) * 4 * numBodies;
    
    CUDA_SAFE_CALL(cudaMalloc((void**)pos, memSize));
  }
  
  void zdeleteNBodyArrays(float* pos, int device_id) {
    cuda_set_device(device_id);
    CUDA_SAFE_CALL(cudaFree((void*)pos));
  }

  void zcopyArrayFromDevice(float* host, 
			   const float* device, 
			   int numBodies, int device_id) {   
    cuda_set_device(device_id);
    CUDA_SAFE_CALL(cudaMemcpy(host, device, numBodies*4*sizeof(float), 
                              cudaMemcpyDeviceToHost));
  }
  
  void zcopyArrayToDevice(float* device, const float* host, int numBodies, int device_id) {
    cuda_set_device(device_id);
    CUDA_SAFE_CALL(cudaMemcpy(device, host, numBodies*4*sizeof(float), 
                              cudaMemcpyHostToDevice));
  }

  void compute_CUDA_gravity(int numBodies, float4 *accel, float* pos, float *vel,
			  int p, int q, int device_id) {

    cuda_set_device(device_id);

    int sharedMemSize = 2 * p * q * sizeof(float4); // 4 floats for pos

    dim3 threads(p,q,1);
    dim3 grid(numBodies/p, 1, 1);

    // execute the kernel:

    // When the numBodies / thread block size is < # multiprocessors 
    // (16 on G80), the GPU is underutilized. For example, with 256 threads per
    // block and 1024 bodies, there will only be 4 thread blocks, so the 
    // GPU will only be 25% utilized.  To improve this, we use multiple threads
    // per body.  We still can use blocks of 256 threads, but they are arranged
    // in q rows of p threads each.  Each thread processes 1/q of the forces 
    // that affect each body, and then 1/q of the threads (those with 
    // threadIdx.y==0) add up the partial sums from the other threads for that 
    // body.  To enable this, use the "--p=" and "--q=" command line options to
    // this example.  e.g.: "nbody.exe --n=1024 --p=64 --q=4" will use 4 
    // threads per body and 256 threads per block. There will be n/p = 16 
    // blocks, so a G80 GPU will be 100% utilized.
    
    // We use a bool template parameter to specify when the number of threads 
    // per body is greater than one, so that when it is not we don't have to 
    // execute the more complex code required!

//     if (threads.y == 1) {
//       get_CUDA_forces<false><<< grid, threads, sharedMemSize >>>
// 	((float4*)accel, (float4*)jerk, (float4*)pos, (float4*)vel, numBodies);
//     } else {
//       get_CUDA_forces<true><<< grid, threads, sharedMemSize >>>
// 	((float4*)accel, (float4*)jerk, (float4*)pos, (float4*)vel, numBodies);
//     }

    get_CUDA_forces<<< grid, threads, sharedMemSize >>>
      ((float4*)accel, (float4*)pos, (float4*)vel, numBodies);
    
    // check if kernel invocation generated an error
    CUT_CHECK_ERROR("Kernel execution failed");
  }


  int get_device_count() {
    initCUDA();
    
    int s_gpuCount = 0;
    
    // Enumerate GPUs.
    CUDA_SAFE_CALL(cudaGetDeviceCount(&s_gpuCount));
    
    return s_gpuCount;
  }
  

}

