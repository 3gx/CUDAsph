#include "CUDAsph.h"

void CUDAsph::allocate_host_memory() {
  kd_tree = (kd_node*)malloc((n_bodies+1)*sizeof(kd_node));
  
  hst.bodies_id  = (int*)malloc(n_dev_bodies*sizeof(int));
  hst.bodies_pos = (float4*)malloc(n_dev_bodies*sizeof(float4));
  hst.bodies_vel = (float4*)malloc(n_dev_bodies*sizeof(float4));
  hst.kd_tree    = (float4*)malloc((n_dev_bodies+1)*sizeof(float4));

  hst.bodies_dots = (float4*)malloc(n_dev_bodies*sizeof(float4));
  hst.hydro_data  = (float4*)malloc(n_dev_bodies*sizeof(float4));
  hst.dots_data   = (float4*)malloc(n_dev_bodies*sizeof(float4));
  hst.hydro_data2 = (float4*)malloc(n_dev_bodies*sizeof(float4));
  hst.grav_data   = (float4*)malloc(n_dev_bodies*sizeof(float4));

  hst.particle_map = (int*)malloc(n_dev_bodies*sizeof(int));
  hst.bodies_hash  = (KeyValuePair*)malloc(n_dev_bodies*sizeof(KeyValuePair));

  hst.n_ngb        = (int*)malloc(n_dev_bodies*sizeof(float4));
  hst.ngb_offset   = (int*)malloc(n_dev_bodies*sizeof(float4)); 

  hst.ngb_list     = (int*)  malloc(ngb_tot_max*sizeof(int)); 
  hst.ngb_acc      = (float*)malloc(ngb_tot_max*sizeof(float)); 
}

void CUDAsph::free_host_memory() {
  free(kd_tree);

  free(hst.bodies_id);
  free(hst.bodies_pos);
  free(hst.bodies_vel);
  free(hst.kd_tree);

  free(hst.bodies_dots);
  free(hst.hydro_data);
  free(hst.dots_data);
  free(hst.hydro_data2);
  free(hst.grav_data);

  free(hst.particle_map);
  free(hst.bodies_hash);

  free(hst.n_ngb);
  free(hst.ngb_offset);

  free(hst.ngb_acc);
  free(hst.ngb_list);
}

void CUDAsph::allocate_device_memory() {

  allocateCUDAarray((void**)&dev.bodies_pos, n_dev_bodies * sizeof(float4), device_id);
  allocateCUDAarray((void**)&dev.bodies_vel, n_dev_bodies * sizeof(float4), device_id);
  allocateCUDAarray((void**)&dev.kd_tree,    (n_dev_bodies+1) * sizeof(float4), device_id);

  allocateCUDAarray((void**)&dev.bodies_dots, n_dev_bodies * sizeof(float4), device_id);
  allocateCUDAarray((void**)&dev.hydro_data,  n_dev_bodies * sizeof(float4), device_id);
  allocateCUDAarray((void**)&dev.dots_data,   n_dev_bodies * sizeof(float4), device_id);
  allocateCUDAarray((void**)&dev.hydro_data2, n_dev_bodies * sizeof(float4), device_id);
  allocateCUDAarray((void**)&dev.grav_data,   n_dev_bodies * sizeof(float4), device_id);

  allocateCUDAarray((void**)&dev.particle_map,   n_dev_bodies * sizeof(float4), device_id);
  allocateCUDAarray((void**)&dev.bodies_hash,   n_dev_bodies * sizeof(KeyValuePair), device_id);
  allocateCUDAarray((void**)&dev.bodies_hash_extra,   n_dev_bodies * sizeof(KeyValuePair), device_id);

  allocateCUDAarray((void**)&dev.ngb_acc,    ngb_tot_max  * sizeof(float), device_id);
  allocateCUDAarray((void**)&dev.ngb_list,   ngb_tot_max  * sizeof(int), device_id);
  allocateCUDAarray((void**)&dev.n_ngb,      n_dev_bodies * sizeof(int), device_id);
  allocateCUDAarray((void**)&dev.ngb_offset, n_dev_bodies * sizeof(int), device_id);
}

void CUDAsph::free_device_memory() {
  deleteCUDAarray((void*)dev.bodies_pos, device_id);
  deleteCUDAarray((void*)dev.bodies_vel, device_id);
  deleteCUDAarray((void*)dev.kd_tree, device_id);

  deleteCUDAarray((void*)dev.bodies_dots, device_id);
  deleteCUDAarray((void*)dev.hydro_data, device_id);
  deleteCUDAarray((void*)dev.dots_data, device_id);
  deleteCUDAarray((void*)dev.hydro_data2, device_id);
  deleteCUDAarray((void*)dev.grav_data, device_id);

  deleteCUDAarray((void*)dev.particle_map, device_id);
  deleteCUDAarray((void*)dev.bodies_hash, device_id);
  deleteCUDAarray((void*)dev.bodies_hash_extra, device_id);

  deleteCUDAarray((void*)dev.ngb_acc, device_id);
  deleteCUDAarray((void*)dev.ngb_list, device_id);
  deleteCUDAarray((void*)dev.n_ngb, device_id);
  deleteCUDAarray((void*)dev.ngb_offset, device_id);
}
