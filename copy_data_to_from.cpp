#include "CUDAsph.h"


void CUDAsph::copy_kd_tree_to_device() {
  copyArrayToDevice((void*)dev.kd_tree, hst.kd_tree, (n_dev_bodies+1) * sizeof(float4), device_id);
}
void CUDAsph::copy_particle_map_to_device() {
  copyArrayToDevice((void*)dev.particle_map, hst.particle_map, (n_dev_bodies+1) * sizeof(int), device_id);
}
void CUDAsph::copy_bodies_hash_to_device() {
  copyArrayToDevice((void*)dev.bodies_hash, hst.bodies_hash, n_dev_bodies * sizeof(KeyValuePair), device_id);
}

void CUDAsph::copy_bodies_pos_vel_to_device() {
  copyArrayToDevice((void*)dev.bodies_pos, hst.bodies_pos, n_dev_bodies * sizeof(float4), device_id);
  copyArrayToDevice((void*)dev.bodies_vel, hst.bodies_vel, n_dev_bodies * sizeof(float4), device_id);
}
void CUDAsph::copy_hydro_data_to_device() {
  copyArrayToDevice((void*)dev.hydro_data, hst.hydro_data, n_dev_bodies * sizeof(float4), device_id);
}
void CUDAsph::copy_bodies_pos_from_device() {
  copyArrayFromDevice((void*)hst.bodies_pos, dev.bodies_pos, n_dev_bodies * sizeof(float4), device_id);
}

void CUDAsph::copy_n_ngb_from_device() {
  copyArrayFromDevice((void*)hst.n_ngb, dev.n_ngb, n_dev_bodies * sizeof(int), device_id);
}
void CUDAsph::copy_ngb_offset_to_device() {
  copyArrayToDevice((void*)dev.ngb_offset, hst.ngb_offset, n_dev_bodies * sizeof(int), device_id);
}


void CUDAsph::copy_bodies_dots_from_device() {
  copyArrayFromDevice((void*)hst.bodies_dots, dev.bodies_dots, n_dev_bodies * sizeof(float4), device_id);
}
void CUDAsph::copy_hydro_data_from_device() {
  copyArrayFromDevice((void*)hst.hydro_data, dev.hydro_data, n_dev_bodies * sizeof(float4), device_id);
}
void CUDAsph::copy_hydro_data2_from_device() {
  copyArrayFromDevice((void*)hst.hydro_data2, dev.hydro_data2, n_dev_bodies * sizeof(float4), device_id);
}

void CUDAsph::copy_ngb_list_from_device() {
  copyArrayFromDevice((void*)hst.ngb_list, dev.ngb_list, ngb_tot * sizeof(int), device_id);
}
void CUDAsph::copy_ngb_acc_from_device() {
  copyArrayFromDevice((void*)hst.ngb_acc, dev.ngb_acc, ngb_tot * sizeof(float), device_id);
}

