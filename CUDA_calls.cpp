#include "CUDAsph.h"

float CUDAsph::CUDA_solve_range() {
  float rel_err = 1.0e-3;
  return host_solve_range(n_dev_bodies, n_bodies, rel_err,
			  dev.bodies_hash,
			  dev.particle_map,
			  dev.bodies_pos, dev.bodies_vel,
			  dev.kd_tree, dev.n_ngb, device_id);
}

float CUDAsph::CUDA_sph_accelerations() {
  float av_alpha = 1.0;
  float av_beta  = 2.0;
  return host_sph_accelerations(n_dev_bodies, ngb_tot, 
				av_alpha, av_beta,
				dev.bodies_hash, dev.bodies_hash_extra,
				dev.particle_map,
				dev.ngb_list,
				dev.ngb_offset,
				dev.hydro_data,
				dev.dots_data,
				dev.bodies_dots,
				dev.hydro_data2,
				dev.ngb_acc, device_id);
}
