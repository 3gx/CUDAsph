#include "CUDAsph.h"

float CUDAsph::compute_timestep() {
  float const cn_1 = 0.3;
  float const cn_2 = 0.2;

  double dt_min = 1.0e30;
  double dt_vel_min = dt_min;
  double dt_eth_min = dt_min;
  for (int i = 0; i < n_bodies; i++) {
    float v_sig = hst.hydro_data2[i].x;

    double dt_vel = cn_1 * fabs(hst.bodies_pos[i].w/v_sig);
    double dt_eth = cn_2 * fabs(hst.hydro_data[i].z / hst.bodies_dots[i].w);
    
    dt_vel_min = min(dt_vel_min, dt_vel);
    dt_eth_min = min(dt_eth_min, dt_eth);
    dt_min = min(dt_min, 1.0/(1.0/dt_vel + 1.0/dt_eth));
  }

  fprintf(stderr, " DTs:  dt_vel= %g  dt_eth= %g  dt= %g\n",
	  dt_vel_min, dt_eth_min, dt_min);
  
  return dt_min;
}
