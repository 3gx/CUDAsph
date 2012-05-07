#ifndef _DEV_PREDICT_CU_
#define _DEV_PREDICT_CU_

#include "dev_textures.h"

__global__ void dev_predictor(int dt,
			      float4 *bodies_pos,
			      float4 *bodies_vel,
			      float4 *hydro_data,
			      float4 *bodies_dots,
			      float4 *grav_data,
			      float4 *bodies_old_vel) {


  int index = blockIdx.x * blockDim.x + threadIdx.x; 

  float4 body_pos  = bodies_pos[index];
  float4 body_vel  = bodies_vel[index];
  float4 body_dots = bodies_dots[index];
  float4 hydro_data = bodies_hydro_data[index];

  float4 old_vel = {b

  float dth = 0.5*dt;
  body_pos.x += bodies_vel.x * dt;
  body_pos.y += bodies_vel.y * dt;
  body_pos.z += bodies_vel.z * dt;
  
  
  float4 body_pos   = tex1Dfetch(bodies_pos_tex, index);
  int n_ngb         = tex1Dfetch(n_ngb_tex, index);
  int offset        = tex1Dfetch(ngb_offset_tex, index);

}

#endif //
