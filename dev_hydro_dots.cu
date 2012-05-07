#ifndef _DEV_HYDRO_DOTS_
#define _DEV_HYDRO_DOTS_

#include <stdio.h>
#include <math.h>
#include "dev_textures.h"
#include "dev_sph_kernels.cu"

#include "dev_art_visc.cu"

#ifndef POW2
#define POW2(x) (x*x)
#endif

#ifndef POW5
#define POW5(x) (x*x*x*x*x)
#endif


struct dots_loop_data {
  float vxdot, vydot, vzdot;
  float u_dot, v_sig;
};

__device__ dots_loop_data compute_dots(int n_ngb, int offset,
				       float4 body_pos,
				       float4 body_vel,
				       float4 hydro_data,
				       float4 dots_data,
				       float  *ngb_acc) {

  dots_loop_data data = {0, 0, 0, 0, 0};
  for (int i = offset; i < offset + n_ngb; i++) {
    
    int j = tex1Dfetch(ngb_list_tex, i);
    float4 body_pos_j   = tex1Dfetch(bodies_pos_tex, j);
    float4 body_vel_j   = tex1Dfetch(bodies_vel_tex, j); 
    float4 hydro_data_j = tex1Dfetch(hydro_data_tex, j);
    float4 dots_data_j  = tex1Dfetch(dots_data_tex, j);
    
    float3 dr;
    dr.x = body_pos.x - body_pos_j.x;
    dr.y = body_pos.y - body_pos_j.y;
    dr.z = body_pos.z - body_pos_j.z;

    float3 dv;
    dv.x = body_vel.x - body_vel_j.x;
    dv.y = body_vel.y - body_vel_j.y;
    dv.z = body_vel.z - body_vel_j.z;
    
    /* do operations on the neighbours */
    float ds = sqrtf(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);
    float u = ds/(0.5*body_pos.w);
 
    float drdv = dr.x*dv.x + dr.y*dv.y + dr.z*dv.z;
    
    float2 art_visc  = dev_art_visc(body_pos.w, body_pos_j.w, 
				    ds, drdv, 
				    hydro_data, dots_data,
				    hydro_data_j, dots_data_j);
    
    data.v_sig = max(data.v_sig, art_visc.y);

    float acc = 
      (hydro_data.y/POW2(hydro_data.x) +
       0.5 * art_visc.x) * dw(u)/POW5(0.5*body_pos.w);
    
    acc -= (hydro_data.y/POW2(hydro_data.x) * dots_data.z +
	    0.5 * dots_data.w) / (body_vel_j.w) * dig(u)/POW2(0.5*body_pos.w);

    data.vxdot -= body_vel_j.w * acc * dr.x;
    data.vydot -= body_vel_j.w * acc * dr.y;
    data.vzdot -= body_vel_j.w * acc * dr.z;
    ngb_acc[i]  = body_vel.w * acc;

    float udot = 
      body_vel_j.w * (hydro_data.y/POW2(hydro_data.x)
		      + 0.5 * art_visc.x) * dw(u)/POW5(0.5*body_pos.w);
    udot -= hydro_data.y/POW2(hydro_data.x) * dots_data.z * dig(u)/POW2(0.5*body_pos.w);
    data.u_dot += udot * drdv;
  }

  return data;
}

__global__ void dev_hydro_dots(float4 *bodies_dots,
			       float4 *hydro_data2,
			       float  *ngb_acc,
			       KeyValuePair *bodies_hash) {
 
//   int index = tex1Dfetch(bodies_map_tex, blockIdx.x * blockDim.x + threadIdx.x);
//  int index = blockIdx.x * blockDim.x + threadIdx.x; 
  int index   = blockIdx.x * blockDim.x + threadIdx.x; 
  index       = bodies_hash[index].value;
  
  float4 body_pos   = tex1Dfetch(bodies_pos_tex, index);
  float4 body_vel   = tex1Dfetch(bodies_vel_tex, index);
  float4 hydro_data = tex1Dfetch(hydro_data_tex, index);
  float4 dots_data  = tex1Dfetch(dots_data_tex, index);
  int n_ngb         = tex1Dfetch(n_ngb_tex, index);
  int offset        = tex1Dfetch(ngb_offset_tex, index);
 

  dots_loop_data data = compute_dots(n_ngb, offset,
				     body_pos, body_vel,
				     hydro_data, dots_data,
				     ngb_acc);

  bodies_dots[index] = make_float4(data.vxdot, data.vydot, data.vzdot, data.u_dot);
  float4 hd2 = hydro_data2[index];
  hd2.x = data.v_sig;
  hydro_data2[index] = hd2;
}

#endif
