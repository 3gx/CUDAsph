#ifndef _DEV_DENSITY_LOOP_
#define _DEV_DENSITY_LOOP_

#include <stdio.h>
#include <math.h>
#include "dev_textures.h"
#include "dev_sph_kernels.cu"

#ifndef POW2
#define POW2(x) (x*x)
#endif

#ifndef POW3
#define POW3(x) (x*x*x)
#endif

#ifndef POW4
#define POW4(x) (x*x*x*x)
#endif

#ifndef POW5
#define POW5(x) (x*x*x*x*x)
#endif

struct density_loop_data {
  float density, omega, Psi, div_v, curl_v;
};

__device__ density_loop_data compute_density(int n_ngb, int offset,
					     float4 body_pos,
					     float4 body_vel) {
  

  density_loop_data data = {0, 0, 0, 0, 0};   // at this moment, curl_v is Omega
  float3 curl_v          = {0, 0, 0};

//   fprintf(stderr, "i= %d n_ngb= %d offset= %d\n",
// 	  blockIdx.x * blockDim.x + threadIdx.x, n_ngb, offset);

  for (int i = offset; i < offset+n_ngb; i++) {
    
    int j = tex1Dfetch(ngb_list_tex, i);
    float4 body_pos_j = tex1Dfetch(bodies_pos_tex, j);
    float4 body_vel_j = tex1Dfetch(bodies_vel_tex, j);
    
    float3 dr;
    dr.x = body_pos.x - body_pos_j.x;
    dr.y = body_pos.y - body_pos_j.y;
    dr.z = body_pos.z - body_pos_j.z;
   
    /* do operations on the neighbours */

    float u = 2 * sqrtf(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z)/body_pos.w;

    data.density += body_vel_j.w *      w(u);
    data.omega   += body_vel_j.w *   dwdh(u);
    data.Psi     += body_vel_j.w * dphidh(u);
    data.curl_v  -= digdh(u);
    
    float4 dv;
    dv.x = body_vel.x - body_vel_j.x;
    dv.y = body_vel.y - body_vel_j.y;
    dv.z = body_vel.z - body_vel_j.z;
    dv.w = body_vel.w * dw(u);
    
    data.div_v += dv.w * (dr.x*dv.x + dr.y*dv.y + dr.z*dv.z);
    curl_v.x   += dv.w * (dr.y*dv.z - dr.z*dv.y);
    curl_v.y   += dv.w * (dr.z*dv.x - dr.x*dv.z);
    curl_v.z   += dv.w * (dr.x*dv.y - dr.y*dv.x);
  }
  
  float h = 0.5 * body_pos.w;

  if (data.curl_v != 0) {
    data.omega   *= 1.0/POW3(h)/data.curl_v;
    data.Psi     *= 1.0/h/data.curl_v;
  } else {
    data.omega = data.Psi = 0;
  }

  data.density *= 1.0/POW3(h);
  data.div_v   *= 1.0/data.density/POW5(h);
  data.curl_v   = sqrtf(curl_v.x*curl_v.x + curl_v.y*curl_v.y + curl_v.z*curl_v.z)/data.density/POW5(h);
  
  return data;
}

__global__ void dev_density_loop(float4 *hydro_data,
				 float4 *dots_data,
				 KeyValuePair *bodies_hash) {
  
//   int index = blockIdx.x * blockDim.x + threadIdx.x; 
//   int index = tex1Dfetch(bodies_map_tex, blockIdx.x * blockDim.x + threadIdx.x);
  int index   = blockIdx.x * blockDim.x + threadIdx.x; 
  index       = bodies_hash[index].value;

  float4 body_pos = tex1Dfetch(bodies_pos_tex, index);
  float4 body_vel = tex1Dfetch(bodies_vel_tex, index);
  int n_ngb       = tex1Dfetch(n_ngb_tex, index);
  int offset      = tex1Dfetch(ngb_offset_tex, index);
  
  density_loop_data data = compute_density(n_ngb, offset,
					   body_pos, body_vel);
  
  float4 hd = hydro_data[index];
  hd.x = data.density;
  hydro_data[index] = hd;
  
  dots_data[index] = make_float4(data.div_v, data.curl_v, data.omega, data.Psi);
  
}

#endif
