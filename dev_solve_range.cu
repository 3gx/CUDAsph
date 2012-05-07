#ifndef _DEV_SOLVE_RANGE_
#define _DEV_SOLVE_RANGE_


#include <stdio.h>
#include <math.h>
#include "dev_textures.h"
#include "dev_sph_kernels.cu"

#ifndef LMEM_STACK_SIZE
#define LMEM_STACK_SIZE 256
#endif

#define DESIRED_NGB 24
#define MAX_ITER 20

__device__ float3 compute_ngb(float4 body_pos) {


  int kd_stack[LMEM_STACK_SIZE];

  int n_stack = 0; 
  kd_stack[n_stack] = 1;
  
  float3 ngb = {0, 0, 0};
  while (n_stack > -1) {
    /* reads from global memory */
    int node_id = kd_stack[n_stack];
    float4 node = tex1Dfetch(kd_tree_tex, node_id); 

    int split = (__float_as_int(node.w) & (3 << (24))) >> (24);
  
    node_id *= 2;
    if (node_id <= n_nodes) {
      switch (split) {
      case 0:
	if (body_pos.x - body_pos.w < node.x)
	  kd_stack[n_stack++] = node_id;
	if (body_pos.x + body_pos.w > node.x)
	  kd_stack[n_stack++] = node_id + 1;
	break;
      case 1:
	if (body_pos.y - body_pos.w < node.y)
	  kd_stack[n_stack++] = node_id;
	if (body_pos.y + body_pos.w > node.y)
	  kd_stack[n_stack++] = node_id + 1;
	break;
      case 2:
	if (body_pos.z - body_pos.w < node.z)
	  kd_stack[n_stack++] = node_id;
	if (body_pos.z + body_pos.w > node.z)
	  kd_stack[n_stack++] = node_id + 1;
	break;
      }
    }
    n_stack--;
    
    float3 dr;
    dr.x = body_pos.x - node.x;
    dr.y = body_pos.y - node.y;
    dr.z = body_pos.z - node.z;
    
    /* do operations on the neighbours */

    float ds2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
    if (ds2 < body_pos.w*body_pos.w) {
      float u = 2 * sqrtf(ds2)/body_pos.w;
      ngb.x +=  ig(u);
      ngb.y -= digdh(u);
      ngb.z += 1;
    }
    /* done ! */
    
    
  }
  
  ngb.x -= DESIRED_NGB;
  ngb.y *= 1.0/body_pos.w;
  if (ngb.y != 0) {
    ngb.y = ngb.x/ngb.y;
  } else {
    if (ngb.x < 0)
      ngb.y = -body_pos.w;
    else
      ngb.y = +body_pos.w;
  }

  return ngb;
}


#define f_scale (0.1)
__device__ float3 dev_bracket_root(float4 body_pos) {
  float3 f = compute_ngb(body_pos);
  float dx = f.y;
  
  if (fabs(dx) < rel_err * body_pos.w) 
    return make_float3(body_pos.w - 0.5*fabs(dx), body_pos.w + 0.5*fabs(dx), f.z);
  
  if (fabs(dx) > f_scale * body_pos.w) {
    if (dx < 0) dx = -f_scale * body_pos.w;
    else        dx = +f_scale * body_pos.w;
  }
  
  float x_left  = body_pos.w;
  float f_try = f.x;
  int iter = 0;
  while ((f_try * f.x >= 0) && (iter++ < MAX_ITER)) {
    x_left = body_pos.w;
    f_try  = f.x;

    body_pos.w -= dx;
    f = compute_ngb(body_pos);
    dx = f.y;
  
    if (fabs(dx) < rel_err * body_pos.w) 
      return make_float3(body_pos.w - 0.5*fabs(dx), body_pos.w + 0.5*fabs(dx), f.z);

    if (fabs(dx) > f_scale * body_pos.w) {
      if (dx < 0) dx = -f_scale * body_pos.w;
      else        dx = +f_scale * body_pos.w;
    }
    
  }
  
  float x_right;
  if (x_left < body_pos.w) {
    x_right = body_pos.w;
  } else {
    x_right = x_left;
    x_left  = body_pos.w;
  }
  return make_float3(x_left, x_right, f.z);
  
}

__device__ float2 solve_range(float4 body_pos) {
  int iter = 0;
  bool keep_solving = true;
  
  float3 root = dev_bracket_root(body_pos);
  body_pos.w = 0.5 * (root.x + root.y);
  
  float3 f = {0,0,root.z};
  float dx = 0;
  if (fabs(root.x - root.y) > rel_err * body_pos.w) {
    while(keep_solving && (iter++ < MAX_ITER)) {
      
      /* range is out of bounds */
      body_pos.w -= dx;
      if ( (body_pos.w - root.y)*(body_pos.w - root.x) >= 0 ){
	dx = 0.5 * (root.y - root.x);
 	body_pos.w = 0.5 * (root.x + root.y);
      }
      
      f = compute_ngb(body_pos);
      
      if (f.x < 0)
	root.x = body_pos.w;
      else
	root.y = body_pos.w;
      
      dx = f.y;

      
      keep_solving = fabs(dx) > rel_err * body_pos.w;
    }
  }
//  if(iter > 5)
//   fprintf(stderr, "id= %d  iter= %d: h= %f  %f %f %f %f\n", blockIdx.x*blockDim.x + threadIdx.x,iter, body_pos.w, f.x, f.y, dx, f.z);
  return make_float2(body_pos.w, f.z);
}

__global__ void dev_solve_range(float4 *bodies_pos,
				int *n_ngb,
				KeyValuePair *bodies_hash) {
  int index   = blockIdx.x * blockDim.x + threadIdx.x; 
  index       = bodies_hash[index].value;
  
  float4 body_pos = bodies_pos[index];
  float2 range = {0,0};
  if (body_pos.w > 0)
    range = solve_range(body_pos);
  
  body_pos.w = range.x;
  bodies_pos[index] = body_pos;
  n_ngb[index] = (int)range.y;
  
}


__global__ void dev_get_ngb(float4 *bodies_pos,
			    int *n_ngb) {
  int index = blockIdx.x * blockDim.x + threadIdx.x; 
  float4 body_pos = bodies_pos[index];

  float3 f = compute_ngb(body_pos);
  
  n_ngb[index] = (int)f.z;
  
}

#endif
