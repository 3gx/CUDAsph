#ifndef _DEV_FIND_NGB_CU_
#define _DEV_FIND_NGB_CU_

#include <stdio.h>
#include <math.h>
#include "dev_textures.h"
#include "radixsort/radixsort.cuh"

#ifndef LMEM_STACK_SIZE
#define LMEM_STACK_SIZE 32
#endif

__device__ int get_ngb(float4 body_pos,
		       int *ngb_list) {
  
  int kd_stack[LMEM_STACK_SIZE];
  
  int n_stack = 0;
  kd_stack[n_stack] = 1;
  
  int n_ngb_list = 0;
  int n_walk = 0;
  while (n_stack > -1) {
    n_walk++;
    /* reads from global memory */
    int node_id = kd_stack[n_stack];
    float4 node = tex1Dfetch(kd_tree_tex, node_id); 

    int split = (__float_as_int(node.w) & (3 << (24))) >> 24;

    node_id *= 2;
    
    /* this *may* help the compiler to generate a non-diverging code */
    if (node_id <= n_nodes) {
      switch (split) {
      case 0:
	if (body_pos.x - body_pos.w < node.x) kd_stack[n_stack++] = node_id;
// 	else                                  kd_stack[n_stack  ] = node_id;
 	if (body_pos.x + body_pos.w > node.x) kd_stack[n_stack++] = node_id + 1;
// 	else                                  kd_stack[n_stack  ] = node_id + 1;
	break;
      case 1:
	if (body_pos.y - body_pos.w < node.y) kd_stack[n_stack++] = node_id;
// 	else                                  kd_stack[n_stack  ] = node_id;
 	if (body_pos.y + body_pos.w > node.y) kd_stack[n_stack++] = node_id + 1;
// 	else                                  kd_stack[n_stack  ] = node_id + 1;
	break;
      case 2:
	if (body_pos.z - body_pos.w < node.z) kd_stack[n_stack++] = node_id;
// 	else                                  kd_stack[n_stack  ] = node_id;
 	if (body_pos.z + body_pos.w > node.z) kd_stack[n_stack++] = node_id + 1;
// 	else                                  kd_stack[n_stack  ] = node_id + 1;
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
      ngb_list[n_ngb_list++] =  __float_as_int(node.w) & ~(3 << (24));
    }
  }

  return n_walk;
}

__global__ void dev_find_ngb(int *ngb_list,
			     KeyValuePair *bodies_hash) {
  
  int index         = blockIdx.x * blockDim.x + threadIdx.x; 
  index  = bodies_hash[index].value; //bhash.value;
  
  float4 body_pos = tex1Dfetch(bodies_pos_tex, index);
  int offset      = tex1Dfetch(ngb_offset_tex, index);
  
  bodies_hash[blockIdx.x * blockDim.x + threadIdx.x].key = 
    get_ngb(body_pos, &ngb_list[offset]);

}

#endif
