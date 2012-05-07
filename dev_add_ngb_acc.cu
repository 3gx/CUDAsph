#ifndef _DEV_ADD_NGB_ACC_
#define _DEV_ADD_NGB_ACC_

#include "dev_textures.h"

__global__ void dev_add_ngb_acc(float4 *bodies_dots,
				int *n_reads) {
  int index = blockIdx.x * blockDim.x + threadIdx.x; 

  float4 body_pos   = tex1Dfetch(bodies_pos_tex, index);
  int n_ngb         = tex1Dfetch(n_ngb_tex, index);
  int offset        = tex1Dfetch(ngb_offset_tex, index);

  float4 dots = bodies_dots[index];
  for (int i = offset; i < offset + n_ngb; i++) {
    float acc         = tex1Dfetch(ngb_acc_tex, i); 
    float4 body_pos_j = tex1Dfetch(bodies_pos_tex, tex1Dfetch(ngb_list_tex, i));
    
    dots.x += acc * (body_pos.x - body_pos_j.x);
    dots.y += acc * (body_pos.y - body_pos_j.y);
    dots.z += acc * (body_pos.z - body_pos_j.z);
  }  
  
  bodies_dots[index] = dots;

}
#endif 
