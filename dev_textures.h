#ifndef _DEV_TEXTURES_H_
#define _DEV_TEXTURES_H_

texture<float4, 1, cudaReadModeElementType> kd_tree_tex;
texture<int,    1, cudaReadModeElementType> bodies_map_tex;

texture<float4, 1, cudaReadModeElementType> bodies_pos_tex;
texture<float4, 1, cudaReadModeElementType> bodies_vel_tex;
texture<float4, 1, cudaReadModeElementType> dots_data_tex;
texture<float4, 1, cudaReadModeElementType> hydro_data_tex;

texture<int,   1, cudaReadModeElementType> ngb_list_tex;
texture<float, 1, cudaReadModeElementType> ngb_acc_tex;
texture<int,   1, cudaReadModeElementType> n_ngb_tex;
texture<int,   1, cudaReadModeElementType> ngb_offset_tex;

__constant__ int   n_nodes;
__constant__ float rel_err;
__constant__ float av_alpha, av_beta;
#endif 
