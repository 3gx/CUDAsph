#ifndef _SPH_GRAV_KERNEL_CU_
#define _SPH_GRAV_KERNEL_CU_

#include <stdio.h>
#include <math.h>

#define _DEVICE_CODE_
#include "dreal.h"

struct acc3 {
  dreal v[4];
};

__device__ struct acc3
bodyBodyInteraction(struct acc3 acc,
		    float4 pos_i, float4 vel_i,
		    float4 pos_j, float4 vel_j) {
  float3 r;

  // r_ij  [3 FLOPS]
  r.x = pos_i.x - pos_j.x;
  r.y = pos_i.y - pos_j.y;
  r.z = pos_i.z - pos_j.z;
  
  float distSqr     = r.x * r.x + r.y * r.y + r.z * r.z;
  float invDist     = rsqrtf(distSqr);
  float invDistCube = invDist * invDist * invDist;
  float s3          = pos_j.w * invDistCube;

  /* if dr > max(range_i, range_j) */
  /*    use usual 1/r gravity */
  if ( (distSqr >= vel_i.w) && (distSqr >= vel_j.w) ) {
    acc.v[0] = dsub(acc.v[0], r.x * s3);
    acc.v[1] = dsub(acc.v[1], r.y * s3);
    acc.v[2] = dsub(acc.v[2], r.z * s3);
    acc.v[3] = dsub(acc.v[3], pos_j.w * invDist);
//     acc.v[3].v[0] -= pos_j.w * invDist;

    
  } else {
    
    /* otherwise */
    s3 = 0.5f*s3;

    /* if dr < range_i */
    /*  use smoothed g(r,h) for gravity half-contribution */
    if (distSqr < vel_i.w) {
      float u2 = distSqr/vel_i.w;
      float h  = sqrtf(vel_i.w);
      float u  = sqrtf(u2);
      float u3 = u2*u;
      float gacc, gpot;
      if (u < 0.5) {
	gacc = (10.666666666667f + u2 * (32.0f * u - 38.4f));
	gpot = -2.8f + u2 * (5.333333333333f + u2 * (6.4f * u - 9.6f));
      } else {
	gacc = (21.333333333333f - 48.0f*u +
		38.4f*u2 - 10.666666666667f*u3 - 0.066666666667f/u3);
	gpot = -3.2f + 0.066666666667f/u + 
	  u2 * (10.666666666667f + u * (-16.0f + u * (9.6f - 2.133333333333f * u)));
      }
      gacc = 0.5 * pos_j.w * gacc / (vel_i.w * h);
      gpot = -gpot;

      acc.v[0] = dsub(acc.v[0], r.x * gacc);
      acc.v[1] = dsub(acc.v[1], r.y * gacc);
      acc.v[2] = dsub(acc.v[2], r.z * gacc);
      acc.v[3] = dsub(acc.v[3], pos_j.w * gpot/h);
      //       acc.v[3].v[0] -= pos_j.w * gpot/h;
	


    } else {
      acc.v[0] = dsub(acc.v[0], r.x * s3);
      acc.v[1] = dsub(acc.v[1], r.y * s3);
      acc.v[2] = dsub(acc.v[2], r.z * s3);
      acc.v[3] = dsub(acc.v[3], pos_j.w * invDist);
//       acc.v[3].v[0] -= pos_j.w * invDist;

    }

    /* if dr < range_j */
    /*  use smoothed g(r,h) for gravity half-contribution */
    if (distSqr < vel_j.w) {
      float u2 = distSqr/vel_j.w;
      float h  = sqrtf(vel_j.w);
      float u  = sqrtf(u2);
      float u3 = u2*u;
      float gacc;
      if (u < 0.5) {
	gacc = (10.666666666667f + u2 * (32.0f * u - 38.4f));
      } else {	
	gacc = (21.333333333333f - 48.0f*u +
		38.4f*u2 - 10.666666666667f*u3 - 0.066666666667f/u3);
      }	
      gacc = 0.5 * pos_j.w * gacc / (vel_j.w * h);
      acc.v[0] = dsub(acc.v[0], r.x * gacc);
      acc.v[1] = dsub(acc.v[1], r.y * gacc);
      acc.v[2] = dsub(acc.v[2], r.z * gacc);
      
    } else {
      acc.v[0] = dsub(acc.v[0], r.x * s3);
      acc.v[1] = dsub(acc.v[1], r.y * s3);
      acc.v[2] = dsub(acc.v[2], r.z * s3);
    }
  }
//   printf ("bId.x= %d  tid.x= %d \n", blockIdx.x, threadIdx.x);
  
  return acc;

} // 22 FLOPS in total


// Macros to simplify shared memory addressing
#define SX_Pos(i) sharedPos[i+blockDim.x*threadIdx.y]
#define SX_Vel(i) sharedVel[i+blockDim.x*threadIdx.y]

// This is the "tile_calculation" function from the GPUG3 article.
__device__ struct acc3
gravitation(struct acc3 acc, float4 myPos, float4 myVel) {
  extern __shared__ float4 shared_mem[];
  float4* sharedPos = (float4*)shared_mem;
  float4* sharedVel = (float4*)&shared_mem[blockDim.x*blockDim.y];

 // Here we unroll the loop
  int i;
#pragma unroll 0
  for (i = 0; i < blockDim.x; i++) 
    acc = bodyBodyInteraction(acc, myPos, myVel, SX_Pos(i), SX_Vel(i));
  

  return acc;
}

// This macro is only used when multithreadBodies is true (below)
#define SX_SUM_Pos(i,j) sharedPos[i+blockDim.x*j]
#define SX_SUM_Vel(i,j) sharedVel[i+blockDim.x*j]

__device__ struct acc3
computeBodyAccel(float4 bodyPos, float4 bodyVel, 
		 float4* positions, float4* velocities, int numBodies) {
  extern __shared__ float4 shared_mem[];
  float4* sharedPos = (float4*)shared_mem;
  float4* sharedVel = (float4*)&shared_mem[blockDim.x*blockDim.y];
  
  struct acc3 acc;
  acc.v[0].v[0] = acc.v[0].v[1] = 0;
  acc.v[1].v[0] = acc.v[1].v[1] = 0;
  acc.v[2].v[0] = acc.v[2].v[1] = 0;
  acc.v[3].v[0] = acc.v[3].v[1] = 0;
  
  int p = blockDim.x;
  int q = blockDim.y;
  int n = numBodies;
  
  int start = n/q * threadIdx.y;
  int finish = start + n/q;
  int tile = 0;
  
  for (int i = start; i < finish; i += p, tile++) {
    sharedPos[threadIdx.x + blockDim.x*threadIdx.y] = 
      positions[tile * blockDim.x + threadIdx.x];
    
    sharedVel[threadIdx.x + blockDim.x*threadIdx.y] = 
      velocities[tile * blockDim.x + threadIdx.x];
    
    __syncthreads();
    
    // This is the "tile_calculation" function from the GPUG3 article.
    
    acc = gravitation(acc, bodyPos, bodyVel);
    __syncthreads();
  }


  return acc;
}

__global__ void
get_CUDA_forces(float4* accel, 
		float4* bodies_pos, float4* bodies_vel, int numBodies) {
  int index  = blockIdx.x * blockDim.x + threadIdx.x; 
  float4 pos = bodies_pos[index];
  float4 vel = bodies_vel[index];

  struct acc3 acc =
    computeBodyAccel(pos, vel, bodies_pos, bodies_vel, numBodies);


  accel[index].x = acc.v[0].v[0];
  accel[index].y = acc.v[1].v[0];
  accel[index].z = acc.v[2].v[0];
  accel[index].w = acc.v[3].v[0];

}

#endif
