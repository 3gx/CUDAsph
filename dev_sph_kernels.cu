#ifndef _DEV_SPH_KERNELS_CU_
#define _DEV_SPH_KERNELS_CU_

#define  PI 3.141592653589793
#define iPI 0.318309886183791

__device__ float w(float u) {
  if (u < 0) 
    return iPI;
  else if (u < 1) 
    return iPI * (1 - 1.5*u*u + 0.75*u*u*u);
  else if (u < 2)
    return iPI*0.25 * (2-u)*(2-u)*(2-u);
  else
    return 0;
}

__device__ float diw(float u) {
  return 12.5663706143592 * u*u * w(u);
}

__device__ float iw(float u) {
  if (u < 0) {
    return 0;
  } else if (u < 1) {
    float u3 = u*u*u;
    return 4 * (u3/3.0 - 0.3*u3*u*u + 0.125*u3*u3);
  } else if (u < 2) {
    float u3 = u*u*u;
    return 4* (-1.0/60 + 2.0/3*u3 - 0.75*u3*u + 0.3*u3*u*u - u3*u3/24.0);
  } else {
    return 1;
  }
}

__device__ float ig(float u) {
  float alpha = 4;
  return iw(alpha * (1 - sqrt((u-1)*(u-1))));

//   /* equivalent to */
//   if (u < 1)
//     return iw(alpha * u);
//   else if (u < 2)
//     return iw(alpha * (2 - u));
//   else
//     return 0;
    
}

__device__ float digdh(float u) {
  float alpha = 4;
  if (u <= 0)
    return 0;
  else if (u < 1)
    return  alpha*u * diw(alpha * u);
  else if (u < 2)
    return -alpha*u * diw(alpha * (2 - u));
  else
    return 0;
}

__device__ float dig(float u) {
  float alpha = 4;
  if (u <= 0)
    return 0;
  else if (u < 1)
    return  alpha/u * diw(alpha * u);
  else if (u < 2)
    return -alpha/u * diw(alpha * (2 - u));
  else
    return 0;
}

__device__ float dw(float u) {
  if (u < 1) 
    return iPI * (-3 + 2.25 * u);
  else if (u < 2)
    return -iPI * 0.75 * (2-u)*(2-u)/u;
  else 
    return 0;
}

__device__ float dwdh(float u) {
  if (u < 1)
    return iPI * (-3 + 7.5*u*u - 4.5*u*u*u);
  else if (u < 2) 
    return iPI * (-6 + 12*u - 7.5*u*u + 1.5*u*u*u);
  else
    return 0;
}

__device__ float dphidh(float u) {
  if (u < 1) 
    return 1.4 - 2*u*u + 1.5*u*u*u*u - 0.6*u*u*u*u*u;
  else if (u < 2)
    return 1.6 - 4*u*u + 4*u*u*u - 1.5*u*u*u*u + 0.2*u*u*u*u*u;
  else
    return 0;
}

#endif
