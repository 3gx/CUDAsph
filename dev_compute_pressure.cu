#include "dev_textures.h"

__device__ float dev_find_root(float q, float r) {

  /*
    Subroutine to solve 4th order equations to determine the temperature x3
    for an equation of state with both ideal gas and radiation pressure.
    Written by Scott Fleming 10/04/02 and James Lombardi 2002-2003
    
    The fourth order equation comes from u_gas+ u_rad = u, with
    u_gas proportional to T and u_rad proportional to T^4
    
    In general, we can transform a 4th order equation to x^4+px^2+qx+r=0
    (see pages 57-58 of Stillwell's "Mathematics and its history" text)
    but we fortunately don't even have an x^2 term (that is, p=0).
    
    Follow Stillwell, we can transform this into a cubic equation:
    First solve for y by using the fact that B^2-4AC=0
    equation is then:  y^3=ry+q^2/8
    using the solution of cubic equations found in Stillwell page 55:

    GPU version (c) E.Gaburov 2008, 
    Sterrenkundig Instituut "Anton Pannekoek" and
    Section Computational Science,
    Universiteit van Amsterdam
  */

  float kh = 0.5 * 0.125 * q*q;

  float piece_1 = kh + sqrtf(kh*kh - r*r*r/27.0);
  float piece_2 = kh - sqrtf(kh*kh - r*r*r/27.0);

  float y = powf(piece_1, 1.0/3) - powf(fabs(piece_2), 1.0/3);
  float b_2 = sqrtf(2.0 * y);
  float c_2 = y - 0.5*q/b_2;
  

  float x_3 = 0.5 * (sqrtf(b_2*b_2 - 4*c_2) - b_2);

  if (piece_1 == -piece_2) {
    /* radiation pressure dominates */
    x_3 = powf(-r - q * powf(-r - q * powf(-r, 0.25), 0.25), 0.25);
  }
  
  if (piece_2 > 0)
    x_3 = -(r + r*r*r*r/(q*q*q*q))/q;
  
  return x_3;
}

// #define DENSITY_UNIT (5.40402671921732)
// #define ENERGY_UNIT (1.59496972099259e+15)
// #define PRESSURE_UNIT (8.61925898858652e+15)

#ifndef DENSITY_UNIT
#define DENSITY_UNIT (5.89994031541978)
#endif

#ifndef ENERGY_UNIT
#define ENERGY_UNIT (1.90698842928778e+15)
#endif

#ifndef PRESSURE_UNIT
#define PRESSURE_UNIT (1.1251117914994e+16)
#endif

#ifndef AMU
#define AMU (1.66054e-24)
#endif

#ifndef BOLTZMANN
#define BOLTZMANN (1.380658e-16)
#endif 

#ifndef RAD_CONST
#define RAD_CONST 7.56591414984829e-15
#endif

#ifndef GAMMA_GAS
#define GAMMA_GAS (5.0/3.0)
#endif 

__device__ float get_pressure(float density, float energy, float mean_mu) {
  
  float temp_base = 1.0e6;
  
  float r = -density * (energy/(temp_base*temp_base)) / (RAD_CONST*temp_base*temp_base);
  float q = (BOLTZMANN/RAD_CONST)/(GAMMA_GAS - 1) * density/ (((mean_mu * temp_base)*temp_base)*temp_base);

  float temperature = temp_base * dev_find_root(q, r);
  
  float p_gas = density * ((BOLTZMANN*temperature)/mean_mu);
  float p_rad = (((RAD_CONST/3.0 * temperature) * temperature) * temperature) * temperature;

  return ((p_gas + p_rad)/PRESSURE_UNIT);
}


__global__ void dev_compute_pressure(float4 *hydro_data) {

  int index = blockIdx.x * blockDim.x + threadIdx.x; 
  float4 hd = hydro_data[index];

  float density = hd.x  * DENSITY_UNIT;
  float energy  = hd.z  * ENERGY_UNIT;
  float mean_mu = hd.w  * AMU;
  
  hd.y = get_pressure(density, energy, mean_mu);
  
  hydro_data[index] = hd;
}
