#ifndef __CUDAsph_H_
#define __CUDAsph_H_

#include <stdio.h>
#include <iostream>
#include <builtin_types.h>
#include <math.h>
#include "kd_tree.h"
#include "radixsort/radixsort.cuh"
#include <sys/time.h>
#include <cstdlib>
#include <vector>

double get_time();

inline int n_norm(int n, int j) {
  n = ((n-1)/j) * j + j;
  if (n == 0) n = j;
  return n;
}

extern "C"
{
  void initCUDA();
  void allocateCUDAarray(void** pos, int n, int);
  void deleteCUDAarray(void* pos, int);
  void copyArrayToDevice(void* device, const void* host, int n, int);
  void copyArrayFromDevice(void* host, const void* device, int n, int);

  void compute_CUDA_gravity(int numBodies, float4 *accel, float *pos, float* vel,
			    int p, int q, int);
  void allocateNBodyArrays(float** pos, int numBodies, int);
  void deleteNBodyArrays(float* pos, int);

  void zallocateNBodyArrays(float** pos, int numBodies, int);
  void zdeleteNBodyArrays(float* pos, int);
  
  void zcopyArrayFromDevice(float* host, const float* device, unsigned int numBodies, int);
  void zcopyArrayToDevice(float* device, const float* host, int numBodies, int);

  float host_solve_range(int, int, float, KeyValuePair*, int*, float4*, float4*, float4*, int*, int);
  float host_sph_accelerations(int, int, float, float, KeyValuePair*, KeyValuePair*, int*, int*, int*,
			       float4*, float4*, float4*, float4*, float*, int);
  int get_device_count();
  void cuda_set_device(int device_id);
  void cuda_set_device0(int device_id);
  int cuda_get_device();
}

struct hydro_data {
  float4  *bodies_pos; // {x, y, z, range}
  float4  *bodies_vel; // {vx, vy, vz, mass}
  float4  *kd_tree;
  int *particle_map;
  
  int *bodies_id;
  float4  *bodies_dots;     //  {vxdot, vydot, vzdot, udot}
  float4  *hydro_data;   //  {rho, pressure, ethermal, mean_mu}
  float4  *dots_data; //  {div_v, curl_v, omega, psi};
  float4  *hydro_data2;   // {max_vsig, ?, ?, ?}
  float4  *grav_data;    // {gx, gy, gz, gpot}

  KeyValuePair *bodies_hash, *bodies_hash_extra;

  float        *ngb_acc;
  int          *n_ngb, *ngb_offset;
  int          *ngb_list;
};

#define N_THREADS 128

class CUDAsph {
private:
  
  int device_id;

  /* legacy variables */
  int nnopt, nav, ngr, nrelax;
  double hmin, hmax, sep0, alpha, beta, eta2, trelax, omega2;
  /* legacy variables */

  kd_node *kd_tree;
  hydro_data dev;
  hydro_data hst;
  
  int n_bodies, n_dev_bodies;
  int ngb_tot, ngb_tot_max;
  
  double dt, global_time, t_end;
  double t_last_output, dt_out;
  int iteration_number;
  int snapshot_number;

  void set_device(int);
  
public:
  
  void read_state(FILE*);
  void write_state(char*);
  void sph_accelerations();
  void stride_setup();

  void output();
  void step();
  float compute_timestep();
  void  system_statistics();
  
  void copy_kd_tree_to_device();
  void copy_particle_map_to_device();
  void copy_bodies_hash_to_device();

  void copy_bodies_pos_vel_to_device();
  void copy_hydro_data_to_device();
  void copy_ngb_offset_to_device();  
  
  void copy_bodies_dots_from_device();
  void copy_hydro_data_from_device();
  void copy_hydro_data2_from_device();
  void copy_n_ngb_from_device();
  void copy_ngb_list_from_device();
  void copy_ngb_acc_from_device();
  void copy_bodies_pos_from_device();
  void sort_bodies();
 
  void allocate_host_memory();
  void free_host_memory();

  void allocate_device_memory();
  void free_device_memory();
  
  float CUDA_solve_range();
  float CUDA_sph_accelerations();
  void  CUDA_gravity();

  CUDAsph(char filename[], int __device_id) {
    set_device(__device_id);

    FILE* fin;
    if (!(fin = fopen(filename, "rb"))) {
      fprintf(stderr, "cannot open file %s\n", filename);
      exit(-1);
    } else {
      fprintf(stderr, "file %s is found\n", filename);
    }
    read_state(fin);
  }
  ~CUDAsph() {
    free_host_memory();
    free_device_memory();
  }

  bool keep_going() {
    FILE* fd;
    if ((fd = fopen("stop", "r"))) {
      fprintf(stderr, " >>> STOP file found. Terminating simulation. <<<  \n");
//      unlink("stop");
      return false;
    }
    
    return global_time < t_end;
  }
  
};

#endif 
