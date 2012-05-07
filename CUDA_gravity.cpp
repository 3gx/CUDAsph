#include "CUDAsph.h"
#include "memory.h"

void CUDAsph::CUDA_gravity() {
  
  int p = 256;
  int q = 1;

  p = p/2;
  
  if (n_dev_bodies < p) {
    printf("shit!");
    exit(-1);
    p = n_dev_bodies;
  }
  
  float* m_hPos;        // CPU data
  float* m_hVel;        // CPU data
  float* m_dPos;        // GPU data
  float* m_dVel;        // GPU data
 
  /* make sure that n_dev_bodies which is sent to GPU
     is divisible by 4 
  */

  m_hPos = new float[4*n_dev_bodies];
  m_hVel = new float[4*n_dev_bodies];
  memset(m_hPos, 0, n_dev_bodies*4*sizeof(float));
  memset(m_hVel, 0, n_dev_bodies*4*sizeof(float));
  
  zallocateNBodyArrays(&m_dPos, n_dev_bodies, device_id);
  zallocateNBodyArrays(&m_dVel, n_dev_bodies, device_id);
  
  int i = 0;
  int j = 0;
  for (int pc = 0; pc < n_bodies; pc++) {
    m_hPos[i++] = hst.bodies_pos[pc].x;
    m_hPos[i++] = hst.bodies_pos[pc].y;
    m_hPos[i++] = hst.bodies_pos[pc].z;
    m_hPos[i++] = hst.bodies_vel[pc].w;

    m_hVel[j++] = hst.bodies_vel[pc].x;
    m_hVel[j++] = hst.bodies_vel[pc].y;
    m_hVel[j++] = hst.bodies_vel[pc].z;
    m_hVel[j++] = hst.bodies_pos[pc].w*hst.bodies_pos[pc].w;
  }
  
  int pc = 0;
  while(i < 4*n_dev_bodies) {
    m_hPos[i++] = hst.bodies_pos[pc].x*1.1;
    m_hPos[i++] = hst.bodies_pos[pc].y*1.1;
    m_hPos[i++] = hst.bodies_pos[pc].z*1.1;
    m_hPos[i++] = 0.0;
    pc++;
    if (pc >= n_bodies) pc = 0;
  }
  
  zcopyArrayToDevice(m_dPos, m_hPos, n_dev_bodies, device_id);
  zcopyArrayToDevice(m_dVel, m_hVel, n_dev_bodies, device_id);
  
  compute_CUDA_gravity(n_dev_bodies, dev.grav_data, m_dPos, m_dVel, p, q, device_id);
  
  zcopyArrayFromDevice((float*)hst.grav_data, (float*)dev.grav_data, n_dev_bodies, device_id);
  
  
  delete[] m_hPos;
  delete[] m_hVel;

  zdeleteNBodyArrays(m_dPos, device_id);
  zdeleteNBodyArrays(m_dVel, device_id);
}
