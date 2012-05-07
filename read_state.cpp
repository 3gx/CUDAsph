#include "CUDAsph.h"

void CUDAsph::read_state(FILE *fin) {
  
  fread(&n_bodies, sizeof(int), 1, fin);
  fread(&nnopt, sizeof(int), 1, fin);
  fread(&hmin,  sizeof(double), 1, fin);
  fread(&hmax,  sizeof(double), 1, fin);
  fread(&sep0,  sizeof(double), 1, fin);
  fread(&t_end,    sizeof(double), 1, fin);
  fread(&dt_out,  sizeof(double), 1, fin);
  fread(&snapshot_number,   sizeof(int), 1, fin);
  fread(&iteration_number,    sizeof(int), 1, fin);
  fread(&global_time,   sizeof(double), 1, fin); 
  fread(&nav,    sizeof(int), 1, fin);
  fread(&alpha,   sizeof(double), 1, fin); 
  fread(&beta,   sizeof(double), 1, fin); 
  fread(&eta2,   sizeof(double), 1, fin); 
  fread(&ngr,    sizeof(int), 1, fin);
  fread(&nrelax,    sizeof(int), 1, fin);
  fread(&trelax,   sizeof(double), 1, fin); 
  fread(&dt,   sizeof(double), 1, fin); 
  fread(&omega2,   sizeof(double), 1, fin); 

  t_last_output = global_time;
  
  n_dev_bodies = n_norm(n_bodies, 256);
  
  ngb_tot_max = n_dev_bodies * 100;

  allocate_host_memory();
  allocate_device_memory();

  for (int i = 0; i < n_bodies; i++) {
    double val;
    
    fread(&val, sizeof(double), 1, fin);    hst.bodies_pos[i].x = val;
    fread(&val, sizeof(double), 1, fin);    hst.bodies_pos[i].y = val;
    fread(&val, sizeof(double), 1, fin);    hst.bodies_pos[i].z = val;

    fread(&val, sizeof(double), 1, fin);    hst.bodies_vel[i].w = val;  //mass
    fread(&val, sizeof(double), 1, fin);    hst.bodies_pos[i].w = 2*val;  //range
    fread(&val, sizeof(double), 1, fin);    hst.hydro_data[i].x = val;  // density

    fread(&val, sizeof(double), 1, fin);    hst.bodies_vel[i].x = val;
    fread(&val, sizeof(double), 1, fin);    hst.bodies_vel[i].y = val;
    fread(&val, sizeof(double), 1, fin);    hst.bodies_vel[i].z = val;

    fread(&val, sizeof(double), 1, fin);    hst.bodies_dots[i].x = val;
    fread(&val, sizeof(double), 1, fin);    hst.bodies_dots[i].y = val;
    fread(&val, sizeof(double), 1, fin);    hst.bodies_dots[i].z = val;

    fread(&val, sizeof(double), 1, fin);    hst.hydro_data[i].z  = val;  // ethermal
    fread(&val, sizeof(double), 1, fin);    hst.bodies_dots[i].w = val;  // udot
    
    fread(&val, sizeof(double), 1, fin);    hst.grav_data[i].x = val; // gx
    fread(&val, sizeof(double), 1, fin);    hst.grav_data[i].y = val; // gy
    fread(&val, sizeof(double), 1, fin);    hst.grav_data[i].z = val; // gz
    fread(&val, sizeof(double), 1, fin);    hst.grav_data[i].w = val; // gpot
    
    fread(&val, sizeof(double), 1, fin);    hst.hydro_data[i].w = val/1.66054e-24; //mean_mu
    
    fread(&val, sizeof(double), 1, fin);  // aa
    fread(&val, sizeof(double), 1, fin);  // bb
    fread(&val, sizeof(double), 1, fin);  // cc
    fread(&val, sizeof(double), 1, fin);    hst.dots_data[i].x = val; // divv;
  }
  
  /* fill hash with data */
  for (int i = 0; i < n_bodies; i++) {
    hst.bodies_hash[i].value = i;
    hst.bodies_hash[i].key   = i;
  }

  for (int i = n_bodies; i < n_dev_bodies; i++) {
    hst.bodies_hash[i].value = i;
    hst.bodies_hash[i].key   = 1 << (24);
    hst.bodies_vel[i].w = 0;
    hst.bodies_pos[i].x = hst.bodies_pos[i].y = hst.bodies_pos[i].z = +1e10;
    hst.bodies_vel[i].x = hst.bodies_vel[i].y = hst.bodies_vel[i].z = +1e10;
    hst.bodies_pos[i].w = 0;
    
    hst.hydro_data[i].w = 0;

    hst.kd_tree[i].x = +1e10;
    hst.kd_tree[i].y = +1e10;
    hst.kd_tree[i].z = +1e10;
    *(int*)&hst.kd_tree[i].w = -1; 
  }

  copy_bodies_hash_to_device();
  
  fclose(fin);
}
