#include "CUDAsph.h"

void CUDAsph::write_state(char filename[]) {
  FILE *fout = fopen(filename, "wb");
  
  fwrite(&n_bodies, sizeof(int), 1, fout);
  fwrite(&nnopt, sizeof(int), 1, fout);
  fwrite(&hmin,  sizeof(double), 1, fout);
  fwrite(&hmax,  sizeof(double), 1, fout);
  fwrite(&sep0,  sizeof(double), 1, fout);
  fwrite(&t_end,   sizeof(double), 1, fout);
  fwrite(&dt_out,  sizeof(double), 1, fout);
  fwrite(&snapshot_number,    sizeof(int), 1, fout);
  fwrite(&iteration_number,    sizeof(int), 1, fout);
  fwrite(&global_time,   sizeof(double), 1, fout); 
  fwrite(&nav,    sizeof(int), 1, fout);
  fwrite(&alpha,   sizeof(double), 1, fout); 
  fwrite(&beta,   sizeof(double), 1, fout); 
  fwrite(&eta2,   sizeof(double), 1, fout); 
  fwrite(&ngr,    sizeof(int), 1, fout);
  fwrite(&nrelax,    sizeof(int), 1, fout);
  fwrite(&trelax,   sizeof(double), 1, fout); 
  fwrite(&dt,   sizeof(double), 1, fout); 
  fwrite(&omega2,   sizeof(double), 1, fout); 

  for (int i = 0; i < n_bodies; i++) {
    double val;
    
    val = hst.bodies_pos[i].x;   fwrite(&val, sizeof(double), 1, fout);
    val = hst.bodies_pos[i].y;   fwrite(&val, sizeof(double), 1, fout);
    val = hst.bodies_pos[i].z;   fwrite(&val, sizeof(double), 1, fout);

    val = hst.bodies_vel[i].w;      fwrite(&val, sizeof(double), 1, fout); //mass
    val = hst.bodies_pos[i].w*0.5; fwrite(&val, sizeof(double), 1, fout);  // range
    val = hst.hydro_data[i].x;   fwrite(&val, sizeof(double), 1, fout); // rho

    val = hst.bodies_vel[i].x;   fwrite(&val, sizeof(double), 1, fout);
    val = hst.bodies_vel[i].y;   fwrite(&val, sizeof(double), 1, fout);
    val = hst.bodies_vel[i].z;   fwrite(&val, sizeof(double), 1, fout);

    val = hst.bodies_dots[i].x;   fwrite(&val, sizeof(double), 1, fout);
    val = hst.bodies_dots[i].y;   fwrite(&val, sizeof(double), 1, fout);
    val = hst.bodies_dots[i].z;   fwrite(&val, sizeof(double), 1, fout);

    val = hst.hydro_data[i].z;   fwrite(&val, sizeof(double), 1, fout);   // ethermal 
    val = hst.bodies_dots[i].w;      fwrite(&val, sizeof(double), 1, fout);  //udot

    val = hst.grav_data[i].x;  fwrite(&val, sizeof(double), 1, fout);  // gx
    val = hst.grav_data[i].y;  fwrite(&val, sizeof(double), 1, fout);  // gy
    val = hst.grav_data[i].z;  fwrite(&val, sizeof(double), 1, fout);  // gz
    val = hst.grav_data[i].w;  fwrite(&val, sizeof(double), 1, fout);  // gpot

    val = hst.hydro_data[i].w*1.66054e-24;   fwrite(&val, sizeof(double), 1, fout);  // mu

    val = -1;
    fwrite(&val, sizeof(double), 1, fout);  // aa
    fwrite(&val, sizeof(double), 1, fout);  // bb
    fwrite(&val, sizeof(double), 1, fout);  // cc

    val = hst.dots_data[i].x;   fwrite(&val, sizeof(double), 1, fout);  // divv 
  }
  fclose(fout);
}
