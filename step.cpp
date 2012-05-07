#include "CUDAsph.h"

double get_time() {
  struct timeval Tvalue;
  struct timezone dummy;
  
  gettimeofday(&Tvalue,&dummy);
  return ((double) Tvalue.tv_sec +
	  1.e-6*((double) Tvalue.tv_usec));
}


void CUDAsph::step() {
  /****************
   *  first half  *
   ***************/

  double t_begin = get_time();

  vector<float>  old_u(n_bodies);
  vector<float3> old_vel(n_bodies);
  
  iteration_number++;
  
  double t1 = 0;
  if (iteration_number%100 == 0) {
    fprintf(stderr, "     Peano-Hilbert sorting ... \n");
    t1 = get_time();
    sort_bodies();
    cerr << "      done in " << get_time() - t1 << " sec\n";
  }

  /***********************************
   *  predict positions & velocities *
   **********************************/

  float dth = 0.5 * dt;
  for (int i = 0; i < n_bodies; i++) {
    hst.bodies_pos[i].x += hst.bodies_vel[i].x * dt;
    hst.bodies_pos[i].y += hst.bodies_vel[i].y * dt;
    hst.bodies_pos[i].z += hst.bodies_vel[i].z * dt;
    
    old_vel[i].x = hst.bodies_vel[i].x;
    old_vel[i].y = hst.bodies_vel[i].y;
    old_vel[i].z = hst.bodies_vel[i].z;
    
    hst.bodies_vel[i].x += hst.bodies_dots[i].x * dth;
    hst.bodies_vel[i].y += hst.bodies_dots[i].y * dth;
    hst.bodies_vel[i].z += hst.bodies_dots[i].z * dth;
    
    old_u[i] = hst.hydro_data[i].z;
    hst.hydro_data[i].z += hst.bodies_dots[i].w * dth;
    
    hst.bodies_dots[i].x = 0;
    hst.bodies_dots[i].y = 0;
    hst.bodies_dots[i].z = 0;
    
    hst.grav_data[i].x = 0;
    hst.grav_data[i].y = 0;
    hst.grav_data[i].z = 0;
  }

  fprintf(stderr, " Computing hydro accelerations ...\n");
  t1 = get_time();
  sph_accelerations();
  cerr << " done in " << get_time() - t1 << " sec\n";
 
//   for (int i = 0; i < n_bodies; i++) {
//     fprintf(stderr, "i= %d: key= %d value= %d\n", 
// 	    i, dev.bodies_hash[i].key, dev.bodies_hash[i].value);
//   }
    
//  for (int i = 0; i < n_bodies; i++){
//     printf("%d  %f %f %f %f   %f %f %f %d  %f %f %f %f\n",
// 	   i, 
// 	   hst.bodies_pos[i].x,
// 	   hst.bodies_pos[i].y,
// 	   hst.bodies_pos[i].z,
// 	   hst.bodies_pos[i].w,
// 	   hst.hydro_data[i].x,
// 	   hst.hydro_data[i].z,
// 	   hst.hydro_data[i].y,
// 	   hst.n_ngb[i],
// 	   hst.bodies_dots[i].x,
// 	   hst.bodies_dots[i].y,
// 	   hst.bodies_dots[i].z,
// 	   hst.bodies_dots[i].w
// 	   );
//   }

  
  t1 = get_time();
  cerr << " Computing gravity ... ";
  CUDA_gravity();
  cerr << " done in " << get_time() - t1 << " sec\n";

  double force[3] = {0,0,0};
  double torque[3] = {0,0,0};
  for (int i = 0; i < n_bodies; i++) {
    hst.bodies_dots[i].x += hst.grav_data[i].x;
    hst.bodies_dots[i].y += hst.grav_data[i].y;
    hst.bodies_dots[i].z += hst.grav_data[i].z;

    force[0] += hst.bodies_vel[i].w * hst.bodies_dots[i].x;
    force[1] += hst.bodies_vel[i].w * hst.bodies_dots[i].y;
    force[2] += hst.bodies_vel[i].w * hst.bodies_dots[i].z;
    torque[0] += hst.bodies_vel[i].w * (hst.bodies_pos[i].y * hst.bodies_dots[i].z -
				    hst.bodies_pos[i].z * hst.bodies_dots[i].y);
    torque[1] += hst.bodies_vel[i].w * (hst.bodies_pos[i].z * hst.bodies_dots[i].x -
				    hst.bodies_pos[i].x * hst.bodies_dots[i].z);
    torque[2] += hst.bodies_vel[i].w * (hst.bodies_pos[i].x * hst.bodies_dots[i].y -
				    hst.bodies_pos[i].y * hst.bodies_dots[i].x);
  }
  fprintf(stderr, " total force  = [ %lg %lg %lg ]\n",
	  force[0], force[1], force[2]);

  fprintf(stderr, " total torque = [ %lg %lg %lg ]\n",
	  torque[0], torque[1], torque[2]);



  global_time += dt;

  fprintf(stderr, "\n   ****************************************************************** \n");
  system_statistics();
  fprintf(stderr, "   ****************************************************************** \n\n");

  dt = compute_timestep();
  for (int i = 0; i < n_bodies; i++) {
    hst.bodies_vel[i].x = old_vel[i].x + hst.bodies_dots[i].x * (0.5*dt + dth);
    hst.bodies_vel[i].y = old_vel[i].y + hst.bodies_dots[i].y * (0.5*dt + dth);
    hst.bodies_vel[i].z = old_vel[i].z + hst.bodies_dots[i].z * (0.5*dt + dth);
    hst.hydro_data[i].z = old_u[i]     + hst.bodies_dots[i].w * (0.5*dt + dth);
  }


  double t_end = get_time();
  cerr << endl;
  if (device_id < 0) {
    cerr << " **** Iteration took " << t_end - t_begin << " seconds. **** " << endl;
  } else {
    cerr << " **** Iteration took " << t_end - t_begin << " seconds on device# " 
	 << device_id << " (" << cuda_get_device() << "). **** " << endl;
  }
  cerr << endl;

}
