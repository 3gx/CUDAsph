#include "CUDAsph.h"

void CUDAsph::stride_setup() {

//   vector<float> u(n_bodies);
//   for (int i = 0; i < n_bodies; i++) {
//     u[i] = host_bodies[i].ethermal;
//   }

//   sph_accelerations();
//   double t1 = get_time();
//   cerr << " Computing gravity ... ";
//   GPU_gravity();
//   cerr << " done in " << get_time() - t1 << " sec\n";


// //   sph_accelerations();
// //   GPU_gravity();

//   dt = compute_timestep(u);

//   dt = 0;
  

//   float dth = 0.5 * dt;
  
//   for (int i = 0; i < n_bodies; i++) {
//     host_bodies[i].ethermal = u[i];
    
//     host_bodies[i].vel.x += host_bodies[i].v_dot.x * dth;
//     host_bodies[i].vel.y += host_bodies[i].v_dot.y * dth;
//     host_bodies[i].vel.z += host_bodies[i].v_dot.z * dth;
//   }

//   build_left_balanced_tree(n_bodies, kd_tree, hst.bodies_pos);
//   for (int i = 0; i < n_bodies; i++) {
//     if (i%100 == 0)
//       fprintf(stderr, "i= %d\n" ,i);
//     int n_ngb = 0;
//     for (int j = 0; j < n_bodies; j++) {
//       float3 dr;
//       dr.x = hst.bodies_pos[i].x - kd_tree[j+1].pos[0];
//       dr.y = hst.bodies_pos[i].y - kd_tree[j+1].pos[1];
//       dr.z = hst.bodies_pos[i].z - kd_tree[j+1].pos[2];
// //       dr.x = hst.bodies_pos[i].x - hst.bodies_pos[j].x;
// //       dr.y = hst.bodies_pos[i].y - hst.bodies_pos[j].y;
// //       dr.z = hst.bodies_pos[i].z - hst.bodies_pos[j].z;
//       float ds2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
//       if (ds2 < hst.bodies_pos[i].w * hst.bodies_pos[i].w)
// 	n_ngb++;
      
//     } 
//     printf("%d\n", n_ngb);
//   }
//   exit(-1);
  
}
