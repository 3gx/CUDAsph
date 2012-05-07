#include "CUDAsph.h"
#include "vector3.h"

void CUDAsph::system_statistics() {

  vec r_min, r_max, pos_com, vel_com, Jtot;
  float m_tot = 0;

  r_min = vec(hst.bodies_pos[0].x,
	      hst.bodies_pos[0].y,
	      hst.bodies_pos[0].z);
  r_max = r_min;

  pos_com = vel_com = Jtot = vec(0,0,0);

  float Etot, Wtot, Utot, Ttot, Stot;
  Etot = Wtot = Utot = Ttot = Stot = 0;

  int n_ngb_min = n_bodies, n_ngb_max = 0;
  double n_ngb_mean = 0, n_ngb_sigma = 0;
  
  for (int i = 0; i < n_bodies; i++) {

    vec pos = vec(hst.bodies_pos[i].x, hst.bodies_pos[i].y, hst.bodies_pos[i].z);
    vec vel = vec(hst.bodies_vel[i].x, hst.bodies_vel[i].y, hst.bodies_vel[i].z);

    r_min.x = min(r_min.x, pos.x);
    r_min.y = min(r_min.y, pos.y);
    r_min.z = min(r_min.z, pos.z);

    r_max.x = max(r_max.x, pos.x);
    r_max.y = max(r_max.y, pos.y);
    r_max.z = max(r_max.z, pos.z);

    float mass = hst.bodies_vel[i].w;
    m_tot += mass;

    pos_com += mass * pos;
    vel_com += mass * vel;
    Jtot   +=  mass * pos%vel;

    Utot +=       mass * hst.hydro_data[i].z;
    Wtot += 0.5 * mass * hst.grav_data[i].w;
    Ttot += 0.5 * mass * vel*vel;

    n_ngb_min = min(n_ngb_min, hst.n_ngb[i]);
    n_ngb_max = max(n_ngb_max, hst.n_ngb[i]);
    n_ngb_mean  += hst.n_ngb[i];
    n_ngb_sigma += hst.n_ngb[i]*hst.n_ngb[i];
    
//     Stot += body.mass * 1.5* (1.380658e-16)/body.mean_mu
//           stot=stot+am(i)*(
//       $        1.5d0*boltz/meanmolecular(i)*
//       $        log(ugas*rho(i)**(-2.d0/3.d0)) +
//       $        4.d0/3.d0*arad*temperature**3.d0/rhocgs)
  }


  n_ngb_mean  *= 1.0/n_bodies;
  n_ngb_sigma *= 1.0/n_bodies;
  n_ngb_sigma  = sqrt(n_ngb_sigma - n_ngb_mean*n_ngb_mean);

  pos_com *= 1.0/m_tot;
  vel_com *= 1.0/m_tot;
  Etot += Ttot + Utot + Wtot;
  
  fprintf(stderr, "    OUTPUT: Iteration %d   time=  %f\n",
	  iteration_number, global_time);

  fprintf(stderr, "     Computational box:  %g < x < %g \n", r_min.x, r_max.x);
  fprintf(stderr, "                         %g < x < %g \n", r_min.y, r_max.y);
  fprintf(stderr, "                         %g < x < %g \n", r_min.z, r_max.z);
  fprintf(stderr, "     Energitics:  W= %g  T= %g  U= %g \n", Wtot, Ttot, Utot);
  fprintf(stderr, "                  Etot= %g  Stot= %g \n", Etot, Stot);
  fprintf(stderr, "     Kinematics:  Rcm= %g  Vcm= %g  Jtot= %g\n",
	  pos_com.abs(), vel_com.abs(), Jtot.abs() );
  fprintf(stderr, "     Neighbours:  <ngb>= %g +/- %g  min= %d  max= %d\n",
	  n_ngb_mean, n_ngb_sigma, n_ngb_min, n_ngb_max);

  fprintf(stdout, "%g  %g  %g  %g  %g  %g  %g \n",
	  global_time, Wtot, Ttot, Utot, Etot, Stot, Jtot.abs());
  fflush(stdout);
  
}
