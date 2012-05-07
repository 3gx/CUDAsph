 #include "CUDAsph.h"

void CUDAsph::sph_accelerations() {

  double t1, t_begin = get_time();
  
  fprintf(stderr, " >>>>> SPH accelerations <<<<< \n");

  /***************************************
   *  prepare arrays for kd-tree build   *
    **************************************/


  cerr << "        building kd-tree ... ";
  t1 = get_time();
  build_left_balanced_tree(n_bodies, kd_tree, hst.bodies_pos);	
  build_kd_hash(n_bodies, kd_tree, hst.particle_map);
  cerr << " done in " << get_time() - t1 << " sec\n";
  
  /* preparing kd-tree data for the device */
  for (int i = 1; i <= n_bodies; i++) {
    hst.kd_tree[i].x = kd_tree[i].pos[0];
    hst.kd_tree[i].y = kd_tree[i].pos[1];
    hst.kd_tree[i].z = kd_tree[i].pos[2];
    // bits 30, 31 & 32 provide split dimension.
    // To extract the data use the following:
    // split   = (data &  (3 << (30))) >> 30
    // body_id = data & ~(3 << (30))

    int data = (kd_tree[i].body_id | (kd_tree[i].split << (24))); 
    *(int*)&hst.kd_tree[i].w = data;

  }


  /*****************
   *  solve range  *
   *****************/

  cerr << "        copying bodies_pos & kd_tree data to the device ... ";
  t1 = get_time();
  copy_kd_tree_to_device();
  copy_particle_map_to_device();
  copy_bodies_pos_vel_to_device();
  cerr << " done in " << get_time() - t1 << " sec\n";

  double dt_gpu = CUDA_solve_range();
  
  cerr << "        copying n_ngb from the device ... ";
  t1 = get_time();
  copy_n_ngb_from_device();
  cerr << " done in " << get_time() - t1 << " sec\n";

  /***********************/
  /* allocate NGB arrays */
  /***********************/
  
  ngb_tot = 0;
  hst.ngb_offset[0] = 0;
  for (int i = 0; i < n_dev_bodies; i++) {
    int n_ngb = n_norm(hst.n_ngb[i], 4);
    hst.ngb_offset[i+1] = hst.ngb_offset[i] + n_ngb;
    ngb_tot += n_ngb;
  }

  if (ngb_tot > ngb_tot_max) {
    fprintf(stderr, "FATAL! not enough memory allocated for nearest neighbour list\n");
    fprintf(stderr, "  ngb_tot_max= %d, but currently we need ngb_tot= %d \n",
	    ngb_tot_max, ngb_tot);
    fprintf(stderr, "  quit \n");
    exit(-1);
  }
  
  cerr << "        copying hydro_data to the device ... ";
  t1 = get_time();
  copy_hydro_data_to_device();
  cerr << " done in " << get_time() - t1 << " sec\n";

  cerr << "        copying ngb_offset to the device ... ";
  t1 = get_time();
  copy_ngb_offset_to_device();
  cerr << " done in " << get_time() - t1 << " sec\n";
  
  cerr << "        sph_accelerations on the device ... \n";  t1 = get_time();
  dt_gpu += CUDA_sph_accelerations();
  cerr << "        total time is " << get_time() - t1 << " sec\n";
  
  cerr << "        copying bodies from the device ... "; t1 = get_time();
  copy_bodies_dots_from_device();		       
  copy_hydro_data_from_device();
  copy_hydro_data2_from_device();
  cerr << " done in " << get_time() - t1 << " sec\n";

  cerr << "        copying ngb_list & ngb_acc from the device ... "; t1 = get_time();
  copy_ngb_list_from_device();
  copy_ngb_acc_from_device();
  cerr << " done in " << get_time() - t1 << " sec\n";

  cerr << "        copying bodies_pos from the device ... "; t1 = get_time();
  copy_bodies_pos_from_device();
  cerr << " done in " << get_time() - t1 << " sec\n";


  cerr << "        add_ngb_acc on the host ... "; t1 = get_time();

  for (int i = 0; i < n_bodies; i++) {
    for (int k = hst.ngb_offset[i]; k < hst.ngb_offset[i] + hst.n_ngb[i]; k++) {
      int     j = hst.ngb_list[k];
      float acc = hst.ngb_acc[k];
      hst.bodies_dots[j].x += acc * (hst.bodies_pos[i].x - hst.bodies_pos[j].x);
      hst.bodies_dots[j].y += acc * (hst.bodies_pos[i].y - hst.bodies_pos[j].y);
      hst.bodies_dots[j].z += acc * (hst.bodies_pos[i].z - hst.bodies_pos[j].z);
    }
  }
  cerr << " done in " << get_time() - t1 << " sec\n";


  
  
  fprintf(stderr,"  Device was used for %f seconds\n", dt_gpu);
  fprintf(stderr, " >>>>> done SPH accelerations in %f sec <<<< \n", get_time() - t_begin);
}


