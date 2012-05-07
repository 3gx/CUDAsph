#include "CUDAsph.h"

void CUDAsph::output() {
  if ((global_time > snapshot_number*dt_out) && (dt_out > 0)) {
    fprintf(stderr, "\n >>>>>>>> writing output file <<<<<<< \n\n");
    char filename[256];
    snapshot_number++;
    sprintf(filename, "out%04d.bin", snapshot_number);
    write_state(filename);
    t_last_output = global_time;
  }
  char filename[256];
  sprintf(filename, "restart.bin");
  write_state(filename);
  
}
