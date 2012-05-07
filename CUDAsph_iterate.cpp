#include "CUDAsph.h"

#include <stdio.h>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "please pass file name \n";
    exit(-1);
  }

  int device_id = -1;
  if (argc > 2) {
    device_id = atoi(argv[2]);
  }

  CUDAsph system(argv[1], device_id);
  system.stride_setup();

     while(system.keep_going()) {
//      for (int i = 0; i < 5; i++) {
      
    system.step();
    system.output();
}
 
//     system.output();
//   }


  cerr << "end-of-simulation\n";
}
