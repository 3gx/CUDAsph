#include "CUDAsph.h"

void CUDAsph::set_device(int __device_id) {
  int n_devices = get_device_count();

  if (n_devices == 0) {
    cerr << "**************************************************************\n";
    cerr << "  No CUDA capable device is found. I am not able to proceed.\n";
    cerr << "***************************************************************\n";
    exit(-1);
  } else if (n_devices == 1) {
    device_id = -1;
    cerr << "*******************************\n";
    cerr << "  Found a CUDA capable device \n";
    cerr << "*******************************\n";
  } else {
    cerr << "********************************\n";
    cerr << "  Found " << n_devices << " CUDA capable devices \n";
    cerr << "--------------------------------\n";
    if ((__device_id < 1) || (__device_id > n_devices)) {
      cerr << "  Please, choose a device# \n";
      cerr << "      in range [1-" << n_devices << "]\n";
      cerr << "********************************\n";
      exit(-1);
    } else {
      device_id = __device_id;
      cuda_set_device0(device_id);
      fprintf(stderr, "    Device# %d will be used\n", device_id);
    }
    cerr << "********************************\n";
  }
}
