CXX = g++
CC  = gcc
LD  = g++
F90  = ifort

.SUFFIXES: .o .f90 .cpp

OFLAGS = -O0 -g -Wall 
OFLAGS = -D_FILE_OFFSET_BITS=64  -O3 -mpreferred-stack-boundary=4 -funroll-loops -fforce-addr
CFLAGS = $(OFLAGS) -I/usr/local/cuda/include
CXXFLAGS = $(CFLAGS)

NVCC      = /usr/local/cuda/bin/nvcc
NVCCFLAGS =  -D_DEBUG -O3 -I/home/jbedorf/NVIDIA_GPU_Computing_SDK/C/common/inc/
# NVCCFLAGS = -D_DEBUG  --maxrregcount=32  -O3 -I/home/nvidia/NVIDIA_CUDA_SDK/common/inc
# NVCCFLAGS = -D_DEBUG --device-emulation   -O0 -g -I/home/nvidia/NVIDIA_CUDA_SDK/common/inc


PROG = CUDAsph_iterate

OBJS = $(PROG).o kd_tree.o read_state.o write_state.o \
	sph_accelerations.o stride_setup.o \
	step.o compute_timestep.o system_statistics.o  \
	memory_alloc_free.o \
	CUDA_calls.o CUDA_gravity.o copy_data_to_from.o \
	output.o sort_bodies.o set_device.o

CUOBJS = host_calls.cu_o radixsort/radixsort.cu_o

#  host_sph_gravity.cu_o
LIBS = -L/usr/local/cuda/lib64 -lcudart

all: $(PROG)

$(PROG): $(OBJS) $(CUOBJS)
	$(LD) $(CXXFLAGS) $(LIBS) $^ -o $@ -lcudart

.cpp.o: 
	$(CXX) $(CXXFLAGS) -c $< -o $@

.f90.o:
	$(F90) $(F90FLAGS) -c $< -o $@

%.cu_o:  %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

convert:
	ifort -O3 -o convert convert.f90

clean:
	/bin/rm -rf *.o *.cu_o radixsort/*.cu_o
	/bin/rm -rf $(PROG)

kd_tree.o: kd_tree.h CUDAsph.h
host_calls.cu_o: dev_solve_range.cu \
		dev_find_ngb.cu \
		dev_density_loop.cu \
		dev_compute_pressure.cu \
		dev_hydro_dots.cu \
		dev_art_visc.cu \
		sph_grav_kernel.cu \
		dev_sph_kernels.cu
