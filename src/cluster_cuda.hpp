#include <stdio.h>
#include <iostream>
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cooperative_groups.h>

void launchKernelCuda(const double* X, const int row, double* coef);
void memAllocate();
void memFree();

static cudaStream_t s1;
static cudaStream_t s2;
static cublasHandle_t handle;
static int* d_idmaxth;
static double* d_x;
static double* d_q;
static int* d_row;
static double* d_max;
static double* d_min;
static double* d_coef;
static double* h_coef;
static double* h_x;




