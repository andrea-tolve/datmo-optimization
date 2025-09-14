#include "cluster_cuda.hpp"

const int d = 25;
const int x_row = 1000;
const float step = (3.14 / 2) / d;
const int MAX_VAL = 10000;
const int MIN_VAL = -10000;

void memAllocate() {
    cudaMalloc((void**)&d_idmaxth, sizeof(int));
    cudaMalloc((void**)&d_x, sizeof(double) * x_row * 2);
    cudaMalloc((void**)&d_row, sizeof(int));
    cudaMalloc((void**)&d_max, sizeof(double) * 2 * d);
    cudaMalloc((void**)&d_min, sizeof(double) * 2 * d);
    cudaMalloc((void**)&d_coef, sizeof(double) * 12);
    cudaMalloc((void**)&d_q, sizeof(double) * 2 * d);
    cudaMallocHost(&h_coef, sizeof(double) * 12);
    cudaMallocHost(&h_x, sizeof(double) * x_row * 2);
    cudaStreamCreate(&s1); 
    cudaStreamCreate(&s2);
    cublasCreate(&handle);

    cublasStatus_t stat;
    stat = cublasSetStream(handle, s1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cout << "errore nell'associazione tra handle e s1\n" << stat << std::endl;;
    }

    // specify that the pointer must be passed by reference to the device
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
}

void memFree() {
    cudaFree(d_x);
    cudaFree(d_row);
    cudaFree(d_max);
    cudaFree(d_min);
    cudaFree(d_coef);
    cudaFree(d_q);
    cudaFree(&d_idmaxth);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cublasDestroy(handle);
}

/*
    atomic operations for double; by default CUDA does not support 
    these operations on double types, but they can be implemented
    using cast to long long int
*/
__device__ void atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmax(val, __longlong_as_double(assumed)))); 
    } while (assumed != old); 
}

__device__ void atomicMinDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmin(val, __longlong_as_double(assumed)))); 
    } while (assumed != old); 
}

__device__ void atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
}

/*
    function to compute the matrix-vector product (X * e1/e2)
    and to store the maximum and minimum values of c1 and c2
*/
__device__ void mulMatrix(double* c1, double* c2, double* max, double* min, double th, double* x, int col, int idx, int idy) {
    double e0, e1, e2, e3;
    double cos_th = cos(th);
    double sin_th = sin(th);

    e0 = cos_th;
    e1 = sin_th;
    e2 = -sin_th;
    e3 = cos_th;

    double x_idy = x[idy];
    double x_idy_row = x[idy + col];
    double sum;

    sum = x_idy * e0 + x_idy_row * e1; 
    c1[idy] = sum;

    atomicMaxDouble(&max[idx * 2], sum);
    atomicMinDouble(&min[idx * 2], sum);

    sum = x_idy * e2 + x_idy_row * e3;
    c2[idy] = sum;

    atomicMaxDouble(&max[idx * 2 + 1], sum);
    atomicMinDouble(&min[idx * 2 + 1], sum);


}

__global__ void calcQKernel(double* x, int* row, double* max, double* min, double* Q) {
    int idx = blockIdx.x;
    int idy = threadIdx.x;
    double th = step * idx;
    int r = row[0];
    if (idy == 0) {
        max[idx * 2] = MIN_VAL;
        max[idx * 2 + 1] = MIN_VAL;
        min[idx * 2] = MAX_VAL;
        min[idx * 2 + 1] = MAX_VAL;
    }

    extern __shared__ double sharedMem[];

    double* c1 = &sharedMem[0];
    double* c2 = &sharedMem[r];
    double* c1max = &sharedMem[2 * r];
    double* c2max = &sharedMem[3 * r];
    double* c1min = &sharedMem[4 * r];
    double* c2min = &sharedMem[5 * r];
   
    mulMatrix(c1, c2, max, min, th, x, r, idx, idy);
    __syncthreads();

    c1max[idy] = max[idx * 2] - c1[idy];
    c1min[idy] = c1[idy] - min[idx * 2];
    c2max[idy] = max[idx * 2 + 1] - c2[idy];
    c2min[idy] = c2[idy] - min[idx * 2 + 1];

    __syncthreads();

    //compute squared norm
    __shared__ double c1maxdata, c1mindata, c2maxdata, c2mindata, b; 
    if (idy == 0) {
        c1maxdata = 0;
        c1mindata = 0;
        c2maxdata = 0;
        c2mindata = 0;
        b = 0;
    }

    atomicAddDouble(&c1maxdata, c1max[idy] * c1max[idy]);
    atomicAddDouble(&c1mindata, c1min[idy] * c1min[idy]);
    atomicAddDouble(&c2maxdata, c2max[idy] * c2max[idy]);
    atomicAddDouble(&c2mindata, c2min[idy] * c2min[idy]);
    __syncthreads();


    double ma, mi, div, d0 = 0.001, val1, val2;

    bool cond1 = c1maxdata >= c1mindata;
    val1 = cond1 * c1min[idy] + !cond1 * c1max[idy]; 

    bool cond2 = c2maxdata >= c2mindata;
    val2 = cond2 * c2min[idy] + !cond2 * c2max[idy]; 


    mi = fmin(val1, val2);
    ma = fmax(mi, d0);
    div = 1 / ma;
    atomicAddDouble(&b, div);

    __syncthreads();

    //reuse the max array to store matrix Q
    if (idy == 0) {
        Q[idx * 2] = th;
        Q[idx * 2 + 1] = b;
    }

    
}


void launchKernelCuda(const double* X,const int num_point, double coef[]) {

    cudaMemcpyAsync(d_row, &num_point, sizeof(int), cudaMemcpyHostToDevice, s1);
    cudaMemcpyAsync(d_x, X, sizeof(double) * num_point * 2, cudaMemcpyHostToDevice, s1);
    int sharedMemorySize = 6 * num_point * sizeof(double);

    calcQKernel << <d, num_point, sharedMemorySize, s1 >> > (d_x, d_row, d_max, d_min, d_q);
   
    int idmax;
	
    //check only column b, starting from the second element and using stride = 2 (skip the element in between)
    cublasIdamax(handle, d, d_q+1, 2, &idmax); //indice che parte da 1
 
    idmax -= 1;
    double th = step * idmax, h_min[2], h_max[2];

    cudaMemcpyAsync(h_min, d_min + idmax * 2, sizeof(double) * 2, cudaMemcpyDeviceToHost, s1);
    cudaMemcpyAsync(h_max, d_max + idmax * 2, sizeof(double) * 2, cudaMemcpyDeviceToHost, s2);

    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);

    coef[0] = cos(th);
    coef[1] = sin(th);
    coef[2] = h_min[0];
    coef[3] = -sin(th);
    coef[4] = coef[0];
    coef[5] = h_min[1];
    coef[6] = coef[0];
    coef[7] = coef[1];
    coef[8] = h_max[0];
    coef[9] = coef[3];
    coef[10] = coef[0];
    coef[11] = h_max[1];


}
