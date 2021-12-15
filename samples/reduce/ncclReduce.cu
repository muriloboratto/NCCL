/*%****************************************************************************80
%  Code: 
%   ncclReduce.cu
%
%  Purpose:
%   Implements sample reduce code using the package NCCL(ncclReduce).
%   Using 'Multiples Devices per Thread'.
%   Implements dot product(scalar product).
%   x = (xo, x1, x2, ..., xn)
%   y = (yo, y1, y2, ..., yn)
%   c = (xo . yo + x1 . y1 + ..., xn . yn)
%
%  Modified:
%   Aug 18 2020 10:57 
%
%  Author:
%   Murilo Boratto <murilo.boratto 'at' fieb.org.br>
%
%  How to Compile:
%   nvcc ncclReduce.cu -o ncclReduce -lnccl  
%
%  Execute: 
%   ./ncclReduce                           
%   
%****************************************************************************80*/

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <nccl.h>

__global__ void Dev_dot(double *x, double *y, int n) {
   
   __shared__ double tmp[512];

   int i = threadIdx.x;
   int t = blockDim.x * blockIdx.x + threadIdx.x;
   
   if (t < n) 
    tmp[i] = x[t];
   
   __syncthreads();

   for (int stride = blockDim.x / 2; stride >  0; stride /= 2) {

      if (i < stride)
         tmp[i] += tmp[i + stride];

      __syncthreads();

   }

   if (threadIdx.x == 0) {
      y[blockIdx.x] = tmp[0];
      printf("\tdot(x,y) = %1.2f\n", y[blockIdx.x]); 
   }

}/*Dev_dot*/     


__global__ void Dev_print(double *x) {
   
   int i = threadIdx.x;
    
   printf("%1.2f\t", x[i]);
   
}/*Dev_print*/     


void print_vector(double *in, int n){

 for(int i=0; i < n; i++)
  printf("%1.2f\t", in[i]);

 printf("\n");

}/*print_vector*/


int main(int argc, char* argv[]) {

  /*Variables*/
  int nGPUs = 0;
  cudaGetDeviceCount(&nGPUs);
  int *DeviceList = (int *) malloc ( nGPUs * sizeof(int));

  int data_size = 8;

  double *x          = (double*)    malloc(data_size * sizeof(double));
  double *y          = (double*)    malloc(data_size * sizeof(double)); 
  double **x_d_data  = (double**)   malloc(nGPUs     * sizeof(double*));
  double **y_d_data  = (double**)   malloc(nGPUs     * sizeof(double*));
  double **Sx_d_data = (double**)   malloc(nGPUs     * sizeof(double*));
  double **Sy_d_data = (double**)   malloc(nGPUs     * sizeof(double*));
 
  for (int i = 0; i < nGPUs; ++i)
      DeviceList[i] = i;
  
  /*Initializing NCCL with Multiples Devices per Thread*/
  ncclComm_t* comms = (ncclComm_t*)  malloc(sizeof(ncclComm_t)  * nGPUs);  
  cudaStream_t* s   = (cudaStream_t*)malloc(sizeof(cudaStream_t)* nGPUs);
  ncclCommInitAll(comms, nGPUs, DeviceList);
      
  /*Population vectors*/
  for(int i = 0; i < data_size; i++){ 
      x[i] = 1;                
      y[i] = 2;
  }                
      
  print_vector(x, data_size); 
  print_vector(y, data_size);
    

  for(int g = 0; g < nGPUs; g++) {
      cudaSetDevice(DeviceList[g]);
      cudaStreamCreate(&s[g]);

      cudaMalloc(&x_d_data[g],    data_size * sizeof(double));
      cudaMalloc(&y_d_data[g],    data_size * sizeof(double));
      
      cudaMalloc(&Sx_d_data[g],   data_size * sizeof(double));
      cudaMalloc(&Sy_d_data[g],   data_size * sizeof(double));
     
      cudaMemcpy(x_d_data[g],  x, data_size * sizeof(double), cudaMemcpyHostToDevice); /*Copy from Host to Devices*/
      cudaMemcpy(y_d_data[g],  y, data_size * sizeof(double), cudaMemcpyHostToDevice);       
    }
      
  ncclGroupStart(); 
  
  	for(int g = 0; g < nGPUs; g++) {
   	  cudaSetDevice(DeviceList[g]);
          ncclReduce(x_d_data[g], Sx_d_data[g], data_size, ncclDouble, ncclSum, 0, comms[g], s[g]); /*Reducing x vector*/
          ncclReduce(y_d_data[g], Sy_d_data[g], data_size, ncclDouble, ncclSum, 0, comms[g], s[g]); /*Reducing y vector*/
        }

  ncclGroupEnd(); 


  for(int g = 0; g < nGPUs; g++) {
      cudaSetDevice(DeviceList[g]);            
      printf("\n This is device %d\n", g);
      Dev_dot <<< 1, data_size >>> (Sy_d_data[g], Sx_d_data[g], data_size); /*Call the CUDA Kernel: dot product*/
      cudaDeviceSynchronize();  
  }
  
  for (int g = 0; g < nGPUs; g++) { /*Synchronizing CUDA Streams*/
      cudaSetDevice(DeviceList[g]);
      cudaStreamSynchronize(s[g]);
  }
  
  for(int g = 0; g < nGPUs; g++) { /*Destroy CUDA Streams*/
      cudaSetDevice(DeviceList[g]);
      cudaStreamDestroy(s[g]);
  }

  for(int g = 0; g < nGPUs; g++) /*Finalizing NCCL*/
     ncclCommDestroy(comms[g]);
  
  /*Freeing memory*/
  free(s);
  free(x);
  free(y);
  free(DeviceList);

  cudaFree(x_d_data);
  cudaFree(y_d_data);
  cudaFree(Sx_d_data);
  cudaFree(Sy_d_data);

  return 0;

}/*main*/