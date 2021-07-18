/*%****************************************************************************80
!  Code: 
!   laplacian-3d-multi-gpu.cu
!
!  Purpose:
!   Implements sample 3D Laplacian Method in C/C++ code using CUDA and NCCL on multi-GPU systems.
!
!  Modified:
!   Aug 13 2020 10:57 
!
!  Author:
!   Murilo do Carmo Boratto <murilo.boratto 'at' fieb.org.br>
!   Reynam da Cruz Pestana <reynam 'at' ufba.br>
!
!  How to Compile:
!   nvcc laplacian-3d-multi-gpu.cu -o object -lnccl -Xcompiler -fopenmp
!     
!    ◆ Print options: 
!         nvcc laplacian-3d-multi-gpu.cu -o object -lnccl -Xcompiler -fopenmp -DprintLabel
!
!  How to Execute: 
!   ./object
!
!  Comments: 
!   1) Approach for the 4 GPUs in the same node with size problem n = m = 8
!****************************************************************************80*/

#include <cuda.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

__global__ void kernel(double *a, double *c, double *stencil, int nx, int ny, int nz, int ista2, int iend2, double dx, double dy, double dz) {

  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  int j = blockIdx.y * blockDim.y + threadIdx.y; 
  int k = blockIdx.z * blockDim.z + threadIdx.z; 

  double sx, sz, sy;
  
  if( k >= 1 && k < (ny - 1) && j >= 1 && j < (nx - 1) && i >= (ista2 - 1) &&  i < iend2 ) {
     sz = a[(i-1) +   j  *ny  +  k   *(nx*ny)]  + stencil[(i+1) +  j    *ny    +  k   *(nx*ny)] + 2 * a[i + j*ny + k*(nx*ny)];
     sx = a[ i    + (j-1)*ny  +  k   *(nx*ny)]  + a[ i    + (j+1) *ny    +  k   *(nx*ny)] + 2 * a[i + j*ny + k*(nx*ny)];
     sy = a[ i    +   j  *ny  + (k-1)*(nx*ny)]  + a[ i    +  j    *ny    + (k+1)*(nx*ny)] + 2 * a[i + j*ny + k*(nx*ny)];
     c[i + j*ny + k*(nx*ny)] = (sz/(dz*dz)) + (sx/(dx*dx)) + (sy/(dy*dy));
  }
      
}/*kernel*/



void show_matrix_3D(double *a, int n){

   for(int k = 0; k < n; k++){
     for(int j = 0; j < n; j++){
         for(int i = 0; i < n; i++){
           printf("%1.2f\t", a[i + j*n + k*(n*n)]); 
         }
       printf("\n");
     }
    printf("\n\n");
   }

}/*show_matrix_3D*/


void PARA_RANGE(int n1,int n2, int nprocs, int myid, int *vector_return){

	int iwork1 = (n2 - n1 + 1) / nprocs;
	int iwork2 = (n2 - n1 + 1) % nprocs;

	int jsta   = (myid * iwork1) + n1 + fmin((double) myid, (double) iwork2);
	int jend   = jsta + iwork1 - 1;

	if (iwork2 > myid)
	 jend = jend + 1;

    vector_return[0] = jsta;
    vector_return[1] = jend;

}/*PARA_RANGE*/


void SUPER_PARA_RANGE(int n, int nprocs, int myid, int *vector_return, double *host_a){   

   	PARA_RANGE(1, n, nprocs, myid, vector_return);

	int jsta = vector_return[0];
	int jend = vector_return[1];
        
    for(int k = 0; k < n; k++)
	   for(int j = 0; j < n; j++)
    	     for(int i = jsta - 1; i < jend; i++)    
                  host_a[i + j*n + k*(n*n)] = (i + j + 2) * 1.;  

}/*SUPER_PARA_RANGE*/


void freeMemoryApp(int *DeviceList, cudaStream_t *s, ncclComm_t *comms, double **device_a, double **device_c, 
                   double **Solution_reduced_device, double **stencil, int nGPUs){

    	 for(int g = 0; g < nGPUs; g++){ 
      	 cudaSetDevice(DeviceList[g]);
         cudaStreamSynchronize(s[g]);
        }
    
        for(int g = 0; g < nGPUs; g++){  
         cudaSetDevice(DeviceList[g]);
         cudaStreamDestroy(s[g]);
        }

        for(int g = 0; g < nGPUs; g++)    
          ncclCommDestroy(comms[g]);
      
        cudaFree(device_a);
        cudaFree(device_c);
        cudaFree(Solution_reduced_device);
        cudaFree(stencil);

}/*freeMemoryApp*/


void definitionParticionNCCLSendRecv(int *partition, int n, int nGPUs){

 int *vector_return = (int *) calloc (2, sizeof(int));
 int i = 0;

 for(int myid=0; myid < nGPUs; ++myid){
   PARA_RANGE(1, n, nGPUs, myid, vector_return);
   partition[i++] = vector_return[1];
 }

 free(vector_return);   

}/*definitionParticionNCCLSendRecv*/


/******************************************** MAIN ***********************************************************************************/

int main(int argc, char *argv[]){

      int n = 8;                                  /*Size problem*/
      int nGPUs = 0;              
      cudaGetDeviceCount(&nGPUs);                /*Definitions the numbers of GPUs of the System*/
      int *DeviceList = (int *) malloc ( nGPUs * sizeof(int));
    
      for(int i = 0; i < nGPUs; ++i)             /*Adding number of GPUs in a list*/
        DeviceList[i] = i;

      /*Inicializing NCCL*/
      ncclComm_t* comms = (ncclComm_t*)   malloc(nGPUs * sizeof(ncclComm_t));  
      cudaStream_t* s   = (cudaStream_t*) malloc(nGPUs * sizeof(cudaStream_t));
      ncclCommInitAll(comms, nGPUs, DeviceList);

      /*General Variables*/
      double **device_a                 = (double**) malloc (nGPUs     * sizeof(double*));
      double **device_c                 = (double**) malloc (nGPUs     * sizeof(double*));
      double **Solution_reduced_device  = (double**) malloc (nGPUs     * sizeof(double*));
      double **stencil                  = (double**) malloc (nGPUs     * sizeof(double*));
     
      /* 3D GRID and SIZEBLOCK definitions*/
      int  sizeblock = n / 2 ;
      int  grid = (int) ceil( (double) n / (double) sizeblock );
      dim3 dimGrid( grid, grid, grid );
      dim3 dimBlock(sizeblock, sizeblock, sizeblock);
   
/**************************************************************************************************************************/


/*Step 1 - Divison of Integral Domain of the Matrix A on GPUs*/

      for(int myid = 0; myid < nGPUs; ++myid){
     
        double  *host_a    =  (double*) calloc (n * n * n, sizeof(double)); 
        int *vector_return = (int *) calloc (2, sizeof(int));
    
        SUPER_PARA_RANGE(n, nGPUs, myid, vector_return, host_a);   

        cudaSetDevice(DeviceList[myid]);

        cudaMalloc(&device_a[myid],  n  * n * n * sizeof(double));  
	      cudaMalloc(&device_c[myid],  n  * n * n * sizeof(double));     
	      cudaMalloc(&stencil[myid],   n  * n * n * sizeof(double));     
	      cudaMalloc(&Solution_reduced_device[myid],  n  * n * n * sizeof(double));     
  
        cudaMemcpy(device_a[myid], host_a,   n * n * n * sizeof(double), cudaMemcpyHostToDevice) ;
 
        free(host_a);
        free(vector_return);
     
       }

      #ifdef printLabel  
      for(int myid = 0; myid < nGPUs; ++myid){
        double  *host_PrintStep1  =   (double*) calloc (n * n * n, sizeof(double)); 
        cudaMemcpy(host_PrintStep1, device_a[myid], n * n * n * sizeof(double), cudaMemcpyDeviceToHost);
        printf("GPU=%d ************************************************************************************\n\n", myid);
        show_matrix_3D(host_PrintStep1, n); 
        free(host_PrintStep1);
      }
      #endif

/**************************************************************************************************************************/

/* Step 2 - Start the NCCL - ncclSend / ncclRecv */

int *partition = (int *) calloc (nGPUs, sizeof(int));
definitionParticionNCCLSendRecv(partition, n, nGPUs);

ncclGroupStart(); 

       for(int myid = 0; myid < nGPUs; ++myid){ 
        //Send/Recv FROM GPU = 1 TO GPU = 0         
        ncclSend(device_a[1]+((partition[0] + 0*n) + 0*n),  n * n * n, ncclDouble, myid, comms[myid], s[myid]);
        ncclRecv(stencil[0] + ((partition[0] + 0*n) + 0*n), n * n * n, ncclDouble, myid, comms[myid], s[myid]); 
       }

    
ncclGroupEnd(); 

ncclGroupStart(); 
 
      for(int myid = 0; myid < nGPUs; ++myid){
        //Send/Recv FROM GPU = 2 TO GPU = 1     
        ncclSend(device_a[2]+((partition[1] +  0*n) + 0*n),  n * n * n, ncclDouble, myid, comms[myid], s[myid]);
        ncclRecv(stencil[1] + ((partition[1] +  0*n) +0*n), n * n * n, ncclDouble, myid, comms[myid], s[myid]);
      }

ncclGroupEnd(); 

ncclGroupStart(); 

      for(int myid = 0; myid < nGPUs; ++myid){
        //Send/Recv FROM GPU = 3 TO GPU = 2     
        ncclSend(device_a[3]+((partition[2] +  0*n) + 0*n), n * n * n, ncclDouble, myid, comms[myid], s[myid]);
        ncclRecv(stencil[2] + ((partition[2] +  0*n) + 0*n), n * n * n, ncclDouble, myid, comms[myid], s[myid]); 
      }

ncclGroupEnd(); 
   
    #ifdef printLabel   
     for(int mystencil = 0; mystencil < nGPUs-1; ++mystencil){      
       double  *host_PrintStep2  =  (double*) calloc (n * n * n, sizeof(double)); 
       cudaMemcpy(host_PrintStep2, stencil[mystencil], n * n * n * sizeof(double), cudaMemcpyDeviceToHost);
       printf("Stencil=%d ************************************************************************************\n\n", mystencil); 
       show_matrix_3D(host_PrintStep2, n); 
       free(host_PrintStep2); 
      }
    #endif

/***************************************************************************************************************************************/

/* Step 3 - Building kernel CUDA*/

ncclGroupStart(); 

    for(int myid = 0; myid < nGPUs; myid++){

        cudaSetDevice(DeviceList[myid]);
        cudaStreamCreate(&s[myid]);  
        
        double dx,dy,dz; 
        dx = dy = dz = 1.0;
 
        int *vector_return = (int *) calloc (2, sizeof(int));
        PARA_RANGE(1, n, nGPUs, myid, vector_return);

        int jsta = vector_return[0];
	      int jend = vector_return[1] + 1;
 
        printf("jsta=%d\tjend=%d\n", jsta, jend);
     
        if(myid == 0)
	        jsta = 2;

        if(myid == (nGPUs - 1))
	        jend = n - 1;
           
        kernel <<< dimGrid, dimBlock >>> (device_a[myid], device_c[myid], stencil[myid], n, n, n, jsta, jend, dx, dy, dz);
        
        ncclReduce(device_c[myid], Solution_reduced_device[myid], n * n * n, ncclDouble, ncclSum, 0, comms[myid], s[myid]);

        free(vector_return);
        
    }
 
ncclGroupEnd();   

       #ifdef printLabel
        double  *host_PrintStep3  =  (double*) calloc(n * n * n, sizeof(double)); 
        cudaMemcpy(host_PrintStep3, Solution_reduced_device[0], n * n * n * sizeof(double), cudaMemcpyDeviceToHost);
        printf("LAPLACIAN ************************************************************************************\n\n");   
        show_matrix_3D(host_PrintStep3, n); 
        free(host_PrintStep3);
       #endif
   
/**************************************************************************************************************************/

/* Step 4 - Free Memory (System Operation, CUDA and NCCL)*/

     freeMemoryApp(DeviceList, s, comms, device_a, device_c, Solution_reduced_device, stencil, nGPUs);

     return 0;

}/*main*/
