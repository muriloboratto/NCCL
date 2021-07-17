/*%****************************************************************************80
!  Code: 
!   laplacian-3d-1GPU.cu
!
!  Purpose:
!   Implements sample 3D Laplacian Method in C/C++ code using CUDA.
!
!  Modified:
!   Aug 18 2020 10:57 
!
!  Author:
!   Murilo Boratto <murilo.boratto 'at' fieb.org.br>
!   Reynam da Cruz Pestana <reynam 'at' ufba.br>
!
!  How to Compile:
!   nvcc laplacian-3d-1GPU.cu -o object
!
!  Execute: 
!   ./object
!
!  Comments: 
!   1) Simple Testbed with size problem n = m = 8.      
!****************************************************************************80*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

__global__ void kernel(double *a, double *c, int nx, int ny, int nz, int ista2, int iend2, int dx, int dy, int dz) {

  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  int j = blockIdx.y * blockDim.y + threadIdx.y; 
  int k = blockIdx.z * blockDim.z + threadIdx.z; 

  double sx, sy, sz;

  if( k >= 1 && k < (ny - 1) && j >= 1 && j < (nx - 1) && i >= (ista2 - 1) &&  i < iend2 ) {
     sz = a[(i-1) +   j  *ny  +  k   *(nx*ny)]  + a[(i+1) +  j    *ny    +  k   *(nx*ny)] + 2 * a[i + j*ny + k*(nx*ny)];
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


void show_matrix(double *a, int n){

   int i, j;

   for(i = 0; i < n; i++){
      for(j = 0; j < n; j++){
        printf("%1.2f\t", a[i + j*n]);
      }
    printf("\n");
   }

   printf("\n\n");

}/*show_matrix*/

void show_vector(int *a, int n){

   int i;

   for(i = 0; i < n; i++)
     printf("%d\t", a[i]);

   printf("\n\n");

}/*show_vector*/

void PARA_RANGE_1(int n1,int n2, int nprocs, int myid,  int *vector_return){

        int ista, iend;
	int iwork1 = (n2 - n1 + 1) / nprocs;
	int iwork2 = (n2 - n1 + 1) % nprocs;

	ista   = (myid * iwork1) + n1 + fmin((double) myid, (double) iwork2);
        iend   = ista + iwork1 - 1;

	if (iwork2 > myid)
	 iend = iend + 1;

        vector_return[0] = ista;
        vector_return[1] = iend;

        printf("%d\t%d\n",vector_return[0], vector_return[1]);
       
} /*PARA_RANGE_1*/


int main(int argc, char *argv[]){
    
    int nx = 8; 
    int myid;
    int nprocs;
    int ny, nz;
    double *a, *c;
    double dx, dy, dz;
    int ista, iend, ista2, iend2;
    int i, j, k;
    int *vector_return = (int *) calloc (2, sizeof(int));
        
    myid = 0; 	 
    nprocs = 1;

    ny = nz = nx;
      
    a  =  (double*) calloc (nz * nx * ny, sizeof(double));
    c  =  (double*) calloc (nz * nx * ny, sizeof(double));

    dx = 1.; dy = 1.; dz = 1.;
	         
    PARA_RANGE_1(1, nz, nprocs, myid, vector_return);

	ista = vector_return[0];
	iend = vector_return[1];

    ista2 = ista;
	iend2 = iend;

	if(myid == 0)
	   ista2 = 2;

    if(myid == (nprocs - 1))
	   iend2 = nz - 1;

    /*Population the matrix*/
    for(k = 0; k < ny; k++)
	   for(j = 0; j < nx; j++)
    	     for(i = ista - 1; i < iend; i++)    
                  a[i + j*ny + k*(nx*ny)] = (i + j + 2) * 1.;  /*stored dates in column*/

    show_matrix_3D(a, ny); 

    printf("\n---------------------------------------------------------------\n\n");


    /*Alloc Device's Variables*/ 
    double *d_a;
 	double *d_c;
 
    cudaMalloc((void **) &d_a,  nx * ny * nz * sizeof(double));
    cudaMalloc((void **) &d_c,  nx * ny * nz * sizeof(double));

	cudaMemset(d_a, 0, nx * ny * nz * sizeof(double));
	cudaMemset(d_c, 0, nx * ny * nz * sizeof(double));

    /*Copy Matrix 'a' from host to device*/
    cudaMemcpy(d_a, a, nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice );

    /* 3D GRID and SIZEBLOCK definitions*/
    int  sizeblock = nx / 2;
    int grid = (int) ceil( (double) nx / (double) sizeblock );
    dim3 dimGrid( grid, grid, grid );
    dim3 dimBlock(sizeblock, sizeblock, sizeblock);
       
    kernel<<< dimGrid, dimBlock >>>(d_a, d_c, nx, ny, nz, ista2, iend2, dx, dy, dz);

    /*Copy Matrix 'd_c' from device to host*/
    cudaMemcpy(c , d_c, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost );

    show_matrix_3D(c, nz); 

    free(a);
    free(c);

    cudaFree(d_a) ;
    cudaFree(d_c) ;
    
    return 0;

}/*main*/
