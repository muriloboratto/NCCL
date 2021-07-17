/*%****************************************************************************80
!  Code: 
!   laplacian-3d-mpi.c
!
!  Purpose:
!   Implements sample 3D Laplacian Method in C/C++ code using MPI.
!
!  Modified:
!   Jan 20 2019 10:57 
!
!  Author:
!   Reynam da Cruz Pestana <reynam 'at' ufba.br>
!
!  How to Compile:
!   mpicc laplacian-3d-mpi.c -o object
!
!  How to Execute: 
!   mpirun -np 4 ./object
!
!  Comments: 
!   1) Simple Testbed with size problem n = m = 8. 
!****************************************************************************80*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

void PARA_RANGE_1(int n1,int n2, int nprocs, int myid, int *vector_return){

	int iwork1 = (n2 - n1 + 1) / nprocs;
	int iwork2 = (n2 - n1 + 1) % nprocs;

	int ista   = (myid * iwork1) + n1 + fmin( (double) myid, (double) iwork2);
	int iend   = ista + iwork1 - 1;

	if (iwork2 > myid)
	 iend = iend + 1;

   vector_return[0] = ista;
   vector_return[1] = iend;

}/*PARA_RANGE_1*/


int main(int argc, char *argv[]){

   int myid;
   int nprocs;
   int nx, ny, nz;
   double  *a, *b, *c;
   double dx, dy, dz;
   double  sx, sy, sz;
   double *r1, *r2, *s1, *s2;
   int ista, iend, ista2, iend2, inext, iprev;
   int i, j, k;
   int *vector_return = (int *) calloc (2, sizeof(int));;
   //FILE *file;

   MPI_Init (&argc, &argv);
   MPI_Comm_rank (MPI_COMM_WORLD, &myid);
   MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
   MPI_Status status;
   MPI_Request isend1, isend2, irecv1, irecv2;

   if (myid == 0)
  	  nx = ny = nz = 8;  /*size problem*/

   MPI_Bcast(&nx, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   MPI_Bcast(&ny, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   MPI_Bcast(&nz, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);

   a  =  (double*) calloc (nz * nx * ny, sizeof(double));
   b  =  (double*) calloc (nz * nx * ny, sizeof(double));
   c  =  (double*) calloc (nz * nx * ny, sizeof(double));

   r1  = (double*) calloc (nx * ny, sizeof(double));
   r2  = (double*) calloc (nx * ny, sizeof(double));

   s1  = (double*) calloc (nx * ny, sizeof(double));
   s2  = (double*) calloc (nx * ny, sizeof(double));

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

	inext = myid + 1;
	iprev = myid - 1;

	if(myid == ( nprocs - 1 ))
	   inext = MPI_PROC_NULL;

	if(myid == 0)
	   iprev = MPI_PROC_NULL;

   for(k = 0; k < ny; k++)
	  for(j = 0; j < nx; j++)
    	    for(i = ista - 1; i < iend; i++)    
               a[i + j*ny + k*(nx*ny)] = (i + j + 2) * 1.; 
         
   if(myid != (nprocs - 1)){   

      for(k = 0; k < ny; k++){
	     for(j = 0; j < nx; j++){
            s1[j + k*ny] = a[(iend - 1) + j*ny + k*(nx*ny)]; 
         }
       }
   }/*if*/


   if(myid != 0){   
      for(k = 0; k < ny; k++){
	     for(j = 0; j < nx; j++){
            s2[j + k*ny] = a[ista - 1 + j*ny + k*(nx*ny)]; 
         }
      }
   }/*if*/

       
   if(myid == 0){      
      for(k = 0; k < ny; k++)  // testbed - only print matrix!
	       for(j = 0; j < nx; j++)
    	      for(i = 0; i < nx; i++)    
               b[i + j*ny + k*(nx*ny)] = (i + j + 2) * 1.; 
         
      show_matrix_3D(b, ny);

      printf("\n---------------------------------------------------------------\n\n");

   }/*if*/

   MPI_Isend(&s1[0 + 0 * ny], nx * ny, MPI_REAL8, inext, 1, MPI_COMM_WORLD, &isend1);
	MPI_Isend(&s2[0 + 0 * ny], nx * ny, MPI_REAL8, iprev, 1, MPI_COMM_WORLD, &isend2);
	MPI_Irecv(&r1[0 + 0 * ny], nx * ny, MPI_REAL8, iprev, 1, MPI_COMM_WORLD, &irecv1);
	MPI_Irecv(&r2[0 + 0 * ny], nx * ny, MPI_REAL8, inext, 1, MPI_COMM_WORLD, &irecv2);

	MPI_Wait(&isend1, &status);
	MPI_Wait(&isend2, &status);
	MPI_Wait(&irecv1, &status);
	MPI_Wait(&irecv2, &status);

   if(myid != 0){   
      for(k = 0; k < ny; k++)
	      for(j = 0; j < nx; j++)
                a[ista - 2 + j*ny + k*(nx*ny)] = r1[j + k*ny];   
   }/*if*/


   if(myid != ( nprocs - 1 )){   
      for(k = 0; k < ny; k++)
	      for(j = 0; j < nx; j++)
            a[iend + j*ny + k*(nx*ny)] = r2[j + k*ny];   
   }/*if*/


  /*Kernel*/

   for(k = 1; k < ny - 1; k++){
      for(j = 1; j < nx - 1; j++){
         for(i = ista2 - 1 ; i < iend2; i++){ 
	        sy = a[ i    +   j  *ny  + (k-1)*(nx*ny)]  + a[ i    +  j    *ny    + (k+1)*(nx*ny)] + 2 * a[i + j*ny + k*(nx*ny)];
	        sx = a[ i    + (j-1)*ny  +  k   *(nx*ny)]  + a[ i    + (j+1) *ny    +  k   *(nx*ny)] + 2 * a[i + j*ny + k*(nx*ny)];
           sz = a[(i-1) +   j  *ny  +  k   *(nx*ny)]  + a[(i+1) +  j    *ny    +  k   *(nx*ny)] + 2 * a[i + j*ny + k*(nx*ny)];
	        c[i + j*ny + k*(nx*ny)] = (sz/(dz*dz)) + (sx/(dx*dx)) + (sy/(dy*dy));
         }	
      }
   }

   MPI_Reduce(c, b, nz * nx * ny, MPI_REAL8, MPI_SUM, 0, MPI_COMM_WORLD);

   if(myid == 0){
      show_matrix_3D(b, ny);
	   //file = fopen("laplaciano_c.dat", "wb");
      //fwrite(b, sizeof(double), n * m, file);
      //fclose(file);
   }/*if*/

   free(a);
   free(b);
   free(c);

   free(s1);
   free(s2);

   free(r1);
   free(r2);

   MPI_Finalize();

   return 0;

}/*main*/
