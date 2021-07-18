/*%****************************************************************************80
!  Code: 
!   laplacian-2d-mpi.c
!
!  Purpose:
!   Implements sample 2D Laplacian Method in C/C++ code using MPI.
!
!  Modified:
!   Aug 18 2020 10:57 
!
!  Author:
!   Murilo Boratto <murilo.boratto 'at' fieb.org.br>
!   Reynam da Cruz Pestana <reynam 'at' ufba.br>
!
!  How to Compile:
!   mpicc aplacian-2d-mpi.c -o object -lm
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

void show_matrix(double *a, int n){

   int i, j;

   for(i = 0; i < n; i++){
      for(j = 0; j < n; j++){
        printf("%1.2f\t", a[i + j*n]);
      }
    printf("\n");
   }

   printf("\n");

}/*show_matrix*/

void show_vector(int *a, int n){

   int i;

   for(i = 0; i < n; i++)
     printf("%d\t", a[i]);

   printf("\n\n");

}/*show_vector*/

void PARA_RANGE_1(int n1,int n2, int nprocs, int myid, int jsta, int jend, int *vector_return){

	int iwork1 = (n2 - n1 + 1) / nprocs;
	int iwork2 = (n2 - n1 + 1) % nprocs;

	jsta   = (myid * iwork1) + n1 + fmin(myid, iwork2);
	jend   = jsta + iwork1 - 1;

	if (iwork2 > myid)
	 jend = jend + 1;

        vector_return[0] = jsta;
        vector_return[1] = jend;

} /*PARA_RANGE_1*/


int main(int argc, char *argv[]){

   int n = 8;
   int m = n;
   int myid;
   int nprocs;
   double dx, dz;
   double *a, *b, *c, *sx, *sz;
   int jsta, jend, jsta2, jend2, inext, iprev ;
   int i, j;
   int *vector_return = (int *) calloc (2, sizeof(int));;
   FILE *file;

   MPI_Init (&argc, &argv);
   MPI_Comm_rank (MPI_COMM_WORLD, &myid);
   MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
   MPI_Status status;
   MPI_Request isend1, isend2, irecv1, irecv2;

   if (myid == 0)
  	  m = n;  

   MPI_Bcast(&n, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   MPI_Bcast(&m, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);

   a  =  (double*) calloc (m * n, sizeof(double));
   b  =  (double*) calloc (m * n, sizeof(double));
   c  =  (double*) calloc (m * n, sizeof(double));
   sx =  (double*) calloc (m * n, sizeof(double));
   sz =  (double*) calloc (m * n, sizeof(double));

   dx = 1.;
   dz = 1.;

   PARA_RANGE_1(1, n, nprocs, myid, jsta, jend, vector_return);

   jsta = vector_return[0];
   jend = vector_return[1];

   jsta2 = jsta;
   jend2 = jend;

  if(myid == 0)
    jsta2 = 2;

   if(myid == (nprocs-1))
     jend2 = n - 1;

   inext = myid + 1;
   iprev = myid - 1;

   if(myid == ( nprocs - 1 ))
     inext = MPI_PROC_NULL;

   if(myid == 0)
     iprev = MPI_PROC_NULL;

   /*Population of the matrix*/
   for(i = 0; i < m; i++)
     for(j = jsta-1; j < jend; j++)
	 a[i + j* n] = (i + j + 2) * 1.;  /*stored dates in column (major column)*/
 
   MPI_Barrier(MPI_COMM_WORLD);

   if(myid == 0){ /*testbed*/
     show_matrix(a, n);
   }

   MPI_Barrier(MPI_COMM_WORLD); /*testbed*/

   if(myid == 1){ 
     show_matrix(a, n);
   }

   MPI_Barrier(MPI_COMM_WORLD); 

   if(myid == 2){ 
      show_matrix(a, n);
   }

   MPI_Barrier(MPI_COMM_WORLD); 

   if(myid == 3){ 
     show_matrix(a, n);
   }

   MPI_Isend(&a[0 + (jend - 1)   * n]  ,   m, MPI_REAL8, inext, 1, MPI_COMM_WORLD, &isend1);
   MPI_Isend(&a[0 + (jsta - 1)   * n]  ,   m, MPI_REAL8, iprev, 1, MPI_COMM_WORLD, &isend2);
   MPI_Irecv(&a[0 + (jsta - 2)   * n]  ,   m, MPI_REAL8, iprev, 1, MPI_COMM_WORLD, &irecv1);
   MPI_Irecv(&a[0 + (jend)       * n]  ,   m, MPI_REAL8, inext, 1, MPI_COMM_WORLD, &irecv2);

   MPI_Wait(&isend1, &status);
   MPI_Wait(&isend2, &status);
   MPI_Wait(&irecv1, &status);
   MPI_Wait(&irecv2, &status);

   for(j = jsta2-1; j < jend2; j++){
      for(i = 1; i < (m - 1); i++){
	     sx[i + j * n] = a[(i-1) + j*n] + a[(i+1)+ j*n] + 2 * a[i + j*n];
	     sz[i + j * n] = a[ i + (j-1)*n] + a[i + (j+1)*n] + 2 * a[i + j*n];
	      c[i + j * n] = (sx[i + j*n]/(dx*dx)) + (sz[i+j*n]/(dz*dz));
      }
   }

   MPI_Reduce(c, b, m * n, MPI_REAL8, MPI_SUM, 0, MPI_COMM_WORLD);

   if (myid == 0){
      show_matrix(b, n);
      //file = fopen("laplaciano_c.dat", "wb");
      //fwrite(b, sizeof(double), n * m, file);
      //fclose(file);
   }

   free(a);
   free(b);
   free(c);
   free(sx);
   free(sz);

   MPI_Finalize();

   return 0;

}/*main*/
