/*%****************************************************************************80
%  Code: 
%   mpiReduce.c
%
%  Purpose:
%   Implements sample code using collective operator Reduce with the package MPI.
%   The code calculate the dot product(scalar product).
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
%   mpicxx mpiReduce.c -o mpiReduce 
%
%  Execute: 
%   mpirun -np 2 ./mpiReduce  <size problem>                            
%   mpirun -np 2 ./mpiReduce      4   
%                          
%****************************************************************************80*/

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
 
void print_vector(double *in, int n){

 for(int i=0; i < n; i++)
  printf("%1.2f\t", in[i]);

 printf("\n");

}/*print_vector*/


int main(int argc, char* argv[]) {

  int i, rank, size;
  double result = 0, result_f;

  MPI_Init (&argc, &argv);                
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);  
  MPI_Comm_size (MPI_COMM_WORLD, &size);  
       
  /*Usage*/ 
  if((argc < 2) && (rank == 0)) {
     printf("Usage:\n");
     printf("mpirun -np [number of processors] %s [size problem]\n", argv[0]);
     exit(-1);
  }

  int data_size = atoi(argv[1]);
  
  double *x  = (double*) malloc(data_size * sizeof(double));
  double *y  = (double*) malloc(data_size * sizeof(double));
   
  for(int i = 0; i < data_size; i++){
      x[i] = 1;
      y[i] = 2;
      result = result + x[i] * y[i]; 
  }        

  if(rank == 0 || rank){
    printf("Rank %d\n", rank);
    print_vector(x, data_size);
    print_vector(y, data_size);
  }

  MPI_Reduce(&result, &result_f, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);   
         
  if(rank == 0)
    printf("dot(x,y) = %f\n", result_f);

  MPI_Finalize();

  return 0;

}/*main*/