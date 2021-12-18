/*%****************************************************************************801
%  Code: 
%   mpiSendRecv.c
%
%  Purpose:
%   Implements sample send/recv code using the package MPI.
%
%  Modified:
%   Aug 17 2020 10:57 
%
%  Author:
%    Murilo Boratto <murilo.boratto 'at' fieb.org.br>
%
%  How to Compile:
%   mpicxx mpiSendRecv.c -o mpiSendRecv 
%
%  How to Execute: 
%   mpirun -np 2 ./mpiSendRecv                                
%****************************************************************************80*/

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
  
int main (int argc, char *argv[]){

 int size = 8;
 int sendbuff[size]; 
 int recvbuff[size];
 int numprocessors, rank, dest, i, tag = 1000;

 MPI_Init(&argc, &argv);
 MPI_Comm_size(MPI_COMM_WORLD, &numprocessors);
 MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 MPI_Status status;

  if (rank == 0){
      
      printf("Rank %d\n", rank);

      for(int i = 0; i < size; i++)
         printf("%d\t", sendbuff[i] = i + 1);

      printf("\n");
       
      for (dest = 1; dest < numprocessors; dest++) 
        MPI_Send(&sendbuff, size, MPI_INT, dest, tag, MPI_COMM_WORLD);  
  
  }else{  
    
     MPI_Recv(&recvbuff, size, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

     printf("Rank %d\n", rank);
     
     for(i = 0; i < size; i++)
       printf("%d\t", recvbuff[i]+10);
       
     printf("\n");

   }
  
   MPI_Finalize();

}/*main*/   
