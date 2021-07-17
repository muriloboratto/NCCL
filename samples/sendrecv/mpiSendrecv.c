/*%****************************************************************************801
%  Code: 
%   mpiSendrecv.c
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
%   mpicxx mpiSendrecv.c -o mpiSendrecv 
%
%  Execute: 
%   mpirun -np 2 ./mpiSendrecv                               
%
%  Comments: 
%  [Execute in System 'Ògún' using SLURM]
%   1) module load intel-xe-2018/2018      
%   2) module load gcc/7.3.0
%   3) srun -p standard mpicxx mpiSendrecv.c -o mpiSendrecv
%   4) srun -p standard mpirun -np 2 ./mpiSendrecv
 
%****************************************************************************80*/

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
  
int main (int argc, char *argv[]){

 int size = 8;
 int sendbuff[size] = {1,2,3,4,5,6,7,8};
 int recvbuff[size];
 int numprocessors, rank, dest, i, tag = 1000;

 MPI_Init(&argc, &argv);
 MPI_Comm_size(MPI_COMM_WORLD, &numprocessors);
 MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 MPI_Status status;

  if (rank == 0){
      for (dest = 1; dest < numprocessors; dest++) 
        MPI_Send(&sendbuff, size, MPI_INT, dest, tag, MPI_COMM_WORLD);  
  }else{  
    
     MPI_Recv(&recvbuff, size, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

     for(i = 0; i < size; i++)
       printf("%d\t", recvbuff[i]);
       
     printf("\n");

   }
  
   MPI_Finalize();

}/*main*/   

