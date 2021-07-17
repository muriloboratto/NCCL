/*%****************************************************************************80
!  Code: 
!   laplacian-3d.c
!
!  Purpose:
!   Implements sample 3D Laplacian Method in C code.
!
!  Modified:
!   Aug 18 2020 10:57 
!
!  Author:
!   Murilo Boratto <murilo.boratto 'at' fieb.org.br>
!   Reynam da Cruz Pestana <reynam 'at' ufba.br>
!
!  How to Compile:
!   gcc laplacian-3d.c -o object
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

int main(int argc, char *argv[]){

  int nx, ny, nz;
  double dx, dy, dz;
  double sx, sy, sz;

  nx = ny = nz = 8;
  dx = dy = dz = 1;
        
  double *a, *c;
 
  a  =  (double*) calloc (nx * ny * nz, sizeof(double));
  c  =  (double*) calloc (nx * ny * nz, sizeof(double));

	for(int k = 0; k < ny; k++)
	   for(int j = 0; j < nx; j++)
        for(int i = 0; i < nz; i++)
  	         a[i + j*ny + k*(nx*ny)] = (i + j +  2) * 1.;  
 
  show_matrix_3D(a, nx);
        
  for(int k = 1; k < ny - 1; k++){
    for(int j = 1; j < nx - 1; j++){
     	for(int i = 1; i < nz - 1; i++){
	      sy = a[ i    +   j  *ny  + (k-1)*(nx*ny)]  + a[ i    +  j    *ny    + (k+1)*(nx*ny)] + 2 * a[i + j*ny + k*(nx*ny)];
	      sx = a[ i    + (j-1)*ny  +  k   *(nx*ny)]  + a[ i    + (j+1) *ny    +  k   *(nx*ny)] + 2 * a[i + j*ny + k*(nx*ny)];
        sz = a[(i-1) +   j  *ny  +  k   *(nx*ny)]  + a[(i+1) +  j    *ny    +  k   *(nx*ny)] + 2 * a[i + j*ny + k*(nx*ny)];
	      c[i + j*ny + k*(nx*ny)] = (sz/(dz*dz)) + (sx/(dx*dx)) + (sy/(dy*dy));
      }	
    }
  }
 
  printf("\n-------------------------------------------------------------\n");

  show_matrix_3D(c, nx);

  return 0;

}/*main*/
