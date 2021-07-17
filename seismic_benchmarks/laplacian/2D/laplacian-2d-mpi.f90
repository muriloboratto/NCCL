!*%****************************************************************************80
!  Code: 
!   laplacian-2d-mpi.f90
!
!  Purpose:
!   Implements sample 2D Laplacian Method in fortran using MPI.
!
!  Modified:
!   Aug 17 2020 10:57 
!
!  Author:
!   Murilo Boratto <murilo.boratto 'at' fieb.org.br>
!   Reynam da Cruz Pestana <reynam 'at' ufba.br>
!
!  How to Compile:
!   mpifort laplacian-2d-mpi.f90 -o object
!
!  How to Execute: 
!   mpirun -np 4 ./object                             
!     
!  Comments: 
!   1) Simple Testbed with size problem n = m = 8.                              
!****************************************************************************80*/ 

PROGRAM laplacian_2d_mpi

	    USE MPI

		IMPLICIT NONE

		INTEGER(kind=4) :: myid   ! number associated with each processor, starting from zero
		INTEGER(kind=4) :: nprocs ! number of processors that will be used
		INTEGER(kind=4) :: ierr   ! error
		INTEGER :: n, m = 8
		REAL :: dx, dz
		REAL*8, DIMENSION(:,:), ALLOCATABLE :: a, b, c, sx, sz
		INTEGER :: status(MPI_STATUS_SIZE)
		INTEGER :: jsta,jend,jsta2,jend2,inext,iprev, isend1, isend2, irecv1, irecv2
		INTEGER :: i, j

		! Initializing MPI
		CALL MPI_init(ierr)
		CALL MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
		CALL MPI_Comm_rank(MPI_COMM_WORLD, myid, ierr)

		IF (myid.eq.0) THEN
		   n = m
		END IF

		CALL MPI_BCAST(n,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
		CALL MPI_BCAST(m,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)

		ALLOCATE(a(m,n),b(m,n),c(m,n),sx(m,n),sz(m,n))

		a  = 0.
		b  = 0.
		c  = 0.
		sx = 0.
		sz = 0.
		dx = 1.
		dz = 1.

		CALL PARA_RANGE_1(1, n, nprocs, myid, jsta, jend)

		jsta2 = jsta
		jend2 = jend

		IF(myid.eq.0) THEN
		   jsta2 = 2
		END IF

		IF(myid.eq.(nprocs-1)) THEN
		   jend2 = n-1
		END IF

		inext = myid + 1
		iprev = myid - 1

		IF(myid.eq.(nprocs-1)) THEN
		   inext = MPI_PROC_NULL
		END IF

		IF(myid.eq.0) THEN
		   iprev = MPI_PROC_NULL
		END IF

		DO i = 1, m
		  DO j = jsta, jend
			a(i,j) = (i + j + 2) * 1.
		  END DO
		END DO

                CALL PRINT_MATRIX(a, n)

		CALL MPI_ISEND(a(1,jend),   m, MPI_REAL8, inext, 1, MPI_COMM_WORLD, isend1, ierr)
		CALL MPI_ISEND(a(1,jsta),   m, MPI_REAL8, iprev, 1, MPI_COMM_WORLD, isend2, ierr)
		CALL MPI_IRECV(a(1,jsta-1), m, MPI_REAL8, iprev, 1, MPI_COMM_WORLD, irecv1, ierr)
		CALL MPI_IRECV(a(1,jend+1), m, MPI_REAL8, inext, 1, MPI_COMM_WORLD, irecv2, ierr)

		CALL MPI_WAIT(isend1, status, ierr)
		CALL MPI_WAIT(isend2, status, ierr)
		CALL MPI_WAIT(irecv1, status, ierr)
		CALL MPI_WAIT(irecv2, status, ierr)

		DO j = jsta2,jend2
		   DO i = 2, m - 1
		     sx(i,j) = a(i-1,j) + a(i+1,j) + 2*a(i,j)
		     sz(i,j) = a(i,j-1) + a(i,j+1) + 2*a(i,j)
	              c(i,j) = sx(i,j)/(dx**2) + sz(i,j)/(dz**2)
		   END DO
		END DO

		CALL MPI_REDUCE(c, b, m*n, MPI_REAL8, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

		IF (myid.eq.0) THEN
	           CALL PRINT_MATRIX(b,n)
		!  OPEN(unit=10, file='laplaciano_fortran.dat', ACCESS='direct', recl=8*n*m, status='UNKNOWN')
		!  WRITE(10,rec=1)real(b,kind=8)
		END IF

		DEALLOCATE (a, b, c, sx, sz)

                ! Finalizing MPI
	        CALL MPI_FINALIZE(ierr)

	        STOP

END PROGRAM laplacian_2d_mpi

! --------------------------------------------------------------------- !


SUBROUTINE PARA_RANGE_1(n1,n2,nprocs,myid,jsta,jend)

	INTEGER :: iwork1, iwork2, n1, n2, nprocs, myid, jsta, jend

	iwork1 = (n2-n1+1)/nprocs
	iwork2 = mod(n2-n1+1,nprocs)
	jsta   = myid*iwork1+n1+min(myid,iwork2)
	jend   = jsta+iwork1-1

	IF (iwork2.gt.myid) THEN
	  jend = jend + 1
	END IF

END SUBROUTINE PARA_RANGE_1


SUBROUTINE PRINT_MATRIX(a,n)

	INTEGER::n
	REAL*8::a(n,n) 

	DO i=1,n
          PRINT '(20f6.2)', a(i,1:n)
        ENDDO

        WRITE(*,*) ' '

END SUBROUTINE PRINT_MATRIX
