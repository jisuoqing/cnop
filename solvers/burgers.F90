!! f2py -c burgers.F90 -m burgers_lib

SUBROUTINE EVOLVE_BURGERS(nx,ui,nt,vis,dt,dx,ut)
implicit none
integer, intent(in) :: nx !number of grid points
double precision, intent(in), dimension(nx) :: ui !initial conditions
integer, intent(in) :: nt !number of time steps
double precision, intent(in) :: vis ! diffusion coefficient
double precision, intent(in) :: dt !time increment
double precision, intent(in) :: dx !space increment
double precision, intent(out), dimension(nx) :: ut !solutions

! double precision ub(nx,nt) !basic states(u)
double precision u(nx,nt) !model solutions

double precision c0,c1
integer i,j

common /com_param/ c0,c1

c0= dt/dx
c1=vis*dt/(dx**2)

!set the initial conditions:
do i=1,nx
  u(i,1)=ui(i)
end do

!set the boundary conditions:
do j=1,nt
  u(1,j)=0.
  u(nx,j)=0.
end do

!integerate the model numerically:-----use central finite difference method on spatial and temporal directions

do i=2,nx-1
  u(i,2)=u(i,1)-0.25*c0*(u(i+1,1)*u(i+1,1)-u(i-1,1)*u(i-1,1))+c1*(u(i+1,1)+u(i-1,1)-2*u(i,1))
end do


do j=3,nt
  do i=2,nx-1
    u(i,j)=u(i,j-2)-0.5*c0*(u(i+1,j-1)*u(i+1,j-1)-u(i-1,j-1)*u(i-1,j-1))+2*c1*(u(i+1,j-1)+u(i-1,j-1)-2*u(i,j-1))
  end do
end do 


!save the nonlinear solutions to the basic fields:
!do j=1,nt
!  do i=1,nx
!    ub(i,j)=u(i,j)
!  end do
!end do

! save final solution to ut
do i=1,nx
  ut(i)=u(i,nt)
end do

return


END SUBROUTINE
