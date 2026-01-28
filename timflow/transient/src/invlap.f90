! this module implements the F.R. de Hoog, J.H. Knight, and A.N. Stokes
! numerical inverse Laplace transform algorithm.
! see "An improved method for numerical inversion of Laplace
!     transforms", SIAM J. Sci. Stat. Comp., 3, 357-366, 1982.

module invlaptrans

contains

    function invlap2( t, tmin, tmax, fp, M, gamma, N ) result (ft)

        real(kind=8), intent(in), dimension(N) :: t   ! vector of times
        real(kind=8), intent(in) :: tmin, tmax
        complex(kind=8), intent(in), dimension(0:2*M) :: fp
        integer, intent(in) :: M, N
        real(kind=8), intent(in) :: gamma
        real(kind=8), dimension(N) :: ft
        
        ft(:) = 3.d0
        
    end function invlap2
  
    function invlap( t, tmin, tmax, fp, M, gamma, Nt ) result (ft)

        real(kind=8), intent(in), dimension(Nt) :: t   ! vector of times
        real(kind=8), intent(in) :: tmin, tmax
        complex(kind=8), intent(in), dimension(0:2*M) :: fp
        integer, intent(in) :: M, Nt
        real(kind=8), intent(in) :: gamma
        real(kind=8), dimension(Nt) :: ft
    
        complex(kind=8), dimension(0:2*M,0:M) :: e
        complex(kind=8), dimension(0:2*M-1,0:M) :: q  ! column 0 is not used
        complex(kind=8), dimension(0:2*M) :: d
        complex(kind=8), dimension(0:2*M+1,Nt) :: A,B
        complex(kind=8), dimension(Nt) :: z,h2M,R2Mz
        integer :: r, rq, n, M
        real(kind=8) :: pi, bigt
            
        pi = 3.1415926535897931d0
        bigt = 2.d0 * tmax

        !a[0] = a[0] / 2.0  # zero term is halved
    
        ! build up e and q tables. superscript is now row index, subscript column
        e(:,:) = cmplx(0.d0,0.d0,kind=8)
        q(:,:) = cmplx(0.d0,0.d0,kind=8)
        q(0,1) = fp(1)/(fp(0)/2.0) ! half first term
        q(1:2*M-1,1) = fp(2:2*M) / fp(1:2*M-1)
        !q(:,1) = fp(1:2*M) / fp(0:2*M-1)

        do r = 1,M  ! step through columns
            e(0:2*(M-r+1)-2,r) = q(1:2*(M-r+1)-1,r) - q(0:2*(M-r+1)-2,r) + e(1:2*(M-r+1)-1,r-1)
            if (r < M) then  ! one column fewer for q
                rq = r+1
                q(0:2*(M-rq+1)-1,rq) = q(1:2*(M-rq+1),rq-1) * e(1:2*(M-rq+1),rq-1) / e(0:2*(M-rq+1)-1,rq-1)
            endif
        end do
    
        ! build up d vector (index shift: 1)
        d(:) = cmplx(0.d0,0.d0,kind=8)
        d(0) = fp(0)/2.0 ! half first term
        d(1:2*M-1:2) = -q(0,1:M) ! these 2 lines changed after niclas
        d(2:2*M:2) = -e(0,1:M) 
    
        ! build A and B vectors (Hollenbeck claims an index shift of 2, but that may be related to the matlab code)
        ! now make into matrices, one row for each time
        A(:,:) = cmplx(0.d0,0.d0,kind=8)
        B(:,:) = cmplx(0.d0,0.d0,kind=8)
        A(1,:) = d(0)
        B(0:1,:) = cmplx(1.d0,0.d0,kind=8)
    
        z = exp( cmplx(0.d0,1.d0,kind=8) * pi * t / bigt )
        ! after niclas back to the paper (not: z = exp(-i*pi*t/T))
        do n = 2, 2*M+1
            A(n,:) = A(n-1,:) + d(n-1) * z * A(n-2,:)  ! different index 
            B(n,:) = B(n-1,:) + d(n-1) * z * B(n-2,:)  ! shift for d!
        end do
        
        ! double acceleration
        h2M = 0.5d0 * ( 1.d0 + ( d(2*M-1) -d(2*M) ) * z )
        R2Mz = -h2M * ( 1.d0 - sqrt( 1.d0 + d(2*M) * z / h2M**2 ) )
    
        A(2*M+1,:) = A(2*M,:) + R2Mz * A(2*M-1,:)
        B(2*M+1,:) = B(2*M,:) + R2Mz * B(2*M-1,:)
        
        ! inversion
        ft = ( 1.d0/bigt * exp(gamma*t) * real( A(2*M+1,:) / B(2*M+1,:) ) )        
    
    end function invlap
  
end module invlaptrans

program invlaptest
    use invlaptrans
    implicit none

    real(kind=8) :: tmin, tmax, alpha, tol, bigt, gamma, pi
    real(kind=8), dimension(3) :: t, ft
    real(kind=8), dimension(0:40) :: run
    complex(kind=8), dimension(0:40) :: p, a
    integer :: M, i, Nt
    
    pi = 3.1415926535897931d0
    M = 20
    Nt = 3
    t(1) = 1.d0; t(2) = 2.d0; t(3) = 3.d0 
    tmin = 1.d0
    tmax = 3.d0
    alpha = 0.d0
    tol = 1.d-9
    do i=0,2*M
        run(i) = float(i)
    end do
    bigt = 2.d0 * tmax
    gamma = alpha - log(tol) / (bigt/2.d0)
    p(:) = gamma + cmplx(0.d0,1.d0,kind=8) * pi * run(:) / bigt
    !print *,p
    a = 1.d0 / p**2
    !a(0) = a(0) / 2.d0
    ft = invlap( t, tmin, tmax, a, M, gamma, Nt )
    print *,'ft ',ft
    
end