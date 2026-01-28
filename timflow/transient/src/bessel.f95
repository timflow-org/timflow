!Copyright (C), 2010, Mark Bakker.
!Module for the computation of bessel functions and bessel line elements
!bessel.f95 is part of the TTim program and is distributed under the MIT license

module bessel

    real(kind=8) :: pi, tiny
    real(kind=8), dimension(0:20) :: a, b, afar, a1, b1
    real(kind=8), dimension(0:20) :: nrange
    real(kind=8), dimension(0:20,0:20) :: gam
    real(kind=8), dimension(8) :: xg, wg

contains

    subroutine initialize

        implicit none
        real(kind=8) :: c, fac, twologhalf
        real(kind=8), dimension(0:20) :: bot
        real(kind=8), dimension(1:21) :: psi
        integer :: n,m

        pi = 3.1415926535897931d0
        tiny = 1.d-10

        do n = 0, 20
            nrange(n) = dble(n)
        end do
        c = log(0.5d0) + 0.577215664901532860d0

        fac = 1.d0
        a(0) = 1.d0
        b(0) = 0.d0
        do n = 1, 20
            fac = n * fac
            a(n) = 1.0 / (4.0d0**nrange(n) * fac**2)
            b(n) = b(n-1) + 1.0d0 / nrange(n)
        end do
        b = (b-c) * a
        a = -0.5d0 * a

        do n = 0,20
            do m = 0,n
                gam(n,m) = product(nrange(m+1:n)) / product(nrange(1:n-m))
            end do
        end do

        afar(0) = sqrt(pi/2.d0)
        do n = 1,20
            afar(n) = -(2.d0*n - 1.d0)**2 / (n * 8.d0) * afar(n-1)
        end do

        ! K1 coefficients

        fac = 1.d0
        bot(0) = 4.d0
        do n = 1, 20
            fac = n * fac
            bot(n) = fac * (n+1)*fac * 4.0**(n+1)
        end do
        psi(1) = 0.d0
        do n = 2,21
            psi(n) = psi(n-1) + 1.d0 / (n-1.d0)
        end do
        psi = psi - 0.577215664901532860d0
        twologhalf = 2.d0 * log(0.5d0)
        do n = 0,20
            a1(n) = 1.d0 / bot(n)
            b1(n) = (twologhalf - (2.d0 * psi(n+1) + 1.d0 / (n+1.d0))) / bot(n)
        end do

        !wg(1) = 0.10122853629037751d0
        !wg(2) = 0.22238103445337376d0
        !wg(3) = 0.3137066458778866d0
        !wg(4) = 0.36268378337836155d0
        !wg(5) = 0.36268378337836205d0
        !wg(6) = 0.3137066458778876d0
        !wg(7) = 0.22238103445337545d0
        !wg(8) = 0.10122853629037616d0
        !
        !xg(1) = -0.9602898564975364d0
        !xg(2) = -0.79666647741362595d0
        !xg(3) = -0.52553240991632855d0
        !xg(4) = -0.18343464249565006d0
        !xg(5) = 0.18343464249565022d0
        !xg(6) = 0.52553240991632888d0
        !xg(7) = 0.79666647741362673d0
        !xg(8) = 0.96028985649753629d0

        wg(1) = 0.101228536290378d0
        wg(2) = 0.22238103445338d0
        wg(3) = 0.31370664587789d0
        wg(4) = 0.36268378337836d0
        wg(5) = 0.36268378337836d0
        wg(6) = 0.313706645877890
        wg(7) = 0.22238103445338d0
        wg(8) = 0.10122853629038d0

        xg(1) = -0.960289856497536d0
        xg(2) = -0.796666477413626d0
        xg(3) = -0.525532409916329d0
        xg(4) = -0.183434642495650d0
        xg(5) = 0.183434642495650d0
        xg(6) = 0.525532409916329d0
        xg(7) = 0.796666477413626d0
        xg(8) = 0.960289856497536d0

        return

    end subroutine initialize

    function besselk0far(z, Nt) result(omega)
        implicit none
        complex(kind=8), intent(in) :: z
        integer, intent(in) :: Nt
        complex(kind=8) :: omega, term
        integer :: n

        term = 1.d0
        omega = afar(0)
        do n = 1, Nt
            term = term / z
            omega = omega + afar(n) * term
        end do
        omega = exp(-z) / sqrt(z) * omega

        return
    end function besselk0far

    function besselk0near(z, Nt) result(omega)
        implicit none
        complex(kind=8), intent(in) :: z
        integer, intent(in) :: Nt
        complex(kind=8) :: omega
        complex(kind=8) :: rsq, log1, term
        integer :: n

        rsq = z**2
        term = cmplx(1.d0,0.d0,kind=8)
        log1 = log(rsq)
        omega = a(0) * log1 + b(0)

        do n = 1, Nt
            term = term * rsq
            omega = omega + (a(n)*log1 + b(n)) * term
        end do

        return
    end function besselk0near

    function besselk1near(z, Nt) result(omega)
        implicit none
        complex(kind=8), intent(in) :: z
        integer, intent(in) :: Nt
        complex(kind=8) :: omega
        complex(kind=8) :: zsq, log1, term
        integer :: n

        zsq = z**2
        term = z
        log1 = log(zsq)
        omega = 1.d0 / z + (a1(0) * log1 + b1(0)) * z

        do n = 1, Nt
            term = term * zsq
            omega = omega + (a1(n)*log1 + b1(n)) * term
        end do

        return
    end function besselk1near

    function besselk0cheb(z, Nt) result (omega)
        implicit none

        complex(kind=8), intent(in) :: z
        integer, intent(in) :: Nt
        complex(kind=8) :: omega

        integer :: n, n2, ts
        real(kind=8) :: a, b, c, A3, u
        complex(kind=8) :: A1, A2, cn, cnp1, cnp2, cnp3
        complex(kind=8) :: z1, z2, S, T

        cnp1 = cmplx( 1.d0, 0.d0, kind=8 )
        cnp2 = cmplx( 0.d0, 0.d0, kind=8 )
        cnp3 = cmplx( 0.d0, 0.d0, kind=8 )
        a = 0.5d0
        c = 1.d0
        b = 1.d0 + a - c

        z1 = 2.d0 * z
        z2 = 2.d0 * z1
        ts = (-1)**(Nt+1)
        S = ts
        T = 1.d0

        do n = Nt, 0, -1
            u = (n+a) * (n+b)
            n2 = 2 * n
            A1 = 1.d0 - ( z2 + (n2+3.d0)*(n+a+1.d0)*(n+b+1.d0) / (n2+4.d0) ) / u
            A2 = 1.d0 - (n2+2.d0)*(n2+3.d0-z2) / u
            A3 = -(n+1.d0)*(n+3.d0-a)*(n+3.d0-b) / (u*(n+2.d0))
            cn = (2.d0*n+2.d0) * A1 * cnp1 + A2 * cnp2 + A3 * cnp3
            ts = -ts
            S = S + ts * cn
            T = T + cn
            cnp3 = cnp2; cnp2 = cnp1; cnp1 = cn
        end do
        cn = cn / 2.d0
        S = S - cn
        T = T - cn
        omega = 1.d0 / cdsqrt(z1) * T / S
        omega = sqrt(pi) * cdexp(-z) * omega

    end function besselk0cheb

    function besselk1cheb(z, Nt) result (omega)
        implicit none

        complex(kind=8), intent(in) :: z
        integer, intent(in) :: Nt
        complex(kind=8) :: omega

        integer :: n, n2, ts
        real(kind=8) :: a, b, c, A3, u
        complex(kind=8) :: A1, A2, cn, cnp1, cnp2, cnp3
        complex(kind=8) :: z1, z2, S, T

        cnp1 = cmplx( 1.d0, 0.d0, kind=8 )
        cnp2 = cmplx( 0.d0, 0.d0, kind=8 )
        cnp3 = cmplx( 0.d0, 0.d0, kind=8 )
        a = 1.5d0
        c = 3.d0
        b = 1.d0 + a - c

        z1 = 2.d0 * z
        z2 = 2.d0 * z1
        ts = (-1)**(Nt+1)
        S = ts
        T = 1.d0

        do n = Nt, 0, -1
            u = (n+a) * (n+b)
            n2 = 2 * n
            A1 = 1.d0 - ( z2 + (n2+3.d0)*(n+a+1.d0)*(n+b+1.d0) / (n2+4.d0) ) / u
            A2 = 1.d0 - (n2+2.d0)*(n2+3.d0-z2) / u
            A3 = -(n+1.d0)*(n+3.d0-a)*(n+3.d0-b) / (u*(n+2.d0))
            cn = (2.d0*n+2.d0) * A1 * cnp1 + A2 * cnp2 + A3 * cnp3
            ts = -ts
            S = S + ts * cn
            T = T + cn
            cnp3 = cnp2; cnp2 = cnp1; cnp1 = cn
        end do
        cn = cn / 2.d0
        S = S - cn
        T = T - cn
        omega = 1.d0 / (sqrt(z1) * z1) * T / S
        omega = 2.d0 * z * sqrt(pi) * cdexp(-z) * omega

    end function besselk1cheb

    function besselk0(x, y, lab) result(omega)
        implicit none
        real(kind=8), intent(in) :: x,y
        complex(kind=8), intent(in) :: lab
        complex(kind=8) :: z, omega
        real(kind=8) :: cond

        z = sqrt(x**2 + y**2) / lab
        cond = abs(z)

        if (cond < 6.d0) then
            omega = besselk0near( z, 17 )
        else
            omega = besselk0cheb( z, 6 )
        end if

        return
    end function besselk0

    function besselk1(x, y, lab) result(omega)
        implicit none
        real(kind=8), intent(in) :: x,y
        complex(kind=8), intent(in) :: lab
        complex(kind=8) :: z, omega
        real(kind=8) :: cond

        z = sqrt(x**2 + y**2) / lab
        cond = abs(z)

        if (cond < 6.d0) then
            omega = besselk1near( z, 20 )
        else
            omega = besselk1cheb( z, 6 )
        end if

        return
    end function besselk1

    function k0bessel(z) result(omega)
        implicit none
        complex(kind=8), intent(in) :: z
        complex(kind=8) :: omega
        real(kind=8) :: cond

        cond = abs(z)

        if (cond < 6.d0) then
            omega = besselk0near( z, 17 )
        else
            omega = besselk0cheb( z, 6 )
        end if

        return
    end function k0bessel

    subroutine besselk0v(x,y,lab,nlab,omega)
        implicit none
        real(kind=8), intent(in) :: x,y
        integer, intent(in) :: nlab
        complex(kind=8), dimension(nlab), intent(in) :: lab
        complex(kind=8), dimension(nlab), intent(inout) :: omega
        integer :: n
        do n = 1,nlab
            omega(n) = besselk0(x, y, lab(n))
        end do
    end subroutine besselk0v

    subroutine k0besselv(z,nlab,omega)
        implicit none
        integer, intent(in) :: nlab
        complex(kind=8), dimension(nlab), intent(in) :: z
        complex(kind=8), dimension(nlab), intent(inout) :: omega
        integer :: n
        do n = 1,nlab
            omega(n) = k0bessel(z(n))
        end do
    end subroutine k0besselv

    function besselk0OLD(x, y, lab) result(omega)
        implicit none
        real(kind=8), intent(in) :: x,y
        complex(kind=8), intent(in) :: lab
        complex(kind=8) :: z, omega
        real(kind=8) :: cond

        z = sqrt(x**2 + y**2) / lab
        cond = abs(z)

        if (cond < 4.d0) then
            omega = besselk0near( z, 12 )  ! Was 10
        else if (cond < 8.d0) then
            omega = besselk0near( z, 18 )
        else if (cond < 12.d0) then
            omega = besselk0far( z, 11 )  ! was 6
        else
            omega = besselk0far( z, 8 )  ! was 4
        end if

        return
    end function besselk0OLD

    function besselcheb(z, Nt) result (omega)
        implicit none
        integer, intent(in) :: Nt
        complex(kind=8), intent(in) :: z
        complex(kind=8) :: omega
        complex(kind=8) :: z2

        z2 = 2.0 * z

        omega = sqrt(pi) * exp(-z) * ucheb(0.5d0,1,z2,Nt)

    end function besselcheb

    function ucheb(a, c, z, n0) result (ufunc)
        implicit none
        integer, intent(in) :: c, n0
        real(kind=8), intent(in) :: a
        complex(kind=8), intent(in) :: z
        complex(kind=8) :: ufunc

        integer :: n, n2, ts
        real(kind=8) :: A3, u, b
        complex(kind=8) :: A1, A2, cn,cnp1,cnp2,cnp3
        complex(kind=8) :: z2, S, T

        cnp1 = 1.d0
        cnp2 = 0.d0
        cnp3 = 0.d0
        ts = (-1)**(n0+1)
        S = ts
        T = 1.d0
        z2 = 2.d0 * z
        b = 1.d0 + a - c

        do n = n0, 0, -1
            u = (n+a) * (n+b)
            n2 = 2 * n
            A1 = 1.d0 - ( z2 + (n2+3)*(n+a+1)*(n+b+1.d0) / (n2+4.d0) ) / u
            A2 = 1.d0 - (n2+2.d0)*(n2+3.d0-z2) / u
            A3 = -(n+1)*(n+3-a)*(n+3-b) / (u*(n+2))
            cn = (2*n+2) * A1 * cnp1 + A2 * cnp2 + A3 * cnp3
            ts = -ts
            S = S + ts * cn
            T = T + cn
            cnp3 = cnp2; cnp2 = cnp1; cnp1 = cn
        end do
        cn = cn / 2.d0
        S = S - cn
        T = T - cn
        ufunc = z**(-a) * T / S

    end function ucheb

    function besselk0complex(x, y) result(phi)
        implicit none
        real(kind=8), intent(in) :: x,y
        real(kind=8) :: phi
        real(kind=8) :: d
        complex(kind=8) :: zeta, zetabar, omega, logdminzdminzbar, dminzeta, term
        complex(kind=8), dimension(0:20) :: zminzbar
        complex(kind=8), dimension(0:20,0:20) :: gamnew
        complex(kind=8), dimension(0:40) :: alpha, beta

        integer :: n

        d = 0.d0

        zeta = cmplx(x,y)
        zetabar = conjg(zeta)
        do n = 0,20
            zminzbar(n) = (zeta-zetabar)**(20-n)  ! Ordered from high power to low power
        end do
        gamnew = gam
        do n = 0,20
            gamnew(n,0:n) = gamnew(n,0:n) * zminzbar(20-n:20)
        end do

        alpha(0:40) = 0.d0
        beta(0:40) = 0.d0
        alpha(0) = a(0)
        beta(0) = b(0)
        do n = 1,20
            alpha(n:2*n) = alpha(n:2*n) + a(n) * gamnew(n,0:n)
            beta(n:2*n)  = beta(n:2*n)  + b(n) * gamnew(n,0:n)
        end do

        omega = 0.d0
        logdminzdminzbar = log( (d-zeta) * (d-zetabar) )
        dminzeta = d - zeta
        term = 1.d0
        do n = 0,40
            omega = omega + ( alpha(n) * logdminzdminzbar + beta(n) ) * term
            term = term * dminzeta
        end do

        phi = real(omega)

        return
    end function besselk0complex

    function lapls_int_ho(x,y,z1,z2,order) result(omega)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y
        complex(kind=8), intent(in) :: z1,z2
        complex(kind=8), dimension(1:order+1) :: omega
        complex(kind=8), dimension(1:order+2) :: qm
        integer :: m, i
        real(kind=8) :: L
        complex(kind=8) :: z, zplus1, zmin1, log1, log2, log3, zpower

        L = abs(z2-z1)
        z = (2.d0 * dcmplx(x,y) - (z1+z2) ) / (z2-z1)
        zplus1 = z + 1.d0; zmin1 = z - 1.d0
        ! Not sure if this gives correct answer at corner point (z also appears in qm); should really be caught in code that calls this function
        if  (abs(zplus1) < tiny) zplus1 = tiny
        if  (abs(zmin1) < tiny) zmin1 = tiny

        qm(1) = 0.d0
        qm(2) = 2.d0
        do m=3,order+1,2
            qm(m+1) = qm(m-1) * z * z + 2.d0 / m
        end do
        do m=2,order+1,2
            qm(m+1) = qm(m) * z
        end do

        log1 = cdlog( (zmin1) / (zplus1) )
        log2 = cdlog(zmin1)
        log3 = cdlog(zplus1)

        zpower = 1.d0
        do i = 1, order+1
            zpower = zpower * z
            omega(i) = -L/(4.d0*pi*i) * ( zpower * log1 + qm(i+1) - log2 + (-1.d0)**i * log3 )
        end do

    end function lapls_int_ho

    function bessellsreal(x,y,x1,y1,x2,y2,lab) result(phi)
        implicit none
        real(kind=8), intent(in) :: x,y,x1,y1,x2,y2,lab
        real(kind=8) :: phi, biglab, biga, L
        complex(kind=8) :: z1, z2, zeta, zetabar, omega, log1, log2, term1, term2, d1minzeta, d2minzeta
        complex(kind=8), dimension(0:20) :: zminzbar
        complex(kind=8), dimension(0:20,0:20) :: gamnew
        complex(kind=8), dimension(0:40) :: alpha, beta

        integer :: n

        z1 = dcmplx(x1,y1); z2 = dcmplx(x2,y2)
        L = abs(z2-z1)
        biga = abs(lab)
        biglab = 2.d0 * biga / L

        zeta = (2.d0 * dcmplx(x,y) - (z1+z2) ) / (z2-z1) / biglab
        zetabar = conjg(zeta)
        do n = 0,20
            zminzbar(n) = (zeta-zetabar)**(20-n)  ! Ordered from high power to low power
        end do
        gamnew = gam
        do n = 0,20
            gamnew(n,0:n) = gamnew(n,0:n) * zminzbar(20-n:20)
        end do

        alpha(0:40) = 0.d0
        beta(0:40) = 0.d0
        alpha(0) = a(0)
        beta(0) = b(0)
        do n = 1,20
            alpha(n:2*n) = alpha(n:2*n) + a(n) * gamnew(n,0:n)
            beta(n:2*n)  = beta(n:2*n)  + b(n) * gamnew(n,0:n)
        end do

        omega = 0.d0
        d1minzeta = -1.d0/biglab - zeta
        d2minzeta = 1.d0/biglab - zeta
        log1 = cdlog(d1minzeta)
        log2 = cdlog(d2minzeta)
        term1 = 1.d0
        term2 = 1.d0
        ! I tried to serialize this, but it didn't speed things up
        do n = 0,40
            term1 = term1 * d1minzeta
            term2 = term2 * d2minzeta
            omega = omega + ( 2.d0 * alpha(n) * log2 - 2.d0 * alpha(n) / (n+1) + beta(n) ) * term2 / (n+1)
            omega = omega - ( 2.d0 * alpha(n) * log1 - 2.d0 * alpha(n) / (n+1) + beta(n) ) * term1 / (n+1)
        end do

        phi = -biga / (2.d0*pi) * real(omega)

        return
    end function bessellsreal

    function bessellsrealho(x,y,x1,y1,x2,y2,lab,order) result(phi)
        implicit none
        real(kind=8), intent(in) :: x,y,x1,y1,x2,y2,lab
        integer, intent(in) :: order
        real(kind=8), dimension(0:order) :: phi
        real(kind=8) :: biglab, biga, L
        complex(kind=8) :: z1, z2, zeta, zetabar, omega, log1, log2, term1, term2, d1minzeta, d2minzeta
        complex(kind=8), dimension(0:20) :: zminzbar
        complex(kind=8), dimension(0:20,0:20) :: gamnew
        complex(kind=8), dimension(0:40) :: alpha, beta
        complex(kind=8) :: cm
        complex(kind=8), dimension(0:50) :: alphanew, betanew ! Maximum programmed order is 10

        integer :: n, m, p

        z1 = dcmplx(x1,y1); z2 = dcmplx(x2,y2)
        L = abs(z2-z1)
        biga = abs(lab)
        biglab = 2.d0 * biga / L

        zeta = (2.d0 * dcmplx(x,y) - (z1+z2) ) / (z2-z1) / biglab
        zetabar = conjg(zeta)
        do n = 0,20
            zminzbar(n) = (zeta-zetabar)**(20-n)  ! Ordered from high power to low power
        end do
        gamnew = gam
        do n = 0,20
            gamnew(n,0:n) = gamnew(n,0:n) * zminzbar(20-n:20)
        end do

        alpha(0:40) = 0.d0
        beta(0:40) = 0.d0
        alpha(0) = a(0)
        beta(0) = b(0)
        do n = 1,20
            alpha(n:2*n) = alpha(n:2*n) + a(n) * gamnew(n,0:n)
            beta(n:2*n)  = beta(n:2*n)  + b(n) * gamnew(n,0:n)
        end do

        d1minzeta = -1.d0/biglab - zeta
        d2minzeta = 1.d0/biglab - zeta
        log1 = cdlog(d1minzeta)
        log2 = cdlog(d2minzeta)

        do p = 0,order
            alphanew(0:40+p) = 0.d0
            betanew(0:40+p) = 0.d0
            do m = 0,p
                cm = biglab**p * gam(p,m) * zeta**(p-m)
                alphanew(m:40+m) = alphanew(m:40+m) + cm * alpha(0:40)
                betanew(m:40+m)  = betanew(m:40+m) + cm * beta(0:40)
            end do

            omega = 0.d0
            term1 = 1.d0
            term2 = 1.d0
            ! I tried to serialize this, but it didn't speed things up
            do n = 0,40+p
                term1 = term1 * d1minzeta
                term2 = term2 * d2minzeta
                omega = omega + ( 2.d0 * alphanew(n) * log2 - 2.d0 * alphanew(n) / (n+1) + betanew(n) ) * term2 / (n+1)
                omega = omega - ( 2.d0 * alphanew(n) * log1 - 2.d0 * alphanew(n) / (n+1) + betanew(n) ) * term1 / (n+1)
            end do

            phi(p) = -biga / (2.d0*pi) * real(omega)
        end do

        return
    end function bessellsrealho

    function bessells_int(x,y,z1,z2,lab) result(omega)
        implicit none
        real(kind=8), intent(in) :: x,y
        complex(kind=8), intent(in) :: z1,z2,lab
        real(kind=8) :: biglab, biga, L, ang, tol
        complex(kind=8) :: zeta, zetabar, omega, log1, log2, term1, term2, d1minzeta, d2minzeta
        complex(kind=8), dimension(0:20) :: zminzbar, anew, bnew, exprange
        complex(kind=8), dimension(0:20,0:20) :: gamnew, gam2
        complex(kind=8), dimension(0:40) :: alpha, beta, alpha2
        integer :: n

        L = abs(z2-z1)
        biga = abs(lab)
        ang = atan2(aimag(lab),real(lab))
        biglab = 2.d0 * biga / L

        tol = 1.d-12

        exprange = exp(-cmplx(0,2,kind=8) * ang * nrange )
        anew = a * exprange
        bnew = (b - a * cmplx(0,2,kind=8) * ang) * exprange

        zeta = (2.d0 * dcmplx(x,y) - (z1+z2) ) / (z2-z1) / biglab
        zetabar = conjg(zeta)
        !do n = 0,20
        !    zminzbar(n) = (zeta-zetabar)**(20-n)  ! Ordered from high power to low power
        !end do
        zminzbar(20) = 1.d0
        do n = 1,20
            zminzbar(20-n) = zminzbar(21-n) * (zeta-zetabar)  ! Ordered from high power to low power
        end do
        gamnew = gam
        do n = 0,20
            gamnew(n,0:n) = gamnew(n,0:n) * zminzbar(20-n:20)
            gam2(n,0:n) = conjg(gamnew(n,0:n))
        end do

        alpha(0:40) = 0.d0
        beta(0:40) = 0.d0
        alpha2(0:40) = 0.d0
        alpha(0) = anew(0)
        beta(0) = bnew(0)
        alpha2(0) = anew(0)
        do n = 1,20
            alpha(n:2*n) = alpha(n:2*n) + anew(n) * gamnew(n,0:n)
            beta(n:2*n)  = beta(n:2*n)  + bnew(n) * gamnew(n,0:n)
            alpha2(n:2*n) = alpha2(n:2*n) + anew(n) * gam2(n,0:n)
        end do

        omega = 0.d0
        d1minzeta = -1.d0/biglab - zeta
        d2minzeta = 1.d0/biglab - zeta
        if (abs(d1minzeta) < tol) d1minzeta = d1minzeta + cmplx(tol,0.d0,kind=8)
        if (abs(d2minzeta) < tol) d2minzeta = d2minzeta + cmplx(tol,0.d0,kind=8)
        log1 = log(d1minzeta)
        log2 = log(d2minzeta)
        term1 = 1.d0
        term2 = 1.d0
        ! I tried to serialize this, but it didn't speed things up
        do n = 0,40
            term1 = term1 * d1minzeta
            term2 = term2 * d2minzeta
            omega = omega + ( alpha(n) * log2 - alpha(n) / (n+1) + beta(n) ) * term2 / (n+1)
            omega = omega - ( alpha(n) * log1 - alpha(n) / (n+1) + beta(n) ) * term1 / (n+1)
            omega = omega + ( alpha2(n) * conjg(log2) - alpha2(n) / (n+1) ) * conjg(term2) / (n+1)
            omega = omega - ( alpha2(n) * conjg(log1) - alpha2(n) / (n+1) ) * conjg(term1) / (n+1)
        end do

        omega = -biga / (2.d0*pi) * omega

        return
    end function bessells_int

    function bessells_int_ho(x,y,z1,z2,lab,order,d1,d2) result(omega)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,d1,d2
        complex(kind=8), intent(in) :: z1,z2,lab
        complex(kind=8), dimension(0:order) :: omega
        real(kind=8) :: biglab, biga, L, ang, tol
        complex(kind=8) :: zeta, zetabar, log1, log2, term1, term2, d1minzeta, d2minzeta, cm
        complex(kind=8), dimension(0:20) :: zminzbar, anew, bnew, exprange
        complex(kind=8), dimension(0:20,0:20) :: gamnew, gam2
        complex(kind=8), dimension(0:40) :: alpha, beta, alpha2
        complex(kind=8), dimension(0:50) :: alphanew, betanew, alphanew2 ! Order fixed to 10

        integer :: m, n, p

        L = abs(z2-z1)
        biga = abs(lab)
        ang = atan2(aimag(lab),real(lab))
        biglab = 2.d0 * biga / L

        tol = 1.d-12

        exprange = exp(-cmplx(0,2,kind=8) * ang * nrange )
        anew = a * exprange
        bnew = (b - a * cmplx(0,2,kind=8) * ang) * exprange

        zeta = (2.d0 * dcmplx(x,y) - (z1+z2) ) / (z2-z1) / biglab
        zetabar = conjg(zeta)
        !do n = 0,20
        !    zminzbar(n) = (zeta-zetabar)**(20-n)  ! Ordered from high power to low power
        !end do
        zminzbar(20) = 1.d0
        do n = 1,20
            zminzbar(20-n) = zminzbar(21-n) * (zeta-zetabar)  ! Ordered from high power to low power
        end do
        gamnew = gam
        do n = 0,20
            gamnew(n,0:n) = gamnew(n,0:n) * zminzbar(20-n:20)
            gam2(n,0:n) = conjg(gamnew(n,0:n))
        end do

        alpha(0:40) = 0.d0
        beta(0:40) = 0.d0
        alpha2(0:40) = 0.d0
        alpha(0) = anew(0)
        beta(0) = bnew(0)
        alpha2(0) = anew(0)
        do n = 1,20
            alpha(n:2*n) = alpha(n:2*n) + anew(n) * gamnew(n,0:n)
            beta(n:2*n)  = beta(n:2*n)  + bnew(n) * gamnew(n,0:n)
            alpha2(n:2*n) = alpha2(n:2*n) + anew(n) * gam2(n,0:n)
        end do

        d1minzeta = d1/biglab - zeta
        d2minzeta = d2/biglab - zeta
        !d1minzeta = -1.d0/biglab - zeta
        !d2minzeta = 1.d0/biglab - zeta
        if (abs(d1minzeta) < tol) d1minzeta = d1minzeta + cmplx(tol,0.d0,kind=8)
        if (abs(d2minzeta) < tol) d2minzeta = d2minzeta + cmplx(tol,0.d0,kind=8)
        log1 = log(d1minzeta)
        log2 = log(d2minzeta)

        do p = 0,order

            alphanew(0:40+p) = 0.d0
            betanew(0:40+p) = 0.d0
            alphanew2(0:40+p) = 0.d0
            do m = 0,p
                cm = biglab**p * gam(p,m) * zeta**(p-m)
                alphanew(m:40+m) = alphanew(m:40+m) + cm * alpha(0:40)
                betanew(m:40+m)  = betanew(m:40+m) + cm * beta(0:40)
                cm = biglab**p * gam(p,m) * zetabar**(p-m)
                alphanew2(m:40+m) = alphanew2(m:40+m) + cm * alpha2(0:40)
            end do

            omega(p) = 0.d0
            term1 = 1.d0
            term2 = 1.d0
            do n = 0,40
                term1 = term1 * d1minzeta
                term2 = term2 * d2minzeta
                omega(p) = omega(p) + ( alphanew(n) * log2 - alphanew(n) / (n+1) + betanew(n) ) * term2 / (n+1)
                omega(p) = omega(p) - ( alphanew(n) * log1 - alphanew(n) / (n+1) + betanew(n) ) * term1 / (n+1)
                omega(p) = omega(p) + ( alphanew2(n) * conjg(log2) - alphanew2(n) / (n+1) ) * conjg(term2) / (n+1)
                omega(p) = omega(p) - ( alphanew2(n) * conjg(log1) - alphanew2(n) / (n+1) ) * conjg(term1) / (n+1)
            end do

        end do

        omega = -biga / (2.d0*pi) * omega

        return
    end function bessells_int_ho

    function bessells_int_ho_qxqy(x,y,z1,z2,lab,order,d1,d2) result(qxqy)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,d1,d2
        complex(kind=8), intent(in) :: z1,z2,lab
        complex(kind=8), dimension(0:2*order+1) :: qxqy
        complex(kind=8), dimension(0:order) :: qx, qy
        real(kind=8) :: biglab, biga, L, ang, angz, tol, bigx, bigy
        complex(kind=8) :: zeta, zetabar, log1, log2, term1, term2, d1minzeta, d2minzeta, bigz
        complex(kind=8) :: cm, biglabcomplex
        complex(kind=8), dimension(0:20) :: zminzbar, anew, bnew, exprange
        complex(kind=8), dimension(0:20,0:20) :: gamnew, gam2
        complex(kind=8), dimension(0:40) :: alpha, beta, alpha2
        complex(kind=8), dimension(0:51) :: alphanew, betanew, alphanew2 ! Order fixed to 10
        complex(kind=8), dimension(0:order+1) :: omega ! To store intermediate result
        complex(kind=8), dimension(0:order) :: omegalap ! To store intermediate result

        integer :: m, n, p

        L = abs(z2-z1)
        bigz = (2.d0 * dcmplx(x,y) - (z1+z2) ) / (z2-z1)
        bigx = real(bigz)
        bigy = aimag(bigz)
        biga = abs(lab)
        ang = atan2(aimag(lab),real(lab))
        angz = atan2(aimag(z2-z1),real(z2-z1))
        biglab = 2.d0 * biga / L
        biglabcomplex = 2.0 * lab / L

        tol = 1.d-12

        exprange = exp(-cmplx(0,2,kind=8) * ang * nrange )
        anew = a1 * exprange
        bnew = (b1 - a1 * cmplx(0,2,kind=8) * ang) * exprange

        zeta = (2.d0 * dcmplx(x,y) - (z1+z2) ) / (z2-z1) / biglab
        zetabar = conjg(zeta)
        zminzbar(20) = 1.d0
        do n = 1,20
            zminzbar(20-n) = zminzbar(21-n) * (zeta-zetabar)  ! Ordered from high power to low power
        end do
        gamnew = gam
        do n = 0,20
            gamnew(n,0:n) = gamnew(n,0:n) * zminzbar(20-n:20)
            gam2(n,0:n) = conjg(gamnew(n,0:n))
        end do

        alpha(0:40) = 0.d0
        beta(0:40) = 0.d0
        alpha2(0:40) = 0.d0
        alpha(0) = anew(0)
        beta(0) = bnew(0)
        alpha2(0) = anew(0)
        do n = 1,20
            alpha(n:2*n) = alpha(n:2*n) + anew(n) * gamnew(n,0:n)
            beta(n:2*n)  = beta(n:2*n)  + bnew(n) * gamnew(n,0:n)
            alpha2(n:2*n) = alpha2(n:2*n) + anew(n) * gam2(n,0:n)
        end do

        d1minzeta = d1/biglab - zeta
        d2minzeta = d2/biglab - zeta
        !d1minzeta = -1.d0/biglab - zeta
        !d2minzeta = 1.d0/biglab - zeta
        if (abs(d1minzeta) < tol) d1minzeta = d1minzeta + cmplx(tol,0.d0,kind=8)
        if (abs(d2minzeta) < tol) d2minzeta = d2minzeta + cmplx(tol,0.d0,kind=8)
        log1 = log(d1minzeta)
        log2 = log(d2minzeta)

        do p = 0,order+1

            alphanew(0:40+p) = 0.d0
            betanew(0:40+p) = 0.d0
            alphanew2(0:40+p) = 0.d0
            do m = 0,p
                cm = biglab**p * gam(p,m) * zeta**(p-m)
                alphanew(m:40+m) = alphanew(m:40+m) + cm * alpha(0:40)
                betanew(m:40+m)  = betanew(m:40+m) + cm * beta(0:40)
                cm = biglab**p * gam(p,m) * zetabar**(p-m)
                alphanew2(m:40+m) = alphanew2(m:40+m) + cm * alpha2(0:40)
            end do

            omega(p) = 0.d0
            term1 = 1.d0
            term2 = 1.d0
            do n = 0,40+p
                term1 = term1 * d1minzeta
                term2 = term2 * d2minzeta
                omega(p) = omega(p) + ( alphanew(n) * log2 - alphanew(n) / (n+1) + betanew(n) ) * term2 / (n+1)
                omega(p) = omega(p) - ( alphanew(n) * log1 - alphanew(n) / (n+1) + betanew(n) ) * term1 / (n+1)
                omega(p) = omega(p) + ( alphanew2(n) * conjg(log2) - alphanew2(n) / (n+1) ) * conjg(term2) / (n+1)
                omega(p) = omega(p) - ( alphanew2(n) * conjg(log1) - alphanew2(n) / (n+1) ) * conjg(term1) / (n+1)
            end do

        end do

        omega = biglab / (2.d0*pi*biglabcomplex**2) * omega
        omegalap = lapld_int_ho_d1d2(x,y,z1,z2,order,d1,d2)

        qx = -( bigx * omega(0:order) - omega(1:order+1) + aimag(omegalap) )  ! multiplication with 2/L inherently included
        qy = -( bigy * omega(0:order) + real(omegalap) )

        qxqy(0:order) = qx * cos(angz) - qy * sin(angz)
        qxqy(order+1:2*order+1) = qx * sin(angz) + qy * cos(angz)

        return
    end function bessells_int_ho_qxqy

    function bessells_gauss(x,y,z1,z2,lab) result(omega)
        implicit none
        real(kind=8), intent(in) :: x,y
        complex(kind=8), intent(in) :: z1,z2
        complex(kind=8), intent(in) :: lab
        complex(kind=8) :: omega
        integer :: n
        real(kind=8) :: L, x0
        complex(kind=8) :: bigz, biglab

        L = abs(z2-z1)
        biglab = 2.d0 * lab / L
        bigz = (2.d0 * cmplx(x,y,kind=8) - (z1+z2) ) / (z2-z1)
        omega = cmplx(0.d0,0.d0,kind=8)
        do n = 1,8
            x0 = real(bigz) - xg(n)
            omega = omega + wg(n) * besselk0( x0, aimag(bigz), biglab )
        end do
        omega = -L/(4.d0*pi) * omega
        return
    end function bessells_gauss

    function bessells_gauss_ho(x,y,z1,z2,lab,order) result(omega)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y
        complex(kind=8), intent(in) :: z1,z2
        complex(kind=8), intent(in) :: lab
        complex(kind=8), dimension(0:order) :: omega
        integer :: n, p
        real(kind=8) :: L, x0
        complex(kind=8) :: bigz, biglab
        complex(kind=8), dimension(8) :: k0

        L = abs(z2-z1)
        biglab = 2.d0 * lab / L
        bigz = (2.d0 * cmplx(x,y,kind=8) - (z1+z2) ) / (z2-z1)
        do n = 1,8
            x0 = real(bigz) - xg(n)
            k0(n) = besselk0( x0, aimag(bigz), biglab )
        end do
        do p = 0,order
            omega(p) = cmplx(0.d0,0.d0,kind=8)
            do n = 1,8
                omega(p) = omega(p) + wg(n) * xg(n)**p * k0(n)
            end do
            omega(p) = -L/(4.d0*pi) * omega(p)
        end do
        return
    end function bessells_gauss_ho

    function bessells_gauss_ho_d1d2(x,y,z1,z2,lab,order,d1,d2) result(omega)
        ! Returns integral from d1 to d2 along real axis while strength is still Delta^order from -1 to +1
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,d1,d2
        complex(kind=8), intent(in) :: z1,z2,lab
        complex(kind=8), dimension(0:order) :: omega, omegac
        integer :: n, m
        real(kind=8) :: xp, yp, dc, fac
        complex(kind=8) :: z1p,z2p,bigz1,bigz2
        bigz1 = dcmplx(d1,0.d0)
        bigz2 = dcmplx(d2,0.d0)
        z1p = 0.5d0 * (z2-z1) * bigz1 + 0.5d0 * (z1+z2)
        z2p = 0.5d0 * (z2-z1) * bigz2 + 0.5d0 * (z1+z2)
        omegac = bessells_gauss_ho(x,y,z1p,z2p,lab,order)
        dc = (d1+d2) / (d2-d1)
        omega(0:order) = 0.d0
        do n = 0, order
            do m = 0, n
                omega(n) = omega(n) + gam(n,m) * dc**(n-m) * omegac(m)
            enddo
            omega(n) = ( 0.5*(d2-d1) )**n * omega(n)
        end do
    end function bessells_gauss_ho_d1d2

    function bessells_gauss_ho_qxqy(x,y,z1,z2,lab,order) result(qxqy)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y
        complex(kind=8), intent(in) :: z1,z2
        complex(kind=8), intent(in) :: lab
        complex(kind=8), dimension(0:2*order+1) :: qxqy
        integer :: n, p
        real(kind=8) :: L, bigy, angz
        complex(kind=8) :: bigz, biglab
        real(kind=8), dimension(8) :: r, xmind
        complex(kind=8), dimension(8) :: k1
        complex(kind=8), dimension(0:order) :: qx,qy


        L = abs(z2-z1)
        biglab = 2.d0 * lab / L
        bigz = (2.d0 * cmplx(x,y,kind=8) - (z1+z2) ) / (z2-z1)
        bigy = aimag(bigz)
        do n = 1,8
            xmind(n) = real(bigz) - xg(n)
            r(n) = sqrt( xmind(n)**2 + aimag(bigz)**2 )
            k1(n) = besselk1( xmind(n), aimag(bigz), biglab )
        end do
        qx = dcmplx(0.d0,0.d0)
        qy = dcmplx(0.d0,0.d0)
        do p = 0,order
            do n = 1,8
                qx(p) = qx(p) + wg(n) * xg(n)**p * xmind(n) * k1(n) / r(n)
                qy(p) = qy(p) + wg(n) * xg(n)**p * bigy * k1(n) / r(n)
            end do
        end do
        qx = -qx * L / (4*pi*biglab) * 2.d0/L
        qy = -qy * L / (4*pi*biglab) * 2.d0/L

        angz = atan2(aimag(z2-z1),real(z2-z1))
        qxqy(0:order) = qx * cos(angz) - qy * sin(angz)
        qxqy(order+1:2*order+1) = qx * sin(angz) + qy * cos(angz)
        return
    end function bessells_gauss_ho_qxqy

    function bessells_gauss_ho_qxqy_d1d2(x,y,z1,z2,lab,order,d1,d2) result(qxqy)
        ! Returns integral from d1 to d2 along real axis while strength is still Delta^order from -1 to +1
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,d1,d2
        complex(kind=8), intent(in) :: z1,z2,lab
        complex(kind=8), dimension(0:2*order+1) :: qxqy, qxqyc
        integer :: n, m
        real(kind=8) :: xp, yp, dc, fac
        complex(kind=8) :: z1p,z2p,bigz1,bigz2
        bigz1 = dcmplx(d1,0.d0)
        bigz2 = dcmplx(d2,0.d0)
        z1p = 0.5d0 * (z2-z1) * bigz1 + 0.5d0 * (z1+z2)
        z2p = 0.5d0 * (z2-z1) * bigz2 + 0.5d0 * (z1+z2)
        qxqyc = bessells_gauss_ho_qxqy(x,y,z1p,z2p,lab,order)
        dc = (d1+d2) / (d2-d1)
        qxqy = 0.d0
        do n = 0, order
            do m = 0, n
                qxqy(n) = qxqy(n) + gam(n,m) * dc**(n-m) * qxqyc(m)
                qxqy(n+order+1) = qxqy(n+order+1) + gam(n,m) * dc**(n-m) * qxqyc(m+order+1)
            enddo
            qxqy(n) = ( 0.5*(d2-d1) )**n * qxqy(n)
            qxqy(n+order+1) = ( 0.5*(d2-d1) )**n * qxqy(n+order+1)
        end do
    end function bessells_gauss_ho_qxqy_d1d2

    function bessells(x,y,z1,z2,lab,order,d1in,d2in) result(omega)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,d1in,d2in
        complex(kind=8), intent(in) :: z1,z2
        complex(kind=8), intent(in) :: lab
        complex(kind=8), dimension(0:order) :: omega

        integer :: Nls, n
        real(kind=8) :: Lnear, L, d1, d2, delta
        complex(kind=8) :: z, delz, za, zb

        Lnear = 3.d0
        z = cmplx(x,y,kind=8)
        omega(0:order) = cmplx(0.d0,0.d0,kind=8)
        L = abs(z2-z1)
        if ( L < Lnear*abs(lab) ) then  ! No need to break integral up
            if ( abs( z - 0.5d0*(z1+z2) ) < 0.5d0 * Lnear * L ) then  ! Do integration
                omega = bessells_int_ho(x,y,z1,z2,lab,order,d1in,d2in)
            else
                omega = bessells_gauss_ho_d1d2(x,y,z1,z2,lab,order,d1in,d2in)
            end if
        else  ! Break integral up in parts
            Nls = ceiling( L / (Lnear*abs(lab)) )
            delta = 2.d0 / Nls
            delz = (z2-z1)/Nls
            L = abs(delz)
            do n = 1,Nls
                d1 = -1.d0 + (n-1) * delta
                d2 = -1.d0 + n * delta
                if ((d2 < d1in) .or. (d1 > d2in)) then
                    cycle
                end if
                d1 = max(d1,d1in)
                d2 = min(d2,d2in)
                za = z1 + (n-1) * delz
                zb = z1 + n * delz
                if ( abs( z - 0.5d0*(za+zb) ) < 0.5d0 * Lnear * L ) then  ! Do integration
                    omega = omega + bessells_int_ho(x,y,z1,z2,lab,order,d1,d2)
                else
                    omega = omega + bessells_gauss_ho_d1d2(x,y,z1,z2,lab,order,d1,d2)
                end if
            end do
        end if
        return
    end function bessells

    function bessellsv(x,y,z1,z2,lab,order,R,nlab) result(omega)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,R
        complex(kind=8), intent(in) :: z1,z2
        integer, intent(in) :: nlab
        real(kind=8) :: d1, d2
        complex(kind=8), dimension(nlab), intent(in) :: lab
        complex(kind=8), dimension(nlab*(order+1)) :: omega
        integer :: n, nterms
        nterms = order+1
        ! Check if endpoints need to be adjusted using the largest lambda (the first one)
        call find_d1d2( z1, z2, dcmplx(x,y), R*abs(lab(1)), d1, d2 )
        do n = 1,nlab
            omega((n-1)*nterms+1:n*nterms) = bessells(x,y,z1,z2,lab(n),order,d1,d2)
        end do
    end function bessellsv

    function bessellsv2(x,y,z1,z2,lab,order,R,nlab) result(omega)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,R
        complex(kind=8), intent(in) :: z1,z2
        integer, intent(in) :: nlab
        real(kind=8) :: d1, d2
        complex(kind=8), dimension(nlab), intent(in) :: lab
        complex(kind=8), dimension(order+1,nlab) :: omega
        integer :: n, nterms
        nterms = order+1
        ! Check if endpoints need to be adjusted using the largest lambda (the first one)
        call find_d1d2( z1, z2, dcmplx(x,y), R*abs(lab(1)), d1, d2 )
        do n = 1,nlab
            omega(1:nterms,n) = bessells(x,y,z1,z2,lab(n),order,d1,d2)
        end do
    end function bessellsv2

    function bessellsqxqy(x,y,z1,z2,lab,order,d1in,d2in) result(qxqy)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,d1in,d2in
        complex(kind=8), intent(in) :: z1,z2
        complex(kind=8), intent(in) :: lab
        complex(kind=8), dimension(0:2*order+1) :: qxqy

        integer :: Nls, n
        real(kind=8) :: Lnear, L, d1, d2, delta
        complex(kind=8) :: z, delz, za, zb

        Lnear = 3.d0
        z = cmplx(x,y,kind=8)
        qxqy = dcmplx(0.d0,0.d0)
        L = abs(z2-z1)
        !print *,'Lnear*abs(lab) ',Lnear*abs(lab)
        if ( L < Lnear*abs(lab) ) then  ! No need to break integral up
            if ( abs( z - 0.5d0*(z1+z2) ) < 0.5d0 * Lnear * L ) then  ! Do integration
                qxqy = bessells_int_ho_qxqy(x,y,z1,z2,lab,order,d1in,d2in)
            else
                qxqy = bessells_gauss_ho_qxqy_d1d2(x,y,z1,z2,lab,order,d1in,d2in)
            end if
        else  ! Break integral up in parts
            Nls = ceiling( L / (Lnear*abs(lab)) )
            !print *,'NLS ',Nls
            delta = 2.d0 / Nls
            delz = (z2-z1)/Nls
            L = abs(delz)
            do n = 1,Nls
                d1 = -1.d0 + (n-1) * delta
                d2 = -1.d0 + n * delta
                if ((d2 < d1in) .or. (d1 > d2in)) then
                    cycle
                end if
                d1 = max(d1,d1in)
                d2 = min(d2,d2in)
                za = z1 + (n-1) * delz
                zb = z1 + n * delz
                if ( abs( z - 0.5d0*(za+zb) ) < 0.5d0 * Lnear * L ) then  ! Do integration
                    qxqy = qxqy + bessells_int_ho_qxqy(x,y,z1,z2,lab,order,d1,d2)
                else
                    qxqy = qxqy + bessells_gauss_ho_qxqy_d1d2(x,y,z1,z2,lab,order,d1,d2)
                end if
            end do
        end if
        return
    end function bessellsqxqy

    function bessellsqxqyv(x,y,z1,z2,lab,order,R,nlab) result(qxqy)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,R
        complex(kind=8), intent(in) :: z1,z2
        integer, intent(in) :: nlab
        real(kind=8) :: d1, d2
        complex(kind=8), dimension(nlab), intent(in) :: lab
        complex(kind=8), dimension(2*nlab*(order+1)) :: qxqy
        complex(kind=8), dimension(0:2*order+1) :: qxqylab
        integer :: n, nterms, nhalf
        nterms = order+1
        nhalf = nlab*(order+1)
        call find_d1d2( z1, z2, dcmplx(x,y), R*abs(lab(1)), d1, d2 )
        do n = 1,nlab
            qxqylab = bessellsqxqy(x,y,z1,z2,lab(n),order,d1,d2)
            qxqy((n-1)*nterms+1:n*nterms) = qxqylab(0:order)
            qxqy((n-1)*nterms+1+nhalf:n*nterms+nhalf) = qxqylab(order+1:2*order+1)
        end do
    end function bessellsqxqyv

    function bessellsqxqyv2(x,y,z1,z2,lab,order,R,nlab) result(qxqy)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,R
        complex(kind=8), intent(in) :: z1,z2
        integer, intent(in) :: nlab
        real(kind=8) :: d1, d2
        complex(kind=8), dimension(nlab), intent(in) :: lab
        complex(kind=8), dimension(2*(order+1),nlab) :: qxqy
        complex(kind=8), dimension(0:2*order+1) :: qxqylab
        integer :: n, nterms, nhalf
        nterms = order+1
        nhalf = nlab*(order+1)
        call find_d1d2( z1, z2, dcmplx(x,y), R*abs(lab(1)), d1, d2 )
        do n = 1,nlab
            qxqylab = bessellsqxqy(x,y,z1,z2,lab(n),order,d1,d2)
            qxqy(1:nterms,n) = qxqylab(0:order)
            qxqy(nterms+1:2*nterms,n) = qxqylab(order+1:2*order+1)
        end do
    end function bessellsqxqyv2

    function bessellsuni(x, y, z1, z2, lab) result(omega)
        ! Uniform strength
        implicit none
        real(kind=8), intent(in) :: x,y
        complex(kind=8), intent(in) :: z1,z2
        complex(kind=8), intent(in) :: lab
        complex(kind=8) :: omega

        integer :: Nls, n
        real(kind=8) :: Lnear, L
        complex(kind=8) :: z, delz, za, zb

        Lnear = 3.d0
        z = cmplx(x,y,kind=8)
        omega = cmplx(0.d0,0.d0,kind=8)
        L = abs(z2-z1)
        if ( L < Lnear*abs(lab) ) then  ! No need to break integral up
            if ( abs( z - 0.5d0*(z1+z2) ) < 0.5d0 * Lnear * L ) then  ! Do integration
                omega = bessells_int(x,y,z1,z2,lab)
            else
                omega = bessells_gauss(x,y,z1,z2,lab)
            end if
        else  ! Break integral up in parts
            Nls = ceiling( L / (Lnear*abs(lab)) )
            delz = (z2-z1)/Nls
            L = abs(delz)
            do n = 1,Nls
                za = z1 + (n-1) * delz
                zb = z1 + n * delz
                if ( abs( z - 0.5d0*(za+zb) ) < 0.5d0 * Lnear * L ) then  ! Do integration
                    omega = omega + bessells_int(x,y,za,zb,lab)
                else
                    omega = omega + bessells_gauss(x,y,za,zb,lab)
                end if
            end do
        end if
        return
    end function bessellsuni

    function bessellsuniv(x, y, z1, z2, lab, nlab) result(omega)
        ! Uniform strength
        implicit none
        real(kind=8), intent(in) :: x,y
        complex(kind=8), intent(in) :: z1,z2
        integer, intent(in) :: nlab
        complex(kind=8), dimension(nlab), intent(in) :: lab
        complex(kind=8), dimension(nlab) :: omega
        integer :: n
        do n = 1, nlab
            omega(n) = bessellsuni(x,y,z1,z2,lab(n))
        end do
    end function bessellsuniv

!!!!!!! Line Doublet Functions
    function lapld_int_ho(x,y,z1,z2,order) result(omega)
        ! Near field only
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y
        complex(kind=8), intent(in) :: z1,z2
        complex(kind=8), dimension(0:order) :: omega, qm
        integer :: m, n
        real(kind=8) :: L
        complex(kind=8) :: z, zplus1, zmin1

        L = abs(z2-z1)
        z = (2.d0 * dcmplx(x,y) - (z1+z2) ) / (z2-z1)
        zplus1 = z + 1.d0; zmin1 = z - 1.d0
        ! Not sure if this gives correct answer at corner point (z also appears in qm); should really be caught in code that calls this function
        if  (abs(zplus1) < tiny) zplus1 = tiny
        if  (abs(zmin1) < tiny) zmin1 = tiny

        omega(0) = cdlog(zmin1/zplus1)
        do n=1,order
            omega(n) = z * omega(n-1)
        end do

        qm(0) = 0.d0
        if (order > 0) qm(1) = 2.d0
        do m=3,order,2
            qm(m) = qm(m-2) * z * z + 2.d0 / m
        end do
        do m=2,order,2
            qm(m) = qm(m-1) * z
        end do

        omega = 1.d0 / (dcmplx(0.d0,2.d0) * pi) * ( omega + qm )
    end function lapld_int_ho

    function lapld_int_ho_d1d2(x,y,z1,z2,order,d1,d2) result(omega)
        ! Near field only
        ! Returns integral from d1 to d2 along real axis while strength is still Delta^order from -1 to +1
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,d1,d2
        complex(kind=8), intent(in) :: z1,z2
        complex(kind=8), dimension(0:order) :: omega, omegac
        integer :: n, m
        real(kind=8) :: xp, yp, dc, fac
        complex(kind=8) :: z1p,z2p,bigz1,bigz2
        bigz1 = dcmplx(d1,0.d0)
        bigz2 = dcmplx(d2,0.d0)
        z1p = 0.5d0 * (z2-z1) * bigz1 + 0.5d0 * (z1+z2)
        z2p = 0.5d0 * (z2-z1) * bigz2 + 0.5d0 * (z1+z2)
        omegac = lapld_int_ho(x,y,z1p,z2p,order)
        dc = (d1+d2) / (d2-d1)
        omega(0:order) = 0.d0
        do n = 0, order
            do m = 0, n
                omega(n) = omega(n) + gam(n,m) * dc**(n-m) * omegac(m)
            enddo
            omega(n) = ( 0.5*(d2-d1) )**n * omega(n)
        end do
    end function lapld_int_ho_d1d2

    function lapld_int_ho_wdis(x,y,z1,z2,order) result(wdis)
        ! Near field only
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y
        complex(kind=8), intent(in) :: z1,z2
        complex(kind=8), dimension(0:order) :: wdis
        complex(kind=8), dimension(0:10) :: qm  ! Max order is 10
        integer :: m, n
        complex(kind=8) :: z, zplus1, zmin1, term1, term2, zterm

        z = (2.d0 * dcmplx(x,y) - (z1+z2) ) / (z2-z1)
        zplus1 = z + 1.d0; zmin1 = z - 1.d0
        ! Not sure if this gives correct answer at corner point (z also appears in qm); should really be caught in code that calls this function
        if  (abs(zplus1) < tiny) zplus1 = tiny
        if  (abs(zmin1) < tiny) zmin1 = tiny

        qm(0:1) = 0.d0
        do m = 2,order
            qm(m) = 0.d0
            do n=1,m/2
                qm(m) = qm(m) + (m-2*n+1) * z**(m-2*n) / (2*n-1)
            end do
        end do

        term1 = 1.d0 / zmin1 - 1.d0 / zplus1
        term2 = cdlog(zmin1/zplus1)
        wdis(0) = term1
        zterm = dcmplx(1.d0,0.d0)
        do m = 1,order
            wdis(m) = m * zterm * term2 + z * zterm * term1 + 2.d0 * qm(m)
            zterm = zterm * z
        end do

        wdis = - wdis / (pi*dcmplx(0.d0,1.d0)*(z2-z1))
    end function lapld_int_ho_wdis

    function lapld_int_ho_wdis_d1d2(x,y,z1,z2,order,d1,d2) result(wdis)
        ! Near field only
        ! Returns integral from d1 to d2 along real axis while strength is still Delta^order from -1 to +1
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,d1,d2
        complex(kind=8), intent(in) :: z1,z2
        complex(kind=8), dimension(0:order) :: wdis, wdisc
        integer :: n, m
        real(kind=8) :: xp, yp, dc, fac
        complex(kind=8) :: z1p,z2p,bigz1,bigz2
        bigz1 = dcmplx(d1,0.d0)
        bigz2 = dcmplx(d2,0.d0)
        z1p = 0.5d0 * (z2-z1) * bigz1 + 0.5d0 * (z1+z2)
        z2p = 0.5d0 * (z2-z1) * bigz2 + 0.5d0 * (z1+z2)
        wdisc = lapld_int_ho_wdis(x,y,z1p,z2p,order)
        dc = (d1+d2) / (d2-d1)
        wdis(0:order) = 0.d0
        do n = 0, order
            do m = 0, n
                wdis(n) = wdis(n) + gam(n,m) * dc**(n-m) * wdisc(m)
            enddo
            wdis(n) = ( 0.5*(d2-d1) )**n * wdis(n)
        end do
    end function lapld_int_ho_wdis_d1d2

    function besselld_int_ho(x,y,z1,z2,lab,order,d1,d2) result(omega)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,d1,d2
        complex(kind=8), intent(in) :: z1,z2,lab
        complex(kind=8), dimension(0:order) :: omega
        real(kind=8) :: biglab, biga, L, ang, tol, bigy
        complex(kind=8) :: zeta, zetabar, log1, log2, term1, term2, d1minzeta, d2minzeta, bigz
        complex(kind=8) :: cm, biglabcomplex
        complex(kind=8), dimension(0:20) :: zminzbar, anew, bnew, exprange
        complex(kind=8), dimension(0:20,0:20) :: gamnew, gam2
        complex(kind=8), dimension(0:40) :: alpha, beta, alpha2
        complex(kind=8), dimension(0:50) :: alphanew, betanew, alphanew2 ! Order fixed to 10

        integer :: m, n, p

        L = abs(z2-z1)
        bigz = (2.d0 * dcmplx(x,y) - (z1+z2) ) / (z2-z1)
        bigy = aimag(bigz)
        biga = abs(lab)
        ang = atan2(aimag(lab),real(lab))
        biglab = 2.d0 * biga / L
        biglabcomplex = 2.0 * lab / L

        tol = 1.d-12

        exprange = exp(-cmplx(0,2,kind=8) * ang * nrange )
        anew = a1 * exprange
        bnew = (b1 - a1 * cmplx(0,2,kind=8) * ang) * exprange

        zeta = (2.d0 * dcmplx(x,y) - (z1+z2) ) / (z2-z1) / biglab
        zetabar = conjg(zeta)
        zminzbar(20) = 1.d0
        do n = 1,20
            zminzbar(20-n) = zminzbar(21-n) * (zeta-zetabar)  ! Ordered from high power to low power
        end do
        gamnew = gam
        do n = 0,20
            gamnew(n,0:n) = gamnew(n,0:n) * zminzbar(20-n:20)
            gam2(n,0:n) = conjg(gamnew(n,0:n))
        end do

        alpha(0:40) = 0.d0
        beta(0:40) = 0.d0
        alpha2(0:40) = 0.d0
        alpha(0) = anew(0)
        beta(0) = bnew(0)
        alpha2(0) = anew(0)
        do n = 1,20
            alpha(n:2*n) = alpha(n:2*n) + anew(n) * gamnew(n,0:n)
            beta(n:2*n)  = beta(n:2*n)  + bnew(n) * gamnew(n,0:n)
            alpha2(n:2*n) = alpha2(n:2*n) + anew(n) * gam2(n,0:n)
        end do

        d1minzeta = d1/biglab - zeta
        d2minzeta = d2/biglab - zeta
        !d1minzeta = -1.d0/biglab - zeta
        !d2minzeta = 1.d0/biglab - zeta
        if (abs(d1minzeta) < tol) d1minzeta = d1minzeta + cmplx(tol,0.d0,kind=8)
        if (abs(d2minzeta) < tol) d2minzeta = d2minzeta + cmplx(tol,0.d0,kind=8)
        log1 = log(d1minzeta)
        log2 = log(d2minzeta)

        do p = 0,order

            alphanew(0:40+p) = 0.d0
            betanew(0:40+p) = 0.d0
            alphanew2(0:40+p) = 0.d0
            do m = 0,p
                cm = biglab**p * gam(p,m) * zeta**(p-m)
                alphanew(m:40+m) = alphanew(m:40+m) + cm * alpha(0:40)
                betanew(m:40+m)  = betanew(m:40+m) + cm * beta(0:40)
                cm = biglab**p * gam(p,m) * zetabar**(p-m)
                alphanew2(m:40+m) = alphanew2(m:40+m) + cm * alpha2(0:40)
            end do

            omega(p) = 0.d0
            term1 = 1.d0
            term2 = 1.d0
            do n = 0,40
                term1 = term1 * d1minzeta
                term2 = term2 * d2minzeta
                omega(p) = omega(p) + ( alphanew(n) * log2 - alphanew(n) / (n+1) + betanew(n) ) * term2 / (n+1)
                omega(p) = omega(p) - ( alphanew(n) * log1 - alphanew(n) / (n+1) + betanew(n) ) * term1 / (n+1)
                omega(p) = omega(p) + ( alphanew2(n) * conjg(log2) - alphanew2(n) / (n+1) ) * conjg(term2) / (n+1)
                omega(p) = omega(p) - ( alphanew2(n) * conjg(log1) - alphanew2(n) / (n+1) ) * conjg(term1) / (n+1)
            end do

        end do

        ! omega = bigy * biglab / (2.d0*pi*biglabcomplex**2) * omega + real( lapld_int_ho_d1d2(x,y,z1,z2,order,d1,d2) )
        omega = bigy * biglab / (2.d0*pi*biglabcomplex**2) * omega + real( lapld_int_ho_d1d2(x,y,z1,z2,order,d1,d2) )


        return
    end function besselld_int_ho

    function besselld_gauss_ho(x,y,z1,z2,lab,order) result(omega)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y
        complex(kind=8), intent(in) :: z1,z2
        complex(kind=8), intent(in) :: lab
        complex(kind=8), dimension(0:order) :: omega
        integer :: n, p
        real(kind=8) :: L, x0, r
        complex(kind=8) :: bigz, biglab
        complex(kind=8), dimension(8) :: k1overr

        L = abs(z2-z1)
        biglab = 2.d0 * lab / L
        bigz = (2.d0 * cmplx(x,y,kind=8) - (z1+z2) ) / (z2-z1)
        do n = 1,8
            x0 = real(bigz) - xg(n)
            r = sqrt( x0**2 + aimag(bigz)**2 )
            k1overr(n) = besselk1( x0, aimag(bigz), biglab ) / r
        end do
        do p = 0,order
            omega(p) = cmplx(0.d0,0.d0,kind=8)
            do n = 1,8
                omega(p) = omega(p) + wg(n) * xg(n)**p * k1overr(n)
            end do
            omega(p) = aimag(bigz)/(2.d0*pi*biglab) * omega(p)
        end do
        return
    end function besselld_gauss_ho

    function besselld_gauss_ho_d1d2(x,y,z1,z2,lab,order,d1,d2) result(omega)
        ! Returns integral from d1 to d2 along real axis while strength is still Delta^order from -1 to +1
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,d1,d2
        complex(kind=8), intent(in) :: z1,z2,lab
        complex(kind=8), dimension(0:order) :: omega, omegac
        integer :: n, m
        real(kind=8) :: xp, yp, dc, fac
        complex(kind=8) :: z1p,z2p,bigz1,bigz2
        bigz1 = dcmplx(d1,0.d0)
        bigz2 = dcmplx(d2,0.d0)
        z1p = 0.5d0 * (z2-z1) * bigz1 + 0.5d0 * (z1+z2)
        z2p = 0.5d0 * (z2-z1) * bigz2 + 0.5d0 * (z1+z2)
        omegac = besselld_gauss_ho(x,y,z1p,z2p,lab,order)
        dc = (d1+d2) / (d2-d1)
        omega(0:order) = 0.d0
        do n = 0, order
            do m = 0, n
                omega(n) = omega(n) + gam(n,m) * dc**(n-m) * omegac(m)
            enddo
            omega(n) = ( 0.5*(d2-d1) )**n * omega(n)
        end do
    end function besselld_gauss_ho_d1d2

    function besselld(x,y,z1,z2,lab,order,d1in,d2in) result(omega)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,d1in,d2in
        complex(kind=8), intent(in) :: z1,z2
        complex(kind=8), intent(in) :: lab
        complex(kind=8), dimension(0:order) :: omega

        integer :: Nls, n
        real(kind=8) :: Lnear, L, d1, d2, delta
        complex(kind=8) :: z, delz, za, zb

        Lnear = 3.d0
        z = cmplx(x,y,kind=8)
        omega(0:order) = cmplx(0.d0,0.d0,kind=8)
        L = abs(z2-z1)
        if ( L < Lnear*abs(lab) ) then  ! No need to break integral up
            if ( abs( z - 0.5d0*(z1+z2) ) < 0.5d0 * Lnear * L ) then  ! Do integration
                omega = besselld_int_ho(x,y,z1,z2,lab,order,d1in,d2in)
            else
                omega = besselld_gauss_ho_d1d2(x,y,z1,z2,lab,order,d1in,d2in)
            end if
        else  ! Break integral up in parts
            Nls = ceiling( L / (Lnear*abs(lab)) )
            delta = 2.d0 / Nls
            delz = (z2-z1)/Nls
            L = abs(delz)
            do n = 1,Nls
                d1 = -1.d0 + (n-1) * delta
                d2 = -1.d0 + n * delta
                if ((d2 < d1in) .or. (d1 > d2in)) then
                    cycle
                end if
                d1 = max(d1,d1in)
                d2 = min(d2,d2in)
                za = z1 + (n-1) * delz
                zb = z1 + n * delz
                if ( abs( z - 0.5d0*(za+zb) ) < 0.5d0 * Lnear * L ) then  ! Do integration
                    omega = omega + besselld_int_ho(x,y,z1,z2,lab,order,d1,d2)
                else
                    omega = omega + besselld_gauss_ho_d1d2(x,y,z1,z2,lab,order,d1,d2)
                end if
            end do
        end if
        return
    end function besselld

    function besselldv(x,y,z1,z2,lab,order,R,nlab) result(omega)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,R
        complex(kind=8), intent(in) :: z1,z2
        integer, intent(in) :: nlab
        real(kind=8) :: d1, d2
        complex(kind=8), dimension(nlab), intent(in) :: lab
        complex(kind=8), dimension(nlab*(order+1)) :: omega
        integer :: n, nterms
        nterms = order+1
        ! Check if endpoints need to be adjusted using the largest lambda (the first one)
        call find_d1d2( z1, z2, dcmplx(x,y), R*abs(lab(1)), d1, d2 )
        do n = 1,nlab
            omega((n-1)*nterms+1:n*nterms) = besselld(x,y,z1,z2,lab(n),order,d1,d2)
        end do
    end function besselldv

    function besselldv2(x,y,z1,z2,lab,order,R,nlab) result(omega)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,R
        complex(kind=8), intent(in) :: z1,z2
        integer, intent(in) :: nlab
        real(kind=8) :: d1, d2
        complex(kind=8), dimension(nlab), intent(in) :: lab
        complex(kind=8), dimension(order+1,nlab) :: omega
        integer :: n, nterms
        nterms = order+1
        ! Check if endpoints need to be adjusted using the largest lambda (the first one)
        call find_d1d2( z1, z2, dcmplx(x,y), R*abs(lab(1)), d1, d2 )
        do n = 1,nlab
            omega(1:nterms,n) = besselld(x,y,z1,z2,lab(n),order,d1,d2)
        end do
    end function besselldv2

    function besselldpart(x,y,z1,z2,lab,order,d1,d2) result(omega)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,d1,d2
        complex(kind=8), intent(in) :: z1,z2,lab
        complex(kind=8), dimension(0:order) :: omega
        real(kind=8) :: biglab, biga, L, ang, tol, bigy
        complex(kind=8) :: zeta, zetabar, log1, log2, term1, term2, d1minzeta, d2minzeta, bigz
        complex(kind=8) :: cm, biglabcomplex
        complex(kind=8), dimension(0:20) :: zminzbar, anew, bnew, exprange
        complex(kind=8), dimension(0:20,0:20) :: gamnew, gam2
        complex(kind=8), dimension(0:40) :: alpha, beta, alpha2
        complex(kind=8), dimension(0:50) :: alphanew, betanew, alphanew2 ! Order fixed to 10

        integer :: m, n, p

        L = abs(z2-z1)
        bigz = (2.d0 * dcmplx(x,y) - (z1+z2) ) / (z2-z1)
        bigy = aimag(bigz)
        biga = abs(lab)
        ang = atan2(aimag(lab),real(lab))
        biglab = 2.d0 * biga / L
        biglabcomplex = 2.0 * lab / L

        tol = 1.d-12

        exprange = exp(-cmplx(0,2,kind=8) * ang * nrange )
        anew = a1 * exprange
        bnew = (b1 - a1 * cmplx(0,2,kind=8) * ang) * exprange
        !anew(0) = 0.d0
        !print *,'anew0 ',anew(0)
        !anew(1:20) = 0.d0
        !bnew(0:20) = 0.d0

        zeta = (2.d0 * dcmplx(x,y) - (z1+z2) ) / (z2-z1) / biglab
        zetabar = conjg(zeta)
        zminzbar(20) = 1.d0
        do n = 1,20
            zminzbar(20-n) = zminzbar(21-n) * (zeta-zetabar)  ! Ordered from high power to low power
        end do
        gamnew = gam
        do n = 0,20
            gamnew(n,0:n) = gamnew(n,0:n) * zminzbar(20-n:20)
            gam2(n,0:n) = conjg(gamnew(n,0:n))
        end do

        alpha(0:40) = 0.d0
        beta(0:40) = 0.d0
        alpha2(0:40) = 0.d0
        alpha(0) = anew(0)
        beta(0) = bnew(0)
        alpha2(0) = anew(0)
        do n = 1,20
            alpha(n:2*n) = alpha(n:2*n) + anew(n) * gamnew(n,0:n)
            beta(n:2*n)  = beta(n:2*n)  + bnew(n) * gamnew(n,0:n)
            alpha2(n:2*n) = alpha2(n:2*n) + anew(n) * gam2(n,0:n)
        end do

        d1minzeta = d1/biglab - zeta
        d2minzeta = d2/biglab - zeta
        !d1minzeta = -1.d0/biglab - zeta
        !d2minzeta = 1.d0/biglab - zeta
        if (abs(d1minzeta) < tol) d1minzeta = d1minzeta + cmplx(tol,0.d0,kind=8)
        if (abs(d2minzeta) < tol) d2minzeta = d2minzeta + cmplx(tol,0.d0,kind=8)
        log1 = log(d1minzeta)
        log2 = log(d2minzeta)

        do p = 0,order

            alphanew(0:40+p) = 0.d0
            betanew(0:40+p) = 0.d0
            alphanew2(0:40+p) = 0.d0
            do m = 0,p
                cm = biglab**p * gam(p,m) * zeta**(p-m)
                alphanew(m:40+m) = alphanew(m:40+m) + cm * alpha(0:40)
                betanew(m:40+m)  = betanew(m:40+m) + cm * beta(0:40)
                cm = biglab**p * gam(p,m) * zetabar**(p-m)
                alphanew2(m:40+m) = alphanew2(m:40+m) + cm * alpha2(0:40)
            end do

            omega(p) = 0.d0
            term1 = 1.d0
            term2 = 1.d0
            do n = 0,40+p
                term1 = term1 * d1minzeta
                term2 = term2 * d2minzeta
                omega(p) = omega(p) + ( alphanew(n) * log2 - alphanew(n) / (n+1) + betanew(n) ) * term2 / (n+1)
                omega(p) = omega(p) - ( alphanew(n) * log1 - alphanew(n) / (n+1) + betanew(n) ) * term1 / (n+1)
                omega(p) = omega(p) + ( alphanew2(n) * conjg(log2) - alphanew2(n) / (n+1) ) * conjg(term2) / (n+1)
                omega(p) = omega(p) - ( alphanew2(n) * conjg(log1) - alphanew2(n) / (n+1) ) * conjg(term1) / (n+1)
            end do

        end do

        omega = biglab / (2.d0*pi*biglabcomplex**2) * omega !+ real( lapld_int_ho(x,y,z1,z2,order) )
        !omega = real( lapld_int_ho(x,y,z1,z2,order) )

        return
    end function besselldpart

    function besselld_int_ho_qxqy(x,y,z1,z2,lab,order,d1,d2) result(qxqy)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,d1,d2
        complex(kind=8), intent(in) :: z1,z2,lab
        complex(kind=8), dimension(0:2*order+1) :: qxqy
        complex(kind=8), dimension(0:order) :: rvz, rvzbar
        real(kind=8) :: biglab, biga, L, ang, angz, tol, bigy
        complex(kind=8) :: zeta, zetabar, log1, log2, term1, term2, d1minzeta, d2minzeta, bigz
        complex(kind=8) :: cm, biglabcomplex, azero
        complex(kind=8), dimension(0:20) :: zminzbar, anew, bnew, exprange
        complex(kind=8), dimension(0:20,0:20) :: gamnew, gam2
        complex(kind=8), dimension(0:40) :: alpha, beta, alpha2
        complex(kind=8), dimension(0:51) :: alphanew, betanew, alphanew2 ! Order fixed to 10
        complex(kind=8), dimension(0:order) :: omegalap, omegaom, wdis, qx, qy ! To store intermediate result
        complex(kind=8), dimension(0:order+1) :: omega ! To store intermediate result


        integer :: m, n, p

        L = abs(z2-z1)
        bigz = (2.d0 * dcmplx(x,y) - (z1+z2) ) / (z2-z1)
        bigy = aimag(bigz)
        biga = abs(lab)
        ang = atan2(aimag(lab),real(lab))
        angz = atan2(aimag(z2-z1),real(z2-z1))
        biglab = 2.d0 * biga / L
        biglabcomplex = 2.0 * lab / L

        tol = 1.d-12

        exprange = exp(-cmplx(0,2,kind=8) * ang * nrange )
        anew = a1 * exprange
        bnew = (b1 - a1 * cmplx(0,2,kind=8) * ang) * exprange
        azero = anew(0)

        do n = 0,19
            bnew(n) = (n+1)*bnew(n+1) + anew(n+1)
            anew(n) = (n+1)*anew(n+1)
        end do
        anew(20) = 0.d0  ! This is a bit lazy
        bnew(20) = 0.d0

        zeta = (2.d0 * dcmplx(x,y) - (z1+z2) ) / (z2-z1) / biglab
        zetabar = conjg(zeta)
        zminzbar(20) = 1.d0
        do n = 1,20
            zminzbar(20-n) = zminzbar(21-n) * (zeta-zetabar)  ! Ordered from high power to low power
        end do
        gamnew = gam
        do n = 0,20
            gamnew(n,0:n) = gamnew(n,0:n) * zminzbar(20-n:20)
            gam2(n,0:n) = conjg(gamnew(n,0:n))
        end do

        alpha(0:40) = 0.d0
        beta(0:40) = 0.d0
        alpha2(0:40) = 0.d0
        alpha(0) = anew(0)
        beta(0) = bnew(0)
        alpha2(0) = anew(0)
        do n = 1,20
            alpha(n:2*n) = alpha(n:2*n) + anew(n) * gamnew(n,0:n)
            beta(n:2*n)  = beta(n:2*n)  + bnew(n) * gamnew(n,0:n)
            alpha2(n:2*n) = alpha2(n:2*n) + anew(n) * gam2(n,0:n)
        end do

        d1minzeta = d1/biglab - zeta
        d2minzeta = d2/biglab - zeta
        !d1minzeta = -1.d0/biglab - zeta
        !d2minzeta = 1.d0/biglab - zeta
        if (abs(d1minzeta) < tol) d1minzeta = d1minzeta + cmplx(tol,0.d0,kind=8)
        if (abs(d2minzeta) < tol) d2minzeta = d2minzeta + cmplx(tol,0.d0,kind=8)
        log1 = log(d1minzeta)
        log2 = log(d2minzeta)

        do p = 0,order+1

            alphanew(0:40+p) = 0.d0
            betanew(0:40+p) = 0.d0
            alphanew2(0:40+p) = 0.d0
            do m = 0,p
                cm = biglab**p * gam(p,m) * zeta**(p-m)
                alphanew(m:40+m) = alphanew(m:40+m) + cm * alpha(0:40)
                betanew(m:40+m)  = betanew(m:40+m) + cm * beta(0:40)
                cm = biglab**p * gam(p,m) * zetabar**(p-m)
                alphanew2(m:40+m) = alphanew2(m:40+m) + cm * alpha2(0:40)
            end do

            omega(p) = 0.d0
            term1 = 1.d0
            term2 = 1.d0
            do n = 0,40
                term1 = term1 * d1minzeta
                term2 = term2 * d2minzeta
                omega(p) = omega(p) + ( alphanew(n) * log2 - alphanew(n) / (n+1) + betanew(n) ) * term2 / (n+1)
                omega(p) = omega(p) - ( alphanew(n) * log1 - alphanew(n) / (n+1) + betanew(n) ) * term1 / (n+1)
                omega(p) = omega(p) + ( alphanew2(n) * conjg(log2) - alphanew2(n) / (n+1) ) * conjg(term2) / (n+1)
                omega(p) = omega(p) - ( alphanew2(n) * conjg(log1) - alphanew2(n) / (n+1) ) * conjg(term1) / (n+1)
            end do

        end do

        omegalap = lapld_int_ho_d1d2(x,y,z1,z2,order,d1,d2) / dcmplx(0.d0,1.d0)
        omegaom = besselldpart(x,y,z1,z2,lab,order,d1,d2)
        wdis = lapld_int_ho_wdis_d1d2(x,y,z1,z2,order,d1,d2)

        rvz =    -biglab * bigy / (2.d0*pi*biglabcomplex**2) * (omega(1:order+1)/biglab - zetabar * omega(0:order)) + &
                 biglab * omegaom / dcmplx(0.d0,2.d0)
        rvzbar = -biglab * bigy / (2.d0*pi*biglabcomplex**2) * (omega(1:order+1)/biglab - zeta * omega(0:order)) - &
                 biglab * omegaom / dcmplx(0.d0,2.d0)
        !qxqy(0:order) = -2.0 / L * ( rvz + rvzbar ) / biglab  ! As we need to take derivative w.r.t. z not zeta
        !qxqy(order+1:2*order+1) = -2.0 / L * dcmplx(0,1) * (rvz-rvzbar) / biglab
        !
        !qxqy(0:order) = qxqy(0:order) - 2.0 / L / biglabcomplex**2 * azero * ( omegalap + conjg(omegalap) )
        !qxqy(order+1:2*order+1) = qxqy(order+1:2*order+1) -  &
        !                          2.0 / L / biglabcomplex**2 * azero * dcmplx(0,1) * (omegalap - conjg(omegalap))
        !
        !qxqy(0:order) = qxqy(0:order) + real(wdis)
        !qxqy(order+1:2*order+1) = qxqy(order+1:2*order+1) - aimag(wdis)

        qx = -2.0 / L * ( rvz + rvzbar ) / biglab  ! As we need to take derivative w.r.t. z not zeta
        qy = -2.0 / L * dcmplx(0,1) * (rvz-rvzbar) / biglab

        qx = qx - 2.0 / L * bigy / biglabcomplex**2 * azero * ( omegalap + conjg(omegalap) )
        qy = qy - 2.0 / L * bigy / biglabcomplex**2 * azero * dcmplx(0,1) * (omegalap - conjg(omegalap))

        !qx = qx + real(wdis * (z2-z1) / L)
        !qy = qy - aimag(wdis * (z2-z1) / L)

        !print *,'angz ',angz
        qxqy(0:order) = qx * cos(angz) - qy * sin(angz) + real(wdis)  ! wdis already includes the correct rotation
        qxqy(order+1:2*order+1) = qx * sin(angz) + qy * cos(angz) - aimag(wdis)

        return
    end function besselld_int_ho_qxqy

    function besselld_gauss_ho_qxqy(x,y,z1,z2,lab,order) result(qxqy)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y
        complex(kind=8), intent(in) :: z1,z2
        complex(kind=8), intent(in) :: lab
        complex(kind=8), dimension(0:2*order+1) :: qxqy
        integer :: n, p
        real(kind=8) :: L, bigy, angz
        complex(kind=8) :: bigz, biglab
        real(kind=8), dimension(8) :: r, xmind
        complex(kind=8), dimension(8) :: k0,k1
        complex(kind=8), dimension(0:order) :: qx,qy


        L = abs(z2-z1)
        biglab = 2.d0 * lab / L
        bigz = (2.d0 * cmplx(x,y,kind=8) - (z1+z2) ) / (z2-z1)
        bigy = aimag(bigz)
        do n = 1,8
            xmind(n) = real(bigz) - xg(n)
            r(n) = sqrt( xmind(n)**2 + aimag(bigz)**2 )
            k0(n) = besselk0( xmind(n), aimag(bigz), biglab )
            k1(n) = besselk1( xmind(n), aimag(bigz), biglab )
        end do
        qx = dcmplx(0.d0,0.d0)
        qy = dcmplx(0.d0,0.d0)
        do p = 0,order
            do n = 1,8
                qx(p) = qx(p) + wg(n) * xg(n)**p * &
                        (-bigy) * xmind(n) / r(n)**3 * ( r(n)*k0(n)/biglab + 2.d0*k1(n))
                qy(p) = qy(p) + wg(n) * xg(n)**p * &
                        ( k1(n)/r(n) - bigy**2 / r(n)**3 * ( r(n)*k0(n)/biglab + 2.d0*k1(n)) )
            end do
        end do
        qx = -qx / (2*pi*biglab) * 2.d0/L
        qy = -qy / (2*pi*biglab) * 2.d0/L

        angz = atan2(aimag(z2-z1),real(z2-z1))
        qxqy(0:order) = qx * cos(angz) - qy * sin(angz)
        qxqy(order+1:2*order+1) = qx * sin(angz) + qy * cos(angz)
        return
    end function besselld_gauss_ho_qxqy

    function besselld_gauss_ho_qxqy_d1d2(x,y,z1,z2,lab,order,d1,d2) result(qxqy)
        ! Returns integral from d1 to d2 along real axis while strength is still Delta^order from -1 to +1
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,d1,d2
        complex(kind=8), intent(in) :: z1,z2,lab
        complex(kind=8), dimension(0:2*order+1) :: qxqy, qxqyc
        integer :: n, m
        real(kind=8) :: xp, yp, dc, fac
        complex(kind=8) :: z1p,z2p,bigz1,bigz2
        bigz1 = dcmplx(d1,0.d0)
        bigz2 = dcmplx(d2,0.d0)
        z1p = 0.5d0 * (z2-z1) * bigz1 + 0.5d0 * (z1+z2)
        z2p = 0.5d0 * (z2-z1) * bigz2 + 0.5d0 * (z1+z2)
        qxqyc = besselld_gauss_ho_qxqy(x,y,z1p,z2p,lab,order)
        dc = (d1+d2) / (d2-d1)
        qxqy = 0.d0
        do n = 0, order
            do m = 0, n
                qxqy(n) = qxqy(n) + gam(n,m) * dc**(n-m) * qxqyc(m)
                qxqy(n+order+1) = qxqy(n+order+1) + gam(n,m) * dc**(n-m) * qxqyc(m+order+1)
            enddo
            qxqy(n) = ( 0.5*(d2-d1) )**n * qxqy(n)
            qxqy(n+order+1) = ( 0.5*(d2-d1) )**n * qxqy(n+order+1)
        end do
    end function besselld_gauss_ho_qxqy_d1d2

    !function besselldqxqy(x,y,z1,z2,lab,order) result(qxqy)
    !    implicit none
    !    integer, intent(in) :: order
    !    real(kind=8), intent(in) :: x,y
    !    complex(kind=8), intent(in) :: z1,z2
    !    complex(kind=8), intent(in) :: lab
    !    complex(kind=8), dimension(0:2*order+1) :: qxqy
    !
    !    integer :: Nls, n
    !    real(kind=8) :: Lnear, L, d1, d2, delta
    !    complex(kind=8) :: z, delz, za, zb
    !
    !    Lnear = 3.d0
    !    z = cmplx(x,y,kind=8)
    !    qxqy = dcmplx(0.d0,0.d0)
    !    L = abs(z2-z1)
    !    !print *,'Lnear*abs(lab) ',Lnear*abs(lab)
    !    if ( L < Lnear*abs(lab) ) then  ! No need to break integral up
    !        if ( abs( z - 0.5d0*(z1+z2) ) < 0.5d0 * Lnear * L ) then  ! Do integration
    !            qxqy = besselld_int_ho_qxqy(x,y,z1,z2,lab,order,-1.d0,1.d0)
    !        else
    !            qxqy = besselld_gauss_ho_qxqy(x,y,z1,z2,lab,order)
    !        end if
    !    else  ! Break integral up in parts
    !        Nls = ceiling( L / (Lnear*abs(lab)) )
    !        !print *,'NLS ',Nls
    !        delta = 2.d0 / Nls
    !        delz = (z2-z1)/Nls
    !        L = abs(delz)
    !        do n = 1,Nls
    !            za = z1 + (n-1) * delz
    !            zb = z1 + n * delz
    !            d1 = -1.d0 + (n-1) * delta
    !            d2 = -1.d0 + n * delta
    !            if ( abs( z - 0.5d0*(za+zb) ) < 0.5d0 * Lnear * L ) then  ! Do integration
    !                qxqy = qxqy + besselld_int_ho_qxqy(x,y,z1,z2,lab,order,d1,d2)
    !            else
    !                qxqy = qxqy + besselld_gauss_ho_qxqy_d1d2(x,y,z1,z2,lab,order,d1,d2)
    !            end if
    !        end do
    !    end if
    !    return
    !end function besselldqxqy

    function besselldqxqy(x,y,z1,z2,lab,order,d1in,d2in) result(qxqy)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,d1in,d2in
        complex(kind=8), intent(in) :: z1,z2
        complex(kind=8), intent(in) :: lab
        complex(kind=8), dimension(0:2*order+1) :: qxqy

        integer :: Nls, n
        real(kind=8) :: Lnear, L, d1, d2, delta
        complex(kind=8) :: z, delz, za, zb

        Lnear = 3.d0
        z = cmplx(x,y,kind=8)
        qxqy = dcmplx(0.d0,0.d0)
        L = abs(z2-z1)
        !print *,'Lnear*abs(lab) ',Lnear*abs(lab)
        if ( L < Lnear*abs(lab) ) then  ! No need to break integral up
            if ( abs( z - 0.5d0*(z1+z2) ) < 0.5d0 * Lnear * L ) then  ! Do integration
                qxqy = besselld_int_ho_qxqy(x,y,z1,z2,lab,order,d1in,d2in)
            else
                qxqy = besselld_gauss_ho_qxqy_d1d2(x,y,z1,z2,lab,order,d1in,d2in)
            end if
        else  ! Break integral up in parts
            Nls = ceiling( L / (Lnear*abs(lab)) )
            !print *,'NLS ',Nls
            delta = 2.d0 / Nls
            delz = (z2-z1)/Nls
            L = abs(delz)
            do n = 1,Nls
                d1 = -1.d0 + (n-1) * delta
                d2 = -1.d0 + n * delta
                if ((d2 < d1in) .or. (d1 > d2in)) then
                    cycle
                end if
                d1 = max(d1,d1in)
                d2 = min(d2,d2in)
                za = z1 + (n-1) * delz
                zb = z1 + n * delz
                if ( abs( z - 0.5d0*(za+zb) ) < 0.5d0 * Lnear * L ) then  ! Do integration
                    qxqy = qxqy + besselld_int_ho_qxqy(x,y,z1,z2,lab,order,d1,d2)
                else
                    qxqy = qxqy + besselld_gauss_ho_qxqy_d1d2(x,y,z1,z2,lab,order,d1,d2)
                end if
            end do
        end if
        return
    end function besselldqxqy

    !function besselldqxqyv(x,y,z1,z2,lab,nlab,order) result(qxqy)
    !    implicit none
    !    integer, intent(in) :: order
    !    real(kind=8), intent(in) :: x,y
    !    complex(kind=8), intent(in) :: z1,z2
    !    integer, intent(in) :: nlab
    !    complex(kind=8), dimension(nlab), intent(in) :: lab
    !    complex(kind=8), dimension(2*nlab*(order+1)) :: qxqy
    !    complex(kind=8), dimension(0:2*order+1) :: qxqylab
    !    integer :: n, nterms, nhalf
    !    nterms = order+1
    !    nhalf = nlab*(order+1)
    !    do n = 1,nlab
    !        qxqylab = besselldqxqy(x,y,z1,z2,lab(n),order)
    !        qxqy((n-1)*nterms+1:n*nterms) = qxqylab(0:order)
    !        qxqy((n-1)*nterms+1+nhalf:n*nterms+nhalf) = qxqylab(order+1:2*order+1)
    !    end do
    !end function besselldqxqyv

    function besselldqxqyv(x,y,z1,z2,lab,order,R,nlab) result(qxqy)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,R
        complex(kind=8), intent(in) :: z1,z2
        integer, intent(in) :: nlab
        real(kind=8) :: d1, d2
        complex(kind=8), dimension(nlab), intent(in) :: lab
        complex(kind=8), dimension(2*nlab*(order+1)) :: qxqy
        complex(kind=8), dimension(0:2*order+1) :: qxqylab
        integer :: n, nterms, nhalf
        nterms = order+1
        nhalf = nlab*(order+1)
        call find_d1d2( z1, z2, dcmplx(x,y), R*abs(lab(1)), d1, d2 )
        do n = 1,nlab
            qxqylab = besselldqxqy(x,y,z1,z2,lab(n),order,d1,d2)
            qxqy((n-1)*nterms+1:n*nterms) = qxqylab(0:order)
            qxqy((n-1)*nterms+1+nhalf:n*nterms+nhalf) = qxqylab(order+1:2*order+1)
        end do
    end function besselldqxqyv

    function besselldqxqyv2(x,y,z1,z2,lab,order,R,nlab) result(qxqy)
        implicit none
        integer, intent(in) :: order
        real(kind=8), intent(in) :: x,y,R
        complex(kind=8), intent(in) :: z1,z2
        integer, intent(in) :: nlab
        real(kind=8) :: d1, d2
        complex(kind=8), dimension(nlab), intent(in) :: lab
        complex(kind=8), dimension(2*(order+1),nlab) :: qxqy
        complex(kind=8), dimension(0:2*order+1) :: qxqylab
        integer :: n, nterms, nhalf
        nterms = order+1
        nhalf = nlab*(order+1)
        call find_d1d2( z1, z2, dcmplx(x,y), R*abs(lab(1)), d1, d2 )
        do n = 1,nlab
            qxqylab = besselldqxqy(x,y,z1,z2,lab(n),order,d1,d2)
            qxqy(1:nterms,n) = qxqylab(0:order)
            qxqy(nterms+1:2*nterms,n) = qxqylab(order+1:2*order+1)
        end do
    end function besselldqxqyv2

    function bessells_circcheck(x,y,z1in,z2in,lab) result(omega)
        implicit none
        real(kind=8), intent(in) :: x,y
        complex(kind=8), intent(in) :: z1in,z2in
        complex(kind=8), intent(in) :: lab
        complex(kind=8) :: omega

        integer :: Npt, Nls, n
        real(kind=8) :: Lnear, Lzero, L, x1, y1, x2, y2
        complex(kind=8) :: z, z1, z2, delz, za, zb

        Lnear = 3.d0
        Lzero = 20.d0
        z = cmplx(x,y,kind=8)
        call circle_line_intersection( z1in, z2in, z, Lzero*abs(lab), x1, y1, x2, y2, Npt )

        z1 = cmplx(x1,y1,kind=8); z2 = cmplx(x2,y2,kind=8) ! f2py has problems with subroutines returning complex variables

        omega = cmplx(0.d0,0.d0,kind=8)

        if (Npt==2) then

            L = abs(z2-z1)
            if ( L < Lnear*abs(lab) ) then  ! No need to break integral up
                if ( abs( z - 0.5d0*(z1+z2) ) < 0.5d0 * Lnear * L ) then  ! Do integration
                    omega = bessells_int(x,y,z1,z2,lab)
                else
                    omega = bessells_gauss(x,y,z1,z2,lab)
                end if
            else  ! Break integral up in parts
                Nls = ceiling( L / (Lnear*abs(lab)) )
                delz = (z2-z1)/Nls
                L = abs(delz)
                do n = 1,Nls
                    za = z1 + (n-1) * delz
                    zb = z1 + n * delz
                    if ( abs( z - 0.5d0*(za+zb) ) < 0.5d0 * Lnear * L ) then  ! Do integration
                        omega = omega + bessells_int(x,y,za,zb,lab)
                    else
                        omega = omega + bessells_gauss(x,y,za,zb,lab)
                    end if
                end do
            end if
        end if
        return
    end function bessells_circcheck

    subroutine circle_line_intersection( z1, z2, zc, R, xouta, youta, xoutb, youtb, N )
        implicit none
        complex(kind=8), intent(in) :: z1, z2, zc
        real(kind=8), intent(in) :: R
        real(kind=8), intent(inout) :: xouta, youta, xoutb, youtb
        integer, intent(inout) :: N
        real(kind=8) :: Lover2, d, xa, xb
        complex(kind=8) :: bigz, za, zb

        N = 0
        za = cmplx(0.d0,0.d0,kind=8)
        zb = cmplx(0.d0,0.d0,kind=8)

        Lover2 = abs(z2-z1) / 2.d0
        bigz = (2*zc - (z1+z2)) * Lover2 / (z2-z1)

        if (abs(aimag(bigz)) < R) then
            d = sqrt( R**2 - aimag(bigz)**2 )
            xa = real(bigz) - d
            xb = real(bigz) + d
            if (( xa < Lover2 ) .and. ( xb > -Lover2 )) then
                N = 2
                if (xa < -Lover2) then
                    za = z1
                else
                    za = ( xa * (z2-z1) / Lover2 + (z1+z2) ) / 2.d0
                end if
                if (xb > Lover2) then
                    zb = z2
                else
                    zb = ( xb * (z2-z1) / Lover2 + (z1+z2) ) / 2.d0
                end if
            end if
        end if

        xouta = real(za); youta = aimag(za)
        xoutb = real(zb); youtb = aimag(zb)

        return
    end subroutine circle_line_intersection
    
    function circle_line_intersection_func(z1, z2, zc, R) result(xayaxbybn)
        implicit none
        complex(kind=8), intent(in) :: z1, z2, zc
        real(kind=8), intent(in) :: R
        real(kind=8), dimension(5) :: xayaxbybn
        real(kind=8) :: xouta, youta, xoutb, youtb
        integer :: N
        call circle_line_intersection(z1, z2, zc, R, xouta, youta, xoutb, youtb, N)
        xayaxbybn(1) = xouta
        xayaxbybn(2) = youta
        xayaxbybn(3) = xoutb
        xayaxbybn(4) = youtb
        xayaxbybn(5) = N
        return
    end function circle_line_intersection_func
        
    subroutine find_d1d2( z1, z2, zc, R, d1, d2 )
        implicit none
        complex(kind=8), intent(in) :: z1, z2, zc
        real(kind=8), intent(in) :: R
        real(kind=8), intent(inout) :: d1, d2
        real(kind=8) :: Lover2, d, xa, xb
        complex(kind=8) :: bigz

        d1 = -1.d0
        d2 = 1.d0

        Lover2 = abs(z2-z1) / 2.d0
        bigz = (2*zc - (z1+z2)) * Lover2 / (z2-z1)

        if (abs(aimag(bigz)) < R) then
            d = sqrt( R**2 - aimag(bigz)**2 )
            xa = real(bigz) - d
            xb = real(bigz) + d
            if (( xa < Lover2 ) .and. ( xb > -Lover2 )) then
                if (xa < -Lover2) then
                    d1 = -1.d0
                else
                    d1 = xa / Lover2
                end if
                if (xb > Lover2) then
                    d2 = 1.d0
                else
                    d2 = xb / Lover2
                end if
            end if
        end if

        return
    end subroutine find_d1d2
    
    function find_d1d2_func(z1, z2, zc, R) result(d1d2)
        implicit none
        complex(kind=8), intent(in) :: z1, z2, zc
        real(kind=8), intent(in) :: R
        real(kind=8), dimension(2) :: d1d2
        real(kind=8) :: d1, d2
        call find_d1d2(z1, z2, zc, R, d1, d2)
        d1d2(1) = d1
        d1d2(2) = d2
        return
    end function find_d1d2_func

    function isinside( z1, z2, zc, R ) result(irv)
        ! Checks whether point zc is within oval with 'radius' R from line element
        implicit none
        complex(kind=8), intent(in) :: z1, z2, zc
        real(kind=8), intent(in) :: R
        integer :: irv
        real(kind=8) :: Lover2, d, xa, xb
        complex(kind=8) :: bigz

        irv = 0
        Lover2 = abs(z2-z1) / 2.d0
        bigz = (2*zc - (z1+z2)) * abs(z2-z1) / (2.d0*(z2-z1))

        if (abs(aimag(bigz)) < R) then
            d = sqrt( R**2 - aimag(bigz)**2 )
            xa = real(bigz) - d
            xb = real(bigz) + d
            if (( xa < Lover2 ) .and. ( xb > -Lover2 )) then
                irv = 1
            endif
        endif

        return
    end function isinside

end module bessel

!!! Compile with gfortran -fbounds-check bessel.f95

program besseltest
    use bessel

    real(kind=8), dimension(0:1) :: omega
    real(kind=8), dimension(0:0) :: omegac
    complex(kind=8), dimension(0:1) :: om0, om1, om2, om3, om4, qxnum, qynum, wdis
    complex(kind=8), dimension(2) :: omv
    complex(kind=8), dimension(2) :: labv
    complex(kind=8), dimension(0:1) :: qxqy, qxqy2
    complex(kind=8), dimension(0:11) :: qxqyv
    complex(kind=8) :: lab, z, om, z1, z2, z3
    real(kind=8) :: d, L, x, y, R, xa, xb, ya, yb, d1, d2
    integer :: order, N, irv
    call initialize()
    z1 = dcmplx(-20.d0,0.d0)
    z2 = dcmplx(1.d0,0.d0)
    x = 2.d0
    y = 3.d0
    R = 5.d0
    labv(1) = dcmplx(1,2)
    labv(2) = dcmplx(3,4)
    om0 = bessellsv(x,y,z1,z2,labv,0,R,2)
    call circle_line_intersection( z1, z2, dcmplx(x,y), R*abs(labv(1)), xa, ya, xb, yb, N )
    print *,'za,zb ',dcmplx(xa,ya),dcmplx(xb,yb)
    !call bessellsuniv(x,y,dcmplx(xa,ya),dcmplx(xb,yb),labv(1),2,om1)
    !print *,'om0 ',om0
    !print *,'om1 ',om1
    !d = 1.d-3
    !lab = dcmplx(1.d0,2.d0)
    !labv(1) = lab
    !labv(2) = 2*lab
    !z1 = dcmplx(-2.d0,2.d0)
    !z2 = dcmplx(4.d0,0.d0)
    !x=4.d0
    !y = 5.d0
    !order = 2
    !om1 = bessellsv(x-d,y,z1,z2,labv,2,order)
    !om2 = bessellsv(x+d,y,z1,z2,labv,2,order)
    !om3 = bessellsv(x,y-d,z1,z2,labv,2,order)
    !om4 = bessellsv(x,y+d,z1,z2,labv,2,order)
    !qxqyv = bessellsqxqyv(x,y,z1,z2,labv,2,order)
    !!om1 = bessells_int_ho(x-d,y,z1,z2,lab,order,-1.d0,1.d0)
    !!om2 = bessells_int_ho(x+d,y,z1,z2,lab,order,-1.d0,1.d0)
    !!om3 = bessells_int_ho(x,y-d,z1,z2,lab,order,-1.d0,1.d0)
    !!om4 = bessells_int_ho(x,y+d,z1,z2,lab,order,-1.d0,1.d0)
    !!qxqy = bessells_int_ho_qxqy(x,y,z1,z2,lab,order,-1.d0,1.d0)
    !qxnum = (om1-om2)/(2*d)
    !qynum = (om3-om4)/(2*d)
    !print *,'qx        ',qxqyv(0:2*order+1)
    !!print *,'qx        ',qxqy(0:order)
    !print *,'numderx   ',qxnum
    !print *,'qy        ',qxqyv(2*order+2:4*order+3)
    !!print *,'qy        ',qxqy(order+1:2*order+1)
    !print *,'numdery   ',qynum
    !print *,'qxqyv ',qxqyv
    !d = 1.d-3
    !lab = dcmplx(4.d0,2.d0)
    !z1 = dcmplx(0.d0,0.d0)
    !z2 = dcmplx(2.d0,0.d0)
    !x=4.d0
    !y = 4.d0
    !order = 0
    !om0 = bpot(x,y,z1,z2,lab,order,-1.d0,1.d0)
    !om1 = bpot(x+d,y,z1,z2,lab,order,-1.d0,1.d0)
    !om2 = bpot(x-d,y,z1,z2,lab,order,-1.d0,1.d0)
    !om3 = bpot(x,y+d,z1,z2,lab,order,-1.d0,1.d0)
    !om4 = bpot(x,y-d,z1,z2,lab,order,-1.d0,1.d0)
    !!om0 = bpart(x,y,z1,z2,lab,order,-1.d0,1.d0)
    !!om1 = bpart(x+d,y,z1,z2,lab,order,-1.d0,1.d0)
    !!om2 = bpart(x-d,y,z1,z2,lab,order,-1.d0,1.d0)
    !!om3 = bpart(x,y+d,z1,z2,lab,order,-1.d0,1.d0)
    !!om4 = bpart(x,y-d,z1,z2,lab,order,-1.d0,1.d0)
    !qxqy = bqxqy(x,y,z1,z2,lab,order,-1.d0,1.d0)
    !print *,'qx        ',qxqy(0:order)
    !print *,'numderx   ',(om2-om1)/(2.d0*d)
    !print *,'qy        ',qxqy(order+1:2*order+1)
    !print *,'numdery   ',(om4-om3)/(2.d0*d)
    !d = 1.d-3
    !lab = dcmplx(0.5d0,0.5d0)
    !z1 = dcmplx(-2.d0,-1.d0)
    !z2 = dcmplx(2.d0,1.d0)
    !z3 = dcmplx(0.d0,0.d0)
    !order = 2
    !om0 = lapld_int_ho_wdis_d1d2(1.d0,2.d0,z1,z2,order,-1.d0,1.d0)
    !om1 = lapld_int_ho_wdis_d1d2(1.d0,2.d0,z1,z2,order,-1.d0,0.d0)
    !om2 = lapld_int_ho_wdis_d1d2(1.d0,2.d0,z1,z2,order,0.d0,1.d0)
    !print *,'om1       ',om1
    !print *,'om2       ',om2
    !print *,'om1+om2   ',om1+om2
    !print *,'om0       ',om0
    !
    !om1 = besselld_int_ho(6.d0,2.d0,z1,z2,lab,order,-1.d0,1.d0)
    !om2 = besselld_gauss_ho(6.d0,2.d0,z1,z2,lab,order)
    !om3 = besselld(6.d0,2.d0,z1,z2,lab,order)
    !print *,'om1       ',om1
    !print *,'om2       ',om2
    !print *,'om3       ',om3
    !om0 = bessells_int(1.d0,2.d0,dcmplx(-1,0),dcmplx(1,0),lab)
    !om1 = bessells_int(1.d0+d,2.d0,dcmplx(-1,0),dcmplx(0,0),lab)
    !om2 = bessells_int(1.d0-d,2.d0,dcmplx(0,0),dcmplx(1,0),lab)
    !print *,'om0       ',om0
    !print *,'om1+om2   ',om1+om2

    !omegac = lapls_int_ho(2.d0,1.d0,dcmplx(-1.d0,-1.d0),dcmplx(1.d0,0.d0),2)
    !print *,omegac
    !d = 1.d-3
    !lab = dcmplx(1.d0,3.d0)
    !z1 = dcmplx(-2.d0,-2.d0)
    !z2 = dcmplx(2.d0,0.d0)
    !order = 5
    !om0 = lapld_int_ho(1.d0,2.d0,z1,z2,order)
    !om1 = lapld_int_ho(1.d0+d,2.d0,z1,z2,order)
    !om2 = lapld_int_ho(1.d0-d,2.d0,z1,z2,order)
    !om3 = lapld_int_ho(1.d0,2.d0+d,z1,z2,order)
    !om4 = lapld_int_ho(1.d0,2.d0-d,z1,z2,order)
    !qxnum = (om2-om1)/(2.d0*d)
    !qynum = (om4-om3)/(2.d0*d)
    !wdis = lapld_int_ho_wdis(1.d0,2.d0,z1,z2,order)
    !print *,'qx        ',real(wdis)
    !print *,'numderx   ',real(qxnum)
    !print *,'qy        ',-aimag(wdis)
    !print *,'numdery   ',real(qynum)
    !d = 1.d-3
    !lab = dcmplx(2.d0,1.d0)*0.5d0
    !labv(1) = lab
    !labv(2) = 2*lab
    !z1 = dcmplx(-2.d0,-2.d0)
    !z2 = dcmplx(2.d0,0.d0)
    !x=4.d0
    !y = 5.d0
    !order = 2
    !!om1 = besselld_int_ho(x+d,y,z1,z2,lab,order,-1.d0,1.d0)
    !!om2 = besselld_int_ho(x-d,y,z1,z2,lab,order,-1.d0,1.d0)
    !!om3 = besselld_int_ho(x,y+d,z1,z2,lab,order,-1.d0,1.d0)
    !!om4 = besselld_int_ho(x,y-d,z1,z2,lab,order,-1.d0,1.d0)
    !!om1 = besselld_int_ho(x,y,z1,z2,lab,order,-1.d0,1.d0)
    !!om2 = besselld_gauss_ho(x,y,z1,z2,lab,order)
    !!om3 = besselld(x,y,z1,z2,lab,order)
    !!om4 = besselld(x,y,z1,z2,labv(2),order)
    !!omv = besselldv(x,y,z1,z2,labv,2,order)
    !qxqy = besselld_int_ho_qxqy(x,y,z1,z2,lab,order,-1.d0,1.d0)
    !qxqy2 = besselld_int_ho_qxqy(x,y,z1,z2,labv(2),order,-1.d0,1.d0)
    !!qxqy2 = besselld_gauss_ho_qxqy_d1d2(x,y,z1,z2,lab,order,-1.d0,1.d0)
    !qxqyv = besselldqxqyv(x,y,z1,z2,labv,2,order)
    !!print *,'qx        ',qxqy(0:order)
    !!print *,'numderx   ',(om2-om1)/(2.d0*d)
    !!print *,'qy        ',qxqy(order+1:2*order+1)
    !!print *,'numdery   ',(om4-om3)/(2.d0*d)
    !!print *,'om1   ',om1
    !!print *,'om2   ',om2
    !!print *,'om3   ',om3
    !!print *,'om4   ',om4
    !!print *,'omv   ',omv
    !print *,'qxqy  ',qxqy
    !print *,'qxqy2 ',qxqy2
    !print *,'qxqyv ',qxqyv(0:5)
    !print *,'qxqyv ',qxqyv(6:11)
    !!omegac = besselld_int_ho(8.d0,2.d0,dcmplx(-1.d0,1.d0),dcmplx(2.d0,0.d0),lab,0)
    !print *,'int2   ',omegac
    !omegac = besselld_gauss_ho(8.d0,2.d0,dcmplx(-1.d0,1.d0),dcmplx(2.d0,0.d0),lab,0)
    !print *,'gauss1 ',omegac
    !omegac = besselld_gauss_ho(8.d0,2.d0,dcmplx(-1.d0,1.d0),dcmplx(2.d0,0.d0),lab,0)
    !print *,'gauss2 ',omegac
    !lab = dcmplx(2.d0,1.d0)
    !om0 = besselld_int_ho(8.d0,2.d0,dcmplx(-1.d0,1.d0),dcmplx(2.d0,0.d0),lab,2)
    !om1 = besselld_int_ho(8.d0+d,2.d0,dcmplx(-1.d0,1.d0),dcmplx(2.d0,0.d0),lab,2)
    !om2 = besselld_int_ho(8.d0,2.d0+d,dcmplx(-1.d0,1.d0),dcmplx(2.d0,0.d0),lab,2)
    !om3 = besselld_int_ho(8.d0-d,2.d0,dcmplx(-1.d0,1.d0),dcmplx(2.d0,0.d0),lab,2)
    !om4 = besselld_int_ho(8.d0,2.d0-d,dcmplx(-1.d0,1.d0),dcmplx(2.d0,0.d0),lab,2)
    !test = (om1+om2+om3+om4-4.d0*om0) / d**2
    !print *,'rhs    ',om0/lab**2
    !print *,'rhsnum ',test
    !
    !lab = dcmplx(2.d0,1.d0)
    !d = 0.001
    !om1 = bessells_int_ho(8.d0-d,2.d0,dcmplx(-1.d0,0.d0),dcmplx(1.d0,0.d0),dcmplx(1.d0,1.d0),2)
    !om2 = bessells_int_ho(8.d0+d,2.d0,dcmplx(-1.d0,0.d0),dcmplx(1.d0,0.d0),dcmplx(1.d0,1.d0),2)
    !qxnum = (om1-om2)/(2*d)
    !om1 = bessells_int_ho(8.d0,2.d0-d,dcmplx(-1.d0,0.d0),dcmplx(1.d0,0.d0),dcmplx(1.d0,1.d0),2)
    !om2 = bessells_int_ho(8.d0,2.d0+d,dcmplx(-1.d0,0.d0),dcmplx(1.d0,0.d0),dcmplx(1.d0,1.d0),2)
    !qynum = (om1-om2)/(2*d)
    !qxqy = bessells_int_ho_qxqy(8.d0,2.d0,dcmplx(-1.d0,0.d0),dcmplx(1.d0,0.d0),dcmplx(1.d0,1.d0),2)
    !print *,'qxnum ',qxnum
    !print *,'qx    ',qxqy(0:2)
    !print *,'qynum ',qynum
    !print *,'qy    ',qxqy(3:5)


    !z = cmplx(2,3,8)
    !omega = besselk1near(z,20)
    !print *,omega
    !z = 5.d0 / cmplx(1,1,8)
    !om = besselk0near(z,10)
    !print *,'k0 near ',om, besselk0( 3.d0, 4.d0, cmplx(1,1,kind=8))
    !om = besselk0cheb(z,10)
    !print *,'k0 cheb ',om, besselk0( 3.d0, 4.d0, cmplx(1,1,kind=8))
    !z = 5.d0 / cmplx(1,1,8)
    !om = besselk1near(z,10)
    !print *,'k1 near ',om, besselk1( 3.d0, 4.d0, cmplx(1,1,kind=8))
    !om = besselk1cheb(z,10)
    !print *,'k1 cheb ',om, besselk1( 3.d0, 4.d0, cmplx(1,1,kind=8))
    !omega = besselcheb(z, 12)
    !print *,'omega cheb ',omega
    !omega = besselk0cheb(z, 12)
    !print *,'omega cheb ',omega
    !za = cmplx(0,0,8); zb = cmplx(0,0,8); N=0
    !call circle_line_intersection(cmplx(-1,-2,kind=8),cmplx(3,1,kind=8),cmplx(-1,0,kind=8),5.d0,za,zb,N)
    !if (N==2) then
    !    print *,'N equals 2!'
    !end if
    !print *,'za,zb,N ',za,zb,N
    !omega = bessells_gauss(3.d0,0.d0,cmplx(-2.d0,0.d0,8),cmplx(2.d0,0.d0,8),cmplx(0.5d0,0.5d0,8))
    !print *,omega
    !omegac = bessells_gauss_ho(3.d0,2.d0,cmplx(-1.d0,-1.d0,8),cmplx(2.d0,0.d0,8),cmplx(2.d0,1.d0,8),3)
    !print *,omegac
    !!omega = bessellsrealho(3.d0,2.d0,-1.d0,-1.d0,2.d0,0.d0,2.d0,3)
    !!print *,omega
    !omegac = 0.d0
    !omegac = bessells_int_ho(3.d0,2.d0,cmplx(-1.d0,-1.d0,8),cmplx(2.d0,0.d0,8),cmplx(2.d0,1.d0,8),3)
    !print *,omegac

    !omega = bessells(3.d0,0.d0,cmplx(-2.d0,0.d0,8),cmplx(2.d0,0.d0,8),cmplx(0.5d0,0.5d0,8))
    !print *,omega
    !omega1 = bessells_gauss(0.d0,3.d0,cmplx(-1.d0,0.d0,8),cmplx(1.d0,0.d0,8),cmplx(0.0d0,1.0d0,8))
    !print *,omega1
    !omega2 = bessells_int(0.d0,3.d0,cmplx(-1.d0,0.d0,8),cmplx(1.d0,0.d0,8),cmplx(0.0d0,1.0d0,8))
    !print *,omega2
    !omega = bessells(0.d0,3.d0,cmplx(-1.d0,0.d0,8),cmplx(1.d0,0.d0,8),cmplx(0.0d0,1.0d0,8))
    !print *,omega
    !lab = cmplx(0.18993748667372698d0,-0.13389092596486057d0, 8)
    !print *,'lab ',lab
    !omega1 = bessells_circcheck(0.d0,1.d0,cmplx(-10.d0,0.d0,8),cmplx(0.d0,0.d0,8),lab)
    !print *,'omega1 ',omega1
    !omega2 = bessells_circcheck(0.d0,1.d0,cmplx(0.d0,0.d0,8),cmplx(10.d0,0.d0,8),lab)
    !print *,'omega2 ',omega2
    !print *,'omega+ ',omega1+omega2
    !omega = bessells_circcheck(0.d0,1.d0,cmplx(-10.d0,0.d0,8),cmplx(10.d0,0.d0,8),lab)
    !print *,'omega  ',omega



    !lab2(1) = cmplx(0.5d0,0.5d0,8)
    !lab2(2) = cmplx(0.5d0,1.5d0,8)
    !lab2(3) = cmplx(0.5d0,2.5d0,8)
    !omega2 = testls(0.d0,3.d0,cmplx(-1.d0,0.d0,8),cmplx(1.d0,0.d0,8),lab2,3)
    !print *,omega2
    !call testls2(0.d0,3.d0,cmplx(-1.d0,0.d0,8),cmplx(1.d0,0.d0,8),lab2,3,omega2)
    !print *,omega2
    !call test3(3,omega2)
    !print *,omega2
end
