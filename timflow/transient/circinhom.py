"""Circular inhomogeneity elements for transient flow.

Defines circular inhomogeneities with different aquifer properties
for transient simulations.

Example::

    CircInhom(ml, xc=0, yc=0, R=100, kaq=[5], Haq=[10])
"""

# flake8: noqa
import numpy as np
from scipy.special import iv  # Needed for K1 in Well class, and in CircInhom
from scipy.special import kv

from timflow.transient.aquifer import AquiferData
from timflow.transient.element import Element
from timflow.transient.equation import InhomEquation


class CircInhomData(AquiferData):
    def __init__(
        self,
        model,
        x0=0,
        y0=0,
        R=1,
        kaq=[1],
        Haq=[1],
        c=[1],
        Saq=[0.1],
        Sll=[0.1],
        topboundary="imp",
    ):
        AquiferData.__init__(
            self, model, kaq, Haq, Hll, c, Saq, Sll, topboundary, phreatictop
        )
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.R = float(R)
        self.Rsq = self.R**2
        self.area = np.pi * self.Rsq
        self.model.addInhom(self)

    def is_inside(self, x, y):
        rv = False
        if (x - self.x0) ** 2 + (y - self.y0) ** 2 < self.Rsq:
            rv = True
        return rv


class CircInhomDataMaq(CircInhomData):
    def __init__(
        self,
        model,
        x0=0,
        y0=0,
        R=1,
        kaq=[1],
        z=[1, 0],
        c=[],
        Saq=[0.001],
        Sll=[0],
        topboundary="imp",
        phreatictop=False,
    ):
        kaq, Haq, Hll, c, Saq, Sll = param_maq(
            kaq, z, c, Saq, Sll, topboundary, phreatictop
        )
        CircInhomData.__init__(
            self, model, x0, y0, R, kaq, Haq, c, Saq, Sll, topboundary, phreatictop
        )


class CircInhomData3D(CircInhomData):
    def __init__(
        self,
        model,
        x0=0,
        y0=0,
        R=1,
        kaq=1,
        z=[4, 3, 2, 1],
        Saq=[0.3, 0.001, 0.001],
        kzoverkh=0.1,
        phreatictop=True,
        topboundary="conf",
        topres=0,
        topthick=0,
        topSll=0,
    ):
        kaq, Haq, Hll, c, Saq, Sll = param_3d(
            kaq, z, Saq, kzoverkh, phreatictop, topboundary, topres, topthick, topSll
        )
        CircInhomData.__init__(self, model, x0, y0, R, kaq, Haq, c, Saq, Sll, "imp")


class BesselRatioApprox:
    # Never fully debugged
    def __init__(self, Norder, Nterms):
        self.Norder = Norder + 1
        self.Nterms = Nterms + 1
        self.krange = np.arange(self.Nterms)
        self.minonek = (-np.ones(self.Nterms)) ** self.krange
        self.hankeltot = np.ones((self.Norder, 2 * self.Nterms), dtype=float)
        self.muk = np.ones((self.Norder, self.Nterms), dtype=float)
        self.nuk = np.ones((self.Norder, self.Nterms), dtype=float)
        for n in range(self.Norder):
            mu = 4.0 * n**2
            for k in range(1, self.Nterms):
                self.hankeltot[n, k] = (
                    self.hankeltot[n, k - 1] * (mu - (2 * k - 1) ** 2) / (4.0 * k)
                )
            for k in range(self.Nterms):
                self.muk[n, k] = (4.0 * n**2 + 16.0 * k**2 - 1.0) / (
                    4.0 * n**2 - (4.0 * k - 1.0) ** 2
                )
                self.nuk[n, k] = (4.0 * n**2 + 4.0 * (2.0 * k + 1.0) ** 2 - 1.0) / (
                    4.0 * n**2 - (4.0 * k + 1.0) ** 2
                )
        self.hankelnk = self.hankeltot[:, : self.Nterms]
        self.hankeln2k = self.hankeltot[:, ::2]
        self.hankeln2kp1 = self.hankeltot[:, 1::2]

    def ivratio(self, rho, R, lab):
        lab = np.atleast_1d(lab)
        rv = np.empty((self.Norder, len(lab)), dtype=complex)
        for k in range(len(lab)):
            top = np.sum(
                self.minonek * self.hankelnk / (2.0 * rho / lab[k]) ** self.krange, 1
            )
            bot = np.sum(
                self.minonek * self.hankelnk / (2.0 * R / lab[k]) ** self.krange, 1
            )
            rv[:, k] = top / bot * np.sqrt(float(R) / rho) * np.exp((rho - R) / lab[k])
        return rv

    def kvratio(self, rho, R, lab):
        lab = np.atleast_1d(lab)
        rv = np.empty((self.Norder, len(lab)), dtype=complex)
        for k in range(len(lab)):
            top = np.sum(self.hankelnk / (2.0 * rho / lab[k]) ** self.krange, 1)
            bot = np.sum(self.hankelnk / (2.0 * R / lab[k]) ** self.krange, 1)
            rv[:, k] = top / bot * np.sqrt(float(R) / rho) * np.exp((R - rho) / lab[k])
        return rv

    def ivratiop(self, rho, R, lab):
        lab = np.atleast_1d(lab)
        rv = np.empty((self.Norder, len(lab)), dtype=complex)
        for k in range(len(lab)):
            top = np.sum(
                self.muk * self.hankeln2k / (2.0 * rho / lab[k]) ** (2 * self.krange), 1
            ) - np.sum(
                self.nuk
                * self.hankeln2kp1
                / (2.0 * rho / lab[k]) ** (2 * self.krange + 1),
                1,
            )
            bot = np.sum(
                self.minonek * self.hankelnk / (2.0 * R / lab[k]) ** self.krange, 1
            )
            rv[:, k] = top / bot * np.sqrt(float(R) / rho) * np.exp((rho - R) / lab[k])
        return rv

    def kvratiop(self, rho, R, lab):
        lab = np.atleast_1d(lab)
        rv = np.empty((self.Norder, len(lab)), dtype=complex)
        for k in range(len(lab)):
            top = np.sum(
                self.muk * self.hankeln2k / (2.0 * rho / lab[k]) ** (2 * self.krange), 1
            ) + np.sum(
                self.nuk
                * self.hankeln2kp1
                / (2.0 * rho / lab[k]) ** (2 * self.krange + 1),
                1,
            )
            bot = np.sum(self.hankelnk / (2.0 * R / lab[k]) ** self.krange, 1)
            rv[:, k] = -top / bot * np.sqrt(float(R) / rho) * np.exp((R - rho) / lab[k])
        return rv


class CircInhomRadial(Element, InhomEquation):
    def __init__(self, model, x0=0, y0=0, R=1.0, label=None):
        Element.__init__(
            self,
            model,
            nparam=2 * model.aq.naq,
            nunknowns=2 * model.aq.naq,
            layers=range(model.aq.naq),
            type="z",
            name="CircInhom",
            label=label,
        )
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.R = float(R)
        self.model.addElement(self)
        self.approx = BesselRatioApprox(0, 2)

    def __repr__(self):
        return self.name + " at " + str((self.x0, self.y0))

    def initialize(self):
        self.xc = np.array([self.x0 + self.R])
        self.yc = np.array([self.y0])
        self.thetacp = np.zeros(1)
        self.ncp = 1
        self.aqin = self.model.aq.findAquiferData(self.x0 + (1 - 1e-8) * self.R, self.y0)
        assert self.aqin.R == self.R, (
            "Radius of CircInhom and CircInhomData must be equal"
        )
        self.aqout = self.model.aq.find_aquifer_data(
            self.x0 + (1 + 1e-8) * self.R, self.y0
        )
        self.setbc()
        self.facin = np.ones_like(self.aqin.lab2)
        self.facout = np.ones_like(self.aqout.lab2)
        # To keep track which circles are small
        self.circ_in_small = np.ones((self.aqin.naq, self.model.nin), dtype="i")
        self.circ_out_small = np.ones((self.aqout.naq, self.model.nin), dtype="i")
        self.Rbig = 700
        # for i in range(self.aqin.Naq):
        #    for j in range(self.model.Nin):
        #        assert self.R / abs(self.aqin.lab2[i,j,0]) < self.Rbig, 'Timflow input error, Radius too big'
        #        assert self.R / abs(self.aqout.lab2[i,j,0]) < self.Rbig, 'Timflow input error, Radius too big'
        #        if self.R / abs(self.aqin.lab2[i,j,0]) < self.Rbig:
        #            self.circ_in_small[i,j] = 1
        #            self.facin[i,j,:] = 1.0 / iv(0, self.R / self.aqin.lab2[i,j,:])
        #        if self.R / abs(self.aqout.lab2[i,j,0]) < self.Rbig:
        #            self.circ_out_small[i,j] = 1
        #            self.facout[i,j,:] = 1.0 / kv(0, self.R / self.aqout.lab2[i,j,:])
        # for i in range(self.aqin.Naq):
        #    for j in range(self.model.Nin):
        #        assert self.R / abs(self.aqin.lab2[i,j,0]) < 900, 'radius too large compared to aqin lab2[i,j,0] '+str((i,j))
        #        assert self.R / abs(self.aqout.lab2[i,j,0]) < 900, 'radius too large compared to aqin lab2[i,j,0] '+str((i,j))
        # self.facin = 1.0 / iv(0, self.R / self.aqin.lab2)
        # self.facout = 1.0 / kv(0, self.R / self.aqout.lab2)
        self.parameters = np.zeros(
            (self.model.Ngvbc, self.Nparam, self.model.Np), dtype=complex
        )

    def potinf(self, x, y, aq=None):
        """Can be called with only one x,y value."""
        if aq is None:
            aq = self.model.aq.findAquiferData(x, y)
        rv = np.zeros(
            (self.nparam, aq.naq, self.model.nin, self.model.npin), dtype=complex
        )
        if aq == self.aqin:
            r = np.sqrt((x - self.x0) ** 2 + (y - self.y0) ** 2)
            for i in range(self.aqin.Naq):
                for j in range(self.model.Nin):
                    if abs(r - self.R) / abs(self.aqin.lab2[i, j, 0]) < self.Rzero:
                        if self.circ_in_small[i, j]:
                            rv[i, i, j, :] = self.facin[i, j, :] * iv(
                                0, r / self.aqin.lab2[i, j, :]
                            )
                        else:
                            print("using approx")
                            rv[i, i, j, :] = self.approx.ivratio(
                                r, self.R, self.aqin.lab2[i, j, :]
                            )
        if aq == self.aqout:
            r = np.sqrt((x - self.x0) ** 2 + (y - self.y0) ** 2)
            for i in range(self.aqout.Naq):
                for j in range(self.model.Nin):
                    if abs(r - self.R) / abs(self.aqout.lab2[i, j, 0]) < self.Rzero:
                        if self.circ_out_small[i, j]:
                            rv[self.aqin.Naq + i, i, j, :] = self.facin[i, j, :] * kv(
                                0, r / self.aqout.lab2[i, j, :]
                            )
                        else:
                            print("using approx")
                            rv[self.aqin.Naq + i, i, j, :] = self.approx.kvratio(
                                r, self.R, self.aqout.lab2[i, j, :]
                            )
        rv.shape = (self.Nparam, aq.Naq, self.model.Np)
        return rv

    def disinf(self, x, y, aq=None):
        """Can be called with only one x,y value."""
        if aq is None:
            aq = self.model.aq.findAquiferData(x, y)
        qx = np.zeros((self.nparam, aq.naq, self.model.np), dtype=complex)
        qy = np.zeros((self.nparam, aq.naq, self.model.np), dtype=complex)
        if aq == self.aqin:
            qr = np.zeros(
                (self.nparam, aq.naq, self.model.nin, self.model.npin), dtype=complex
            )
            r = np.sqrt((x - self.x0) ** 2 + (y - self.y0) ** 2)
            if r < 1e-20:
                r = 1e-20  # As we divide by that on the return
            for i in range(self.aqin.Naq):
                for j in range(self.model.Nin):
                    if abs(r - self.R) / abs(self.aqin.lab2[i, j, 0]) < self.Rzero:
                        if self.circ_in_small[i, j]:
                            qr[i, i, j, :] = (
                                -self.facin[i, j, :]
                                * iv(1, r / self.aqin.lab2[i, j, :])
                                / self.aqin.lab2[i, j, :]
                            )
                        else:
                            qr[i, i, j, :] = (
                                -self.approx.ivratiop(r, self.R, self.aqin.lab2[i, j, :])
                                / self.aqin.lab2[i, j, :]
                            )
            qr.shape = (self.nparam, aq.naq, self.model.np)
            qx[:] = qr * (x - self.x0) / r
            qy[:] = qr * (y - self.y0) / r
        if aq == self.aqout:
            qr = np.zeros(
                (self.Nparam, aq.Naq, self.model.Nin, self.model.Npin), dtype=complex
            )
            r = np.sqrt((x - self.x0) ** 2 + (y - self.y0) ** 2)
            for i in range(self.aqout.Naq):
                for j in range(self.model.Nin):
                    if abs(r - self.R) / abs(self.aqout.lab2[i, j, 0]) < self.Rzero:
                        if self.circ_out_small[i, j]:
                            qr[self.aqin.Naq + i, i, j, :] = (
                                self.facin[i, j, :]
                                * kv(1, r / self.aqout.lab2[i, j, :])
                                / self.aqout.lab2[i, j, :]
                            )
                        else:
                            qr[self.aqin.Naq + i, i, j, :] = (
                                self.approx.kvratiop(r, self.R, self.aqout.lab2[i, j, :])
                                / self.aqout.lab2[i, j, :]
                            )
            qr.shape = (self.Nparam, aq.Naq, self.model.Np)
            qx[:] = qr * (x - self.x0) / r
            qy[:] = qr * (y - self.y0) / r
        return qx, qy

    def layout(self):
        alpha = np.linspace(0, 2 * np.pi, 100)
        return (
            "line",
            self.x0 + self.R * np.cos(alpha),
            self.y0 + self.R * np.sin(alpha),
        )


# class CircInhom(Element,InhomEquation):
#     def __init__(self,model,x0=0,y0=0,R=1.0,order=0,label=None,test=False):
#         Element.__init__(self, model, Nparam=2*model.aq.Naq*(2*order+1), Nunknowns=2*model.aq.Naq*(2*order+1), layers=range(model.aq.Naq), type='z', name='CircInhom', label=label)
#         self.x0 = float(x0); self.y0 = float(y0); self.R = float(R)
#         self.order = order
#         self.approx = BesselRatioApprox(0,3)
#         self.test=test
#         self.model.addElement(self)
#     def __repr__(self):
#         return self.name + ' at ' + str((self.x0,self.y0))
#     def initialize(self):
#         self.Ncp = 2*self.order + 1
#         self.thetacp = np.arange(0,2*np.pi,(2*np.pi)/self.Ncp)
#         self.xc = self.x0 + self.R * np.cos( self.thetacp )
#         self.yc = self.y0 + self.R * np.sin( self.thetacp )
#         self.aqin = self.model.aq.findAquiferData(self.x0 + (1-1e-10)*self.R,self.y0)
#         self.aqout = self.model.aq.findAquiferData(self.x0+(1.0+1e-8)*self.R,self.y0)
#         assert self.aqin.Naq == self.aqout.Naq, 'Timflow input error: Number of layers needs to be the same inside and outside circular inhomogeneity'
#         # Now that aqin is known, check that radii of circles are the same
#         assert self.aqin.R == self.R, 'Timflow Input Error: Radius of CircInhom and CircInhomData must be equal'
#         self.setbc()
#         self.facin = np.zeros((self.order+1,self.aqin.Naq,self.model.Nin,self.model.Npin),dtype='D')
#         self.facout = np.zeros((self.order+1,self.aqin.Naq,self.model.Nin,self.model.Npin),dtype='D')
#         self.circ_in_small = np.zeros((self.aqin.Naq,self.model.Nin),dtype='i') # To keep track which circles are small
#         self.circ_out_small = np.zeros((self.aqout.Naq,self.model.Nin),dtype='i')
#         self.besapprox = BesselRatioApprox(self.order,2) # Nterms = 2 is probably enough
#         self.Rbig = 200
#         for i in range(self.aqin.Naq):
#             for j in range(self.model.Nin):
#                 # When the circle is too big, an assertion is thrown. In the future, the approximation of the ratio of bessel functions needs to be completed
#                 # For now, the logic is there, but not used
#                 if self.test:
#                     print('inside  relative radius: ',self.R / abs(self.aqin.lab2[i,j,0]))
#                     print('outside relative radius: ',self.R / abs(self.aqout.lab2[i,j,0]))
#                 #assert self.R / abs(self.aqin.lab2[i,j,0]) < self.Rbig, 'Timflow input error, Radius too big'
#                 #assert self.R / abs(self.aqout.lab2[i,j,0]) < self.Rbig, 'Timflow input error, Radius too big'
#                 if self.R / abs(self.aqin.lab2[i,j,0]) < self.Rbig:
#                     self.circ_in_small[i,j] = 1
#                     for n in range(self.order+1):
#                         self.facin[n,i,j,:] = 1.0 / iv(n, self.R / self.aqin.lab2[i,j,:])
#                 if self.R / abs(self.aqout.lab2[i,j,0]) < self.Rbig:
#                     self.circ_out_small[i,j] = 1
#                     for n in range(self.order+1):
#                         self.facout[n,i,j,:] = 1.0 / kv(n, self.R / self.aqout.lab2[i,j,:])
#         self.parameters = np.zeros( (self.model.Ngvbc, self.Nparam, self.model.Np), 'D' )
#     def potinf(self,x,y,aq=None):
#         '''Can be called with only one x,y value'''
#         if aq is None: aq = self.model.aq.findAquiferData( x, y )
#         rv = np.zeros((2*aq.Naq,1+2*self.order,aq.Naq,self.model.Nin,self.model.Npin),'D')
#         if aq == self.aqin:
#             r = np.sqrt( (x-self.x0)**2 + (y-self.y0)**2 )
#             alpha = np.arctan2(y-self.y0, x-self.x0)
#             for i in range(self.aqin.Naq):
#                 for j in range(self.model.Nin):
#                     if abs(r-self.R) / abs(self.aqin.lab2[i,j,0]) < self.Rzero:
#                         if self.circ_in_small[i,j]:
#                             pot = np.zeros((self.model.Npin),'D')
#                             rv[i,0,i,j,:] = iv( 0, r / self.aqin.lab2[i,j,:] ) * self.facin[0,i,j,:]
#                             for n in range(1,self.order+1):
#                                 pot[:] = iv( n, r / self.aqin.lab2[i,j,:] ) * self.facin[n,i,j,:]
#                                 rv[i,2*n-1,i,j,:] = pot * np.cos(n*alpha)
#                                 rv[i,2*n  ,i,j,:] = pot * np.sin(n*alpha)
#                         else:
#                             pot = self.besapprox.ivratio(r,self.R,self.aqin.lab2[i,j,:])
#                             rv[i,0,i,j,:] = pot[0]
#                             for n in range(1,self.order+1):
#                                 rv[i,2*n-1,i,j,:] = pot[n] * np.cos(n*alpha)
#                                 rv[i,2*n  ,i,j,:] = pot[n] * np.sin(n*alpha)
#         if aq == self.aqout:
#             r = np.sqrt( (x-self.x0)**2 + (y-self.y0)**2 )
#             alpha = np.arctan2(y-self.y0, x-self.x0)
#             for i in range(self.aqout.Naq):
#                 for j in range(self.model.Nin):
#                     if abs(r-self.R) / abs(self.aqout.lab2[i,j,0]) < self.Rzero:
#                         if self.circ_out_small[i,j]:
#                             pot = np.zeros((self.model.Npin),'D')
#                             rv[aq.Naq+i,0,i,j,:] = kv( 0, r / self.aqout.lab2[i,j,:] ) * self.facout[0,i,j,:]
#                             for n in range(1,self.order+1):
#                                 pot[:] = kv( n, r / self.aqout.lab2[i,j,:] ) * self.facout[n,i,j,:]
#                                 rv[aq.Naq+i,2*n-1,i,j,:] = pot * np.cos(n*alpha)
#                                 rv[aq.Naq+i,2*n  ,i,j,:] = pot * np.sin(n*alpha)
#                         else:
#                             pot = self.besapprox.kvratio(r,self.R,self.aqout.lab2[i,j,:])
#                             rv[aq.Naq+i,0,i,j,:] = pot[0]
#                             for n in range(1,self.order+1):
#                                 rv[aq.Naq+i,2*n-1,i,j,:] = pot[n] * np.cos(n*alpha)
#                                 rv[aq.Naq+i,2*n  ,i,j,:] = pot[n] * np.sin(n*alpha)
#         rv.shape = (self.Nparam,aq.Naq,self.model.Np)
#         return rv
#     def disinf(self,x,y,aq=None):
#         '''Can be called with only one x,y value'''
#         if aq is None: aq = self.model.aq.findAquiferData( x, y )
#         qx = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
#         qy = np.zeros((self.Nparam,aq.Naq,self.model.Np),'D')
#         if aq == self.aqin:
#             r = np.sqrt( (x-self.x0)**2 + (y-self.y0)**2 )
#             alpha = np.arctan2(y-self.y0, x-self.x0)
#             qr = np.zeros((aq.Naq,1+2*self.order,aq.Naq,self.model.Nin,self.model.Npin),'D')
#             qt = np.zeros((aq.Naq,1+2*self.order,aq.Naq,self.model.Nin,self.model.Npin),'D')
#             if r < 1e-20: r = 1e-20  # As we divide by that on the return
#             for i in range(self.aqin.Naq):
#                 for j in range(self.model.Nin):
#                     if abs(r-self.R) / abs(self.aqin.lab2[i,j,0]) < self.Rzero:
#                         if self.circ_in_small[i,j]:
#                             pot = np.zeros((self.order+2,self.model.Npin),'D')
#                             for n in range(self.order+2):
#                                 pot[n] = iv( n, r / self.aqin.lab2[i,j,:] )
#                             qr[i,0,i,j,:] = -pot[1] / self.aqin.lab2[i,j,:] * self.facin[0,i,j,:]
#                             for n in range(1,self.order+1):
#                                 qr[i,2*n-1,i,j,:] = -(pot[n-1] + pot[n+1]) / 2 / self.aqin.lab2[i,j,:] * np.cos(n*alpha) * self.facin[n,i,j,:]
#                                 qr[i,2*n  ,i,j,:] = -(pot[n-1] + pot[n+1]) / 2 / self.aqin.lab2[i,j,:] * np.sin(n*alpha) * self.facin[n,i,j,:]
#                                 qt[i,2*n-1,i,j,:] =   pot[n] * np.sin(n*alpha) * n / r * self.facin[n,i,j,:]
#                                 qt[i,2*n  ,i,j,:] =  -pot[n] * np.cos(n*alpha) * n / r * self.facin[n,i,j,:]
#                         else:
#                             pot  = self.besapprox.ivratio(r,self.R,self.aqin.lab2[i,j,:])
#                             potp = self.besapprox.ivratiop(r,self.R,self.aqin.lab2[i,j,:])
#                             qr[i,0,i,j,:] = -potp[0] / self.aqin.lab2[i,j,:]
#                             for n in range(1,self.order+1):
#                                 qr[i,2*n-1,i,j,:] = -potp[n] / self.aqin.lab2[i,j,:] * np.cos(n*alpha)
#                                 qr[i,2*n  ,i,j,:] = -potp[n] / 2 / self.aqin.lab2[i,j,:] * np.sin(n*alpha)
#                                 qt[i,2*n-1,i,j,:] =  pot[n] * np.sin(n*alpha) * n / r
#                                 qt[i,2*n  ,i,j,:] = -pot[n] * np.cos(n*alpha) * n / r
#             qr.shape = (self.Nparam/2,aq.Naq,self.model.Np)
#             qt.shape = (self.Nparam/2,aq.Naq,self.model.Np)
#             qx[:self.Nparam/2,:,:] = qr * np.cos(alpha) - qt * np.sin(alpha);
#             qy[:self.Nparam/2,:,:] = qr * np.sin(alpha) + qt * np.cos(alpha);
#         if aq == self.aqout:
#             r = np.sqrt( (x-self.x0)**2 + (y-self.y0)**2 )
#             alpha = np.arctan2(y-self.y0, x-self.x0)
#             qr = np.zeros((aq.Naq,1+2*self.order,aq.Naq,self.model.Nin,self.model.Npin),'D')
#             qt = np.zeros((aq.Naq,1+2*self.order,aq.Naq,self.model.Nin,self.model.Npin),'D')
#             if r < 1e-20: r = 1e-20  # As we divide by that on the return
#             for i in range(self.aqout.Naq):
#                 for j in range(self.model.Nin):
#                     if abs(r-self.R) / abs(self.aqout.lab2[i,j,0]) < self.Rzero:
#                         if self.circ_out_small[i,j]:
#                             pot = np.zeros((self.order+2,self.model.Npin),'D')
#                             for n in range(self.order+2):
#                                 pot[n] = kv( n, r / self.aqout.lab2[i,j,:] )
#                             qr[i,0,i,j,:] = pot[1] / self.aqout.lab2[i,j,:] * self.facout[0,i,j,:]
#                             for n in range(1,self.order+1):
#                                 qr[i,2*n-1,i,j,:] = (pot[n-1] + pot[n+1]) / 2 / self.aqout.lab2[i,j,:] * np.cos(n*alpha) * self.facout[n,i,j,:]
#                                 qr[i,2*n  ,i,j,:] = (pot[n-1] + pot[n+1]) / 2 / self.aqout.lab2[i,j,:] * np.sin(n*alpha) * self.facout[n,i,j,:]
#                                 qt[i,2*n-1,i,j,:] =   pot[n] * np.sin(n*alpha) * n / r * self.facout[n,i,j,:]
#                                 qt[i,2*n  ,i,j,:] =  -pot[n] * np.cos(n*alpha) * n / r * self.facout[n,i,j,:]
#                         else:
#                             pot  = self.besapprox.kvratio(r,self.R,self.aqout.lab2[i,j,:])
#                             potp = self.besapprox.kvratiop(r,self.R,self.aqout.lab2[i,j,:])
#                             qr[i,0,i,j,:] = -potp[0] / self.aqout.lab2[i,j,:]
#                             for n in range(1,self.order+1):
#                                 qr[i,2*n-1,i,j,:] = -potp[n] / self.aqout.lab2[i,j,:] * np.cos(n*alpha)
#                                 qr[i,2*n  ,i,j,:] = -potp[n] / self.aqout.lab2[i,j,:] * np.sin(n*alpha)
#                                 qt[i,2*n-1,i,j,:] =  pot[n] * np.sin(n*alpha) * n / r
#                                 qt[i,2*n  ,i,j,:] = -pot[n] * np.cos(n*alpha) * n / r
#             qr.shape = (self.Nparam/2,aq.Naq,self.model.Np)
#             qt.shape = (self.Nparam/2,aq.Naq,self.model.Np)
#             qx[self.Nparam/2:,:,:] = qr * np.cos(alpha) - qt * np.sin(alpha);
#             qy[self.Nparam/2:,:,:] = qr * np.sin(alpha) + qt * np.cos(alpha);
#         return qx,qy
#     def layout(self):
#         return 'line', self.x0 + self.R * np.cos(np.linspace(0,2*np.pi,100)), self.y0 + self.R * np.sin(np.linspace(0,2*np.pi,100))

# def CircInhomMaq(model,x0=0,y0=0,R=1,order=1,kaq=[1],z=[1,0],c=[],Saq=[0.001],Sll=[0],topboundary='imp',phreatictop=False,label=None,test=False):
#     CircInhomDataMaq(model,x0,y0,R,kaq,z,c,Saq,Sll,topboundary,phreatictop)
#     return CircInhom(model,x0,y0,R,order,label,test)

# def CircInhom3D(model,x0=0,y0=0,R=1,order=1,kaq=[1,1,1],z=[4,3,2,1],Saq=[0.3,0.001,0.001],kzoverkh=[.1,.1,.1],phreatictop=True,label=None):
#     CircInhomData3D(model,x0,y0,R,kaq,z,Saq,kzoverkh,phreatictop)
#     return CircInhom(model,x0,y0,R,order,label)


#

# ml = timflow.transient.ModelMaq(kaq=[4,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=1,tmax=10,M=20)
##ls = MscreenLineSinkDitchString(ml,[(-1,0),(0,0),(1,0)],tsandQ=[(0.0,1.0)],layers=[2])
# e1a = EllipseInhomDataMaq(ml,0,0,along=2.0,bshort=1.0,angle=0.0,kaq=[10,2],z=[4,2,1,0],c=[200],Saq=[2e-3,2e-4],Sll=[1e-5])
# e1 = EllipseInhom(ml,0,0,along=2.0,bshort=1.0,angle=0.0,order=5)
# e1 = EllipseInhomMaq(ml,0,0,along=2.0,bshort=1.0,angle=0.0,order=5,kaq=[10,2],z=[4,2,1,0],c=[200],Saq=[2e-3,2e-4],Sll=[1e-5])
## Same inside and outside
# c1 = CircInhomMaq(ml,0,0,2.0,order=5,kaq=[4,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6])
# c1 = CircInhomMaq(ml,0,0,2.0,order=5,kaq=[10,.1],z=[4,2,1,0],c=[200],Saq=[2e-3,2e-4],Sll=[1e-5])
##c2 = CircInhomMaq(ml,0,0,5000.0,order=1,kaq=[10,2],z=[4,2,1,0],c=[200],Saq=[2e-3,2e-4],Sll=[1e-5])
##ml.initialize()
##c2.circ_in_small[:] = 0
##c2.circ_out_small[:] = 0
# w = DischargeWell(ml,xw=.5,yw=0,rw=.1,tsandQ=[0,5.0],layers=1)
# ml.solve()

# ml.solve()
# h1,h2 = np.zeros((2,e1.Ncp)), np.zeros((2,e1.Ncp))
# qn1,qn2 = np.zeros((2,e1.Ncp)), np.zeros((2,e1.Ncp))
# for i in range(e1.Ncp):
#    h1[:,i] = ml.head(e1.xc[i],e1.yc[i],2,aq=e1.aqin)[:,0]
#    h2[:,i] = ml.head(e1.xc[i],e1.yc[i],2,aq=e1.aqout)[:,0]
#    qx1,qy1 = ml.discharge(e1.xc[i],e1.yc[i],2,aq=e1.aqin)
#    qx2,qy2 = ml.discharge(e1.xc[i],e1.yc[i],2,aq=e1.aqout)
#    a = e1a.outwardnormalangle(e1.xc[i],e1.yc[i])
#    qn1[:,i] = qx1[:,0]*np.cos(a) + qy1[:,0]*np.sin(a)
#    qn2[:,i] = qx2[:,0]*np.cos(a) + qy2[:,0]*np.sin(a)


# ml = timflow.transient.ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=.1,tmax=10)
# w1 = timflow.transient.Well(ml,0,2,.1,tsandQ=[(0,10)],layers=[1])
# ls2 = ZeroHeadLineSinkString(ml,xy=[(-10,-2),(0,-4),(4,0)],layers=[1])
# ls1 = MscreenLineSinkDitchString(ml,xy=[(-10,0),(0,0),(10,10)],tsandQ=[(0.0,7.0)],res=0.0,wh='H',layers=[2],label=None)
# ml.solve()

# ml = timflow.transient.ModelMaq([1,20,2],[25,20,18,10,8,0],c=[1000,2000],Saq=[0.1,1e-4,1e-4],Sll=[0,0],phreatictop=True,tmin=1e-6,tmax=10,M=30)
# w1 = timflow.transient.Well(ml,0,0,.1,tsandQ=[(0,1000)],layers=[2])
# ls1 = ZeroMscreenLineSink(ml,10,-5,10,5,layers=[1,2,3],res=0.5,wh=1,vres=3,wv=1)
# w2 = ZeroMscreenWell(ml,10,0,res=1.0,layers=[1,2,3],vres=1.0)
# w3 = timflow.transient.Well(ml,0,-10,.1,tsandQ=[(0,700)],layers=[2])
# ml.solve()
##ml1 = timflow.transient.ModelMaq([1,20,2],[25,20,18,10,8,0],c=[1000,2000],Saq=[1e-4,1e-4,1e-4],Sll=[0,0],tmin=0.1,tmax=10000,M=30)
##w1 = timflow.transient.Well(ml1,0,0,.1,tsandQ=[(0,1000)],layers=[2],res=0.1)
##ml1.solve()
# t = np.logspace(-1,3,100)
# h0 = ml.head(50,0,t)
##h1 = ml1.head(50,0,t)
##w = MscreenWell(ml,0,0,.1,tsandQ=[(0,1000),(100,0),(365,1000),(465,0)],layers=[2,3])
##w2 = HeadWell(ml,50,0,.2,tsandh=[(0,1)],layers=[2])
##y = [-500,-300,-200,-100,-50,0,50,100,200,300,500]
##x = 50 * np.ones(len(y))
##ls = ZeroHeadLineSinkString(ml,xy=zip(x,y),layers=[1])
##w = timflow.transient.Well(ml,0,0,.1,tsandQ=[(0,1000),(100,0)],layers=[2])
##ml.solve()


# ml = timflow.transient.Model3D( kaq=[2,1,5,10,4], z=[10,8,6,4,2,0], Saq=[.1,.0001,.0002,.0002,.0001], phreatictop=True, kzoverkh=0.1, tmin=1e-3, tmax=1e3 )
# w = MscreenWell(ml,0,-25,rw=.3,tsandQ=[(0,100),(100,50)],layers=[2,3])
# ml.solve()

##ml = timflow.transient.Model3D(kaq=2.0,z=[10,5,0],Saq=[.002,.001],kzoverkh=0.2,phreatictop=False,tmin=.1,tmax=10,M=15)
# ml = timflow.transient.ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=100,tmax=300,M=50)
# w = HeadWellNew(ml,0,0,.1,tsandh=[(0.0,1.0)],layers=1)
# ml.solve()
##L1 = np.sqrt(10**2+5**2)
##ls1 = LineSink(ml,-10,-10,0,-5,tsandQ=[(0,.05*L1),(1,.02*L1)],res=1.0,layers=[1,2],label='mark1')
# w = MscreenWell(ml,-5,-5,.1,[0,5],layers=[1,2])
# L2 = np.sqrt(10**2+15**2)
# ls2 = LineSink(ml,0,-5,10,10,tsandQ=[(0,.03*L2),(2,.07*L2)],layers=[1],label='mark2')
##ls3a = ZeroHeadLineSink(ml,-10,5,-5,5,res=1.0,layers=[1,2])
##ls3b = ZeroHeadLineSink(ml,-5,5,0,5,res=1.0,layers=[1,2])
##ls3c = ZeroHeadLineSink(ml,0,5,5,5,res=1.0,layers=[1,2])
##lss = HeadLineSinkString(ml,[(-10,5),(-5,5),(0,5)],tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])
# lss = ZeroHeadLineSinkString(ml,[(-10,5),(-5,5),(0,5),(5,5)],res=1.0,layers=[1,2])
##lss = MscreenLineSinkString(ml,[(-10,5),(-5,5),(0,5)],tsandQ=[(0,0.2),(3,0.1)],res=1.0,layers=[1,2])
##lss = ZeroMscreenLineSinkString(ml,[(-10,5),(-5,5),(0,5)],res=1.0,layers=[1,2])
##ml.initialize()
# ml.solve()
# print ml.potential(50,50,[0.5,5])

# ml2 = timflow.transient.ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=.1,tmax=10,M=15)
# L1 = np.sqrt(10**2+5**2)
# ls1b = LineSink(ml2,-10,-10,0,-5,tsandQ=[(0,.05*L1),(1,.02*L1)],res=1.0,layers=[1,2],label='mark1')
# L2 = np.sqrt(10**2+15**2)
# ls2b = LineSink(ml2,0,-5,10,10,tsandQ=[(0,.03*L2),(2,.07*L2)],layers=[1],label='mark2')
##ls3a = HeadLineSink(ml2,-10,5,-5,5,tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])
##ls3b = HeadLineSink(ml2,-5,5,0,5,tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])
##ls3a = ZeroHeadLineSink(ml2,-10,5,-5,5,res=1.0,layers=[1,2])
##ls3b = ZeroHeadLineSink(ml2,-5,5,0,5,res=1.0,layers=[1,2])
##ls3a = MscreenLineSink(ml2,-10,5,-5,5,tsandQ=[(0,0.2),(3,0.1)],res=1.0,layers=[1,2])
##ls3b = MscreenLineSink(ml2,-5,5,0,5,tsandQ=[(0,0.2),(3,0.1)],res=1.0,layers=[1,2])
# ls3a = ZeroMscreenLineSink(ml2,-10,5,-5,5,res=1.0,layers=[1,2])
# ls3b = ZeroMscreenLineSink(ml2,-5,5,0,5,res=1.0,layers=[1,2])
##lssb = HeadLineSinkStringOld(ml2,[(-10,5),(-5,5),(0,5)],tsandh=[(0,0.02),(3,0.01)],res=0.0,layers=[1,2])
# ml2.solve()
# print ml2.potential(50,50,[0.5,5])

# lss = HeadLineSinkString(ml,[(-10,5),(-5,5),(0,5)],tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])
# lss = MscreenLineSinkString(ml,[(-10,5),(-5,5),(0,5)],tsandQ=[(0,.03*5),(2,.07*5)],res=0.5,layers=[1,2])
# ls3a = MscreenLineSink(ml,-10,5,-5,5,tsandQ=[(0,.03*5),(2,.07*5)],res=0.5,layers=[1,2])
# ls3b = MscreenLineSink(ml,-5,5,0,5,tsandQ=[(0,.03*5),(2,.07*5)],res=0.5,layers=[1,2])
#
# ml2 = timflow.transient.ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=.1,tmax=10,M=15)
# L1 = np.sqrt(10**2+5**2)
# ls1a = LineSink(ml2,-10,-10,0,-5,tsandQ=[(0,.05*L1),(1,.02*L1)],res=1.0,layers=[1,2],label='mark1')
# L2 = np.sqrt(10**2+15**2)
# ls2a = LineSink(ml2,0,-5,10,10,tsandQ=[(0,.03*L2),(2,.07*L2)],layers=[1],label='mark2')
# ls3a = HeadLineSink(ml2,-10,5,-5,5,tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])
# ls3b = HeadLineSink(ml2,-5,5,0,5,tsandh=[(0,0.02),(3,0.01)],res=1.0,layers=[1,2])


##lss = HeadLineSinkString(ml,[(-10,5),(-5,5),(0,5)],tsandh=[(0,0.02),(3,0.01)],res=0.0,layers=[1,2])
##ls3 = ZeroMscreenLineSink(ml,-10,5,0,5,res=1.0,layers=[1,2])
# ml = timflow.transient.ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=.1,tmax=10,M=15)
# w1 = timflow.transient.Well(ml,0,0,.1,tsandQ=[(0,5),(1,2)],res=1.0,layers=[1,2])
# w2 = timflow.transient.Well(ml,100,0,.1,tsandQ=[(0,3),(2,7)],layers=[1])
##w3 = MscreenWell(ml,0,100,.1,tsandQ=[(0,2),(3,1)],res=2.0,layers=[1,2])
# w3 = ZeroMscreenWell(ml,0,100,.1,res=2.0,layers=[1,2])
##w3 = ZeroHeadWell(ml,0,100,.1,res=1.0,layers=[1,2])
##w3 = HeadWell(ml,0,100,.1,tsandh=[(0,2),(3,1)],res=1.0,layers=[1,2])
# ml.solve()
###print ml.potential(2,3,[.5,5])
# print ml.potential(50,50,[0.5,5])
# ml2.solve()
# print ml2.potential(50,50,[.5,5])
# print lss.strength([.5,5])
#
# ml2 = timflow.transient.ModelMaq(kaq=[10,5],z=[4,2,1,0],c=[100],Saq=[1e-3,1e-4],Sll=[1e-6],tmin=0.1,tmax=10,M=15)
# ls1a = LineSink(ml2,-10,-10,0,-5,tsandsig=[(0,.05),(1,.02)],res=1.0,layers=[1,2],label='mark1')
# ls2a = LineSink(ml2,0,-5,10,10,tsandsig=[(0,.03),(2,.07)],layers=[1],label='mark2')
# ls3a = HeadLineSinkStringOld(ml2,[(-10,5),(-5,5),(0,5)],tsandh=[(0,0.02),(3,0.01)],res=0.0,layers=[1,2])
# ml2.solve()
# print ml2.potential(50,50,[0.5,5])

# print 'Q from strength:  ',w3.strength(.5)
# print 'Q from head diff: ',(ml.head(w3.xc,w3.yc,.5)-w3.headinside(.5))/w3.res*2*np.pi*w3.rw*ml.aq.Haq[:,np.newaxis]
# print 'Q from head diff: ',(ml.head(w3.xc,w3.yc,.5)-2.0)/w3.res*2*np.pi*w3.rw*ml.aq.Haq[:,np.newaxis]
# print w3.strength([.5,5])
# print ls3.strength([.5,5])
# print sum(ls3.strength([.5,5]),0)
# Q = w3.strength([.5,5])
# print sum(Q,0)
# print ml.potential(w3.xc,w3.yc,[.5,5])
