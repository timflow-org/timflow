"""Equation assembly helpers for transient flow.

Defines mixins to build linear systems of equations for transient problems.
"""

import numpy as np


class HeadEquation:
    def equation(self):
        """Matrix rows for head-specified conditions.

        Really written as constant potential element.
        Works for nunknowns = 1
        Returns matrix part nunknowns,neq,npval, complex.

        Returns rhs part nunknowns,nvbc,npval, complex
        Phi_out - c*T*q_s = Phi_in
        Well: q_s = Q / (2*pi*r_w*H)
        LineSink: q_s = sigma / H = Q / (L*H)
        """
        mat = np.empty((self.nunknowns, self.model.neq, self.model.npval), dtype=complex)
        # rhs needs be initialized zero
        rhs = np.zeros(
            (self.nunknowns, self.model.ngvbc, self.model.npval), dtype=complex
        )
        for icp in range(self.ncp):
            istart = icp * self.nlayers
            ieq = 0
            for e in self.model.elementlist:
                if e.nunknowns > 0:
                    mat[istart : istart + self.nlayers, ieq : ieq + e.nunknowns, :] = (
                        e.potinflayers(self.xc[icp], self.yc[icp], self.layers)
                    )
                    if e == self:
                        for i in range(self.nlayers):
                            mat[istart + i, ieq + istart + i, :] -= (
                                self.resfacp[istart + i]
                                * e.dischargeinflayers[istart + i]
                            )
                    ieq += e.nunknowns
            for i in range(self.model.ngbc):
                rhs[istart : istart + self.nlayers, i, :] -= self.model.gbclist[
                    i
                ].unitpotentiallayers(self.xc[icp], self.yc[icp], self.layers)
            if self.type == "v":
                iself = self.model.vbclist.index(self)
                for i in range(self.nlayers):
                    rhs[istart + i, self.model.ngbc + iself, :] = (
                        self.pc[istart + i] / self.model.p
                    )
        return mat, rhs


class WellBoreStorageEquation:
    def equation(self):
        """Matrix rows for multi-aquifer well element.

        Element with total given discharge, uniform but unknown head and
        InternalStorageEquation.
        """
        mat = np.zeros((self.nunknowns, self.model.neq, self.model.npval), dtype=complex)
        rhs = np.zeros(
            (self.nunknowns, self.model.ngvbc, self.model.npval), dtype=complex
        )
        ieq = 0
        for e in self.model.elementlist:
            if e.nunknowns > 0:
                head = (
                    e.potinflayers(self.xc[0], self.yc[0], self.layers)
                    / self.aq.T[self.layers][:, np.newaxis, np.newaxis]
                )
                mat[:-1, ieq : ieq + e.nunknowns, :] = head[:-1, :] - head[1:, :]
                mat[-1, ieq : ieq + e.nunknowns, :] -= (
                    np.pi * self.rc**2 * self.model.p * head[0, :]
                )
                if e == self:
                    disterm = self.dischargeinflayers * self.resfach[:, np.newaxis]
                    if self.nunknowns > 1:  # Multiple layers
                        for i in range(self.nunknowns - 1):
                            mat[i, ieq + i, :] -= disterm[i]
                            mat[i, ieq + i + 1, :] += disterm[i + 1]
                    mat[-1, ieq : ieq + self.nunknowns, :] += self.dischargeinflayers
                    mat[-1, ieq, :] += np.pi * self.rc**2 * self.model.p * disterm[0]
                ieq += e.nunknowns
        for i in range(self.model.ngbc):
            head = (
                self.model.gbclist[i].unitpotentiallayers(
                    self.xc[0], self.yc[0], self.layers
                )
                / self.aq.T[self.layers][:, np.newaxis]
            )
            rhs[:-1, i, :] -= head[:-1, :] - head[1:, :]
            rhs[-1, i, :] += np.pi * self.rc**2 * self.model.p * head[0, :]
        if self.type == "v":
            iself = self.model.vbclist.index(self)
            rhs[-1, self.model.ngbc + iself, :] += self.flowcoef
            if self.hdiff is not None:
                # head[0] - head[1] = hdiff
                rhs[:-1, self.model.ngbc + iself, :] += (
                    self.hdiff[:, np.newaxis] / self.model.p
                )
        return mat, rhs


class HeadEquationNores:
    def equation(self):
        """Mix-in class that returns matrix rows for head-specified conditions.

        (really written as constant potential element) Returns matrix part nunknowns,
        neq, npval, complex Returns rhs part nunknowns, nvbc, npval, complex
        """
        mat = np.empty((self.nunknowns, self.model.neq, self.model.npval), dtype=complex)
        rhs = np.zeros(
            (self.nunknowns, self.model.ngvbc, self.model.npval), dtype=complex
        )
        for icp in range(self.ncp):
            istart = icp * self.nlayers
            ieq = 0
            for e in self.model.elementlist:
                if e.nunknowns > 0:
                    mat[istart : istart + self.nlayers, ieq : ieq + e.nunknowns, :] = (
                        e.potinflayers(self.xc[icp], self.yc[icp], self.layers)
                    )
                    ieq += e.nunknowns
            for i in range(self.model.ngbc):
                rhs[istart : istart + self.nlayers, i, :] -= self.model.gbclist[
                    i
                ].unitpotentiallayers(self.xc[icp], self.yc[icp], self.layers)
            if self.type == "v":
                iself = self.model.vbclist.index(self)
                for i in range(self.nlayers):
                    rhs[istart + i, self.model.ngbc + iself, :] = (
                        self.pc[istart + i] / self.model.p
                    )
        return mat, rhs


class LeakyWallEquation:
    def equation(self):
        """Mix-in class that returns matrix rows for leaky-wall condition.

        Returns matrix part (nunknowns,neq,npval), complex
        Returns rhs part (nunknowns,nvbc,npval), complex.
        """
        mat = np.empty((self.nunknowns, self.model.neq, self.model.npval), dtype=complex)
        rhs = np.zeros(
            (self.nunknowns, self.model.ngvbc, self.model.npval), dtype=complex
        )
        for icp in range(self.ncp):
            istart = icp * self.nlayers
            ieq = 0
            for e in self.model.elementlist:
                if e.nunknowns > 0:
                    qx, qy = e.disvecinflayers(self.xc[icp], self.yc[icp], self.layers)
                    mat[istart : istart + self.nlayers, ieq : ieq + e.nunknowns, :] = (
                        qx * self.cosout[icp] + qy * self.sinout[icp]
                    )
                    if e == self:
                        hmin = (
                            e.potinflayers(self.xcneg[icp], self.ycneg[icp], self.layers)
                            / self.aq.T[self.layers][:, np.newaxis, np.newaxis]
                        )
                        hplus = (
                            e.potinflayers(self.xc[icp], self.yc[icp], self.layers)
                            / self.aq.T[self.layers][:, np.newaxis, np.newaxis]
                        )
                        mat[
                            istart : istart + self.nlayers, ieq : ieq + e.nunknowns, :
                        ] -= self.resfac[:, np.newaxis, np.newaxis] * (hplus - hmin)
                    ieq += e.nunknowns
            for i in range(self.model.ngbc):
                qx, qy = self.model.gbclist[i].unitdisveclayers(
                    self.xc[icp], self.yc[icp], self.layers
                )
                rhs[istart : istart + self.nlayers, i, :] -= (
                    qx * self.cosout[icp] + qy * self.sinout[icp]
                )
            # if self.type == 'v':
            #    iself = self.model.vbclist.index(self)
            #    for i in range(self.nlayers):
            #        rhs[istart+i,self.model.ngbc+iself,:] = \
            #             self.pc[istart+i] / self.model.p
        return mat, rhs


class MscreenEquation:
    def equation(self):
        """Matrix rows for multi-screen conditions where total discharge is specified.

        Mix-in class that returns matrix rows for multi-screen conditions where total
        discharge is specified. Works for nunknowns = 1 Returns matrix part nunknowns,
        neq, npval, complex.

        Returns rhs part nunknowns, nvbc, npval, complex
        head_out - c * q_s = h_in
        Set h_i - h_(i + 1) = 0 and Sum Q_i = Q
        """
        mat = np.zeros((self.nunknowns, self.model.neq, self.model.npval), dtype=complex)
        rhs = np.zeros(
            (self.nunknowns, self.model.ngvbc, self.model.npval), dtype=complex
        )
        ieq = 0
        for icp in range(self.ncp):
            istart = icp * self.nlayers
            ieq = 0
            for e in self.model.elementlist:
                if e.nunknowns > 0:
                    head = (
                        e.potinflayers(self.xc[icp], self.yc[icp], self.layers)
                        / self.aq.T[self.layers][:, np.newaxis, np.newaxis]
                    )
                    mat[
                        istart : istart + self.nlayers - 1, ieq : ieq + e.nunknowns, :
                    ] = head[:-1, :] - head[1:, :]
                    if e == self:
                        for i in range(self.nlayers - 1):
                            mat[istart + i, ieq + istart + i, :] -= (
                                self.resfach[istart + i]
                                * e.dischargeinflayers[istart + i]
                            )
                            mat[istart + i, ieq + istart + i + 1, :] += (
                                self.resfach[istart + i + 1]
                                * e.dischargeinflayers[istart + i + 1]
                            )
                            mat[istart + i, ieq + istart : ieq + istart + i + 1, :] -= (
                                self.vresfac[istart + i]
                                * e.dischargeinflayers[istart + i]
                            )
                        mat[
                            istart + self.nlayers - 1,
                            ieq + istart : ieq + istart + self.nlayers,
                            :,
                        ] = 1.0
                    ieq += e.nunknowns
            for i in range(self.model.ngbc):
                head = (
                    self.model.gbclist[i].unitpotentiallayers(
                        self.xc[icp], self.yc[icp], self.layers
                    )
                    / self.aq.T[self.layers][:, np.newaxis]
                )
                rhs[istart : istart + self.nlayers - 1, i, :] -= (
                    head[:-1, :] - head[1:, :]
                )
            if self.type == "v":
                iself = self.model.vbclist.index(self)
                rhs[istart + self.nlayers - 1, self.model.ngbc + iself, :] = 1.0
            # If self.type == 'z', it should sum to zero,
            # which is the default value of rhs
        return mat, rhs


class MscreenDitchEquation:
    def equation(self):
        """Matrix rows for multi-screen conditions where total discharge is specified.

        Returns matrix part nunknowns,neq,npval, complex.
        Returns rhs part nunknowns,nvbc,npval, complex
        head_out - c*q_s = h_in
        Set h_i - h_(i+1) = 0 and Sum Q_i = Q
        I would say
        headin_i - headin_(i+1) = 0
        headout_i - c*qs_i - headout_(i+1) + c*qs_(i+1) = 0
        In case of storage:
        Sum Q_i - A * p^2 * headin = Q
        """
        mat = np.zeros((self.nunknowns, self.model.neq, self.model.npval), dtype=complex)
        rhs = np.zeros(
            (self.nunknowns, self.model.ngvbc, self.model.npval), dtype=complex
        )
        ieq = 0
        for icp in range(self.ncp):
            istart = icp * self.nlayers
            ieq = 0
            for e in self.model.elementlist:
                if e.nunknowns > 0:
                    head = (
                        e.potinflayers(self.xc[icp], self.yc[icp], self.layers)
                        / self.aq.T[self.layers][:, np.newaxis, np.newaxis]
                    )
                    if self.nlayers > 1:
                        mat[
                            istart : istart + self.nlayers - 1,
                            ieq : ieq + e.nunknowns,
                            :,
                        ] = head[:-1, :] - head[1:, :]
                    # Store head in top layer in 2nd to last equation
                    # of this control point
                    mat[istart + self.nlayers - 1, ieq : ieq + e.nunknowns, :] = head[
                        0, :
                    ]
                    if e == self:
                        # Correct head in top layer in second to last equation
                        # to make it head inside
                        mat[istart + self.nlayers - 1, ieq + istart, :] -= (
                            self.resfach[istart] * e.dischargeinflayers[istart]
                        )
                        if icp == 0:
                            istartself = ieq  # Needed to build last equation
                        for i in range(self.nlayers - 1):
                            mat[istart + i, ieq + istart + i, :] -= (
                                self.resfach[istart + i]
                                * e.dischargeinflayers[istart + i]
                            )
                            mat[istart + i, ieq + istart + i + 1, :] += (
                                self.resfach[istart + i + 1]
                                * e.dischargeinflayers[istart + i + 1]
                            )
                            # vresfac not yet used here; it is set to zero as
                            # I don't quite now what is means yet
                            # mat[istart + i, ieq + istart:ieq+istart+i+1,:] -= \
                            # self.vresfac[istart + i] * \
                            # e.dischargeinflayers[istart + i]
                    ieq += e.nunknowns
            for i in range(self.model.ngbc):
                head = (
                    self.model.gbclist[i].unitpotentiallayers(
                        self.xc[icp], self.yc[icp], self.layers
                    )
                    / self.aq.T[self.layers][:, np.newaxis]
                )
                if self.nlayers > 1:
                    rhs[istart : istart + self.nlayers - 1, i, :] -= (
                        head[:-1, :] - head[1:, :]
                    )
                # Store minus the head in top layer in second to last equation
                # for this control point
                rhs[istart + self.nlayers - 1, i, :] -= head[0, :]
        # Modify last equations
        for icp in range(self.ncp - 1):
            ieq = (icp + 1) * self.nlayers - 1
            # Head first layer control point icp - Head first layer control
            # point icp + 1
            mat[ieq, :, :] -= mat[ieq + self.nlayers, :, :]
            rhs[ieq, :, :] -= rhs[ieq + self.nlayers, :, :]
        # Last equation setting the total discharge of the ditch
        mat[-1, :, :] = 0.0
        mat[-1, istartself : istartself + self.nparam, :] = 1.0
        if self.Astorage is not None:
            # Used to store last equation in case of ditch storage
            matlast = np.zeros((self.model.neq, self.model.npval), dtype=complex)
            rhslast = np.zeros((self.model.npval), dtype=complex)
            ieq = 0
            for e in self.model.elementlist:
                head = (
                    e.potinflayers(self.xc[0], self.yc[0], self.layers)
                    / self.aq.T[self.layers][:, np.newaxis, np.newaxis]
                )
                matlast[ieq : ieq + e.nunknowns] -= (
                    self.Astorage * self.model.p**2 * head[0, :]
                )
                if e == self:
                    # only need to correct first unknown
                    matlast[ieq] += (
                        self.Astorage
                        * self.model.p**2
                        * self.resfach[0]
                        * e.dischargeinflayers[0]
                    )
                ieq += e.nunknowns
            for i in range(self.model.ngbc):
                head = (
                    self.model.gbclist[i].unitpotentiallayers(
                        self.xc[0], self.yc[0], self.layers
                    )
                    / self.aq.T[self.layers][:, np.newaxis]
                )
                rhslast += self.Astorage * self.model.p**2 * head[0]
            mat[-1] += matlast
        rhs[-1, :, :] = 0.0
        if self.type == "v":
            iself = self.model.vbclist.index(self)
            rhs[-1, self.model.ngbc + iself, :] = 1.0
            # If self.type == 'z', it should sum to zero, which is the default
            # value of rhs
            if self.Astorage is not None:
                rhs[-1, self.model.ngbc + iself, :] += rhslast
        return mat, rhs


class InhomEquation:
    def equation(self):
        """Mix-in class that returns matrix rows for inhomogeneity conditions."""
        mat = np.zeros((self.nunknowns, self.model.neq, self.model.npval), dtype=complex)
        rhs = np.zeros(
            (self.nunknowns, self.model.ngvbc, self.model.npval), dtype=complex
        )
        for icp in range(self.ncp):
            istart = icp * 2 * self.nlayers
            ieq = 0
            for e in self.model.elementlist:
                if e.nunknowns > 0:
                    mat[istart : istart + self.nlayers, ieq : ieq + e.nunknowns, :] = (
                        e.potinflayers(self.xc[icp], self.yc[icp], self.layers, self.aqin)
                        / self.aqin.T[self.layers][:, np.newaxis, np.newaxis]
                        - e.potinflayers(
                            self.xc[icp], self.yc[icp], self.layers, self.aqout
                        )
                        / self.aqout.T[self.layers][:, np.newaxis, np.newaxis]
                    )
                    qxin, qyin = e.disinflayers(
                        self.xc[icp], self.yc[icp], self.layers, self.aqin
                    )
                    qxout, qyout = e.disinflayers(
                        self.xc[icp], self.yc[icp], self.layers, self.aqout
                    )
                    mat[
                        istart + self.nlayers : istart + 2 * self.nlayers,
                        ieq : ieq + e.nunknowns,
                        :,
                    ] = (qxin - qxout) * np.cos(self.thetacp[icp]) + (
                        qyin - qyout
                    ) * np.sin(self.thetacp[icp])
                    ieq += e.nunknowns
            for i in range(self.model.ngbc):
                rhs[istart : istart + self.nlayers, i, :] -= (
                    self.model.gbclist[i].unitpotentiallayers(
                        self.xc[icp], self.yc[icp], self.layers, self.aqin
                    )
                    / self.aqin.T[self.layers][:, np.newaxis]
                    - self.model.gbclist[i].unitpotentiallayers(
                        self.xc[icp], self.yc[icp], self.layers, self.aqout
                    )
                    / self.aqout.T[self.layers][:, np.newaxis]
                )
                qxin, qyin = self.model.gbclist[i].unitdischargelayers(
                    self.xc[icp], self.yc[icp], self.layers, self.aqin
                )
                qxout, qyout = self.model.gbclist[i].unitdischargelayers(
                    self.xc[icp], self.yc[icp], self.layers, self.aqout
                )
                rhs[istart + self.nlayers : istart + 2 * self.nlayers, i, :] -= (
                    qxin - qxout
                ) * np.cos(self.thetacp[icp]) + (qyin - qyout) * np.sin(self.thetacp[icp])
        return mat, rhs


class HeadDiffEquation:
    def equation(self):
        """Mix-in class that returns matrix rows for continuity of head."""
        mat = np.empty((self.nunknowns, self.model.neq, self.model.npval), dtype=complex)
        rhs = np.zeros(
            (self.nunknowns, self.model.ngvbc, self.model.npval), dtype=complex
        )
        for icp in range(self.ncp):
            istart = icp * self.nlayers
            ieq = 0
            for e in self.model.elementlist:
                if e.nunknowns > 0:
                    mat[istart : istart + self.nlayers, ieq : ieq + e.nunknowns, :] = (
                        e.potinflayers(
                            self.xcin[icp], self.ycin[icp], self.layers, self.aqin
                        )
                        / self.aqin.T[self.layers][:, np.newaxis, np.newaxis]
                        - e.potinflayers(
                            self.xcout[icp], self.ycout[icp], self.layers, self.aqout
                        )
                        / self.aqout.T[self.layers][:, np.newaxis, np.newaxis]
                    )
                    ieq += e.nunknowns
            for i in range(self.model.ngbc):
                rhs[istart : istart + self.nlayers, i, :] -= (
                    self.model.gbclist[i].unitpotentiallayers(
                        self.xcin[icp], self.ycin[icp], self.layers, self.aqin
                    )
                    / self.aqin.T[self.layers][:, np.newaxis]
                    - self.model.gbclist[i].unitpotentiallayers(
                        self.xcout[icp], self.ycout[icp], self.layers, self.aqout
                    )
                    / self.aqout.T[self.layers][:, np.newaxis]
                )
        return mat, rhs


class FluxDiffEquation:
    def equation(self):
        """Mix-in class that returns matrix rows for continuity of flow."""
        mat = np.zeros((self.nunknowns, self.model.neq, self.model.npval), dtype=complex)
        rhs = np.zeros(
            (self.nunknowns, self.model.ngvbc, self.model.npval), dtype=complex
        )
        for icp in range(self.ncp):
            istart = icp * self.nlayers
            ieq = 0
            for e in self.model.elementlist:
                if e.nunknowns > 0:
                    qxin, _ = e.disvecinflayers(
                        self.xcin[icp], self.ycin[icp], self.layers, self.aqin
                    )
                    qxout, _ = e.disvecinflayers(
                        self.xcout[icp], self.ycout[icp], self.layers, self.aqout
                    )
                    mat[
                        istart : istart + self.nlayers,
                        ieq : ieq + e.nunknowns,
                        :,
                    ] = qxin - qxout
                    ieq += e.nunknowns
            for i in range(self.model.ngbc):
                qxin, _ = self.model.gbclist[i].unitdisveclayers(
                    self.xc[icp], self.yc[icp], self.layers, self.aqin
                )
                qxout, _ = self.model.gbclist[i].unitdisveclayers(
                    self.xc[icp], self.yc[icp], self.layers, self.aqout
                )
                rhs[istart : istart + self.nlayers, i, :] -= qxin - qxout
        return mat, rhs
