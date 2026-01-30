"""Inverse Laplace transform implementations.

Provides multiple inverse Laplace transform algorithms including
Talbot and Stehfest methods for transient flow computations.

Copyright 2019 Kristopher L. Kuhlman <klkuhlm _at_ sandia _dot_ gov>
"""
# Copyright 2019 Kristopher L. Kuhlman <klkuhlm _at_ sandia _dot_ gov>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


class InverseLaplaceTransform(object):
    def __init__(self):
        self.talbot_cache = {}
        self.stehfest_cache = {}

    def clear(self):
        self.talbot_cache = {}
        self.stehfest_cache = {}

    def calc_laplace_parameter(self, t, **kwargs):
        raise NotImplementedError

    def calc_time_domain_solution(self, fp):
        raise NotImplementedError


class FixedTalbot(InverseLaplaceTransform):
    def calc_laplace_parameter(self, t, **kwargs):
        import numpy as np

        # required
        # ------------------------------
        # time of desired approximation
        self.t = t

        # optional
        # ------------------------------
        # maximum time desired (used for scaling) default is requested
        # time.
        self.tmax = kwargs.get("tmax", self.t)

        self.degree = int(kwargs.get("degree", 14))
        M = self.degree

        # Abate & Valko rule of thumb for r parameter
        if "r" in kwargs:
            self.r = kwargs["r"]
            r_default = False
        else:
            self.r = 2.0 / 5.0 * M
            r_default = True

        if r_default and self.degree in self.talbot_cache:
            self.theta, self.cot_theta, self.delta = self.talbot_cache[self.degree]
        else:
            self.theta = np.linspace(0.0, np.pi, M + 1, dtype=np.float64)
            self.cot_theta = np.empty((M,), dtype=np.float64)

            self.cot_theta[0] = 0.0
            self.cot_theta[1:] = 1.0 / np.tan(self.theta[1:-1])

            # all but time-dependent part of p
            self.delta = np.empty((M,), dtype=np.complex64)
            self.delta[0] = self.r
            self.delta[1:] = self.r * self.theta[1:-1] * (self.cot_theta[1:] + 1j)

            self.talbot_cache[self.degree] = self.theta, self.cot_theta, self.delta

        self.p = self.delta / self.tmax

        # NB: p is complex

    def calc_time_domain_solution(self, fp, t):
        import numpy as np

        # required
        # ------------------------------
        self.t = t

        # assume fp was computed from p matrix returned from
        # calc_laplace_parameter()

        # these were computed in previous call to
        # calc_laplace_parameter()
        theta = self.theta
        delta = self.delta
        M = self.degree
        # p = self.p
        # r = self.r

        ans = np.empty((M,), dtype=np.complex64)
        ans[0] = np.exp(delta[0]) * fp[0] / 2.0

        ans[1:] = np.exp(delta[1:]) * fp[1:]
        ans[1:] *= (
            1.0
            + 1j * theta[1:-1] * (1.0 + self.cot_theta[1:] ** 2)
            - 1j * self.cot_theta[1:]
        )

        result = 2.0 / 5.0 * np.sum(ans) / self.t

        # ignore any small imaginary part
        return result.real


# ****************************************


class Stehfest(InverseLaplaceTransform):
    def calc_laplace_parameter(self, t, **kwargs):
        import numpy as np

        # required
        # ------------------------------
        # time of desired approximation
        self.t = t

        # optional
        # ------------------------------

        self.degree = int(kwargs.get("degree", 16))

        self.ln2 = np.log(2.0)

        # _coeff routine requires even degree
        if self.degree % 2 > 0:
            self.degree += 1

        M = self.degree

        if self.degree in self.stehfest_cache:
            self.V = self.stehfest_cache[self.degree]
        else:
            self.V = kwargs.get("V", self._coeff())
            self.stehfest_cache[self.degree] = self.V

        self.p = np.arange(1, M + 1) * self.ln2 / self.t

        # NB: p is real

    def _coeff(self):
        """Compute the Salzer summation weights.

        Salzer summation weights (aka, "Stehfest coefficients") only depends on the
        approximation order (M) and the precision.
        """
        import numpy as np
        from scipy.misc import factorial

        M = self.degree
        M2 = int(M / 2.0)  # checked earlier that M is even

        V = np.empty((M,), dtype=np.float64)

        fac = lambda x: float(factorial(x, exact=True))  # noqa: E731

        # Salzer summation weights
        # get very large in magnitude and oscillate in sign,
        # if the precision is not high enough, there will be
        # catastrophic cancellation
        for k in range(1, M + 1):
            z = np.zeros((min(k, M2) + 1,), dtype=np.float64)
            for j in range(int((k + 1) / 2.0), min(k, M2) + 1):
                z[j] = (
                    j**M2
                    * fac(2 * j)
                    / (fac(M2 - j) * fac(j) * fac(j - 1) * fac(k - j) * fac(2 * j - k))
                )
            V[k - 1] = (-1) ** (k + M2) * np.sum(z)

        return V

    def calc_time_domain_solution(self, fp, t):
        import numpy as np

        # required
        self.t = t

        # assume fp was computed from p matrix returned from
        # calc_laplace_parameter()

        result = np.dot(self.V, fp) * self.ln2 / self.t

        # ignore any small imaginary part
        return result.real


# ****************************************


class deHoog(InverseLaplaceTransform):
    def calc_laplace_parameter(self, t, **kwargs):
        import numpy as np

        # NB: too many parameters, and simple p
        # nothing is cached here.

        self.t = t
        self.tmax = kwargs.get("tmax", self.t)

        self.degree = int(kwargs.get("degree", 17))

        # 2*M+1 terms in approximation
        # M = self.degree

        self.alpha = kwargs.get("alpha", 1.0e-16)

        # desired tolerance (here simply related to alpha)
        self.tol = kwargs.get("tol", self.alpha * 10.0)
        self.nump = 2 * self.degree + 1  # number of terms in approximation

        # scaling factor (likely tune-able, but 2 is typical)
        self.scale = kwargs.get("scale", 2.0)
        self.T = kwargs.get("T", self.scale * self.tmax)

        self.gamma = self.alpha - np.log(self.tol) / (self.scale * self.T)
        self.p = self.gamma + 1j * np.pi * np.arange(self.nump) / self.T

        # NB: p is complex (mpc)

    def calc_time_domain_solution(self, fp, t):
        import numpy as np

        M = self.degree
        NP = self.nump
        T = self.T

        self.t = t

        # would it be useful to try re-using
        # space between e&q and A&B?
        e = np.empty((NP, M + 1), dtype=np.complex64)
        q = np.empty((NP, M), dtype=np.complex64)
        d = np.empty((NP,), dtype=np.complex64)
        A = np.empty((NP + 2,), dtype=np.complex64)
        B = np.empty((NP + 2,), dtype=np.complex64)

        # initialize Q-D table
        e[0 : 2 * M, 0] = 0.0
        q[0, 0] = fp[1] / (fp[0] / 2.0)
        for i in range(1, 2 * M):
            q[i, 0] = fp[i + 1] / fp[i]

        # rhombus rule for filling triangular Q-D table (e & q)
        for r in range(1, M + 1):
            # start with e, column 1, 0:2*M-2
            mr = 2 * (M - r)
            e[0:mr, r] = q[1 : mr + 1, r - 1] - q[0:mr, r - 1] + e[1 : mr + 1, r - 1]
            if not r == M:
                rq = r + 1
                mr = 2 * (M - rq) + 1
                for i in range(mr):
                    q[i, rq - 1] = q[i + 1, rq - 2] * e[i + 1, rq - 1] / e[i, rq - 1]

        # build up continued fraction coefficients (d)
        d[0] = fp[0] / 2.0
        for r in range(1, M + 1):
            d[2 * r - 1] = -q[0, r - 1]  # even terms
            d[2 * r] = -e[0, r]  # odd terms

        # seed A and B for recurrence
        A[0] = 0.0
        A[1] = d[0]
        B[0:2] = 1.0

        # base of the power series
        z = np.exp(1j * np.pi * self.t / T)

        # coefficients of Pade approximation (A & B)
        # using recurrence for all but last term
        for i in range(1, 2 * M):
            A[i + 1] = A[i] + d[i] * A[i - 1] * z
            B[i + 1] = B[i] + d[i] * B[i - 1] * z

        # "improved remainder" to continued fraction
        brem = (1.0 + (d[2 * M - 1] - d[2 * M]) * z) / 2.0
        rem = -brem * (1.0 - np.sqrt(1.0 + d[2 * M] * z / brem**2))

        # last term of recurrence using new remainder
        A[NP] = A[2 * M] + rem * A[2 * M - 1]
        B[NP] = B[2 * M] + rem * B[2 * M - 1]

        # diagonal Pade approximation
        # F=A/B represents accelerated trapezoid rule
        result = np.exp(self.gamma * self.t) / T * (A[NP] / B[NP]).real

        return result


# ****************************************

# initialize classes
_fixed_talbot = FixedTalbot()
_stehfest = Stehfest()
_de_hoog = deHoog()


def invertlaplace(f, t, **kwargs):
    rule = kwargs.get("method", "dehoog")
    if isinstance(rule, str):
        lrule = rule.lower()
        if lrule == "talbot":
            rule = _fixed_talbot
        elif lrule == "stehfest":
            rule = _stehfest
        elif lrule == "dehoog":
            rule = _de_hoog
        else:
            raise ValueError("unknown invlap algorithm: %s" % rule)
    else:
        rule = rule()

    # determine the vector of Laplace-space parameter
    # needed for the requested method and desired time
    rule.calc_laplace_parameter(t, **kwargs)

    # compute the Laplace-space function evalutations
    # at the required abscissa.
    fp = [f(p) for p in rule.p]

    # compute the time-domain solution from the
    # Laplace-space function evaluations
    return rule.calc_time_domain_solution(fp, t)


# shortcuts for the above function for specific methods
def invlaptalbot(*args, **kwargs):
    kwargs["method"] = "talbot"
    return invertlaplace(*args, **kwargs)


def invlapstehfest(*args, **kwargs):
    kwargs["method"] = "stehfest"
    return invertlaplace(*args, **kwargs)


def invlapdehoog(*args, **kwargs):
    kwargs["method"] = "dehoog"
    return invertlaplace(*args, **kwargs)
