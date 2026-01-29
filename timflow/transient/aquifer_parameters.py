"""Parameter helpers for transient aquifer models.

Utilities to compute derived parameters for multi-layer and 3D aquifer setups
for transient flow simulations.
"""

import numpy as np


def param_maq(
    kaq=[1],
    z=[1, 0],
    c=[],
    Saq=[0.001],
    Sll=[0],
    poraq=[0.3],
    porll=[0.3],
    topboundary="conf",
    phreatictop=False,
):
    # Computes the parameters for a TimModel from input for a maq model
    z = np.atleast_1d(z).astype(float)
    kaq = np.atleast_1d(kaq).astype(float)
    Saq = np.atleast_1d(Saq).astype(float)
    poraq = np.atleast_1d(poraq).astype(float)
    c = np.atleast_1d(c).astype(float)
    Sll = np.atleast_1d(Sll).astype(float)
    porll = np.atleast_1d(porll).astype(float)
    H = z[:-1] - z[1:]
    assert np.all(H >= 0), (
        "Error: Not all layer thicknesses are" + " non-negative" + str(H)
    )
    if topboundary[:3] == "con":
        assert len(z) % 2 == 0, (
            "Error: Length of z must be 2 * number of aquifers "
            "when topboundary is confined in MaqModel"
        )
        naq = int(len(z) / 2)
        if len(kaq) == 1:
            kaq = kaq * np.ones(naq)
        if len(Saq) == 1:
            Saq = Saq * np.ones(naq)
        if len(poraq) == 1:
            poraq = poraq * np.ones(naq)
        if len(c) == 1:
            c = c * np.ones(naq - 1)
        if len(Sll) == 1:
            Sll = Sll * np.ones(naq - 1)
        if len(porll) == 1:
            porll = porll * np.ones(naq - 1)
        assert len(kaq) == naq, "Error: Length of kaq needs to be " + str(naq)
        assert len(Saq) == naq, "Error: Length of Saq needs to be " + str(naq)
        assert len(poraq) == naq, "Error: Length of poraq needs to be " + str(naq)
        assert len(c) == naq - 1, "Error: Length of c needs to be " + str(naq - 1)
        assert len(Sll) == naq - 1, "Error: Length of Sll needs to be " + str(naq - 1)
        assert len(porll) == naq - 1, "Error: Length of porll needs to be " + str(naq - 1)
        Haq = H[::2]
        assert np.all(Haq > 0), (
            "Error: Some thicknesses of aquifer layers " + "are negative"
        )
        Hll = H[1::2]
        Hll = np.maximum(Hll, 1e-20)  # make sure none are negative
        assert np.all(Hll > 0), (
            "Error: Some thicknesses of leaky layers " + "are negative"
        )
        c = np.hstack((1e100, c))
        Sll = np.hstack((1e-20, Sll))
        Hll = np.hstack((1e-20, Hll))
        porll = np.hstack((1e-20, porll))
        # layertype
        nlayers = len(z) - 1
        ltype = np.array(nlayers * ["a"])
        ltype[1::2] = "l"
    else:  # leaky layers on top
        assert len(z) % 2 == 1, (
            "Error: Length of z must be 2 * number of aquifers + 1 "
            "when topboundary is leaky in MaqModel"
        )
        naq = int((len(z) - 1) / 2)
        if len(kaq) == 1:
            kaq = kaq * np.ones(naq)
        if len(Saq) == 1:
            Saq = Saq * np.ones(naq)
        if len(poraq) == 1:
            poraq = poraq * np.ones(naq)
        if len(c) == 1:
            c = c * np.ones(naq)
        if len(Sll) == 1:
            Sll = Sll * np.ones(naq)
        if len(porll) == 1:
            porll = porll * np.ones(naq)
        assert len(kaq) == naq, "Error: Length of kaq needs to be " + str(naq)
        assert len(Saq) == naq, "Error: Length of Saq needs to be " + str(naq)
        assert len(poraq) == naq, "Error: Length of poraq needs to be " + str(naq)
        assert len(c) == naq, "Error: Length of c needs to be " + str(naq)
        assert len(Sll) == naq, "Error: Length of Sll needs to be " + str(naq)
        assert len(porll) == naq, "Error: Length of porll needs to be " + str(naq)
        Haq = H[1::2]
        Hll = H[::2]
        # layertype
        nlayers = len(z) - 1
        ltype = np.array(nlayers * ["a"])
        ltype[0::2] = "l"
    return kaq, Haq, Hll, c, Saq, Sll, poraq, porll, ltype


def param_3d(
    kaq=[1],
    z=[1, 0],
    Saq=[0.001],
    kzoverkh=1,
    poraq=0.3,
    phreatictop=False,
    topboundary="conf",
    topres=0,
    topthick=0,
    topSll=0,
    toppor=0.3,
):
    # Computes the parameters for a TimModel from input for a 3D model
    kaq = np.atleast_1d(kaq).astype(float)
    z = np.atleast_1d(z).astype(float)
    naq = len(z) - 1
    if len(kaq) == 1:
        kaq = kaq * np.ones(naq)
    Saq = np.atleast_1d(Saq).astype(float)
    if len(Saq) == 1:
        Saq = Saq * np.ones(naq)
    kzoverkh = np.atleast_1d(kzoverkh).astype(float)
    if len(kzoverkh) == 1:
        kzoverkh = kzoverkh * np.ones(naq)
    poraq = np.atleast_1d(poraq).astype(float)
    if len(poraq) == 1:
        poraq = poraq * np.ones(naq)
    Haq = z[:-1] - z[1:]
    c = 0.5 * Haq[:-1] / (kzoverkh[:-1] * kaq[:-1]) + 0.5 * Haq[1:] / (
        kzoverkh[1:] * kaq[1:]
    )
    # Saq = Saq * H
    # if phreatictop:
    #    Saq[0] = Saq[0] / H[0]
    c = np.hstack((1e100, c))
    Hll = 1e-20 * np.ones(len(c))
    Sll = 1e-20 * np.ones(len(c))
    porll = np.zeros(len(c))
    nlayers = len(z) - 1
    ltype = np.array(nlayers * ["a"])
    if (topboundary[:3] == "sem") or (topboundary[:3] == "lea"):
        c[0] = np.max([1e-20, topres])
        Hll[0] = np.max([1e-20, topthick])
        Sll[0] = np.max([1e-20, topSll])
        porll[0] = toppor
        ltype = np.hstack(("l", ltype))
        z = np.hstack((z[0] + topthick, z))
    return kaq, Haq, Hll, c, Saq, Sll, poraq, porll, ltype, z
