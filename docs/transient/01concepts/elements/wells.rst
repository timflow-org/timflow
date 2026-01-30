Wells
=====

Wells may be defined by specifying the discharge of the well or by specifying the
(desired) head inside the well. Wells may have an entry resistance (skin effect)
defined by the resistance :math:`c` (dimension: time). The discharge :math:`Q_i` in
layer :math:`i` is a function of the head :math:`h_i` in layer :math:`i` just outside
the well and the head :math:`h_w` inside the well, and the aquifer thicknes :math:`H_i`
of layer :math:`i`:

    .. math::
        Q_i = 2\pi r_wH_i(h_i - h_w)/c
        
The following types of wells are implemented:

1. :class:`~timflow.transient.well.Well` is a well for which the total discharge is specified. The
   discharge may be continuous between specified times or may be a slug. The total
   discharge is distributed across the layers in which the well is screened such that the
   head inside the well is the same in each screened layer. Skin effect and wellbore
   storage may be included.

2. :class:`~timflow.transient.well.HeadWell` is a well for which the head inside the well is
   specified. The discharge in each layer is computed such that the head in all screened
   layers is equal to the specified head. Skin effect may be included.