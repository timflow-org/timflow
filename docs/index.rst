timflow
=======

`timflow` is a Python package for the modeling of multi-layer flow with analytic
elements. Is is split into two main submodules `timflow.steady` for steady-state flow
and `timflow.transient` for modeling transient flow. Both modules may be applied to an
arbitrary number of aquifers and leaky layers. The head, flow, and leakage between
aquifers may be computed semi-analytically at any point in space and time. The design
of timflow is object-oriented and has been kept simple and flexible.

New analytic elements may be added to the code without making any changes in the
existing part of the code. `timflow` is coded in Python and uses numba to speed up
evaluation of the line elements and inverse laplace transforms.

The transient modeling submodule is based on the Laplace-transform analytic element
method. The solution is computed analytically in the Laplace domain and converted back
to the time domain numerically usig the algorithm of De Hoog, Stokes, and Knight.

Use the links below to navigate the documentation for steady or transient flow, or to
view the code documentation.

.. grid::

    .. grid-item-card:: Steady-state flow
        :link: steady/index
        :link-type: doc

        Steady-state multi-layer groundwater flow.

    .. grid-item-card:: Transient flow
        :link: transient/index
        :link-type: doc

        Transient multi-layer groundwater flow.

.. grid::
    
    .. grid-item-card:: Code Reference
        :link: api/index
        :link-type: doc

        Timflow code reference.

    .. grid-item-card:: Cite
        :link: about/index
        :link-type: doc

        References for citing timflow in publications.


.. toctree::
   :maxdepth: 1
   :hidden:

   Steady <steady/index>
   Transient <transient/index>
   Code Reference <api/index>
   Cite <about/index>
