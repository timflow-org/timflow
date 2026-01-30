Introduction
============

Timflow transient is a submodule for the modeling of transient multi-layer groundwater
flow with analytic elements.

The head, flow, and leakage between aquifer layers may be computed analytically at any
point in the aquifer system and at any time.

.. grid::

    .. grid-item-card:: Tutorials
        :link: 00userguide/index
        :link-type: doc

        Tutorials and how-to guides for getting started with timflow transient.


    .. grid-item-card:: Concepts
        :link: 01concepts/index
        :link-type: doc

        Timflow transient basic concepts and elements explained.

    .. grid-item-card:: Examples
        :link: 02examples/index
        :link-type: doc

        Timflow transient example notebooks.

.. grid::

    .. grid-item-card:: Cross-section
        :link: 03xsections/index
        :link-type: doc

        Transient cross-section models.

    .. grid-item-card:: Pumping tests
        :link: 04pumpingtests/index
        :link-type: doc

        Pumping test benchmark notebooks.
    
    .. grid-item-card:: Benchmarks
        :link: 05benchmarks/index
        :link-type: doc

        Comparing timflow transient to known solutions.


Quick Example
-------------

.. tab-set::

    .. tab-item:: Python

        In this example a well is modelled near a river in a single aquifer.

        .. code-block:: python

            # Import python packages
            import numpy as np
            from timflow import transient as ft

            # Create model
            ml = ft.ModelMaq(kaq=10, z=[20, 0], Saq=[0.1], phreatictop=True, tmin=1e-3, tmax=100)
            
            # Add a river with a fixed water level
            yls = np.arange(-100.0, 101, 20)
            xls = 50.0 * np.ones_like(yls)
            river = ft.HeadLineSinkString(ml, xy=list(zip(xls, yls)), tsandh='fixed')
            
            # Add a well
            well = ft.Well(ml, 0.0, 0.0, rw=0.3, tsandQ=[(0, 1000)])
            
            # Solve model
            ml.solve()

            # Plot head contours at t=2 days
            ml.plots.contour(win=[-30, 55, -30, 30], ngr=40, t=2, labels=True, decimals=1)
            

    .. tab-item:: Result

        In this example a well is modelled near a river in a single aquifer.

        .. figure:: _static/example_output_transient.png
            :figwidth: 500px


Approximations
--------------

The Dupuit approximation is applied to aquifer layers, while flow in leaky layers is
approximated as vertical.


.. toctree::
    :maxdepth: 2
    :hidden:

    User-guide <00userguide/index>
    Concepts <01concepts/index>
    Examples <02examples/index>
    Cross-sections <03xsections/index>
    Pumping tests <04pumpingtests/index>
    Benchmarks <05benchmarks/index>
