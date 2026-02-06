Introduction
============

Timflow steady is a submodule for the modeling of steady-state multi-layer groundwater
flow with analytic elements.

Timflow steady may be applied to an arbitrary number of layers and an arbitrary sequence of
aquifers and leaky layers. The head, flow, and leakage between aquifer layers may be
computed analytically at any point in the aquifer system. The Dupuit approximation is
applied to flow in aquifer layers (i.e., the resistance to flow in the vertical
direction is neglected), while flow in leaky layers is approximated as vertical.

.. grid::

    .. grid-item-card:: User-guide
        :link: 00userguide/index
        :link-type: doc

        Tutorials and how-to guides for getting started with timflow steady.

    .. grid-item-card:: Concepts
        :link: 01concepts/index
        :link-type: doc

        Timflow steady fundamental concepts explained.

    .. grid-item-card:: Examples
        :link: 02examples/index
        :link-type: doc

        Timflow steady example notebooks.

.. grid::

    .. grid-item-card:: Cross-sections
        :link: 03xsections/index
        :link-type: doc

        Cross-sectional models.

    .. grid-item-card:: Benchmarks
        :link: 04benchmarks/index
        :link-type: doc

        Notebooks testing/benchmarking timflow steady implementations.


Quick Example
-------------

.. tab-set::

    .. tab-item:: Python

        In this example a well is modelled near a river in a single aquifer.

        .. code-block:: python

            # import python packages
            import numpy as np
            import timflow.steady as tfs

            # create model
            ml = tfs.ModelMaq(kaq=10, z=[20, 0]) # single layer model
            
            # add a river with a fixed water level
            yls = np.arange(-100, 101, 20) # 20 points, so 19 segments
            xls = 50 * np.ones_like(yls)
            river = tfs.RiverString(ml, xy=list(zip(xls, yls)), hls=0.0)
            
            # add a well
            well = tfs.Well(ml, 0, 0, rw=0.3, Qw=1000)
            
            # solve model
            ml.solve()

            # plot head contours
            ml.plots.contour(win=[-30, 55, -30, 30], ngr=40, labels=True, decimals=1)
            

    .. tab-item:: Result

        In this example a well is modelled near a river in a single aquifer.

        .. figure:: ../_static/example_output_steady.png
            :figwidth: 500px


.. toctree::
   :maxdepth: 2
   :hidden:

    User-guide <00userguide/index>
    Concepts <01concepts/index>
    Examples <02examples/index>
    Cross-sections <03xsections/index>
    Benchmarks <04benchmarks/index>
