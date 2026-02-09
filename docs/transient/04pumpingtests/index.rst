Pumping tests
=============

This series of benchmark problems demonstrate the features and capabilities of TTim for
simulating and analyzing transient groundwater hydraulic problems such as pumping tests
and slug tests.


Confined Pumping Tests
----------------------

1. `Oude Korendijk`_ - One layer confined pumping test with two observation wells
2. `Grindley`_ - One layer confined pumping test with data from both observation well and pumping well.
3. `Sioux`_ - One Layer confined aquifer test with three observation wells
4. `Schroth`_ - Three layers (Aquifer - Aquitard - Aquifer) confined pumping test with one observation well.
5. `Nevada`_ - One layer, fractured confined aquifer test with data from both observation well and pumping well.

.. _Oude Korendijk: confined1_oude_korendijk.html
.. _Grindley: confined2_grindley.html
.. _Sioux: confined3_sioux.html
.. _Schroth: confined4_schroth.html
.. _Nevada: confined5_nevada.html


Leaky Pumping Tests (Semi-confined)
-----------------------------------


1. `Dalem`_ - One layer, semi-confined aquifer test with four observation wells.
2. `Hardixveld`_ - Four layers (Aquitard - Aquifer - Aquitard - Aquifer) semi-confined pumping test with data from the pumping well.
3. `Texas Hill`_ - One layer, semi-confined aquifer test with three observation wells.

.. _Dalem: leaky1_dalem.html
.. _Hardixveld: leaky2_hardixveld.html
.. _Texas Hill: leaky3_texashill.html

Unconfined Pumping Tests
------------------------

1. `Vennebulten`_ - Two layers, unconfined aquifer test with data two observation wells screened at different depths.
2. `Moench`_ - Solving the Analytical model from Moench in TTim. Pumping test in unconfined aquifer with four piezometers screened at two different depths and two different distances
3. `Neuman`_ - Constant-rate pumping test performed in an unconfined aquifer with delayed gravity response.

.. _Vennebulten: unconfined1_vennebulten.html
.. _Moench: unconfined2_moench.html
.. _Neuman: unconfined3_neuman.html

Slug Tests
----------

1. `Pratt County`_ - One layer partially penetrated slug test with data from the test well.
2. `Falling Head`_ - One layer partially penetrated slug test with data from the test well.
3. `Multi-Well`_ - One layer confined slug test with data from the test well and from an observation well.
4. `Dawsonville`_ - One layer fully-penetrated confined slug test with data from the test well.

.. _Pratt County: slug1_pratt_county.html
.. _Falling Head: slug2_falling_head.html
.. _Multi-Well: slug3_multiwell.html
.. _Dawsonville: slug4_dawsonville.html


.. toctree::
    :maxdepth: 1
    :hidden:

    synthetic0_data
    synthetic1_2aquifers
    synthetic2_test_calibrate
    confined1_oude_korendijk
    confined2_grindley
    confined3_sioux
    confined4_schroth
    confined5_nevada
    leaky1_dalem
    leaky2_hardixveld
    leaky3_texashill
    unconfined1_vennebulten
    unconfined2_moench
    unconfined3_neuman
    slug1_pratt_county
    slug2_falling_head
    slug3_multiwell
    slug4_dawsonville
