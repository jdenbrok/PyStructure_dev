.. PyStructure documentation master file, created by
   sphinx-quickstart on Mon Oct 30 00:34:21 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyStructure Documentation
==========================

PyStructure is an astronomical analysis software script build on Python. It's intended for the analysis of a set of 3D and 2D astrophysical data. The script will homogenize the data sets and produce a science-ready table that makes it possible to compare the various observations sightline-by-sightline.

.. important::

    This code is still active work in progress and changes to the workflow might occur.


Acknowledgements
----------------
The code builds on past IDL scripts that had similar functionalities. The routines have been updated, improved, and translated to Python.

List of Papers
--------------
The scripts have been employed by several peer-reviewed publications:

* Stuber et al. (2025), A&A, 702A, 66S

* Galić et al. (2025), arXiv250815901G

* Zhang et al. (2025), ApJ, 982, 21Z 

* den Brok et al. (2025), ApJ, 988, 162D

* Kovačić et al. (2025), A&A, 694A, 87K

* den Brok et al. (2023), MNRAS, 526, 6347

* Eibensteiner et al. (2023), A&A, 675, 37

* Neumann et al. (2023),MNRAS, 521, 3348

* den Brok et al. (2022), A&A, 662, 89

* Eibensteiner et al. (2022), A&A, 659, 173

* den Brok et al. (2021), MNRAS, 504, 3221

.. _Sphinx: http://www.sphinx-doc.org
.. _Read the Docs: http://www.readthedocs.org

.. toctree::
   :caption: Getting Started
   :maxdepth: 2
   :hidden:

   installing
   codeStruct
   quickstart

.. toctree::
    :maxdepth: 2
    :caption: Running the Script
    :hidden:

    simpleExample
    advanced

.. toctree::
    :maxdepth: 2
    :caption: Analysis
    :hidden:

    reading
    PyStructFunctions
