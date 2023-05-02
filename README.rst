.. Copyright (c) 2013-2023, Julien Seguinot (juseg.dev)
.. GNU General Public License v3.0+ (https://www.gnu.org/licenses/gpl-3.0.txt)

PyPDD
=====

.. image:: https://github.com/juseg/pypdd/actions/workflows/test.yaml/badge.svg
   :target: https://github.com/juseg/pypdd/actions?query=workflow%3Atest
.. image:: https://codecov.io/github/juseg/pypdd/branch/main/graph/badge.svg
   :target: https://codecov.io/github/juseg/pypdd
.. image:: https://img.shields.io/pypi/v/pypdd.svg
   :target: https://pypi.python.org/pypi/pypdd
.. image:: https://zenodo.org/badge/8483394.svg
   :target: https://zenodo.org/badge/latestdoi/8483394

A Python positive degree day model for glacier surface mass balance.

This module provides a simple model to compute accumulation and melt on a
glacier using near-surface air temperature and precipitation time series. The
model assumes that melt is proportional to the number of positive degree-days,
which corresponds to the integral of temperature above 0°C. Temperature
variability is included by assuming a normal temperature distribution around the
mean. The model optionally includes refreezing of melted snow and ice at the
glacier surface.

PyPDD_ can be used as a module within Python to operate on Numpy_ arrays. In
addition, it reads and writes netCDF_ files directly from the command line, and
provides a raster module for `GRASS GIS`_. The PDD model is based on an
algorithm that was initially developed for the `Parallel Ice Sheet Model`_ and
adopted here with very few changes.

Installation::

   pip install pypdd


The PDDModel class
------------------

**Requires:** NumPy_, SciPy_, Xarray_.

A PDD model instance can be created by::

   from pypdd import PDDModel
   pdd = PDDModel()

Several model parameters can be set at initialization. See ``help(PDDModel)``
for a list. Provided two arrays ``temp`` and ``prec`` of shape ``(t, x, y)``
containing temperature and precipitation data, the PDD model can be called
with::

   pdd(temp, prec)

This will return a dictionary containing a number of two- and three-dimensional
arrays, including the number of positive degree day ``'pdd'`` and total surface
mass balance ``'smb'``. Temperature variability can be included in a third array
``stdv`` containing temperature standard deviation values::

	pdd(temp, prec, stdv)

If any of ``temp``, ``prec``, or ``stdv`` has shape ``(x, y)``, it will be
interpreted as constant in time and expanded along the time dimension. Floats
with be interpreted as constant in time and space and expanded along all
dimensions.

NetCDF interface
----------------

**Requires:** netCDF4-Python_.

The PDDModel class holds a netCDF operator, which can be called by::

   pdd.nco('input.nc', 'output.nc')

The file ``'input.nc'`` should contain temperatures and precipitation in
variables ``'temp'`` and ``'prec'``. The calculated number of positive degree
days and total surface mass balance are stored in variables ``'pdd'`` and
``'smb'`` of ``'output.nc'``. Keyword argument ``output_size`` or
``output_variables`` can be used to produce more output.

The netCDF interface can be used directly from the command line by executing the
module as a script::

   python pypdd.py -i 'input.nc' -o 'output.nc'

If no input file is provided, an artificial climate will be generated under
``atm.nc`` and used by the model. By default, output is saved as ``smb.nc``.
Many more command-line options are available. For an overview type::

   python pypdd.py --help


GRASS GIS interface
-------------------

**Requires:** `GRASS GIS`_.

PyPDD can also operate on GRASS raster maps using the attached module ``r.pdd``.
Temperature, precipitation and standard deviation maps should be provided as
comma-separated lists::

   r.pdd.py temp=list,of,temp,maps prec=list,of,prec,maps pdd=pdd_map smb=smb_map

All time-independent PyPDD output variables can currently be exported as raster
maps. Alike any other GRASS module, a graphical prompt can be invoked by calling
``r.pdd`` without arguments, and a list of options can be obtained with::

   r.pdd.py --help


References
----------

Applications of PyPDD:

* N. Gandy, L. J. Gregoire, J. C. Ely, C. D. Clark, D. M. Hodgson, V. Lee,
  T. Bradwell, and R. F. Ivanovic.
  Marine ice sheet instability and ice shelf buttressing of the Minch Ice
  Stream, northwest Scotland.
  *The Cryosphere*, 12(11):3635--3651,
  https://doi.org/10.5194/tc-12-3635-2018, 2018.

* A. Plach, K. H. Nisancioglu, S. Le clec'h, A. Born, P. M. Langebroek, C. Guo,
  M. Imhof, and T. F. Stocker.
  Eemian Greenland SMB strongly sensitive to model choice.
  *Clim. Past*, 14:1463--1485,
  https://doi.org/10.5194/cp-14-1463-2018, 2018.

* J. Seguinot.
  Spatial and seasonal effects of temperature variability in a positive
  degree-day glacier surface mass-balance model.
  *J. Glaciol.*, 59(218):1202--1204,
  http://doi.org/10.3189/2013JoG13J081, 2013.


.. links

.. _GRASS GIS: http://grass.osgeo.org
.. _netCDF: http://www.unidata.ucar.edu/software/netcdf
.. _netCDF4-Python: https://github.com/Unidata/netcdf4-python
.. _NumPy: http://numpy.scipy.org
.. _Parallel Ice Sheet Model: http://www.pism-docs.org
.. _PyPDD: https://github.com/jsegu/pypdd
.. _SciPy: http://www.scipy.org
.. _Xarray: https://xarray.dev
