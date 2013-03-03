PyPDD
=====

A Python Positive Degree Day (PDD) model for glacier surface mass-balance.

Using an algorythm inspired by `PISM`_

The PDDModel class
------------------

**Requires:** `NumPy`_

A PDD model instance can be created by::

  from pypdd import PDDModel
  pdd = PDDModel()

Provided two arrays ``temp`` and ``prec`` of dimension ``(12, x, y)`` and containing monthly temperature and precipitation data,

::

  pdd(temp, prec)

will return an array of dimension ``(x, y)`` containing the calculated mass balance.

NetCDF interface
----------------

**Requires:** `NetCDF4-Python`_

A NetCDF operator can be called by::

  pdd.nco('input.nc', 'output.nc')

The file ``input.nc`` should contain monthly temperatures and precipitation in variables ``air_temp`` and ``precipitation`` and the calculated surface mass balance is stored as ``climatic_mass_balance`` in ``output.nc``.

Alternatively the module can be executed as a script::

  python pypdd.py -i 'input.nc' -o 'output.nc'

If no input file is provided, an artificial climate will be generated under ``atm.nc``. By default output is saved as ``smb.nc``.


GRASS GIS interface
-------------------

**Requires:** `GRASS GIS`_

To run a PDD model in GRASS enter::

  r.pdd.py temp=list,of,temp,maps prec=list,of,prec,maps smb=smb_map

.. links

.. _NumPy: http://numpy.scipy.org
.. _NetCDF4-Python: http://netcdf4-python.googlecode.com
.. _GRASS GIS: http://grass.osgeo.org
.. _PISM: http://www.pism-docs.org

