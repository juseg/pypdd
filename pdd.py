#!/usr/bin/env python

import numpy as np
from netCDF4 import Dataset as NC

# PDD model parameters

temp_snow = 0
temp_rain = 2
pdd_factor_snow = 0.003
pdd_factor_ice = 0.008
pdd_refreeze = 0.6
pdd_std_dev = 5

# PDD model computations

def pdd(temp):
		"""Compute positive degree days from temperature time series"""
		return sum(np.greater(temp,0)*temp)*365.242198781/12

def snowfrac(temp):
		"""Compute snow fraction from temperature"""
		return np.clip((temp_rain-temp)/(temp_rain-temp_snow), 0, 1)

def snow(temp, prec):
		"""Compute snow fall from temperature and precipitation"""
		return sum(snowfrac(temp)*prec)

def smb(snow, pdd):
		"""Compute surface mass balance from snow fall and pdd sum"""
		return np.where(pdd_factor_snow*pdd < snow,
			snow - pdd_factor_snow*pdd,
			pdd_refreeze*snow - pdd_factor_ice*(pdd-snow/pdd_factor_snow))

# netCDF IO

def init():
		"""Create an artificial PISM atmosphere file"""

		from math import cos, pi

		# open netcdf file
		nc = NC('atm.nc', 'w')

		# create dimensions
		tdim = nc.createDimension('time', 12)
		xdim = nc.createDimension('x', 8)
		ydim = nc.createDimension('y', 8)

		# prepare coordinate arrays
		x = range(len(xdim))
		y = range(len(ydim))
		t = range(len(tdim))
		(xx, yy) = np.meshgrid(x, y)

		# create air temperature variable
		temp = nc.createVariable('air_temp', 'f4', ('time', 'x', 'y'))
		temp.units = 'degC'

		# create precipitation variable
		prec = nc.createVariable('precipitation', 'f4', ('time', 'x', 'y'))
		prec.units = "m yr-1"

		# assign temperature and precipitation values
		for i in t:
			temp[i] = xx - 10 * cos(i*2*pi/12)
			prec[i] = yy * (1 - cos(i*2*pi/12))/4

		# close netcdf file
		nc.close()

def main():
		"""Read atmosphere file and output surface mass balance"""

		# open netcdf files
		i = NC('atm.nc', 'r')
		o = NC('clim.nc', 'w')

		# read input data
		temp = i.variables['air_temp'][:]
		prec = i.variables['precipitation'][:]

		# create dimensions
		for dimname, dim in i.dimensions.items():
			print dimname, len(dim)
			o.createDimension(dimname, len(dim))

		# compute the number of positive degree days
		pddvar = o.createVariable('pdd', 'f4', ('x', 'y'))
		pddvar[:] = pdd(temp)

		# compute snowfall
		snowvar = o.createVariable('snow', 'f4', ('x', 'y'))
		snowvar[:] = snow(temp, prec)

		# compute surface mass balance
		smbvar = o.createVariable('smb', 'f4', ('x', 'y'))
		smbvar[:] = smb(snowvar[:], pddvar[:])

# Called at execution

if __name__ == "__main__":

		# prepare dummy input dataset
		init()

		# run the mass balance model
		main()

