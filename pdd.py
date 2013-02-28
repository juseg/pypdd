#!/usr/bin/env python

import numpy as np
from netCDF4 import Dataset as NC

# PDD model class

class PDDModel():
	"""A positive degree-day model for glacier surface mass balance"""

	def __init__(self,
		pdd_factor_snow = 0.003,
		pdd_factor_ice  = 0.008,
		pdd_refreeze    = 0.6,
		pdd_std_dev     = 5,
		temp_snow       = 0,
		temp_rain       = 2):
		"""Initiate PDD model with given parameters"""
		
		# set pdd model parameters
		self.pdd_factor_snow = pdd_factor_snow
		self.pdd_factor_ice  = pdd_factor_ice
		self.pdd_refreeze    = pdd_refreeze
		self.pdd_std_dev     = pdd_std_dev
		self.temp_snow       = temp_snow
		self.temp_rain       = temp_rain

	def pdd(self, temp):
		"""Compute positive degree days from temperature time series"""
		return sum(np.greater(temp,0)*temp)*365.242198781/12

	def snowfrac(self, temp):
		"""Compute snow fraction from temperature"""
		reduced_temp = (self.temp_rain-temp) / (self.temp_rain-self.temp_snow)
		return np.clip(reduced_temp, 0, 1)

	def snow(self, temp, prec):
		"""Compute snow fall from temperature and precipitation"""
		return sum(self.snowfrac(temp)*prec)

	def smb(self, snow, pdd):
		"""Compute surface mass balance from snow precipitation and pdd sum"""

		# compute snow accumulation after melt
		snow_acc = snow - pdd * self.pdd_factor_snow

		# if positive, return accumulated snow
		# if negative, compute refreezing and ice melt
		return np.where(snow_acc > 0, snow_acc, snow*self.pdd_refreeze
			+ snow_acc*self.pdd_factor_ice/self.pdd_factor_snow)

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
			o.createDimension(dimname, len(dim))

		# initiate PDD model
		pdd=PDDModel()

		# compute the number of positive degree days
		pddvar = o.createVariable('pdd', 'f4', ('x', 'y'))
		pddvar[:] = pdd.pdd(temp)

		# compute snowfall
		snowvar = o.createVariable('snow', 'f4', ('x', 'y'))
		snowvar[:] = pdd.snow(temp, prec)

		# compute surface mass balance
		smbvar = o.createVariable('smb', 'f4', ('x', 'y'))
		smbvar[:] = pdd.smb(snowvar[:], pddvar[:])

# Called at execution

if __name__ == "__main__":

		# prepare dummy input dataset
		init()

		# run the mass balance model
		main()

