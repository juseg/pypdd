#!/usr/bin/env python

"""A Python Positive Degree Day (PDD) model for glacier surface mass balance"""

import numpy as np

# Default model parameters
# ------------------------

default_pdd_factor_snow = 0.003
default_pdd_factor_ice  = 0.008
default_pdd_refreeze    = 0.6
default_temp_snow       = 0.
default_temp_rain       = 2.
default_integrate_rule  = 'rectangle'
default_interpolate_rule= 'linear'
default_interpolate_n   = 53

# PDD model class
# ---------------

class PDDModel():
  """A Positive Degree Day (PDD) model for glacier surface mass balance"""

  def __init__(self,
    pdd_factor_snow = default_pdd_factor_snow,
    pdd_factor_ice  = default_pdd_factor_ice,
    pdd_refreeze    = default_pdd_refreeze,
    temp_snow       = default_temp_snow,
    temp_rain       = default_temp_rain,
    integrate_rule  = default_integrate_rule,
    interpolate_rule= default_interpolate_rule,
    interpolate_n   = default_interpolate_n):
    """Initiate a PDD model with given parameters"""
    
    # set pdd model parameters
    self.pdd_factor_snow = pdd_factor_snow
    self.pdd_factor_ice  = pdd_factor_ice
    self.pdd_refreeze    = pdd_refreeze
    self.temp_snow       = temp_snow
    self.temp_rain       = temp_rain
    self.integrate_rule  = integrate_rule
    self.interpolate_rule= interpolate_rule
    self.interpolate_n   = interpolate_n

  def __call__(self, temp, prec, stdv=0., big=False):
    """Run the PDD model"""

    # interpolate time-series
    newtemp = self._interpolate(temp)
    newprec = self._interpolate(prec)

    # expand stdv
    if type(stdv) == float:
      newstdv = np.ones_like(newtemp) * stdv
    else:
      newstdv = self._interpolate(stdv)

    # compute accumulation and pdd
    accu_rate = self.accu_rate(newtemp, newprec)
    inst_pdd  = self.inst_pdd(newtemp, newstdv)

    # compute snow depth and melt rates
    snow_depth     = np.zeros_like(newtemp)
    snow_melt_rate = np.zeros_like(newtemp)
    ice_melt_rate  = np.zeros_like(newtemp)
    melt_rate      = np.zeros_like(newtemp)
    for i in range(len(newtemp)):
      if i > 0: snow_depth[i] = snow_depth[i-1]
      snow_depth[i] += accu_rate[i]
      snow_melt_rate[i], ice_melt_rate[i] = self.melt_rates(snow_depth[i], inst_pdd[i])
      snow_depth[i] -= snow_melt_rate[i]

    # compute comulative quantities
    pdd       = self._integrate(inst_pdd)
    accu      = self._integrate(accu_rate)
    snow_melt = self._integrate(snow_melt_rate)
    ice_melt  = self._integrate(ice_melt_rate)
    melt   = snow_melt + ice_melt
    runoff = melt - self.pdd_refreeze * melt
    smb    = accu - runoff

    # output
    if big:
      return dict(
        temp           = newtemp,
        prec           = newprec,
        stdv           = newstdv,
        inst_pdd       = inst_pdd,
        accu_rate      = accu_rate,
        snow_melt_rate = snow_melt_rate,
        ice_melt_rate  = ice_melt_rate,
        melt_rate      = melt_rate,
        snow_depth     = snow_depth,
        pdd            = pdd,
        accu           = accu,
        snow_melt      = snow_melt,
        ice_melt       = ice_melt,
        melt           = melt,
        runoff         = runoff,
        smb            = smb,
      )
    else:
      return smb

  def _integrate(self, a):
    """Integrate an array over one year"""

    rule = self.integrate_rule
    dx = 1./(self.interpolate_n-1)

    if rule == 'rectangle':
      return np.sum(a, axis=0)*dx

    if rule == 'trapeze':
      from scipy.integrate import trapz
      a = np.append(a, [a[0]], axis=0)
      return trapz(a, axis=0, dx=dx)

    if rule == 'simpson':
      from scipy.integrate import simps
      a = np.append(a, [a[0]], axis=0)
      return simps(a, axis=0, dx=dx)

  def _interpolate(self, a):
    """Interpolate an array through one year"""

    from scipy.interpolate import interp1d

    x = np.linspace(0, 1, 13)
    y = np.append(a, [a[0]], axis=0)
    newx = np.linspace(0, 1, self.interpolate_n)
    newy = interp1d(x, y, kind=self.interpolate_rule, axis=0)(newx)
    return newy[:-1]

  def inst_pdd(self, temp, stdv):
    """Compute instantaneous positive degree days from temperature and its standard deviation"""

    from math import exp, pi, sqrt
    from scipy.special import erfc

    # if sigma is zero scalar, use positive part of temperature
    def positivepart(temp):
      return np.greater(temp,0)*temp

    # otherwise use the Calov and Greve (2005) formula
    def calovgreve(temp, stdv):
      z = temp / (sqrt(2)*stdv)
      return stdv / sqrt(2*pi) * np.exp(-z**2) + temp/2 * erfc(-z)

    teff = np.where(stdv == 0., positivepart(temp), calovgreve(temp, stdv))

    # convert to degree-days
    return teff*365.242198781

  def accu_rate(self, temp, prec):
    """Compute accumulation rate from temperature and precipitation"""

    # compute snow fraction as a function of temperature
    reduced_temp = (self.temp_rain-temp)/(self.temp_rain-self.temp_snow)
    snowfrac     = np.clip(reduced_temp, 0, 1)

    # return accumulation rate
    return snowfrac*prec

  def melt_rates(self, snow, pdd):
    """Compute melt rate from snow precipitation and pdd sum"""

    # parse model parameters for readability
    ddf_snow = self.pdd_factor_snow
    ddf_ice  = self.pdd_factor_ice

    # compute a potential snow melt
    pot_snow_melt = ddf_snow * pdd

    # effective snow melt can't exceed amount of snow
    snow_melt = np.minimum(snow, pot_snow_melt)

    # ice melt is proportional to excess snow melt
    ice_melt = (pot_snow_melt - snow_melt) * ddf_ice/ddf_snow

    # return melt rates
    return (snow_melt, ice_melt)

  def nco(self, input_file, output_file, big=False, stdv=None):
    """NetCDF operator"""

    from netCDF4 import Dataset as NC

    # open netcdf files
    i = NC(input_file, 'r')
    o = NC(output_file, 'w', format='NETCDF3_CLASSIC')

    # read input data
    temp = i.variables['air_temp'][:]
    prec = i.variables['precipitation'][:]
    if stdv is None:
      try:
        stdv = i.variables['air_temp_stdev'][:]
      except KeyError:
        stdv = 0.

    # convert to degC
    # TODO: handle unit conversion better
    if i.variables['air_temp'].units == 'K': temp = temp - 273.15

    # get dimensions tuple from temp variable
    txydim = i.variables['air_temp'].dimensions
    xydim = txydim[1:]

    # create dimensions
    o.createDimension(txydim[0], self.interpolate_n - 1)
    for dimname in xydim:
      o.createDimension(dimname, len(i.dimensions[dimname]))

    # copy coordinates
    for varname,ivar in i.variables.iteritems():
      if varname in xydim:
        ovar = o.createVariable(varname, ivar.dtype, ivar.dimensions)
        for attname in ivar.ncattrs():
          setattr(ovar,attname,getattr(ivar,attname))
        ovar[:] = ivar[:]

    # create surface mass balance variable
    smb_var = o.createVariable('climatic_mass_balance', 'f4', xydim)
    smb_var.long_name     = 'instantaneous ice-equivalent surface mass balance (accumulation/ablation) rate'
    smb_var.standard_name = 'land_ice_surface_specific_mass_balance'
    smb_var.units         = 'm yr-1'

    # create more variables for 'big' output
    if big:

      # create snow precipitation variable
      pdd_var = o.createVariable('pdd', 'f4', xydim)
      pdd_var.long_name = 'number of positive degree days'
      pdd_var.units = 'degC day'

      # create snow precipitation variable
      accu_var = o.createVariable('saccum', 'f4', xydim)
      accu_var.long_name = 'instantaneous ice-equivalent surface accumulation rate (precipitation minus rain)'
      accu_var.units = 'm yr-1'

      # create melt variable
      snow_melt_var = o.createVariable('snow_melt', 'f4', xydim)
      snow_melt_var.long_name = 'cumulative melt of snow'
      snow_melt_var.units = 'm yr-1'
      ice_melt_var = o.createVariable('ice_melt', 'f4', xydim)
      ice_melt_var.long_name = 'cumulative melt rate'
      ice_melt_var.units = 'm yr-1'
      melt_var = o.createVariable('smelt', 'f4', xydim)
      melt_var.long_name = 'instantaneous ice-equivalent surface melt rate'
      melt_var.units = 'm yr-1'

      # create runoff variable
      runoff_var = o.createVariable('srunoff', 'f4', xydim)
      runoff_var.long_name = 'instantaneous ice-equivalent surface meltwater runoff rate'
      runoff_var.units = 'm yr-1'

      # interpolated climatic variables
      temp_var = o.createVariable('air_temp', 'f4', txydim)
      temp_var.long_name = 'near-surface air temperature'
      temp_var.units     = 'degC'
      prec_var = o.createVariable('precipitation', 'f4', txydim)
      prec_var.long_name = 'ice-equivalent precipitation rate'
      prec_var.units     = 'm yr-1'
      stdv_var = o.createVariable('air_temp_stdv', 'f4', txydim)
      stdv_var.long_name = 'standard deviation of near-surface air temperature'
      stdv_var.units     = 'degC'

      # instantaneous pdd
      inst_pdd_var = o.createVariable('inst_pdd', 'f4', txydim)
      inst_pdd_var.long_name = 'instantaneous positive degree days'
      inst_pdd_var.units = 'degC day'

      # accumulation rate variable
      accu_rate_var = o.createVariable('accu_rate', 'f4', txydim)
      accu_rate_var.long_name = 'instantaneous accumulation rate'
      accu_rate_var.units = 'm yr-1'
      snow_melt_rate_var = o.createVariable('snow_melt_rate', 'f4', txydim)
      snow_melt_rate_var.long_name = 'instantaneous melt rate of snow'
      snow_melt_rate_var.units = 'm yr-1'
      ice_melt_rate_var = o.createVariable('ice_melt_rate', 'f4', txydim)
      ice_melt_rate_var.long_name = 'instantaneous melt rate of ice'
      ice_melt_rate_var.units = 'm yr-1'
      melt_rate_var = o.createVariable('melt_rate', 'f4', txydim)
      melt_rate_var.long_name = 'instantaneous melt rate of snow and ice'
      melt_rate_var.units = 'm yr-1'
      snow_depth_var = o.createVariable('snow_depth', 'f4', txydim)
      snow_depth_var.long_name = 'depth of snow cover'
      snow_depth_var.units = 'm'

    # run PDD model
    smb = self(temp, prec, stdv=stdv, big=big)

    # assign variables values
    smb_var[:]            = smb['smb'],
    if big:
      pdd_var[:]            = smb['pdd'],
      accu_var[:]           = smb['accu'],
      snow_melt_var[:]      = smb['snow_melt'],
      ice_melt_var[:]       = smb['ice_melt'],
      melt_var[:]           = smb['melt'],
      runoff_var[:]         = smb['runoff'],
      temp_var[:]           = smb['temp'],
      prec_var[:]           = smb['prec'],
      stdv_var[:]           = smb['stdv'],
      inst_pdd_var[:]       = smb['inst_pdd'],
      accu_rate_var[:]      = smb['accu_rate'],
      snow_melt_rate_var[:] = smb['snow_melt_rate'],
      ice_melt_rate_var[:]  = smb['ice_melt_rate'],
      melt_rate_var[:]      = smb['melt_rate'],
      snow_depth_var[:]     = smb['snow_depth'],

    # close netcdf files
    i.close()
    o.close()

# Command-line interface
# ----------------------

def make_fake_climate(filename):
    """Create an artificial temperature and precipitation file"""

    from math import cos, pi
    from netCDF4 import Dataset as NC

    # open netcdf file
    nc = NC(filename, 'w')

    # create dimensions
    tdim = nc.createDimension('time', 12)
    xdim = nc.createDimension('x', 201)
    ydim = nc.createDimension('y', 201)
    ndim = nc.createDimension('nv', 2)

    # create x coordinate variable
    xvar = nc.createVariable('x', 'f4', ('x',))
    xvar.axis          = 'X'
    xvar.long_name     = 'x-coordinate in Cartesian system'
    xvar.standard_name = 'projection_x_coordinate'
    xvar.units         = 'm'

    # create y coordinate variable
    yvar = nc.createVariable('y', 'f4', ('y',))
    yvar.axis          = 'Y'
    yvar.long_name     = 'y-coordinate in Cartesian system'
    yvar.standard_name = 'projection_y_coordinate'
    yvar.units         = 'm'
    
    # create time coordinate and time bounds
    tvar = nc.createVariable('time', 'f4', ('time',))
    tvar.axis          = 'T'
    tvar.long_name     = 'time'
    tvar.standard_name = 'time'
    tvar.units         = 'month'
    tvar.bounds        = 'time_bounds'
    tboundsvar = nc.createVariable('time_bounds', 'f4', ('time','nv'))

    # create air temperature variable
    temp = nc.createVariable('air_temp', 'f4', ('time', 'x', 'y'))
    temp.long_name = 'near-surface air temperature'
    temp.units     = 'degC'

    # create precipitation variable
    prec = nc.createVariable('precipitation', 'f4', ('time', 'x', 'y'))
    prec.long_name = 'ice-equivalent precipitation rate'
    prec.units     = "m yr-1"

    # assign coordinate values
    lx = ly = 750000
    xvar[:] = np.linspace(-lx, lx, len(xdim))
    yvar[:] = np.linspace(-ly, ly, len(ydim))
    tvar[:] = np.arange(len(tdim))
    tboundsvar[:,0] = tvar[:]
    tboundsvar[:,1] = tvar[:]+1

    # assign temperature and precipitation values
    (xx, yy) = np.meshgrid(xvar[:], yvar[:])
    for i in range(len(tdim)):
      temp[i] = -10 * yy/ly - 5 * cos(i*2*pi/12)
      prec[i] = xx/lx * (np.sign(xx) - cos(i*2*pi/12))

    # close netcdf file
    nc.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
      description='''A Python Positive Degree Day (PDD) model
        for glacier surface mass balance''',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', metavar='input.nc',
      help='input file')
    parser.add_argument('-o', '--output', metavar='output.nc',
      help='output file',
      default='smb.nc')
    parser.add_argument('-b', '--big', action='store_true',
      help='produce big output (more variables)')
    parser.add_argument('--pdd-factor-snow', metavar='F', type=float,
      help='PDD factor for snow',
      default=default_pdd_factor_snow)
    parser.add_argument('--pdd-factor-ice', metavar='F', type=float,
      help='PDD factor for ice',
      default=default_pdd_factor_ice)
    parser.add_argument('--pdd-refreeze', metavar='R', type=float,
      help='PDD model refreezing fraction',
      default=default_pdd_refreeze)
    parser.add_argument('--pdd-std-dev', metavar='S', type=float,
      help='Use constant standard deviation of temperature')
    parser.add_argument('--temp-snow', metavar='T', type=float,
      help='Temperature at which all precip is snow',
      default=default_temp_snow)
    parser.add_argument('--temp-rain', metavar='T', type=float,
      help='Temperature at which all precip is rain',
      default=default_temp_rain)
    parser.add_argument('--integrate-rule',
      help='Rule for integrations',
      default = default_integrate_rule,
      choices = ('rectangle', 'trapeze', 'simpson'))
    parser.add_argument('--interpolate-rule',
      help='Rule for interpolations',
      default = default_interpolate_rule,
      choices = ('linear','nearest','zero','slinear','quadratic','cubic'))
    parser.add_argument('--interpolate-n', metavar='N',
      help='Number of points used in interpolations.',
      default = default_interpolate_n)
    args = parser.parse_args()

    # if no input file was given, prepare a dummy one
    if not args.input:
      make_fake_climate('atm.nc')

    # initiate PDD model
    pdd=PDDModel(
      pdd_factor_snow = args.pdd_factor_snow,
      pdd_factor_ice  = args.pdd_factor_ice,
      pdd_refreeze    = args.pdd_refreeze,
      temp_snow       = args.temp_snow,
      temp_rain       = args.temp_rain,
      integrate_rule  = args.integrate_rule,
      interpolate_rule= args.interpolate_rule,
      interpolate_n   = args.interpolate_n)

    # compute surface mass balance
    pdd.nco(args.input or 'atm.nc', args.output, big=args.big, stdv=args.pdd_std_dev)

