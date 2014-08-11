#!/usr/bin/env python

"""A Python Positive Degree Day (PDD) model for glacier surface mass balance"""

import numpy as np


# Default model parameters
# ------------------------

defaults = {
    'pdd_factor_snow':  0.003,
    'pdd_factor_ice':   0.008,
    'refreeze_snow':    0.0,
    'refreeze_ice':     0.0,
    'temp_snow':        0.0,
    'temp_rain':        2.0,
    'interpolate_rule': 'linear',
    'interpolate_n':    52}


# Default variable names
# ----------------------
names = {

    # coordinate variables
    'x': {
        'axis': 'X',
        'long_name': 'x-coordinate in Cartesian system',
        'standard_name': 'projection_x_coordinate',
        'units': 'm'},
    'y': {
        'axis': 'Y',
        'long_name': 'y-coordinate in Cartesian system',
        'standard_name': 'projection_y_coordinate',
        'units': 'm'},
    'time': {
        'axis': 'T',
        'long_name': 'time',
        'standard_name': 'time',
        'bounds': 'time_bounds',
        'units': 'yr'},
    'time_bounds': {},

    # climatic variables
    'temp': {
        'long_name': 'near-surface air temperature',
        'units':     'degC'},
    'prec': {
        'long_name': 'ice-equivalent precipitation rate',
        'units':     'm yr-1'},
    'stdv': {
        'long_name': 'standard deviation of near-surface air temperature',
        'units':     'K'},

    # cumulative quantities
    'smb': {
        'standard_name': 'land_ice_surface_specific_mass_balance',
        'long_name': 'cumulative ice-equivalent surface mass balance',
        'units':     'm yr-1'},
    'pdd': {
        'long_name': 'cumulative number of positive degree days',
        'units':     'degC day'},
    'accu': {
        'long_name': 'cumulative ice-equivalent surface accumulation',
        'units':     'm'},
    'snow_melt': {
        'long_name': 'cumulative ice-equivalent surface melt of snow',
        'units':     'm'},
    'ice_melt': {
        'long_name': 'cumulative ice-equivalent surface melt of ice',
        'units':     'm'},
    'melt': {
        'long_name': 'cumulative ice-equivalent surface melt',
        'units':     'm'},
    'runoff': {
        'long_name': 'cumulative ice-equivalent surface meltwater runoff',
        'units':     'm yr-1'},

    # instantaneous quantities
    'inst_pdd': {
        'long_name': 'instantaneous positive degree days',
        'units':     'degC day'},
    'accu_rate': {
        'long_name': 'instantaneous ice-equivalent surface accumulation rate',
        'units':     'm yr-1'},
    'snow_melt_rate': {
        'long_name': 'instantaneous ice-equivalent surface melt rate of snow',
        'units':     'm yr-1'},
    'ice_melt_rate': {
        'long_name': 'instantaneous ice-equivalent surface melt rate of ice',
        'units':     'm yr-1'},
    'melt_rate': {
        'long_name': 'instantaneous ice-equivalent surface melt rate',
        'units':     'm yr-1'},
    'runoff_rate': {
        'long_name': 'instantaneous ice-equivalent surface runoff rate',
        'units':     'm yr-1'},
    'inst_smb': {
        'long_name': 'instantaneous ice-equivalent surface mass balance',
        'units':     'm yr-1'},
    'snow_depth': {
        'long_name': 'depth of snow cover',
        'units':     'm'}}


def _create_nc_variable(nc, varname, dtype, dimensions):
    """Create netCDF variable and apply default attributes"""
    var = nc.createVariable(varname, dtype, dimensions)
    for (attr, value) in names[varname].iteritems():
        setattr(var, attr, value)
    return var


# PDD model class
# ---------------

class PDDModel():
    """Return a callable Positive Degree Day (PDD) model instance.

    Model parameters are held as public attributes, and can be set using
    corresponding keyword arguments at initialization time:

    *pdd_factor_snow* : float
        Positive degree-day factor for snow.
    *pdd_factor_ice* : float
        Positive degree-day factor for ice.
    *refreeze_snow* : float
        Refreezing fraction of melted snow.
    *refreeze_ice* : float
        Refreezing fraction of melted ice.
    *temp_snow* : float
        Temperature at which all precipitation falls as snow.
    *temp_rain* : float
        Temperature at which all precipitation falls as rain.
    *interpolate_rule* : [ 'linear' | 'nearest' | 'zero' |
                           'slinear' | 'quadratic' | 'cubic' ]
        Interpolation rule passed to `scipy.interpolate.interp1d`.
    *interpolate_n*: int
        Number of points used in interpolations.
    """

    def __init__(self,
                 pdd_factor_snow=defaults['pdd_factor_snow'],
                 pdd_factor_ice=defaults['pdd_factor_ice'],
                 refreeze_snow=defaults['refreeze_snow'],
                 refreeze_ice=defaults['refreeze_ice'],
                 temp_snow=defaults['temp_snow'],
                 temp_rain=defaults['temp_rain'],
                 interpolate_rule=defaults['interpolate_rule'],
                 interpolate_n=defaults['interpolate_n']):

        # set pdd model parameters
        self.pdd_factor_snow = pdd_factor_snow
        self.pdd_factor_ice = pdd_factor_ice
        self.refreeze_snow = refreeze_snow
        self.refreeze_ice = refreeze_ice
        self.temp_snow = temp_snow
        self.temp_rain = temp_rain
        self.interpolate_rule = interpolate_rule
        self.interpolate_n = interpolate_n

    def __call__(self, temp, prec, stdv=0.0):
        """Run the positive degree day model.

        Use temperature, precipitation, and standard deviation of temperature
        to compute the number of positive degree days, accumulation and melt
        surface mass fluxes, and the resulting surface mass balance.

        *temp*: array_like
            Input near-surface air temperature in degrees Celcius.
        *prec*: array_like
            Input precipitation rate in meter per year.
        *stdv*: array_like (default 0.0)
            Input standard deviation of near-surface air temperature in Kelvin.

        By default, inputs are N-dimensional arrays whose first dimension is
        interpreted as time and as periodic. Arrays of dimensions
        N-1 are interpreted as constant in time and expanded to N dimensions.
        Arrays of dimension 0 and numbers are interpreted as constant in time
        and space and will be expanded too. The largest input array determines
        the number of dimensions N.

        Return the number of positive degree days ('pdd'), surface mass balance
        ('smb'), and many other output variables in a dictionary.
        """

        # ensure numpy arrays
        temp = np.asarray(temp)
        prec = np.asarray(prec)
        stdv = np.asarray(stdv)

        # expand arrays to the largest shape
        maxshape = max(temp.shape, prec.shape, stdv.shape)
        temp = self._expand(temp, maxshape)
        prec = self._expand(prec, maxshape)
        stdv = self._expand(stdv, maxshape)

        # interpolate time-series
        temp = self._interpolate(temp)
        prec = self._interpolate(prec)
        stdv = self._interpolate(stdv)

        # compute accumulation and pdd
        accu_rate = self.accu_rate(temp, prec)
        inst_pdd = self.inst_pdd(temp, stdv)

        # initialize snow depth and melt rates
        snow_depth = np.zeros_like(temp)
        snow_melt_rate = np.zeros_like(temp)
        ice_melt_rate = np.zeros_like(temp)

        # compute snow depth and melt rates
        for i in range(len(temp)):
            if i > 0:
                snow_depth[i] = snow_depth[i-1]
            snow_depth[i] += accu_rate[i]
            snow_melt_rate[i], ice_melt_rate[i] = self.melt_rates(
                snow_depth[i], inst_pdd[i])
            snow_depth[i] -= snow_melt_rate[i]
        melt_rate = snow_melt_rate + ice_melt_rate
        runoff_rate = melt_rate - self.refreeze_snow * snow_melt_rate \
                                - self.refreeze_ice * ice_melt_rate
        inst_smb = accu_rate - runoff_rate

        # output
        return {'temp':           temp,
                'prec':           prec,
                'stdv':           stdv,
                'inst_pdd':       inst_pdd,
                'accu_rate':      accu_rate,
                'snow_melt_rate': snow_melt_rate,
                'ice_melt_rate':  ice_melt_rate,
                'melt_rate':      melt_rate,
                'runoff_rate':    runoff_rate,
                'inst_smb':       inst_smb,
                'snow_depth':     snow_depth,
                'pdd':            self._integrate(inst_pdd),
                'accu':           self._integrate(accu_rate),
                'snow_melt':      self._integrate(snow_melt_rate),
                'ice_melt':       self._integrate(ice_melt_rate),
                'melt':           self._integrate(melt_rate),
                'runoff':         self._integrate(runoff_rate),
                'smb':            self._integrate(inst_smb)}

    def _expand(self, a, shape):
        """Expand an array to the given shape"""
        if a.shape == shape:
            return a
        elif a.shape == (1, shape[1], shape[2]):
            return np.asarray([a[0]]*shape[0])
        elif a.shape == shape[1:]:
            return np.asarray([a]*shape[0])
        elif a.shape == ():
            return a * np.ones(shape)
        else:
            raise ValueError('could not expand array of shape %s to %s'
                             % (a.shape, shape))

    def _integrate(self, a):
        """Integrate an array over one year"""
        return np.sum(a, axis=0)/(self.interpolate_n-1)

    def _interpolate(self, a):
        """Interpolate an array through one year."""
        from scipy.interpolate import interp1d
        rule = self.interpolate_rule
        n = self.interpolate_n
        x = (np.arange(len(a)+2)-0.5) / len(a)
        y = np.vstack(([a[-1]], a, [a[0]]))
        newx = (np.arange(n)+0.5) / n  # change to 0.0 for PISM-like behaviour
        newy = interp1d(x, y, kind=rule, axis=0)(newx)
        return newy

    def inst_pdd(self, temp, stdv):
        """Compute instantaneous positive degree days from temperature.

        Use near-surface air temperature and standard deviation to compute
        instantaneous positive degree days (effective temperature for melt,
        unit degrees C) using an integral formulation (Calov and Greve, 2005).

        *temp*: array_like
            Near-surface air temperature in degrees Celcius.
        *stdv*: array_like
            Standard deviation of near-surface air temperature in Kelvin.
        """
        from scipy.special import erfc

        # compute positive part of temperature everywhere
        positivepart = np.greater(temp, 0)*temp

        # compute Calov and Greve (2005) integrand, ignoring division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            z = temp / (np.sqrt(2)*stdv)
        calovgreve = stdv/np.sqrt(2*np.pi)*np.exp(-z**2) + temp/2*erfc(-z)

        # use positive part where sigma is zero and Calov and Greve elsewhere
        teff = np.where(stdv == 0., positivepart, calovgreve)

        # convert to degree-days
        return teff*365.242198781

    def accu_rate(self, temp, prec):
        """Compute accumulation rate from temperature and precipitation.

        The fraction of precipitation that falls as snow decreases linearly
        from one to zero between temperature thresholds defined by the
        `temp_snow` and `temp_rain` attributes.

        *temp*: array_like
            Near-surface air temperature in degrees Celcius.
        *prec*: array_like
            Precipitation rate in meter per year.
        """

        # compute snow fraction as a function of temperature
        reduced_temp = (self.temp_rain-temp)/(self.temp_rain-self.temp_snow)
        snowfrac = np.clip(reduced_temp, 0, 1)

        # return accumulation rate
        return snowfrac*prec

    def melt_rates(self, snow, pdd):
        """Compute melt rates from snow precipitation and pdd sum.

        Snow melt is computed from the number of positive degree days (*pdd*)
        and the `pdd_factor_snow` model attribute. If all snow is melted and
        some energy (PDD) remains, ice melt is computed using `pdd_factor_ice`.

        *snow*: array_like
            Snow precipitation rate.
        *pdd*: array_like
            Number of positive degree days.
        """

        # parse model parameters for readability
        ddf_snow = self.pdd_factor_snow
        ddf_ice = self.pdd_factor_ice

        # compute a potential snow melt
        pot_snow_melt = ddf_snow * pdd

        # effective snow melt can't exceed amount of snow
        snow_melt = np.minimum(snow, pot_snow_melt)

        # ice melt is proportional to excess snow melt
        ice_melt = (pot_snow_melt - snow_melt) * ddf_ice/ddf_snow

        # return melt rates
        return (snow_melt, ice_melt)

    def nco(self, input_file, output_file,
            output_size='small', output_variables=None):
        """NetCDF operator.

        Read near-surface air temperature, precipitation rate, and standard
        deviation of near-surface air temperature from *input_file*, compute
        number of positive degree days and surface mass balance, and write
        results in *output_file*.

        *input_file*: str
            Name of input netCDF file. The input file should contain
            near-surface air temperature in variable 'temp', precipitation rate
            in variable 'prec', and optionally, standard deviation of
            near-surface air temperature in variable 'stdv'. If variable 'stdv'
            is not provided, and argument *stdv* is None, a constant value of
            zero is used.
        *outut_file*: str
            Name of output netCDF file.
        *output_size*: ['small', 'medium', 'big']
            Control which variables are written in the output file. If 'small',
            export only the number of positive degree days and total surface
            mass balance. If 'medium', export all cumulative (time-independent)
            variables. If 'big', output all cumulative and instantaneous
            (time-dependent) variables computed by the model.
        *output_variables*: list of str
            List of output variables to write in the output file. Prevails
            over any choice of *output_size*.
        """
        from netCDF4 import Dataset as NC

        # open netcdf files
        i = NC(input_file, 'r')
        o = NC(output_file, 'w', format='NETCDF3_CLASSIC')

        # read input temperature data
        try:
            temp = i.variables['temp'][:]
        except KeyError:
            raise KeyError('could not find input variable %s (%s) in file %s.'
                           % ('temp', names['temp']['long_name'], input_file))

        # read input precipitation data
        try:
            prec = i.variables['prec'][:]
        except KeyError:
            raise KeyError('could not find input variable %s (%s) in file %s.'
                           % ('prec', names['prec']['long_name'], input_file))

        # read input standard deviation, warn and use zero if absent
        try:
            stdv = i.variables['stdv'][:]
        except KeyError:
            import warnings
            warnings.warn('Variable stdv not found, assuming zero everywhere.')
            stdv = 0.0

        # convert to degC
        # TODO: handle unit conversion better
        if i.variables['temp'].units == 'K':
            temp = temp - 273.15

        # get dimensions tuple from temp variable
        txydim = i.variables['temp'].dimensions
        xydim = txydim[1:]

        # create dimensions
        o.createDimension(txydim[0], self.interpolate_n)
        for dimname in xydim:
            o.createDimension(dimname, len(i.dimensions[dimname]))

        # copy spatial coordinates
        for varname, ivar in i.variables.iteritems():
            if varname in xydim:
                ovar = o.createVariable(varname, ivar.dtype, ivar.dimensions)
                for attname in ivar.ncattrs():
                    setattr(ovar, attname, getattr(ivar, attname))
                ovar[:] = ivar[:]

        # create time coordinate
        var = _create_nc_variable(o, 'time', 'f4', ('time',))
        var[:] = (np.arange(self.interpolate_n)+0.5) / self.interpolate_n

        # run PDD model
        smb = self(temp, prec, stdv=stdv)

        # if output_variables was not defined, use output_size
        if output_variables is None:
            output_variables = ['pdd', 'smb']
            if output_size in ('medium', 'big'):
                output_variables += ['accu', 'snow_melt', 'ice_melt', 'melt',
                                     'runoff']
            if output_size in ('big'):
                output_variables += ['temp', 'prec', 'stdv', 'inst_pdd',
                                     'accu_rate', 'snow_melt_rate',
                                     'ice_melt_rate', 'melt_rate',
                                     'runoff_rate', 'inst_smb', 'snow_depth']

        # write output variables
        for varname in output_variables:
            if varname not in smb:
                raise KeyError("%s is not a valid variable name" % varname)
            dim = (txydim if smb[varname].ndim == 3 else xydim)
            var = _create_nc_variable(o, varname, 'f4', dim)
            var[:] = smb[varname]

        # close netcdf files
        i.close()
        o.close()


# Command-line interface
# ----------------------

def make_fake_climate(filename):
    """Create an artificial temperature and precipitation file.

    This function is used if pypdd.py is called as a script without an input
    file. The file produced contains an idealized, three-dimensional (t, x, y)
    distribution of near-surface air temperature, precipitation rate and
    standard deviation of near-surface air temperature to be read by
    `PDDModel.nco`.

    filename: str
        Name of output file.
    """
    from netCDF4 import Dataset as NC

    # open netcdf file
    nc = NC(filename, 'w')

    # create dimensions
    tdim = nc.createDimension('time', 12)
    xdim = nc.createDimension('x', 201)
    ydim = nc.createDimension('y', 201)
    nc.createDimension('nv', 2)

    # create coordinates and time bounds
    xvar = _create_nc_variable(nc, 'x', 'f4', ('x',))
    yvar = _create_nc_variable(nc, 'y', 'f4', ('y',))
    tvar = _create_nc_variable(nc, 'time', 'f4', ('time',))
    tboundsvar = _create_nc_variable(nc, 'time_bounds', 'f4', ('time', 'nv'))

    # create temperature and precipitation variables
    for varname in ['temp', 'prec', 'stdv']:
        _create_nc_variable(nc, varname, 'f4', ('time', 'x', 'y'))

    # assign coordinate values
    lx = ly = 750000
    xvar[:] = np.linspace(-lx, lx, len(xdim))
    yvar[:] = np.linspace(-ly, ly, len(ydim))
    tvar[:] = (np.arange(12)+0.5) / 12
    tboundsvar[:, 0] = tvar[:] - 1.0/24
    tboundsvar[:, 1] = tvar[:] + 1.0/24

    # assign temperature and precipitation values
    (xx, yy) = np.meshgrid(xvar[:], yvar[:])
    for i in range(len(tdim)):
        nc.variables['temp'][i] = -10 * yy/ly - 5 * np.cos(i*2*np.pi/12)
        nc.variables['prec'][i] = xx/lx * (np.sign(xx) - np.cos(i*2*np.pi/12))
        nc.variables['stdv'][i] = (2+xx/lx-yy/ly) * (1-np.cos(i*2*np.pi/12))

    # close netcdf file
    nc.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='A Python Positive Degree Day (PDD) model '
                    'for glacier surface mass balance.')
    parser.add_argument('-i', '--input', metavar='input.nc',
                        help='name of netCDF input file containing '
                             'air temperature (temp), precipitation (prec), '
                             '[and temperature standard deviation (stdv)]')
    parser.add_argument('-o', '--output', metavar='output.nc',
                        help='name of netCDF output file (default smb.nc)',
                        default='smb.nc')
    parser.add_argument('-l', '--list-variables',
                        help='list possible output variables and exit',
                        action='store_true')
    parser.add_argument('-s', '--output-size', metavar='SIZE',
                        help='size of output file, '
                             'ignored if -v/--output-variables is set '
                             '(default small; other choices medium, big)',
                        choices=('small', 'medium', 'big'), default='small')
    parser.add_argument('-v', '--output-variables', metavar='VAR', nargs='+',
                        help='output variables (use -l to list choices)',
                        choices=names.keys())
    parser.add_argument('--pdd-factor-snow', metavar='FS', type=float,
                        help='positive degree-day factor for snow '
                             '(default %s)' % defaults['pdd_factor_snow'],
                        default=defaults['pdd_factor_snow'])
    parser.add_argument('--pdd-factor-ice', metavar='FI', type=float,
                        help='positive degree-day factor for ice '
                             '(default %s)' % defaults['pdd_factor_ice'],
                        default=defaults['pdd_factor_ice'])
    parser.add_argument('--refreeze-snow', metavar='RS', type=float,
                        help='refreezing fraction of melted snow '
                             '(default %s)' % defaults['refreeze_snow'],
                        default=defaults['refreeze_snow'])
    parser.add_argument('--refreeze-ice', metavar='RI', type=float,
                        help='refreezing fraction of melted ice '
                             '(default %s)' % defaults['refreeze_ice'],
                        default=defaults['refreeze_ice'])
    parser.add_argument('--temp-snow', metavar='TS', type=float,
                        help='temperature at which all precip is snow '
                             '(default %s)' % defaults['temp_snow'],
                        default=defaults['temp_snow'])
    parser.add_argument('--temp-rain', metavar='TI', type=float,
                        help='temperature at which all precip is rain '
                             '(default %s)' % defaults['temp_rain'],
                        default=defaults['temp_rain'])
    parser.add_argument('--interpolate-rule', metavar='R',
                        help='rule used for time interpolations '
                             '(default %s)' % defaults['interpolate_rule'],
                        default=defaults['interpolate_rule'],
                        choices=('linear', 'nearest', 'zero', 'slinear',
                                 'quadratic', 'cubic'))
    parser.add_argument('--interpolate-n', type=int, metavar='N',
                        help='number of points used in interpolations '
                             '(default %s)' % defaults['interpolate_n'],
                        default=defaults['interpolate_n'])

    args = parser.parse_args()

    # if asked, list output variables and exit
    if args.list_variables:
        print 'currently available output variables:'
        for varname, vardict in sorted(names.iteritems()):
            if varname != 'time_bounds':
                print '  %-16s %s' % (varname, vardict['long_name'])
        import sys
        sys.exit()

    # if no input file was given, prepare a dummy one
    if not args.input:
        make_fake_climate('atm.nc')

    # initiate PDD model
    pdd = PDDModel(pdd_factor_snow=args.pdd_factor_snow,
                   pdd_factor_ice=args.pdd_factor_ice,
                   refreeze_snow=args.refreeze_snow,
                   refreeze_ice=args.refreeze_ice,
                   temp_snow=args.temp_snow,
                   temp_rain=args.temp_rain,
                   interpolate_rule=args.interpolate_rule,
                   interpolate_n=args.interpolate_n)

    # compute surface mass balance
    pdd.nco(args.input or 'atm.nc', args.output,
            output_size=args.output_size,
            output_variables=args.output_variables)
