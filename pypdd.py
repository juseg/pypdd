#!/usr/bin/env python
# Copyright (c) 2013-2023, Julien Seguinot (juseg.dev)
# GNU General Public License v3.0+ (https://www.gnu.org/licenses/gpl-3.0.txt)

"""
A positive degree day model for glacier surface mass balance
"""

import numpy as np
import xarray as xr


# Default model parameters
# ------------------------

PARAMETERS = {
    'pdd_factor_snow':  0.003,
    'pdd_factor_ice':   0.008,
    'refreeze_snow':    0.0,
    'refreeze_ice':     0.0,
    'temp_snow':        0.0,
    'temp_rain':        2.0,
    'interpolate_rule': 'linear',
    'interpolate_n':    52}


# Default variable attributes
# ---------------------------

ATTRIBUTES = {

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


def _create_nc_variable(dataset, varname, dtype, dimensions):
    """Create netCDF variable and apply default attributes"""
    var = dataset.createVariable(varname, dtype, dimensions)
    for (attr, value) in ATTRIBUTES[varname].items():
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
                 pdd_factor_snow=PARAMETERS['pdd_factor_snow'],
                 pdd_factor_ice=PARAMETERS['pdd_factor_ice'],
                 refreeze_snow=PARAMETERS['refreeze_snow'],
                 refreeze_ice=PARAMETERS['refreeze_ice'],
                 temp_snow=PARAMETERS['temp_snow'],
                 temp_rain=PARAMETERS['temp_rain'],
                 interpolate_rule=PARAMETERS['interpolate_rule'],
                 interpolate_n=PARAMETERS['interpolate_n']):

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
        # FIXME use data arrays instead
        temp = np.asarray(temp)
        prec = np.asarray(prec)
        stdv = np.asarray(stdv)

        # expand arrays to the largest shape
        # FIXME use xarray auto-broadcasting instead
        maxshape = max(temp.shape, prec.shape, stdv.shape)
        temp = self._expand(temp, maxshape)
        prec = self._expand(prec, maxshape)
        stdv = self._expand(stdv, maxshape)

        # interpolate time-series
        # FIXME propagate data arrays, coordinates
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

        # make a dataset
        # FIXME add coordinate variables
        ds = xr.Dataset(
            data_vars={
                'temp': (['time', 'x', 'y'], temp),
                'prec': (['time', 'x', 'y'], prec),
                'stdv': (['time', 'x', 'y'], stdv),
                'inst_pdd': (['time', 'x', 'y'], inst_pdd),
                'accu_rate': (['time', 'x', 'y'], accu_rate),
                'snow_melt_rate': (['time', 'x', 'y'], snow_melt_rate),
                'ice_melt_rate': (['time', 'x', 'y'], ice_melt_rate),
                'melt_rate': (['time', 'x', 'y'], melt_rate),
                'runoff_rate': (['time', 'x', 'y'], runoff_rate),
                'inst_smb': (['time', 'x', 'y'], inst_smb),
                'snow_depth': (['time', 'x', 'y'], snow_depth),
                'pdd': (['x', 'y'], self._integrate(inst_pdd)),
                'accu': (['x', 'y'], self._integrate(accu_rate)),
                'snow_melt': (['x', 'y'], self._integrate(snow_melt_rate)),
                'ice_melt': (['x', 'y'], self._integrate(ice_melt_rate)),
                'melt': (['x', 'y'], self._integrate(melt_rate)),
                'runoff': (['x', 'y'], self._integrate(runoff_rate)),
                'smb': (['x', 'y'], self._integrate(inst_smb))})

        # return dataset
        return ds

    def _expand(self, array, shape):
        """Expand an array to the given shape"""
        if array.shape == shape:
            res = array
        elif array.shape == (1, shape[1], shape[2]):
            res = np.asarray([array[0]]*shape[0])
        elif array.shape == shape[1:]:
            res = np.asarray([array]*shape[0])
        elif array.shape == ():
            res = array * np.ones(shape)
        else:
            raise ValueError('could not expand array of shape %s to %s'
                             % (array.shape, shape))
        return res

    def _integrate(self, array):
        """Integrate an array over one year"""
        return np.sum(array, axis=0)/(self.interpolate_n-1)

    def _interpolate(self, array):
        """Interpolate an array through one year."""
        from scipy.interpolate import interp1d
        rule = self.interpolate_rule
        npts = self.interpolate_n
        oldx = (np.arange(len(array)+2)-0.5) / len(array)
        oldy = np.vstack(([array[-1]], array, [array[0]]))
        newx = (np.arange(npts)+0.5) / npts  # use 0.0 for PISM-like behaviour
        newy = interp1d(oldx, oldy, kind=rule, axis=0)(newx)
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
        import scipy.special as sp

        # compute positive part of temperature everywhere
        positivepart = np.greater(temp, 0)*temp

        # compute Calov and Greve (2005) integrand, ignoring division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            normtemp = temp / (np.sqrt(2)*stdv)
        calovgreve = (stdv/np.sqrt(2*np.pi)*np.exp(-normtemp**2) +
                      temp/2*sp.erfc(-normtemp))

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

        # open atmosphere file
        atm = xr.open_dataset(input_file)

        # check for missing variables (move this to __call__(self, ds))
        if 'temp' not in atm:
            raise KeyError('could not find input variable %s (%s) in file %s.'
                           % ('temp', ATTRIBUTES['temp']['long_name'], input_file))
        if 'prec' not in atm:
            raise KeyError('could not find input variable %s (%s) in file %s.'
                           % ('prec', ATTRIBUTES['prec']['long_name'], input_file))
        if 'stdv' not in atm:
            import warnings
            warnings.warn('Variable stdv not found, assuming zero everywhere.')

        # convert to degC
        # TODO: handle unit conversion better
        if atm.temp.units in ('K', 'Kelvin'):
            atm['temp'] -= 273.15

        # run PDD model
        smb = self(atm.temp, atm.prec, stdv=atm.get('stdv'))

        # if output_variables was not defined, use output_size
        if output_variables is None:
            output_variables = ['pdd', 'smb']
            if output_size in ('medium', 'big'):
                output_variables += ['accu', 'snow_melt', 'ice_melt', 'melt',
                                     'runoff']
            if output_size == 'big':
                output_variables += ['temp', 'prec', 'stdv', 'inst_pdd',
                                     'accu_rate', 'snow_melt_rate',
                                     'ice_melt_rate', 'melt_rate',
                                     'runoff_rate', 'inst_smb', 'snow_depth']

        # write netcdf file
        smb = smb[output_variables]
        smb.to_netcdf(output_file)


# Command-line interface
# ----------------------

def make_fake_climate(filename=None):
    """Create an artificial temperature and precipitation file.

    This function is used if pypdd.py is called as a script without an input
    file. The file produced contains an idealized, three-dimensional (t, x, y)
    distribution of near-surface air temperature, precipitation rate and
    standard deviation of near-surface air temperature to be read by
    `PDDModel.nco`.

    filename: str, optional
        Name of output file.
    """

    # FIXME code could be simplified a lot more but we need a better test not
    # relying on exact reproducibility of this toy climate data.

    # assign coordinate values
    lx = ly = 750000
    x = xr.DataArray(np.linspace(-lx, lx, 201, dtype='f4'), dims='x')
    y = xr.DataArray(np.linspace(-ly, ly, 201, dtype='f4'), dims='y')
    time = xr.DataArray((np.arange(12, dtype='f4')+0.5) / 12, dims='time')
    tboundsvar = np.empty((12, 2), dtype='f4')
    tboundsvar[:, 0] = time[:] - 1.0/24
    tboundsvar[:, 1] = time[:] + 1.0/24

    # seasonality index from winter to summer
    season = xr.DataArray(-np.cos(np.arange(12)*2*np.pi/12), dims='time')

    # order of operation is dictated by test md5sum and legacy f4 dtype
    temp = 5 * season - 10 * x / lx + 0 * y
    prec = y / ly * (season.astype('f4') + 0 * x + np.sign(y))
    stdv = (2+y/ly-x/lx) * (1+season)

    # this is also why transpose is needed here, and final type conversion
    temp = temp.transpose('time', 'x', 'y').astype('f4')
    prec = prec.transpose('time', 'x', 'y').astype('f4')
    stdv = stdv.transpose('time', 'x', 'y').astype('f4')

    # assign variable attributes
    temp.attrs.update(ATTRIBUTES['temp'])
    prec.attrs.update(ATTRIBUTES['prec'])
    stdv.attrs.update(ATTRIBUTES['stdv'])

    # make a dataset
    ds = xr.Dataset(
        data_vars={'temp': temp, 'prec': prec, 'stdv': stdv},
        coords={
            'time': time, 'x': x, 'y': y,
            'time_bounds': (['time', 'nv'], tboundsvar[:]),
        },
    )

    # write dataset to file
    if filename is not None:
        ds.to_netcdf(filename)

    # return dataset
    return ds


def main():
    """Main program for command-line execution."""

    import argparse

    # parse arguments
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
                        choices=ATTRIBUTES.keys())
    parser.add_argument('--pdd-factor-snow', metavar='FS', type=float,
                        help='positive degree-day factor for snow '
                             '(default %s)' % PARAMETERS['pdd_factor_snow'],
                        default=PARAMETERS['pdd_factor_snow'])
    parser.add_argument('--pdd-factor-ice', metavar='FI', type=float,
                        help='positive degree-day factor for ice '
                             '(default %s)' % PARAMETERS['pdd_factor_ice'],
                        default=PARAMETERS['pdd_factor_ice'])
    parser.add_argument('--refreeze-snow', metavar='RS', type=float,
                        help='refreezing fraction of melted snow '
                             '(default %s)' % PARAMETERS['refreeze_snow'],
                        default=PARAMETERS['refreeze_snow'])
    parser.add_argument('--refreeze-ice', metavar='RI', type=float,
                        help='refreezing fraction of melted ice '
                             '(default %s)' % PARAMETERS['refreeze_ice'],
                        default=PARAMETERS['refreeze_ice'])
    parser.add_argument('--temp-snow', metavar='TS', type=float,
                        help='temperature at which all precip is snow '
                             '(default %s)' % PARAMETERS['temp_snow'],
                        default=PARAMETERS['temp_snow'])
    parser.add_argument('--temp-rain', metavar='TI', type=float,
                        help='temperature at which all precip is rain '
                             '(default %s)' % PARAMETERS['temp_rain'],
                        default=PARAMETERS['temp_rain'])
    parser.add_argument('--interpolate-rule', metavar='R',
                        help='rule used for time interpolations '
                             '(default %s)' % PARAMETERS['interpolate_rule'],
                        default=PARAMETERS['interpolate_rule'],
                        choices=('linear', 'nearest', 'zero', 'slinear',
                                 'quadratic', 'cubic'))
    parser.add_argument('--interpolate-n', type=int, metavar='N',
                        help='number of points used in interpolations '
                             '(default %s)' % PARAMETERS['interpolate_n'],
                        default=PARAMETERS['interpolate_n'])
    args = parser.parse_args()

    # if asked, list output variables and exit
    if args.list_variables:
        print('currently available output variables:')
        for varname, vardict in sorted(ATTRIBUTES.items()):
            if varname != 'time_bounds':
                print('  %-16s %s' % (varname, vardict['long_name']))
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


def test_pdd():

    import hashlib

    # compute smb from fake climate
    ds = make_fake_climate()
    pdd = PDDModel()
    smb = pdd(ds.temp, ds.prec, ds.stdv)

    # check md5 sums against v0.3.0
    hashes = {
        'pdd': 'c314959f12e41fd6c68ea619da71d000',
        'smb': '631c50ad64f268f82d530cc25e764c74'}
    for name, hash in hashes.items():
        var = smb[name].data.astype('f4')
        assert hashlib.md5(var).hexdigest() == hash


def test_nco():

    import hashlib

    # compute smb from fake climate
    make_fake_climate('atm.nc')
    pdd = PDDModel()
    pdd.nco('atm.nc', 'smb.nc')
    smb = xr.open_dataset('smb.nc')

    # check md5 sums against v0.3.0
    hashes = {
        'pdd': 'c314959f12e41fd6c68ea619da71d000',
        'smb': '631c50ad64f268f82d530cc25e764c74'}
    for name, hash in hashes.items():
        var = smb[name].data.astype('f4')
        assert hashlib.md5(var).hexdigest() == hash


if __name__ == '__main__':
    main()
