#!/usr/bin/env python

"""
MODULE:      r.in.pdd

AUTHOR(S):   Julien Seguinot

PURPOSE:     Positive Degree Day (PDD) model for glacier mass balance

COPYRIGHT:   (c) 2013-2014 Julien Seguinot

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

#%Module
#% description: Positive Degree Day (PDD) model for glacier mass balance
#% keywords: raster pdd
#%End

#%option
#% key: temp
#% type: string
#% gisprompt: old,cell,raster
#% description: Name of input temperature raster maps
#% required: yes
#% multiple: yes
#%end
#%option
#% key: prec
#% type: string
#% gisprompt: old,cell,raster
#% description: Name of input precipitation raster maps
#% required: yes
#% multiple: yes
#%end
#%option
#% key: stdv
#% type: string
#% gisprompt: old,cell,raster
#% description: Name of input precipitation raster maps
#% required: no
#% multiple: yes
#%end

#%option
#% key: pdd
#% type: string
#% gisprompt: new,cell,raster
#% description: Name for output number of positive degree days map
#% required: no
#%end
#%option
#% key: accu
#% type: string
#% gisprompt: new,cell,raster
#% description: Name for output ice-equivalent surface accumulation map
#% required: no
#%end
#%option
#% key: snow_melt
#% type: string
#% gisprompt: new,cell,raster
#% description: Name for output ice-equivalent surface melt of snow map
#% required: no
#%end
#%option
#% key: ice_melt
#% type: string
#% gisprompt: new,cell,raster
#% description: Name for output ice-equivalent surface melt of ice map
#% required: no
#%end
#%option
#% key: melt
#% type: string
#% gisprompt: new,cell,raster
#% description: Name for output ice-equivalent surface melt map
#% required: no
#%end
#%option
#% key: runoff
#% type: string
#% gisprompt: new,cell,raster
#% description: Name for output ice-equivalent surface meltwater runoff map
#% required: no
#%end
#%option
#% key: smb
#% type: string
#% gisprompt: new,cell,raster
#% description: Name for output ice-equivalent surface mass balance map
#% required: no
#%end

#%option
#% key: pdd_factor_snow
#% type: double
#% description: Positive degree-day factor for snow
#% required: no
#%end
#%option
#% key: pdd_factor_ice
#% type: double
#% description: Positive degree-day factor for ice
#% required: no
#%end
#%option
#% key: refreeze_snow
#% type: double
#% description: Refreezing fraction of melted snow
#% required: no
#%end
#%option
#% key: refreeze_ice
#% type: double
#% description: Refreezing fraction of melted ice
#% required: no
#%end
#%option
#% key: temp_snow
#% type: double
#% description: Temperature at which all precipitation falls as snow
#% required: no
#%end
#%option
#% key: temp_rain
#% type: double
#% description: Temperature at which all precipitation falls as rain
#% required: no
#%end
#%option
#% key: interpolate_rule
#% type: string
#% description: Rule used for interpolations
#% options: linear,nearest,zero,slinear,quadratic,cubic
#% required: no
#%end
#%option
#% key: interpolate_n
#% type: integer
#% description: Number of points used in interpolations
#% required: no
#%end

from grass.script import core as grass
from grass.script import array as garray
import numpy as np          # scientific module Numpy [1]
from pypdd import PDDModel  # positive degree day model PyPDD [2]


### Main function ###

def main():
    """main function, called at execution time"""

    # parse arguments
    temp_maps = options['temp'].split(',')
    prec_maps = options['prec'].split(',')
    stdv_maps = options['stdv'].split(',')

    # check that we have compatible number of input maps
    ntemp = len(temp_maps)
    nprec = len(prec_maps)
    nstdv = len(stdv_maps)
    if nprec not in (1, ntemp):
        grass.fatal('Got %i prec maps, expected 1 (constant) or %i (as temp)'
                    % (nprec, ntemp))
    if nstdv not in (1, ntemp):
        grass.fatal('Got %i stdv maps, expected 1 (constant) or %i (as temp)'
                    % (nstdv, ntemp))

    # read temperature maps
    grass.info('reading temperature maps...')
    temp = [garray.array()] * ntemp
    for i, m in enumerate(temp_maps):
        temp[i].read(m)
        grass.percent(i, ntemp, 1)
    temp = np.asarray(temp)

    # read precipitation maps
    grass.info('reading precipitation maps...')
    prec = [garray.array()] * nprec
    for i, m in enumerate(prec_maps):
        prec[i].read(m)
        grass.percent(i, nprec, 1)
    prec = np.asarray(prec)

    # read standard deviation maps
    if stdv_maps != ['']:
        grass.info('reading standard deviation maps...')
        stdv = [garray.array()] * nstdv
        for i, m in enumerate(stdv_maps):
            stdv[i].read(m)
            grass.percent(i, nstdv, 1)
        stdv = np.asarray(stdv)
    else:
        stdv = 0.0

    # initialize PDD model
    pdd = PDDModel()
    for param in ('pdd_factor_snow', 'pdd_factor_ice',
                  'refreeze_snow', 'refreeze_ice', 'temp_snow', 'temp_rain',
                  'interpolate_rule', 'interpolate_n'):
        if options[param]:
            setattr(pdd, param, float(options[param]))
    for param in ('interpolate_rule',):
        if options[param]:
            setattr(pdd, param, str(options[param]))

    # run PDD model
    grass.info('running PDD model...')
    smb = pdd(temp, prec, stdv)

    # write output maps
    grass.info('writing output maps...')
    for varname in ['pdd', 'accu', 'snow_melt', 'ice_melt', 'melt',
                    'runoff', 'smb']:
        if options[varname]:
            a = garray.array()
            a[:] = smb[varname]


### Main program ###

if __name__ == "__main__":
    options, flags = grass.parser()
    main()

# Links
# [1] http://numpy.scipy.org
# [2] http://github.com/jsegu/pypdd

