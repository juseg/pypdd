#!/usr/bin/env python
# Copyright (c) 2018, Julien Seguinot <seguinot@vaw.baug.ethz.ch>
# GNU General Public License v3.0+ (https://www.gnu.org/licenses/gpl-3.0.txt)

"""Build script for pypdd."""

import setuptools

with open('README.rst', 'r') as f:
    README = f.read()

setuptools.setup(
    name='pypdd',
    version='0.3.0',
    author='Julien Seguinot',
    author_email='seguinot@vaw.baug.ethz.ch',
    description='A positive degree day model for glacier surface mass balance',
    long_description=README,
    long_description_content_type='text/x-rst',
    url='http://github.com/juseg/aftershocks',
    license='gpl-3.0',
    install_requires=['numpy', 'scipy'],
    extras_require = {'NetCDF interface': 'netCDF4'},
    py_modules=['pypdd'],
    scripts=['pypdd.py'],
)
