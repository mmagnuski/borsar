#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2016 Mikołaj Magnuski
# <mmagnuski@swps.edu.pl>


import os
from setuptools import setup


DISTNAME = 'borsar'
DESCRIPTION = "Tools for (mostly eeg) data analysis with mne-python."
MAINTAINER = u'Mikołaj Magnuski'
MAINTAINER_EMAIL = 'mmagnuski@swps.edu.pl'
URL = 'https://github.com/mmagnuski/borsar'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/mmagnuski/mypy'
VERSION = '0.1dev1'


if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=['Intended Audience :: Science/Research',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS'],
          platforms='any',
          packages=['borsar'],
          )
