#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2018 Mikołaj Magnuski
# <mmagnuski@swps.edu.pl>


import os
import os.path as op
from setuptools import setup


DISTNAME = 'borsar'
DESCRIPTION = "Tools for (mostly eeg) data analysis with mne-python."
MAINTAINER = u'Mikołaj Magnuski'
MAINTAINER_EMAIL = 'mmagnuski@swps.edu.pl'
URL = 'https://github.com/mmagnuski/borsar'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/mmagnuski/borsar'
VERSION = '0.1dev1'


def package_tree(pkgroot):
    """Get the submodule list."""
    # Adapted from VisPy
    path = op.dirname(__file__)
    subdirs = [op.relpath(i[0], path).replace(op.sep, '.')
               for i in os.walk(op.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return sorted(subdirs)


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
          packages=package_tree('borsar'),
          )
