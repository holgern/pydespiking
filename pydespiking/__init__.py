# -*- coding: utf-8 -*-

# Copyright (c) 2015 Holger Nahrstaedt
# See COPYING for license details.


"""
despiking
"""

from __future__ import division, print_function, absolute_import


from .phasespace import *

from pydespiking.version import version as __version__

from numpy.testing import Tester

__all__ = [s for s in dir() if not s.startswith('_')]
test = Tester().test