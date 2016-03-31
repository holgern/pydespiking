#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pydespiking

x = np.arange(0,10*np.pi,0.1)
fi = np.sin(x)
fi[99] = 100
[fo, ip] = pydespiking.despike_phasespace3d( fi, 9 )
print("- plot process and length(fo)=length(fi)   fo contains NaN")
[fo, ip] = pydespiking.despike_phasespace3d( fi, 9, 1 )
print(" plot process and length(fo)<length(fi)   NaN is excluded from fo")
[fo, ip] = pydespiking.despike_phasespace3d( fi, 9, 2 )
print(" plot process and length(fo)=length(fi)    NaN in fo is interpolated")
