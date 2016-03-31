# -*- coding: utf-8 -*-

# Copyright (c) 2016 Holger Nahrstaedt
# See COPYING for license details.

"""
Helper function for annotations
"""

from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

__all__ = ['gradient', 'despike_phasespace3d','excludeoutlier_ellipsoid3d']

def gradient(f):
    return np.concatenate((np.array([0]),np.diff(f)))

def excludeoutlier_ellipsoid3d(xi,yi,zi,theta):
    """
 This program excludes the points outside of ellipsoid in two-
 dimensional domain

 Input
   xi : input x data
   yi : input y data
   zi : input z data
   theta  : angle between xi and zi

 Output
   xp : excluded x data
   yp : excluded y data
   zp : excluded y data
   ip : excluded array element number in xi and yi
   coef : coefficients for ellipsoid

 Example: 
   [xp,yp,zp,ip,coef] = func_excludeoutlier_ellipsoid3d(f,f_t,f_tt,theta);


 Copyright: 

       Nobuhito Mori, Kyoto University
       Holger Nahrstaedt

    """
    n = np.max(xi.shape)
    _lambda = np.sqrt(2*np.log(n))

    xp = np.array([])
    yp = np.array([])
    zp = np.array([])
    ip = np.array([])

    #
    # --- rotate data
    #

    #theta = atan2( sum(xi.*zi), sum(xi.^2) );

    if theta == 0:
        X = xi
        Y = yi
        Z = zi
    else:
        R = np.zeros((3,3))
        R[0,:] = [np.cos(theta), 0,  np.sin(theta)]
        R[1,:] = [0,1,0]
        R[2,:] = [-np.sin(theta), 0, np.cos(theta)]
        X = xi*R[0,0] + yi*R[0,1] + zi*R[0,2]
        Y = xi*R[1,0] + yi*R[1,1] + zi*R[1,2]
        Z = xi*R[2,0] + yi*R[2,1] + zi*R[2,2]
    

    #test
    #plot3(xi,yi,zi,'b*')
    #hold on
    #  plot3(X,Y,Z,'r*')
    #hold off
    #pause

    #
    # --- preprocess
    #

    a = _lambda*np.nanstd(X)
    b = _lambda*np.nanstd(Y)
    c = _lambda*np.nanstd(Z)

    #
    # --- main
    #

    for i in np.arange(n):
        x1 = X[i]
        y1 = Y[i]
        z1 = Z[i]
        # point on the ellipsoid
        x2 = a*b*c*x1/np.sqrt((a*c*y1)**2+b**2*(c**2*x1**2+a**2*z1**2))
        y2 = a*b*c*y1/np.sqrt((a*c*y1)**2+b**2*(c**2*x1**2+a**2*z1**2))
        zt = c**2* ( 1 - (x2/a)**2 - (y2/b)**2 )
        if z1 < 0:
            z2 = -np.sqrt(zt)
        elif z1 > 0:
            z2 = np.sqrt(zt)
        else:
            z2 = 0

        # check outlier from ellipsoid
        dis = (x2**2+y2**2+z2**2) - (x1**2+y1**2+z1**2)
        if dis < 0:
            ip = np.append(ip,i)
            xp = np.append(xp,xi[i])
            yp = np.append(yp,yi[i])
            zp = np.append(zp,zi[i])


    coef = np.zeros(3)
    coef[0] = a
    coef[1] = b
    coef[2] = c
    return (xp,yp,zp,ip,coef)

def despike_phasespace3d( fi, i_plot = 0, i_opt=0 ):
    """
This subroutine excludes spike noise from Acoustic Doppler 
 Velocimetry (ADV) data using phase-space method, using 
 modified Goring and Nikora (2002) method by Nobuhito Mori (2005).
 Further modified by Joseph Ulanowski to remove offset in output (2014).
 

 Input
   fi     : input data with dimension (n,1)
   i_plot : =9 plot results (optional)
   i_opt : = 0 or not specified  ; return spike noise as NaN
           = 1            ; remove spike noise and variable becomes shorter than input length
           = 2            ; interpolate NaN using cubic polynomial

 Output
   fo     : output (filtered) data
   ip     : excluded array element number in fi

 Example: 
   [fo, ip] = func_despike_phasespace3d( fi, 9 );
     or
   [fo, ip] = func_despike_phasespace3d( fi, 9, 2 );



 Copyright:
       Holger Nahrstaedt - 2016
       Nobuhito Mori
           Disaster Prevention Research Institue
           Kyoto University
           mori@oceanwave.jp
"""
    #
    # --- initial setup
    #
    fi = fi.flatten()
    # number of maximum iternation
    n_iter = 20
    n_out  = 999

    n      = np.size(fi)
    f_mean = 0     # do not calculate f_mean here, as it will be affected by spikes (was: f_mean = nanmean(fi);)
    f      = fi    # this offset subtraction is unnecessary now (was: f = fi - f_mean;)
    _lambda = np.sqrt(2*np.log(n))
    #
    # --- loop
    #

    n_loop = 1

    while (n_out != 0) and (n_loop <= n_iter):

        #
        # --- main
        #

        # step 0
        f_mean=f_mean+np.nanmean(f) # accumulate offset value at each step [J.U.]
        f = f - np.nanmean(f)
        #nanstd(f)

        # step 1: first and second derivatives
        #f_t  = gradient(f);
        #f_tt = gradient(f_t);
        f_t  = gradient(f)
        f_tt = gradient(f_t)

        # step 2: estimate angle between f and f_tt axis
        if n_loop==1:
            theta = np.arctan2( np.sum(f*f_tt), np.sum(f**2) )
        

        # step 3: checking outlier in the 3D phase space
        [xp,yp,zp,ip,coef] = excludeoutlier_ellipsoid3d(f,f_t,f_tt,theta)

        #
        # --- excluding data
        #

        n_nan_1 = np.size(np.where(np.isnan(f)))
        f[ip.astype(np.int)]  = np.NAN
        n_nan_2 = np.size(np.where(np.isnan(f)))
        n_out   = n_nan_2 - n_nan_1;

        #
        # --- end of loop
        #

        n_loop = n_loop + 1;

        
    #
    # --- post process
    #

    go = f + f_mean;    # add offset back
    ip = np.where(np.isnan(go))[0]

    if n_loop < n_iter:
        print('>> Number of outlier   =  %d,  Number of iteration = %d'%(np.sum(np.isnan(f)),n_loop-1))
    else:
        print('>> Number of outlier   =  %d,  Number of iteration = %d !!! exceed maximum value !!!'%(np.sum(np.isnan(f)),n_loop-1))

    #
    # --- interpolation or shorten NaN data
    #
    
    if i_opt >= 1:
        # remove NaN from data
        inan = np.where(~np.isnan(go))[0]
        fo = go[inan]
        # interpolate NaN data
        if i_opt == 2:
            x   = np.where(~np.isnan(go))[0]
            y   = go[x]
            xi  = np.arange(np.size(fi))
            fo = interp1d(x, y, kind='cubic')(xi)
    else:
        # output despiked value as NaN
        fo = go
    if i_plot == 9:

        #theta/pi*180
        F    = fi - f_mean
        F_t  = gradient(F)
        F_tt = gradient(F_t)
        RF = np.zeros((3,3))
        RF[0,:] = [np.cos(theta), 0,  np.sin(theta)]
        RF[1,:] = [0,1,0]
        RF[2,:] = [-np.sin(theta), 0, np.cos(theta)]
        RB = np.zeros((3,3))
        RB[0,:] = [np.cos(theta), 0,  -np.sin(theta)]
        RB[1,:] = [0,1,0]
        RB[2,:] = [np.sin(theta), 0, np.cos(theta)]        


        # making ellipsoid data

        a = coef[0]
        b = coef[1]
        c = coef[2]
        ne  = 32;
        dt  = 2*np.pi/ne
        dp  = np.pi/ne
        t   = np.arange(0,2*np.pi,dt)
        p   = np.arange(0,2*np.pi,dp)
        n_t = np.size(t)
        n_p = np.size(p)

        # making ellipsoid
        xe = np.zeros(n_p*n_t+n_p)
        ye = np.zeros(n_p*n_t+n_p)
        ze = np.zeros(n_p*n_t+n_p)
        for it in np.arange(n_t):
            for _is in np.arange(n_p):
                xe[n_p*it+_is] = a*np.sin(p[_is])*np.cos(t[it])
                ye[n_p*it+_is] = b*np.sin(p[_is])*np.sin(t[it])
                ze[n_p*it+_is] = c*np.cos(p[_is])

        xer = xe*RB[0,0] + ye*RB[0,1] + ze*RB[0,2]
        yer = xe*RB[1,0] + ye*RB[1,1] + ze*RB[1,2]
        zer = xe*RB[2,0] + ye*RB[2,1] + ze*RB[2,2]
        
        # plot figures
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(f,f_t,f_tt,'b*',markersize=3)
        #hold on
        ax.plot(F[ip],F_t[ip],F_tt[ip],'ro',markerfacecolor='r',markersize=5)
        ax.plot(xer,yer,zer,'k-');

        plt.xlabel('u');
        plt.ylabel('\Delta u');
        #plt.zlabel('\Delta^2 u');


        fig2 = plt.figure()
        plt.plot(fi,'k-')

        plt.plot(ip,fi[ip],'ro')
        if i_opt==2:
            plt.plot(fo,'r-')
    return (fo, ip)
