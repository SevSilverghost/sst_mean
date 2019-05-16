#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Jan 20 2019

Purpose: 1) to read data from dumped masked arrays (averaged SST fields).
         2) to make 2013-2017 mean for each month (image)
         3) to make mean 2013-2017 for each year (image)
         4) to make 2013-2017 mean (image)

@author: andrei
"""

#import netCDF4 as netCDF
import calendar
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
#import pandas as pd
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%

#set the limits of the polygon
long_min = 18.0
long_max = 22.0
lat_min = 54.0
lat_max = 56.5

#set bins
Xbins = np.linspace(long_min,long_max,80)
Ybins = np.linspace(lat_min,lat_max,60)


XC = np.zeros(shape=(Xbins.shape[0]-1,Ybins.shape[0]-1))
YC = np.zeros(shape=(Xbins.shape[0]-1,Ybins.shape[0]-1))
WGHTS = np.cos(np.radians(YC))


#calc bins centers
for i,x in enumerate(Xbins[:-1]):
    for j,y in enumerate(Ybins[:-1]):
        XC[i,j] = (x+Xbins[i+1])/2.0
        YC[i,j] = (y+Ybins[j+1])/2.0

#...

#-------------
#set the resolution for output images
defdpi = 300
#-------------

#makeup for the maps
pars=np.arange(lat_min,lat_max+1.0,1.)
mers=np.arange(long_min, long_max+1.0,1.)
col='grey'
outline=[1, 1, 0, 1]
merslabels=[0, 0, 0, 1]
parslabels=[1, 0, 0, 0]

#map instance
mymap = Basemap(llcrnrlon=long_min,
                 llcrnrlat=lat_min,
                 urcrnrlon=long_max,
                 urcrnrlat=lat_max,
                 projection='tmerc',
                 lat_0 = 55.0,
                 lon_0 = 19.0,
                 resolution='i',
                 area_thresh=0)

#path to the files with the data
path = './'
#%%

#preparation of the folders and files for output
if not os.path.exists(path+'FIGURES'):
    os.makedirs(path+'FIGURES')

#if not os.path.exists(path+'DATA'):
#    os.makedirs(path+'DATA')
    
#mean_file = open(path+'DATA/mean_SST.dat','w')
#%%

#%%
for MONTH in ['01','02','03','04','05','06','07','08','09','10','11','12']:
    MMEAN = []
    for YEAR in np.arange(2003,2017+1,1):
        print YEAR
        if os.path.exists(path+'/'+str(YEAR)+'/monthly_mean_'+MONTH+'.dat'):
            print MONTH
            fname = open(path+'/'+str(YEAR)+'/monthly_mean_'+MONTH+'.dat','rb')
            dop = ma.load(fname)
            MMEAN.append(dop)
        
    if len(MMEAN)>0:
        MMEAN = ma.average(MMEAN,axis=0)
    #prepare figure
        plt.close('all')
        plt.figure(figsize=(8,10),dpi=defdpi)
        ax = plt.gca()
        
        # Do plot   
        mymap.drawcoastlines(ax=ax,zorder=500)
        #mymap.fillcontinents('0.9',ax=ax,zorder=499)
        
        mymap.drawparallels(pars, dashes=(1, 1), 
                        linewidth=0.15, labels=parslabels, ax=ax,zorder=501)
        mymap.drawmeridians(mers, dashes=(1, 1), 
                        linewidth=0.15, labels=merslabels, ax=ax,zorder=501)
        
        #do plot
        XMAP,YMAP = mymap(XC,YC)
        im = mymap.pcolormesh(XMAP,YMAP,MMEAN,cmap='jet',vmin=1, vmax=20)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        plt.colorbar(im, cax=cax)
        ax.set_title('SST 2003-2017 mean for '+calendar.month_name[int(MONTH)])
        
        plt.savefig(path+'/FIGURES/'+MONTH+'mean_SST_'+calendar.month_name[int(MONTH)]+'.png',dpi=defdpi)
#%%        
        
TOTAL_MEAN = []        
for YEAR in np.arange(2003,2017+1,1):
    MMEAN = []
    for MONTH in ['01','02','03','04','05','06','07','08','09','10','11','12']:
        print YEAR
        if os.path.exists(path+'/'+str(YEAR)+'/monthly_mean_'+MONTH+'.dat'):
            print MONTH
            fname = open(path+'/'+str(YEAR)+'/monthly_mean_'+MONTH+'.dat','rb')
            dop = ma.load(fname)
            MMEAN.append(dop)
        
    if len(MMEAN)>0:
        MMEAN = ma.average(MMEAN,axis=0)
        TOTAL_MEAN.append(MMEAN)
        #prepare figure
        plt.close('all')
        plt.figure(figsize=(8,10),dpi=defdpi)
        ax = plt.gca()
        
        # Do plot   
        mymap.drawcoastlines(ax=ax,zorder=500)
        #mymap.fillcontinents('0.9',ax=ax,zorder=499)
        
        mymap.drawparallels(pars, dashes=(1, 1), 
                        linewidth=0.15, labels=parslabels, ax=ax,zorder=501)
        mymap.drawmeridians(mers, dashes=(1, 1), 
                        linewidth=0.15, labels=merslabels, ax=ax,zorder=501)
        
        #do plot
        XMAP,YMAP = mymap(XC,YC)
        im = mymap.pcolormesh(XMAP,YMAP,MMEAN,cmap='jet',vmin=5, vmax=15)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        plt.colorbar(im, cax=cax)
        
        ax.set_title('SST mean for '+str(YEAR))
        plt.savefig(path+'/FIGURES/'+'mean_SST_'+str(YEAR))
        
#%%

TOTAL_MEAN = ma.average(TOTAL_MEAN,axis=0)
        
plt.close('all')
plt.figure(figsize=(8,10),dpi=defdpi)
ax = plt.gca()

# Do plot   
mymap.drawcoastlines(ax=ax,zorder=500)
#mymap.fillcontinents('0.9',ax=ax,zorder=499)

mymap.drawparallels(pars, dashes=(1, 1), 
            linewidth=0.15, labels=parslabels, ax=ax,zorder=501)
mymap.drawmeridians(mers, dashes=(1, 1), 
            linewidth=0.15, labels=merslabels, ax=ax,zorder=501)

#do plot
XMAP,YMAP = mymap(XC,YC)
im = mymap.pcolormesh(XMAP,YMAP,TOTAL_MEAN,cmap='jet',vmin=10, vmax=15)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)

ax.set_title('TOTAL SST mean for 2003-2017')
plt.savefig(path+'/FIGURES/'+'mean_SST_TOTAL')