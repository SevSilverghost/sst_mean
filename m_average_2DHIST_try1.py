#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:56:22 2019

@author: andrei
"""

import netCDF4 as netCDF
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import seaborn
import time
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%%

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

#%%

data_file = netCDF.Dataset('./2016/05/A20160503_113500.L2_LAC_SST.nc')

#%%

#get the data
sst_data = data_file.groups['geophysical_data'].variables['sst']
long_data = data_file.groups['navigation_data'].variables['longitude']
lat_data = data_file.groups['navigation_data'].variables['latitude']
mask_data = data_file.groups['geophysical_data'].variables['flags_sst']

#tmp empty vars
LON_subset = []
LAT_subset = []
DATA_subset = []
LINES_subset = []

#select the data in the region
for line in np.arange(sst_data.shape[0]):

    LONindices = (long_data[line,:] > long_min) & (long_data[line,:] < long_max)
    LATindices = (lat_data[line,:] > lat_min) & (lat_data[line,:] < lat_max)
    DATAindices = np.logical_and(LONindices,LATindices)
    DATAindices = np.logical_and(DATAindices,mask_data[line].mask)
    if np.any(DATAindices):
        LON_subset.append(np.array(long_data[line,DATAindices],dtype=np.float64))
        LAT_subset.append(np.array(lat_data[line,DATAindices],dtype=np.float64))
        DATA_subset.append(sst_data[line,DATAindices])
        LINES_subset.append(int(line))
#---
#prepare data containers
AVG = ma.zeros(shape=(Xbins.shape[0]-1,Ybins.shape[0]-1))

#...

#main cycle of averaging to bins
for i,x in enumerate(Xbins[:-1]):
    for j,y in enumerate(Ybins[:-1]):

        SUM = []
        for line in np.arange(len(LINES_subset)):
            LONindices = (LON_subset[line] > x) & (LON_subset[line] <= Xbins[i+1])
            LATindices = (LAT_subset[line] > y) & (LAT_subset[line] <= Ybins[j+1])
            DATAindices = np.logical_and(LONindices,LATindices)

            if np.any(DATAindices):
                local_mean = ma.mean(DATA_subset[line][DATAindices])
                if ma.is_masked(local_mean):
                    pass
                else:
                    SUM.append(float(local_mean))
        if len(SUM)==0:
            AVG[i,j] = 1e+20
        else:
            AVG[i,j] = np.mean(SUM)




#...
#set mask
AVG = ma.masked_where(AVG>=1e+19,AVG)
#XC = ma.masked_where(AVG>=1e+19,XC)
#YC = ma.masked_where(AVG>=1e+19,YC)
#---



#prepare current figure
plt.close('all')
plt.figure(figsize=(8,10),dpi=defdpi)
ax = plt.gca()

#prepare map
mymap.drawcoastlines(ax=ax,zorder=500)
#mymap.fillcontinents('0.9',ax=ax,zorder=499)

mymap.drawparallels(pars, dashes=(1, 1), 
                        linewidth=0.15, labels=parslabels, ax=ax,zorder=501)
mymap.drawmeridians(mers, dashes=(1, 1), 
                        linewidth=0.15, labels=merslabels, ax=ax,zorder=501)

#do plot   
XMAP,YMAP = mymap(XC,YC)
im = mymap.pcolormesh(XMAP,YMAP,AVG,cmap='jet',vmin=np.min(AVG), vmax=np.max(AVG))

#plot makeup
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)
#save figure
plt.savefig('dummy.png',dpi=defdpi)

#save data to temp var
#month_mean_T.append(ma.average(AVG,weights=WGHTS))
#month_AVG.append(AVG)

#write space-averaged data to file
tmp_average = ma.average(AVG,weights=WGHTS)
tmp_variance = ma.average((AVG-tmp_average)**2, weights=WGHTS)
tmp_std = ma.sqrt(tmp_variance)
#mean_file.write(filename+','+str(tmp_average)+','+str(tmp_std)+'\n')

#%%
plt.style.use('classic')
seaborn.set()
#%%
FLAT_LON_subset = []
for item in LON_subset:
    FLAT_LON_subset.extend(item)
    
FLAT_LAT_subset = []
for item in LAT_subset:
    FLAT_LAT_subset.extend(item)
#
#print LON_subset.shape

plt.close('all')
plt.figure(figsize=(8,10),dpi=defdpi)
ax = plt.gca()

#prepare map
mymap.drawcoastlines(ax=ax,zorder=500)
#mymap.fillcontinents('0.9',ax=ax,zorder=499)

mymap.drawparallels(pars, dashes=(1, 1), 
                        linewidth=0.15, labels=parslabels, ax=ax,zorder=501)
mymap.drawmeridians(mers, dashes=(1, 1), 
                        linewidth=0.15, labels=merslabels, ax=ax,zorder=501)
#do plot   
#XMAP,YMAP = mymap(XC,YC)
#matplotlib.pyplot.hist2d(x, y, bins=10, range=None, normed=False, 
#weights=None, cmin=None, cmax=None, *, data=None, **kwargs)

#h = plt.hist2d(FLAT_LON_subset,FLAT_LAT_subset,bins=[Xbins,Ybins],cmap='jet')
counts, xedges, yedges = np.histogram2d(FLAT_LON_subset,FLAT_LAT_subset,bins=[Xbins,Ybins])

XMAP,YMAP = mymap(XC,YC)
im = mymap.pcolormesh(XMAP,YMAP,counts,cmap='Greens')


#plot makeup
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
#
plt.colorbar(im, cax=cax,label='points count')
#save figure
plt.savefig('2dhist.png',dpi=defdpi)
plt.close()