#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 02.10.2018

Purpose: 1) to read a set of NETCDF files with SST data from one folder,
         presumed to be for one month.
         2) to average each file to bins
         3) to mask lagoons
         4) to make time-averaging of space-averaged fields -> one value
         5) to make time-averaged field for the month
         

@author: andrei
"""

import netCDF4 as netCDF
import numpy as np
import os
#import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
#import pandas as pd
import numpy.ma as ma
#from mpl_toolkits.axes_grid1 import make_axes_locatable
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
defdpi = 100
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
path = './05_2016/'
#%%

#preparation of the folders and files for output
#if not os.path.exists(path+'FIGURES'):
#    os.makedirs(path+'FIGURES')

if not os.path.exists(path+'DATA'):
    os.makedirs(path+'DATA')
    
mean_file_v = open(path+'DATA/mean_SST_Vistula.dat','w')
mean_file_c = open(path+'DATA/mean_SST_Curonian.dat','w')

#%%

#empty containers for temporary values
#month_mean_T = []
month_AVG_v = []
month_AVG_c = []



#main cycle
for filename in os.listdir(path):
    if filename.endswith(".nc"): 
        print(os.path.join(path, filename))
        #start = time.time()
        #---
        #open file
        data_file = netCDF.Dataset(path+filename)
        
        #---
        #get the data
        sst_data = data_file.groups['geophysical_data'].variables['sst']
        long_data = data_file.groups['navigation_data'].variables['longitude']
        lat_data = data_file.groups['navigation_data'].variables['latitude']
        mask_data = data_file.groups['geophysical_data'].variables['flags_sst']
        
        #
        
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
        
        
        '''
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
        #plt.savefig(path+'./FIGURES/'+filename[:-2]+'png',dpi=defdpi)
        plt.show()
        '''

        


        #
        MMM = AVG.mask
        invMMM_v = ~np.array(MMM)
        invMMM_c = ~np.array(MMM)
        #
        invMMM_v[26,6] = False
        invMMM_v[25:30,7] = False
        invMMM_v[27:33,8] = False
        invMMM_v[30:34,9] = False
        invMMM_v[32:35,10] = False
        invMMM_v[34:38,11] = False
        invMMM_v[35:39,12] = False
        invMMM_v[36:42,13] = False
        invMMM_v[37:44,14] = False
        invMMM_v[39:41,15] = False
        invMMM_v[42:46,15] = False
        invMMM_v[39,16] = False
        #
        invMMM_c[49:65,21] = False
        invMMM_c[49:65,22] = False
        invMMM_c[49:65,23] = False
        invMMM_c[50:65,24] = False
        invMMM_c[52:65,25] = False
        invMMM_c[54:65,26] = False
        invMMM_c[55:65,27] = False
        invMMM_c[56:65,28] = False
        invMMM_c[57:65,29] = False
        invMMM_c[58:65,30] = False
        invMMM_c[59:65,31] = False
        invMMM_c[60:65,32] = False
        
        resMMM_v = np.logical_or(MMM,invMMM_v)
        resMMM_c = np.logical_or(MMM,invMMM_c)
        
        #
        newAVG_v = np.ma.masked_where(resMMM_v, AVG)
        newAVG_c = np.ma.masked_where(resMMM_c, AVG)
        
        month_AVG_v.append(newAVG_v)
        month_AVG_c.append(newAVG_c)
        

        #save data
        curr_monthly_mean_T_v = ma.average(newAVG_v,weights=WGHTS)
        curr_monthly_mean_T_c = ma.average(newAVG_c,weights=WGHTS)
        if ma.is_masked(curr_monthly_mean_T_v):
            #month_mean_T.append(np.NaN)
            mean_file_v.write(filename+',--,--\n')
        else:
            #print curr_monthly_mean_T_v
            #month_mean_T.append(curr_monthly_mean_T)
            
            tmp_variance = ma.average((newAVG_v-curr_monthly_mean_T_v)**2, weights=WGHTS)
            tmp_std = ma.sqrt(tmp_variance)
            
            #write space-averaged data to file
            mean_file_v.write(filename+','+str(curr_monthly_mean_T_v)+','+str(tmp_std)+'\n')
        
        if ma.is_masked(curr_monthly_mean_T_c):
            #month_mean_T.append(np.NaN)
            mean_file_c.write(filename+',--,--\n')
        else:
            #print curr_monthly_mean_T_v
            tmp_variance = ma.average((newAVG_c-curr_monthly_mean_T_c)**2, weights=WGHTS)
            tmp_std = ma.sqrt(tmp_variance)
            #write space-averaged data to file
            #print tmp_variance,tmp_std
            mean_file_c.write(filename+','+str(curr_monthly_mean_T_c)+','+str(tmp_std)+'\n')
        
        #
        
        
#write monthly-averaged data to file
MMEAN_v = ma.average(month_AVG_v,axis=0)
MMEAN_c = ma.average(month_AVG_c,axis=0)
#print 'Monthly average for Vistula lagoon',ma.average(MMEAN)
#tmp_average = ma.average(AVG,weights=np.cos(YC))
#tmp_variance = np.average((AVG-tmp_average)**2, weights=np.cos(YC))
#tmp_std = np.sqrt(tmp_variance)
#mean_file.write(filename+','+str(tmp_average)+','+str(tmp_std)+'\n')
mean_file_v.write('Monthly average for Vistula lagoon,'+str(ma.average(MMEAN_v))+','+str(ma.std(MMEAN_v)))
mean_file_c.write('Monthly average for Curonian lagoon,'+str(ma.average(MMEAN_c))+','+str(ma.std(MMEAN_c)))


#%%
#prepare figure
'''
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
im = mymap.pcolormesh(XMAP,YMAP,MMEAN,cmap='jet',vmin=np.min(AVG), vmax=np.max(AVG))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)

plt.show()
#plt.savefig(path+'/FIGURES/monthly_mean_SST.png',dpi=defdpi)
'''

#%%
#close data file
mean_file_v.close()
mean_file_c.close()

#%%
#save array with monthly-averaged SST
#(binary file to be read from python script)
#MMEAN.dump(path+'DATA/monthly_mean_'+path[2:-1]+'.dat')

#%%