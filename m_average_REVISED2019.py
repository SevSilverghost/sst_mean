#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:27:13 2018
Modified on Feb 27 2019
Modified on March 04 2019


Purpose: 1) to read a set of NETCDF files with SST data from all folders
         2) to average each file to bins and create figure -> many figures
         3) to make time-averaging of space-averaged fields -> one value
         4) to make time-averaged field for the month -> one figure & 1 2-d array
         5) to output averaged scalars to file -> one file with many scalars
         6) extra_1: calculate, plot and dump 2dhist ('counts' as array) of
            data availability by bins: monthly, yearly and total
         7) 

@author: andrei
"""

import netCDF4 as netCDF
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
#import pandas as pd
#import seaborn
import time
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%
#seaborn.set()
start = time.time()
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
#%%


for Y_dirname in os.listdir('./'):
    if os.path.isdir(Y_dirname) and Y_dirname[0]!='.':
        #print(Y_dirname)
        if os.listdir(Y_dirname):
            print Y_dirname+'/'
            
            for M_dirname in os.listdir(Y_dirname):
                if os.path.isdir('./'+Y_dirname+'/'+M_dirname) and M_dirname[0]!='.':

                    if os.listdir('./'+Y_dirname+'/'+M_dirname):
                        print '\t'+M_dirname+'/'
                        
                        
                        path = './'+Y_dirname+'/'+M_dirname+'/'
                        #preparation of the folders and files for output
                        if not os.path.exists(path+'FIGURES'):
                            os.makedirs(path+'FIGURES')
                        if not os.path.exists(path+'DATA'):
                            os.makedirs(path+'DATA')
                        mean_file = open(path+'DATA/mean_SST.dat','w')
                        
                        #empty containers for temporary values
                        month_mean_T = []
                        month_AVG = []
                        
                        FLAT_LON_subset = []
                        FLAT_LAT_subset = []

                        #=========================== MAIN CODE START =====================
                        for filename in os.listdir('./'+Y_dirname+'/'+M_dirname):
                            if os.path.isfile('./'+Y_dirname+'/'+M_dirname+'/'+filename) and filename[0]!='.' \
                               and filename[-3:]=='.nc':
                                print '\t\t'+filename
                                
                                
                                
                                
                                ################################################
                                data_file = netCDF.Dataset(path+filename)
                                
                                #---
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
                                plt.savefig(path+'FIGURES/'+filename[:-2]+'png',dpi=defdpi)
                                
                                #save data to temp var
                                month_mean_T.append(ma.average(AVG,weights=WGHTS))
                                month_AVG.append(AVG)
                                for item in LON_subset:
                                    FLAT_LON_subset.extend(item)
                                for item in LAT_subset:
                                    FLAT_LAT_subset.extend(item)
                                
                                #write space-averaged data to file
                                tmp_average = ma.average(AVG,weights=WGHTS)
                                tmp_variance = ma.average((AVG-tmp_average)**2, weights=WGHTS)
                                tmp_std = ma.sqrt(tmp_variance)
                                mean_file.write(filename+','+str(tmp_average)+','+str(tmp_std)+'\n')
                                ################################################
                                        
                        #write monthly-averaged data to file
                        mean_file.write('monthly mean SST,'+str(ma.average(month_mean_T))+','+str(ma.std(month_mean_T)))
                        #%
                        
                        MMEAN = ma.average(month_AVG,axis=0)
                        
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
                        im = mymap.pcolormesh(XMAP,YMAP,MMEAN,cmap='jet',vmin=np.min(AVG), vmax=np.max(AVG))
                        
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        
                        plt.colorbar(im, cax=cax)
                        
                        plt.savefig(path+'FIGURES/monthly_mean_SST.png',dpi=defdpi)
                        #---------------- DRAW DENSITY 2D HIST -------------------------
                        
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
                        
                        #XMAP,YMAP = mymap(XC,YC) #no need to repeat
                        im = mymap.pcolormesh(XMAP,YMAP,counts,cmap='Blues')
                        
                        
                        #plot makeup
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        #
                        plt.colorbar(im, cax=cax,label='points count')
                        #save figure
                        plt.savefig(path+'FIGURES/monthly_mean_SST_2dhist.png',dpi=defdpi)
                        #------------------------------------------------------
                        
                        #close data file
                        mean_file.close()
                        
                        #%
                        #save array with monthly-averaged SST
                        #(binary file to be read from python script)
                        MMEAN.dump(path+'DATA/monthly_mean_'+path[-3:-1]+'.dat')
                                
                        #=========================== MAIN CODE FINISH=====================
                    else:
                        print '\t'+M_dirname+'/ is empty!'
        else:
            print Y_dirname + '/ is empty!'
        #path = './'+filename

end = time.time()
print 'execution time: '+str(end - start)+' sec'




