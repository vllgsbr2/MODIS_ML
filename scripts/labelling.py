#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:36:02 2020

@author: jesserl2
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as matCol
from matplotlib.colors import ListedColormap
import random

from matplotlib.widgets import Slider

def get_random_sample(data_directory):
    """
    TODO
    """
    file_list = os.listdir(data_directory)
    valid_file_list = [fname for fname in file_list if '.hdf' in fname]
    
    valid_image = None
    maxiter = 1000
    iter1 = 0
    while valid_image is None or iter1 > maxiter:
        
        random_file = random.choice(valid_file_list)
        File = h5py.File(os.path.join(data_directory, random_file) ,'r+')
        image_list = File.keys()
        
        if len(image_list) > 0:
            image = random.sample(list(image_list), 1)
            ClassAccuracy = File[image[0]]['ClassificationAccuracy'][()]
            if ClassAccuracy == -1:
                valid_image = image[0]
            else:    
                File.close()
        else:
            File.close()
        iter1 += 1
        
    assert valid_image is not None, 'wow, the code should find an image faster than this'

    Features = File[valid_image]['ImageFeatures'][:]
    FeatureLabels = File[valid_image]['FeatureLabels'][:]
    
    CloudMask = File[valid_image]['ImageClassification'][:]
    return Features, FeatureLabels, CloudMask, ClassAccuracy, File, valid_image


def pick_visualization_channels(Features, FeatureLabels, visualization_bands= ['Band_1.0', 'Band_26.0','Band_31.0']):
    """
    TODO
    """
    vis_out = []
    for band in visualization_bands:
        position = np.where(FeatureLabels==np.string_(band))
        image = Features[position[0],...][0]
        vis_out.append(image)
        
    return vis_out


def classify(data_list, ClassAccuracy, valid_image, File):
    
    n = 7
    f, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(n,n))
    f.subplots_adjust(wspace=0.1) 
    #f.subplots_adjust(left=0.25, bottom=0.25)
    f.subplots_adjust(hspace = 0.4)
    
    
    im1 = ax[0,0].imshow(data_list[0], cmap='Greys_r')
    ax[0,0].set_title('MODIS BRF RGB')
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    
        #1.38 microns grey scale inverted
    im2=ax[0,1].imshow(data_list[1], cmap='Greys_r')
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])
    ax[0,1].set_title('1.38 microns')
    
    im3 = ax[1,0].imshow(data_list[2], cmap='Greys')
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,0].set_title('11 microns')
    
    cmap = ListedColormap(['white', 'green', 'blue','black'])
    norm = matCol.BoundaryNorm(np.arange(0,5,1), cmap.N)
    im4   = ax[1,1].imshow(data_list[3], cmap=cmap, norm=norm)
        #l,b,w,h
    #cbar_ax = f.add_axes([0.89, 0.1, 0.017, 0.3])
    cbar = plt.colorbar(im4, ax=ax[1,1])#cax=cbar_ax)
    cbar.set_ticks([0.5,1.5,2.5,3.5])
    cbar.set_ticklabels(['cloudy', 'uncertain\nclear', \
                         'probably\nclear', 'confident\nclear'])
    ax[1,1].set_title('MODIS Cloud Mask')
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])
    
    
    
    
    data = data_list[0]
    
    c_max = data.max()
    c_min = data.min()
            
    pos = ax[0,0].get_position()

    ax_cmin = plt.axes([pos.x0, pos.y0-0.03, pos.x1-0.2, 0.01])
    ax_cmax  = plt.axes([pos.x0, pos.y0-0.06, pos.x1-0.2, 0.01])
            
    s_cmin1 = Slider(ax_cmin, 'min', c_min, c_max, valinit=c_min)
    s_cmax1 = Slider(ax_cmax, 'max', c_min, c_max, valinit=c_max)
            
    def update1(val, s=None):
        _cmin = s_cmin1.val
        _cmax = s_cmax1.val
        im1.set_clim([_cmin, _cmax])
        plt.draw()
        
    s_cmin1.on_changed(update1)
    s_cmax1.on_changed(update1)
    
    data = data_list[1]
    
    c_max = data.max()
    c_min = data.min()
            
    pos = ax[0,1].get_position()

    ax_cmin = plt.axes([pos.x0, pos.y0-0.03, pos.x1-pos.x0-0.1, 0.01])
    ax_cmax  = plt.axes([pos.x0, pos.y0-0.06, pos.x1-pos.x0-0.1, 0.01])
            
    s_cmin2 = Slider(ax_cmin, 'min', c_min, c_max, valinit=c_min)
    s_cmax2 = Slider(ax_cmax, 'max', c_min, c_max, valinit=c_max)
            
    def update2(val, s=None):
        _cmin = s_cmin2.val
        _cmax = s_cmax2.val
        im2.set_clim([_cmin, _cmax])
        plt.draw()
        
    s_cmin2.on_changed(update2)
    s_cmax2.on_changed(update2)    
    

    data = data_list[2]
    
    c_max = data.max()
    c_min = data.min()
            
    pos = ax[1,0].get_position()

    ax_cmin = plt.axes([pos.x0, pos.y0-0.03, pos.x1-pos.x0-0.1, 0.01])
    ax_cmax  = plt.axes([pos.x0, pos.y0-0.06, pos.x1-pos.x0-0.1, 0.01])
            
    s_cmin3 = Slider(ax_cmin, 'min', c_min, c_max, valinit=c_min)
    s_cmax3 = Slider(ax_cmax, 'max', c_min, c_max, valinit=c_max)
            
    def update3(val, s=None):
        _cmin = s_cmin3.val
        _cmax = s_cmax3.val
        im3.set_clim([_cmin, _cmax])
        plt.draw()
        
    s_cmin3.on_changed(update3)
    s_cmax3.on_changed(update3)    
    
    data = data_list[3]
    
    c_max = data.max()
    c_min = data.min()
            
    pos = ax[1,1].get_position()

    ax_cmin = plt.axes([pos.x0, pos.y0-0.03, pos.x1-pos.x0-0.1, 0.01])
    ax_cmax  = plt.axes([pos.x0, pos.y0-0.06, pos.x1-pos.x0-0.1, 0.01])
            
    s_cmin4 = Slider(ax_cmin, 'min', c_min, c_max, valinit=c_min)
    s_cmax4 = Slider(ax_cmax, 'max', c_min, c_max, valinit=c_max)
            
    def update4(val, s=None):
        _cmin = s_cmin4.val
        _cmax = s_cmax4.val
        im4.set_clim([_cmin, _cmax])
        plt.draw()
        
    s_cmin4.on_changed(update4)
    s_cmax4.on_changed(update4) 
        
    f.show()
    plt.pause(10**6)

if __name__ == '__main__':
    
    data_directory = '/Users/jesserl2/Documents/Data/MODIS_ML/output'
    Features, FeatureLabels, CloudMask, ClassAccuracy, File, valid_image = get_random_sample(data_directory)
    vis_out = pick_visualization_channels(Features, FeatureLabels)
    vis_out.append(CloudMask)
    
    classify(vis_out, ClassAccuracy, valid_image, File)
    
    print('Enter Label for {} {}. 1 for good 0 for bad:'.format(File,valid_image))
    label = input()
    
    while label not in ['0', '1']:
        print('Your label {} was not in 0 or 1. Please Try Again'.format(label))
        classify(vis_out, ClassAccuracy, valid_image, File)        
        label = input()
                
    File[valid_image]['ClassificationAccuracy'][()] = float(label)
    
    #plt.close(f)
    File.flush()
    File.close()
    
    
    
    
    
    
    
    

    
    
    
    
    