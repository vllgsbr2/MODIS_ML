#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 09:02:30 2019

@author: jesserl2
"""

import numpy as np
import os
import mpi4py.MPI as MPI
import sys

from tesselate_image import tesselate
from plt_MODIS_02 import get_data, prepare_data
from plt_MODIS_03 import get_solarZenith, get_sensorZenith, get_relativeAzimuth, get_lat, get_lon
from read_MODIS_35 import get_bits, decode_byte_1

import h5py


def get_matching_file(MOD02, base_data_directory):
    """
    TODO
    """
    #include path.join with base_directory for direct opening.
    
    datetime = MOD02[10:22]
    
    filename_MOD03 = None
    filename_MOD35 = None
    filename_MOD02 = None
    
    for fname in os.listdir(os.path.join(base_data_directory,'MOD03')):
        if datetime in fname:
            assert filename_MOD03 is not None, 'files appear to be duplicated. A: {}, B: {}'.format(filename_MOD03, fname)
            filename_MOD03 = os.path.join(base_data_directory,'MOD03',fname)
            
    for fname in os.listdir(os.path.join(base_data_directory,'MOD35')):
        if datetime in fname:
            assert filename_MOD35 is not None, 'files appear to be duplicated. A: {}, B: {}'.format(filename_MOD35, fname)
            filename_MOD35 = os.path.join(base_data_directory,'MOD35',fname)
            
    assert filename_MOD03 is not None, 'No Matching MOD03 file found for {}'.format(MOD02)
    assert filename_MOD35 is not None, 'No Matching MOD35 file found for {}'.format(MOD02)            
    
    filename_MOD02 = os.path.join(base_data_directory, 'MOD02', MOD02)
    
    return filename_MOD03, filename_MOD35, filename_MOD02, datetime

def get_band_numbers(filename_MOD02, variable_names):
    """
    TODO
    """
    band_numbers = []
    with SD(filename_MOD02) as f:
        
        for variable_name in variable_names:
            if '1KM_Emissive' in variable_name:
                b_nums =  f['Band_1KM_Emissive']
            elif '250' in variable_name:
                b_nums = f['Band_250M']
            elif '500' in variable_name:
                b_nums = f['Band_500M']
            elif '1KM_RefSB' in variable_name:
                b_nums = f['Band_1KM_RefSB']
            band_numbers.append(b_nums)
    
    band_numbers = np.concatenate(band_numbers)
    
    return band_numbers

def get_day_of_year(filename_MOD02, image_shape = (64, 64)):
    
    fname = filename_MOD02.split['/'][-1]
    doy = float(fname[14:17])
    
    DOY = np.full((1,) + image_shape, fill_value = doy)
    
    return DOY


def open_MOD35(filename_MOD35):
    """
    TODO
    """
    data_SD = get_data(filename_MOD35, 'Cloud_Mask', 2)
    fast_bits = get_bits(data_SD, 0)
    cloud_mask = decode_byte_1(fast_bits)[1]
    
    return cloud_mask
    
def subset_by_latitude(tesselated_features, tesselated_mask, lat_min=15.0, lat_max=45.0):
    """
    TODO
    """
    
    # if tesselated images don't contain ANY lat < max or lat > min then
    #throw them away.
    
    latitude_min = np.min(tesselated_features[-3],axis=(1,2))
    latitude_max = np.max(tesselated_features[-3],axis=(1,2))#-3 index is latitude (see open_data())
    
    condition = np.where((latitude_min < lat_max) & (latitude_max > lat_min))
    
    subset_features = tesselated_features[...,condition[0]]
    subset_mask = tesselated_mask[...,condition[0]]
    
    return subset_features, subset_mask
    

def open_data(filename_MOD03, filename_MOD35, filename_MOD02, image_shape = (64, 64), 
              variable_names = ['EV_1KM_RefSB','EV_1KM_Emissive','EV_250_Aggr1km_RefSB',
                                'EV_500_Aggr1km_RefSB'],
              lat_min=15.0, lat_max = 45.0):
    """
    TODO
    """        

    radiances = np.stack([prepare_data(filename_MOD02, name) for name in variable_names], axis=0)
    band_numbers = get_band_numbers(filename_MOD02, variable_names)
    
    MOD03_data = np.concatenate([get_solarZenith(filename_MOD03), get_sensorZenith(filename_MOD03),
                                 get_relativeAzimuth(filename_MOD03), get_lat(filename_MOD03),
                                 get_lon(filename_MOD03)], axis=0)
    
    
    MOD35_data = np.concatenate([open_MOD35(filename_MOD35)], axis=0)
    
    DOY = get_day_of_year(filename_MOD02, image_shape)
    
    feature_data = np.stack([radiances, MOD03_data, DOY])
    
    tesselated_features = []
    for feature in feature_data:
        tesselated = tesselate(feature, image_shape[0], image_shape[1])
        tesselated_features.append(tesselated)
        
    tesselated_features = np.concatenate(tesselated_features)
    tesselated_mask = tesselate(MOD35_data)
    
    subset_features, subset_mask = subset_by_latitude(tesselated_features, tesselated_mask, lat_min=15.0, lat_max=45.0)
    
    feature_labels = np.stack([band_numbers,['SolarZenithAngle','SensorZenithAngle',
                                             'RelativeAzimuthAngle','Latitude','Longitude','DayOfYear']])
    
    return subset_features, subset_mask, feature_labels
    

def process_file(MOD02_filename, base_data_directory, save_directory,image_shape=(64,64), lat_min=15.0, lat_max = 45.0):
    
    """Saves the recreated data from old files (MOD03, MOD35, MOD02)
    Main function."""
    fail = False
    try:
        filename_MOD03, filename_MOD35, filename_MOD02, datetime = get_matching_file(MOD02_filename, base_data_directory)
        tesselated_features, tesselated_mask, feature_labels = open_data(filename_MOD03, filename_MOD35, filename_MOD02)
        
        #Save Data.
        
        hdfFile = h5py.File('MODIS_MLData_Shape_64x64_{}_.hdf'.format(datetime), SDC.WRITE|SDC.CREATE)
        hdfFile.attrs.create('description','Data repackaged by Di Girolamo Research Group, UIUC for machine learning purposes. Original MODIS Granules were retrieved from \'https://ladsweb.modaps.eosdis.nasa.gov\ \
        Original Granule IDs: {} {} {}'.format(filename_MOD03, filename_MOD35, filename_MOD02),dtype=np.str)
        
        for i in range(tesselated_features.shape[-1]):
            
            group = hdfFile.create_group('Image_{}'.format(i))
            group.create_dataset('ImageFeatures', tesselated_features[...,-1].shape, 'f')
            group.create_dataset('ImageClassification', tesselated_mask[...,-1].shape, np.int8)
            
            group['ImageFeatures'][:] = tesselated_features[...,i]
            group['ImageClassification'][:] = tesselated_mask[...,i]
            
            group['ImageClassification'].dims[1].label = 'image_dim_1'  
            group['ImageClassification'].dims[2].label = 'image_dim_2'
            group['ImageClassification'].attrs.create('Interpretation','0 = , 1 = , 2 = , 3 = . For binary classification group 0 with 1 and 2 with 3.',dtype=np.str)
            
            group['FeatureLabels'][:] = feature_labels
            group['ImageFeatures'].make_scale('FeatureLabels')
            group['ImageFeatures'].dims[0].attach_scale(group['FeatureLabels'])
            
            group['ImageFeatures'].dims[0].label = 'Features'
            group['ImageFeatures'].dims[1].label = 'image_dim_1'            
            group['ImageFeatures'].dims[2].label = 'image_dim_2'
            
            group['FeatureLabels'].attrs.create('units', ['Watts/m^2/micrometer/steradian']*36 + ['degrees']*5 + ['days'])
            group['FeatureLabels'].attrs.create('valid_range', ['Watts/m^2/micrometer/steradian']*36 + ['degrees']*5 + ['days'])            
            
    except:
        fail = True
        
    return fail
        
if __name__ == '__main__':
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    base_data_directory = sys.argv[1]
    save_directory = sys.argv[2]
    try:
        image_shape = (sys.argv[3], sys.argv[4])
        print('image shape set to {}'.format(image_shape))
    except:
        if rank == 0:
            print('image shape set to default of (64, 64)')
        image_shape = (64, 64)
            
    lat_min = 15.0#sys.argv[5]
    lat_max = 45.0#sys.argv[6]
    # sub directories should be MOD03, MOD35, MOD02.
    
    file_list_02 = os.listdir(os.path.join(base_data_directory, 'MOD02'))

    
    for i in range(0,len(file_list_02),size):
        idd = i + rank
        if idd < len(file_list_02):
            success = process_file(file_list_02[idd], base_data_directory, save_directory, image_shape=image_shape, lat_min=lat_min,
                         lat_max=lat_max)
            print('{} has completed with status'.format(file_list_02[idd], success))