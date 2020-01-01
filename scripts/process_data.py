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

from pyhdf.SD import SD
from tesselate_image import tesselate
from plt_MODIS_02 import get_data, prepare_data, get_scale_and_offset
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
    
    #print(os.listdir(os.path.join(base_data_directory,'MOD03')))
    #print(os.listdir(os.path.join(base_data_directory,'MOD35')))    
    for fname in os.listdir(os.path.join(base_data_directory,'MOD03')):     
        if datetime in fname:
            filename_MOD03 = os.path.join(base_data_directory,'MOD03',fname)
            #assert filename_MOD03 is not None, 'files appear to be duplicated. A: {}, B: {}'.format(filename_MOD03, fname)

            
    for fname in os.listdir(os.path.join(base_data_directory,'MOD35')):
        if datetime in fname:
            #assert filename_MOD35 is not None, 'files appear to be duplicated. A: {}, B: {}'.format(filename_MOD35, fname)
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
    f = SD(filename_MOD02)#.select('MODIS_SWATH_Type_L1B')
    #dsets = f.datasets()

    #print(f.select(variable_names[0]).attributes.keys)
    
    valid_maxs = []
    for variable_name in variable_names:
        if '1KM_Emissive' in variable_name:
            b_nums =  f.select('Band_1KM_Emissive')[:]
        elif '250' in variable_name:
            b_nums = f.select('Band_250M')[:]
        elif '500' in variable_name:
            b_nums = f.select('Band_500M')[:]
        elif '1KM_RefSB' in variable_name:
            b_nums = f.select('Band_1KM_RefSB')[:]
        radiance_scale, offsets = get_scale_and_offset(f.select(variable_name), True)
        max_val = (32767 - np.array(offsets)) * np.array(radiance_scale)
        valid_maxs.append(max_val)
        band_numbers.append(b_nums)
    f.end()
    band_numbers = np.concatenate(band_numbers)
    valid_maxs = np.concatenate(valid_maxs,axis=0)
    valid_mins = np.zeros(valid_maxs.shape)
    valid_range = np.stack([valid_mins, valid_maxs], axis=-1)
    return band_numbers, valid_range

def get_day_of_year(filename_MOD02, dset_shape):
    
    fname = filename_MOD02.split('/')[-1]
    doy = float(fname[14:17])

    DOY = np.full(dset_shape, dtype=np.int, fill_value = doy)
    
    return DOY


def open_MOD35(filename_MOD35):
    """
    TODO
    """
    data_SD = get_data(filename_MOD35, 'Cloud_Mask', 2)
    fast_bits = get_bits(data_SD, 0)
    cloud_mask = decode_byte_1(fast_bits)
    return cloud_mask

    
def subset_by_latitude_and_failed(tesselated_features, tesselated_mask, feature_labels, valid_ranges, lat_min=15.0, lat_max=45.0,
                                  bands_to_remove=['36.0']):
    """
    TODO
    """
    
    # if tesselated images don't contain ANY lat < max or lat > min then
    #throw them away.

    maximum_sub_images = tesselated_features.shape[-1]
    indices = []
    for band in bands_to_remove:
        indices.append(np.where(feature_labels==band)[0])


    tesselated_features = np.delete(tesselated_features, np.array(indices), axis=0)
    feature_labels = np.delete(feature_labels, np.array(indices), axis=0)
    valid_ranges = np.delete(valid_ranges, np.array(indices),axis=0)

    
    latitude_min = np.min(tesselated_features[-3],axis=(0,1))
    latitude_max = np.max(tesselated_features[-3],axis=(0,1))#-3 index is latitude (see open_data())

    
    band_bad = (np.sum(np.isnan(tesselated_features), axis=(1,2,3)))
    band_nans = feature_labels[np.where(band_bad > 0)]
    
    condition = np.where((latitude_min < lat_max) & (latitude_max > lat_min) & np.all(np.isfinite(tesselated_features), axis=(0,1,2)))
    subset_features = tesselated_features[...,condition[0]]
    subset_mask = tesselated_mask[...,condition[0]]
    
    sub_image_index = condition[0]

    return subset_features, subset_mask, feature_labels, len(condition[0]), band_nans, valid_ranges, sub_image_index, maximum_sub_images
    
        


def open_data(filename_MOD03, filename_MOD35, filename_MOD02, image_shape = (64, 64), 
              variable_names = ['EV_1KM_RefSB','EV_1KM_Emissive','EV_250_Aggr1km_RefSB',
                                'EV_500_Aggr1km_RefSB'],
              lat_min=15.0, lat_max = 45.0):
    """
    TODO
    """        

    radiances = np.concatenate([prepare_data(filename_MOD02, name, True) for name in variable_names], axis=0)
    band_numbers, valid_range = get_band_numbers(filename_MOD02, variable_names)
    
    other_ranges = np.array([[0,90],[0,90],[-360,360],[-90,90],[-180, 180],[1,366]])
    valid_ranges = np.append(valid_range, other_ranges, axis=0)

    MOD03_data = np.stack([get_solarZenith(filename_MOD03), get_sensorZenith(filename_MOD03),
                                 get_relativeAzimuth(filename_MOD03), get_lat(filename_MOD03),
                                 get_lon(filename_MOD03)], axis=0)
    
    #print(radiances.min(), radiances.max())
    MOD35_data = open_MOD35(filename_MOD35)
    
    DOY = get_day_of_year(filename_MOD02, (radiances.shape[1], radiances.shape[2]))
    #print(DOY.shape, radiances.shape, MOD03_data.shape)
    feature_data = np.concatenate([radiances, MOD03_data, np.expand_dims(DOY,0)], axis=0)
    
    tesselated_features = []
    for feature in feature_data:
        tesselated, num_cols, num_rows = tesselate(feature, image_shape[0], image_shape[1])

        tesselated_features.append(tesselated)
        
    
    tesselated_features = np.stack(tesselated_features, axis=0)
    #print(tesselated_features.shape)
    
    
    #print(MOD35_data.shape)
    tesselated_mask, num_cols, num_rows = tesselate(MOD35_data, image_shape[0], image_shape[1])
    #print(tesselated_mask.shape)
    feature_labels = np.append(band_numbers, np.array(['SolarZenithAngle','SensorZenithAngle',
                                             'RelativeAzimuthAngle','Latitude','Longitude','DayOfYear']))
    
    subset_features, subset_mask,feature_labels,num_valid_images,band_nans, valid_ranges, sub_image_index, maximum_subimages = subset_by_latitude_and_failed(tesselated_features, tesselated_mask,
                                                                                                           feature_labels, valid_ranges,lat_min=15.0, lat_max=45.0)    
    

    
    #print(feature_labels)
    
    return subset_features, subset_mask, feature_labels, num_valid_images, band_nans, valid_ranges, sub_image_index, maximum_subimages
    

def process_file(MOD02_filename, base_data_directory, save_directory,image_shape=(64,64), lat_min=15.0, lat_max = 45.0):
    
    """Saves the recreated data from old files (MOD03, MOD35, MOD02)
    Main function."""
    #fail = False

    filename_MOD03, filename_MOD35, filename_MOD02, datetime = get_matching_file(MOD02_filename, base_data_directory)
    tesselated_features, tesselated_mask, feature_labels, num_valid_images, band_nans, valid_ranges, sub_image_index, maximum_subimages = open_data(filename_MOD03, filename_MOD35, filename_MOD02)
        
        #Save Data.
        
    hdfFile = h5py.File(os.path.join(save_directory,'MODIS_MLData_Shape_64x64_{}_.hdf'.format(datetime)),'w')
    label ='Data repackaged by Di Girolamo Research Group, UIUC for machine learning purposes. Original MODIS Granules were retrieved from \'https://ladsweb.modaps.eosdis.nasa.gov\ \
        Original Granule IDs: {} {} {}'.format(filename_MOD03.split('/')[-1], filename_MOD35.split('/')[-1], filename_MOD02.split('/')[-1])
    hdfFile.attrs.create('description',label, dtype='S{}'.format(len(label)))
        
    
    for i,f in enumerate(feature_labels):
        if '.' in f:
            feature_labels[i] = 'Band_{}'.format(f)
    feature_labels = [np.string_(feature) for feature in feature_labels]
    
    for i in range(tesselated_features.shape[-1]):
        group = hdfFile.create_group('Image_{:03}'.format(i))
        group.create_dataset('ImageFeatures', tesselated_features[...,-1].shape, 'f')
        group.create_dataset('ImageClassification', tesselated_mask[...,-1].shape, 'f')
            
        group['ImageFeatures'][:] = tesselated_features[...,i]
        group['ImageClassification'][:] = tesselated_mask[...,i]
            
        group['ImageClassification'].dims[0].label = 'image_dim_1'  
        group['ImageClassification'].dims[1].label = 'image_dim_2'
        label2 = '0 = confident cloudy, 1 = probably cloudy, 2 = probably clear, 3 = confident clear. For binary classification group 0 with 1 and 2 with 3.'
        group['ImageClassification'].attrs.create('Interpretation',label2,dtype='S{}'.format(len(label2)))
        
        
        group.create_dataset('FeatureLabels', data = feature_labels)
        #group['FeatureLabels'][:] = feature_labels
        #group['ImageFeatures'].make_scale('FeatureLabels')
        group['ImageFeatures'].dims[0].attach_scale(group['FeatureLabels'])
            
        group['ImageFeatures'].dims[0].label = 'Features'
        group['ImageFeatures'].dims[1].label = 'image_dim_1'            
        group['ImageFeatures'].dims[2].label = 'image_dim_2'
            
        group['FeatureLabels'].attrs.create('units', ['Watts/m^2/micrometer/steradian']*(len(feature_labels)-6) + ['degrees']*5 + ['days'])
        group['FeatureLabels'].attrs.create('valid_range', data=valid_ranges)
            
        label3 = '-1: not classified, 0: bad, 1: good'
        #group.attrs.create('Interpretation',)
        group.create_dataset('ClassificationAccuracy',data=-1, dtype=np.int)
        #group['ClassificationAccuracy'][:] = -1
        group['ClassificationAccuracy'].attrs.create('Interpretation',label3,dtype='S{}'.format(len(label3)))
        
        group.attrs.create('SubImageIndex', sub_image_index[i])
        group.attrs.create('MaximumSubImage', maximum_subimages)
        
    hdfFile.flush()
    hdfFile.close()
            
#    except:
#        fail = True
    return num_valid_images, band_nans
        
if __name__ == '__main__':
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    base_data_directory = sys.argv[1]
    save_directory = sys.argv[2]
    try:
        image_shape = (int(sys.argv[3]), int(sys.argv[4]))
        print('image shape set to {}'.format(image_shape))
    except:
        if rank == 0:
            print('image shape set to default of (64, 64)')
        image_shape = (64, 64)
            
    lat_min = 15.0#sys.argv[5]
    lat_max = 45.0#sys.argv[6]
    # sub directories should be MOD03, MOD35, MOD02.
    
    file_list_02 = os.listdir(os.path.join(base_data_directory, 'MOD02'))
    file_list_02 = [f for f in file_list_02 if '.hdf' in f]
    
    for i in range(0,len(file_list_02),size):
        idd = i + rank
        if idd < len(file_list_02):
            success = process_file(file_list_02[idd], base_data_directory, save_directory, image_shape=image_shape, lat_min=lat_min,
                         lat_max=lat_max)
            print('{} has completed with status {} {}'.format(file_list_02[idd], success[0], success[1]))