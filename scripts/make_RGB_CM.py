'''
make_RGB_CM.py (CM = Cloud Mask)

This script will take in MODIS reflectances, cloud mask and saves them.
The purpose of this code is to preprocess the data into images to view 
in inspect_MODIS_CMQ.py
'''

import matplotlib.pyplot as plt
import numpy as np
import os
from rgb_enhancement import get_BRF_RGB, get_enhanced_RGB
from read_MODIS_35 import get_data, get_bits, decode_byte_1
from plt_MODIS_02 import prepare_data
#grab files
home = '/data/keeling/a/vllgsbr2/c/MAIA_Threshold_Dev/LA_PTA_MODIS_Data/'

filename_MOD_03 = os.listdir(home + 'MOD_03/')
filename_MOD_02 = os.listdir(home + 'MOD_02/')
filename_MOD_35 = os.listdir(home + 'MOD_35/')

#sort so all file indicies correspond
filename_MOD_02 = np.sort(filename_MOD_02)[100:]
filename_MOD_03 = np.sort(filename_MOD_03)[100:]
filename_MOD_35 = np.sort(filename_MOD_35)[100:]

time_stamp = [ x[10:22] for x in filename_MOD_02]

#add home to the path
filename_MOD_03 = [home + 'MOD_03/' + file for file in filename_MOD_03]
filename_MOD_02 = [home + 'MOD_02/' + file for file in filename_MOD_02]
filename_MOD_35 = [home + 'MOD_35/' + file for file in filename_MOD_35]

error_file = open('../images/error_file.txt', 'w')

#do every 12 granules for ~102 granules per year
for mod02, mod03, mod35, time_stamp in zip(filename_MOD_02[::12], filename_MOD_03[::12], filename_MOD_35[::12], time_stamp[::12]):
    try:
        #grab 11 and 1.38 microns channels
        fieldnames = ['EV_1KM_RefSB', 'EV_1KM_Emissive']
        rad_or_ref = False #False for Ref
        emissive_bands_refsb = prepare_data(mod02, fieldnames[0], rad_or_ref)
        emissive_bands = prepare_data(mod02, fieldnames[1], rad_or_ref)
        R_11 = emissive_bands[-6,:,:]
        R_1_38 = emissive_bands_refsb[-1,:,:]

        #process enhanced RGB from BRF
        RGB = get_BRF_RGB(mod02, mod03)
        RGB_enhanced  = get_enhanced_RGB(RGB)

        #process cloud mask from MOD_35 file
        data_SD       = get_data(mod35, 'Cloud_Mask', 2)
        mod_35_byte_1 = get_bits(data_SD, 0)

        #Unobstructed_FOV_Quality_Flag is thecloud mask
        #values 0,1,2,3 -> cloudy, uncertain clear, probably clear, confident clear
        Unobstructed_FOV_Quality_Flag = decode_byte_1(mod_35_byte_1)

        np.savez('../images/RGB/'+ time_stamp +'.npz', RGB_enhanced)
        np.savez('../images/CM/'+ time_stamp +'.npz', Unobstructed_FOV_Quality_Flag)
        
        np.savez('../images/R_1_38/'+ time_stamp +'.npz', R_1_38)
        np.savez('../images/R_11/'  + time_stamp +'.npz', R_11)

        print(time_stamp)
    except Exception as e:
        #error_file.writelines('Corrupt: '+ time_stamp + '\n')
        print(e)
error_file.close()
