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

#grab files
home = '/data/keeling/a/vllgsbr2/c/MAIA_Threshold_Dev/LA_PTA_MODIS_Data/'

filename_MOD_03 = os.listdir(home + 'MOD_03/')
filename_MOD_02 = os.listdir(home + 'MOD_02/')
filename_MOD_35 = os.listdir(home + 'MOD_35/')

#sort so all file indicies correspond
filename_MOD_02 = np.sort(filename_MOD_02)
filename_MOD_03 = np.sort(filename_MOD_03)
filename_MOD_35 = np.sort(filename_MOD_35)

time_stamp = [ x[10:22] for x in filename_MOD_02]

#add home to the path
filename_MOD_03 = [home + 'MOD_03/' + file for file in filename_MOD_03]
filename_MOD_02 = [home + 'MOD_02/' + file for file in filename_MOD_02]
filename_MOD_35 = [home + 'MOD_35/' + file for file in filename_MOD_35]

error_file = open('../images/error_file.txt', 'w')

#do every 12 granules for ~102 granules per year
for mod02, mod03, mod35, time_stamp in zip(filename_MOD_02[::12], filename_MOD_03[::12], filename_MOD_35[::12], time_stamp[::12]):
    try:
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

    except:
        error_file.writelines('Corrupt: '+ time_stamp + '\n')

error_file.close()
