'''
inspect_MODIS_CMQ (CMQ = Cloud Mask Quality)

This script will produce a 2 panel plot. An enhanced RGB and a cloud mask that share an axis. The
Purpose of this code is to efficiently find good candidate granules to use as
training data for a MODIS cloud mask model based in AI.

Simply run and the plot will show; then follow the prompts on the terminal.
You type 1 for good or 0 for bad training candidate based on how well the
cloud mask matches the RGB simply by inspection. This repeats until all the data
has been choosen. Your answer is recorded in a file named label_csv. It has
two columns: time stamp, label

NOTE: ADJUST 'figsize' IF FIGURE IS TOO BIG/SMALL
'''

import matplotlib.pyplot as plt
import matplotlib.colors as matCol
from matplotlib.colors import ListedColormap
import numpy as np
import os
import time
#grab files
home = '/data/keeling/a/vllgsbr2/c/MAIA_Threshold_Dev/LA_PTA_MODIS_Data/'

filename_MOD_02 = os.listdir(home + 'MOD_02/')

#sort so all file indicies correspond
filename_MOD_02 = np.sort(filename_MOD_02)

#grab time stamps
#time_stamps = [ x[10:22] for x in filename_MOD_02]

#get RGB and CM images
home = '/data/keeling/a/vllgsbr2/MODIS_ML/images/'

RGB    = np.sort(os.listdir(home + 'RGB'))
CM     = np.sort(os.listdir(home + 'CM'))
R_1_38 = np.sort(os.listdir(home + 'R_1_38'))
R_11   = np.sort(os.listdir(home + 'R_11'))
time_stamps = [ x for x in RGB]
##find matching time stamps and only use those which all have available
#time_stamps_try = [ '2004073.2000.npz',  '2005118.1935.npz',  '2006228.1745.npz',\
#'2002004.2040.npz',  '2002327.1755.npz',  '2003197.1910.npz',  '2004076.2025.npz',  '2005122.1910.npz',  '2006232.2030.npz',\
#'2002008.2015.npz',  '2002329.1920.npz',  '2003198.1955.npz',  '2004077.1935.npz',  '2005126.1850.npz',  '2006238.1955.npz']

label_csv = open('../images/labels.csv', 'w')

height, width = 400, 400//2

#plot
for rgb, cm, r_1_38, r_11, time_stamp in zip(RGB, CM, R_1_38, R_11, time_stamps):
#for time_stamp in time_stamps_try:

    r_1_38 = np.load(home + 'R_1_38/' + time_stamp)['arr_0']
    r_11   = np.load(home + 'R_11/'   + time_stamp)['arr_0']
    rgb    = np.load(home + 'RGB/'    + time_stamp)['arr_0']
    cm     = np.load(home + 'CM/'     + time_stamp)['arr_0']

    for i in range(1, 6):

        if i != 5:
            r_1_38_temp = r_1_38[height * (i-1):height * i ,677-width:677+width]
            r_11_temp   = r_11[height   * (i-1):height * i ,677-width:677+width]
            rgb_temp    = rgb[height    * (i-1):height * i ,677-width:677+width]
            cm_temp     = cm[height     * (i-1):height * i ,677-width:677+width]
        else:
            r_1_38_temp = r_1_38[height * (i-1): ,677-width:677+width]
            r_11_temp   = r_11[height   * (i-1): ,677-width:677+width]
            rgb_temp    = rgb[height    * (i-1): ,677-width:677+width]
            cm_temp     = cm[height     * (i-1): ,677-width:677+width]

        n = 17
        f, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(n, n))
        f.subplots_adjust(wspace=0)
        f.suptitle('section '+ str(i) +'of 5\n' + time_stamp)

        #11 microns grey scale inverted
        ax[1,0].imshow(r_11_temp, cmap='Greys_r')
        ax[1,0].set_xticks([])
        ax[1,0].set_yticks([])
        ax[1,0].set_title('11 microns')

        #1.38 microns grey scale inverted
        ax[0,1].imshow(r_1_38_temp, cmap='Greys_r')
        ax[0,1].set_xticks([])
        ax[0,1].set_yticks([])
        ax[0,1].set_title('1.38 microns')

        #plot the RGB image
        ax[0,0].imshow(rgb_temp)
        ax[0,0].set_title('MODIS BRF RGB')
        ax[0,0].set_xticks([])
        ax[0,0].set_yticks([])

        #plot cloud mask with convoluted code for colorbar
        cmap = ListedColormap(['white', 'green', 'blue','black'])
        norm = matCol.BoundaryNorm(np.arange(0,5,1), cmap.N)
        im   = ax[1,1].imshow(cm_temp, cmap=cmap, norm=norm)
        #l,b,w,h
        cbar_ax = f.add_axes([0.89, 0.1, 0.017, 0.3])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_ticks([0.5,1.5,2.5,3.5])
        cbar.set_ticklabels(['cloudy', 'uncertain\nclear', \
                         'probably\nclear', 'confident\nclear'])
        ax[1,1].set_title('MODIS Cloud Mask')
        ax[1,1].set_xticks([])
        ax[1,1].set_yticks([])
        #plt.tight_layout()
        f.show()
        plt.pause(0.0001)
        print('enter label for ' + time_stamp + '\n1 for good 0 for bad')
        label = time_stamp + ','  + input()
        label_csv.writelines(label + '\n')
        plt.close(f)


label_csv.close()

