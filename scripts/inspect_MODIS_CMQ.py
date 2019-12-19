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
time_stamps = [ x[10:22] for x in filename_MOD_02]

#get RGB and CM images
home = '/data/keeling/a/vllgsbr2/MODIS_ML/images/'
RGB = os.listdir(home + 'RGB')
CM  = os.listdir(home + 'CM')

label_csv = open('../images/labels.csv', 'w')

#plot
for rgb, cm, time_stamp in zip(RGB, CM, time_stamps):
    print(rgb)    
    #rgb = imageio.imread(home + 'RGB/'  + rgb)
    #cm  = imageio.imread(home + 'CM/'  + cm)
    rgb = np.load(home + 'RGB/'  + rgb)['arr_0']
    cm  = np.load(home + 'CM/'  + cm)['arr_0']

    f, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(19, 13.5))
    f.subplots_adjust(wspace=0)
    f.suptitle('time stamp' + time_stamp)
    #plot the RGB image
    ax[0].imshow(rgb)
    ax[0].set_title('MODIS BRF RGB')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    #plot cloud mask with convoluted code for colorbar
    cmap = ListedColormap(['white', 'green', 'blue','black'])
    norm = matCol.BoundaryNorm(np.arange(0,5,1), cmap.N)
    im   = ax[1].imshow(cm, cmap=cmap, norm=norm)

    cbar_ax = f.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0.5,1.5,2.5,3.5])
    cbar.set_ticklabels(['cloudy', 'uncertain\nclear', \
                         'probably\nclear', 'confident\nclear'])
    ax[1].set_title('MODIS Cloud Mask')
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    f.show()
    plt.pause(0.0001)
    print('enter label for ' + time_stamp + '\n1 for good 0 for bad')
    label = time_stamp + ','  + input()
    label_csv.writelines(label + '\n')    
    plt.close(f)

    
label_csv.close()
