import numpy as np
import matplotlib.pyplot as plt
from pyhdf.SD import SD
from plt_MODIS_02 import get_data
import h5py


def get_bits(data_SD, N, cMask_or_QualityAssur=True):
    '''
    INPUT:
          data_SD               - 3D numpy array  - cloud mask SD from HDF
          N                     - int             - byte to work on
          cMask_or_QualityAssur - boolean         - True for mask, False for QA
    RETURNS:
          numpy.bytes array of byte stack of shape 2030x1354
    '''
    shape = np.shape(data_SD)

    #convert MODIS 35 signed ints to unsigned ints
    if cMask_or_QualityAssur:
        data_unsigned = np.bitwise_and(data_SD[N, :, :], 0xff)
    else:
        data_unsigned = np.bitwise_and(data_SD[:, :, N], 0xff)



    #type is int16, but unpackbits taks int8, so cast array
    data_unsigned = data_unsigned.astype(np.uint8)#data_unsigned.view('uint8')

    #return numpy array of length 8 lists for every element of data_SD
    data_bits = np.unpackbits(data_unsigned)

    if cMask_or_QualityAssur:
        data_bits = np.reshape(data_bits, (shape[1], shape[2], 8))
    else:
        data_bits = np.reshape(data_bits, (shape[0], shape[1], 8))

    return data_bits

def save_mod35(data, save_path):
    hf = h5py.File(save_path, 'w')
    hf.create_dataset('MOD_35_decoded', data=data)
    hf.close()

#save_mod35(data_SD_bit, '/Users/vllgsbr2/Desktop/MODIS_Training/Data/mod_35/MOD_35.h5')

#recover bit array
#shape = (2030, 1354, 8)
#data = np.array(h5py.File('/Users/vllgsbr2/Desktop/MODIS_Training/Data/mod_35/MOD_35.h5', 'r').get('MOD_35_decoded'))

def decode_byte_1(decoded_mod35_hdf):
    '''
    INPUT:
          decoded_mod35_hdf: - numpy array (2030, 1354, 8) - bit representation
                               of MOD_35
    RETURN:
          Cloud_Mask_Flag,
          new_Unobstructed_FOV_Quality_Flag,
          Day_Night_Flag,
          Sun_glint_Flag,
          Snow_Ice_Background_Flag,
          new_Land_Water_Flag
                           : - numpy array (6, 2030, 1354) - first 6 MOD_35
                                                             products from byte1

    '''
    data = decoded_mod35_hdf
    shape = np.shape(data)

    #0,1,2,or 3 fill
    #cloudy, uncertain clear, probably clear, confident clear
    Unobstructed_FOV_Quality_Flag = data[:,:, 5:7]
    #find index of each cloud possibility
    #new list to stuff new laues into and still perform a good search
    new_Unobstructed_FOV_Quality_Flag = np.empty((shape[0], shape[1]))

    cloudy_index          = np.where((Unobstructed_FOV_Quality_Flag[:,:, 0]==0)\
                                   & (Unobstructed_FOV_Quality_Flag[:,:, 1]==0))
    uncertain_clear_index = np.where((Unobstructed_FOV_Quality_Flag[:,:, 0]==0)\
                                   & (Unobstructed_FOV_Quality_Flag[:,:, 1]==1))
    probably_clear_index  = np.where((Unobstructed_FOV_Quality_Flag[:,:, 0]==1)\
                                   & (Unobstructed_FOV_Quality_Flag[:,:, 1]==0))
    confident_clear_index = np.where((Unobstructed_FOV_Quality_Flag[:,:, 0]==1)\
                                   & (Unobstructed_FOV_Quality_Flag[:,:, 1]==1))

    new_Unobstructed_FOV_Quality_Flag[cloudy_index]          = 0
    new_Unobstructed_FOV_Quality_Flag[uncertain_clear_index] = 1
    new_Unobstructed_FOV_Quality_Flag[probably_clear_index]  = 2
    new_Unobstructed_FOV_Quality_Flag[confident_clear_index] = 3

    return new_Unobstructed_FOV_Quality_Flag


def decode_Quality_Assurance(data_SD_Quality_Assurance):
    '''
    INPUT:
          data_SD_Quality_Assurance - numpy array (2030,1354,10) - HDF SD of QA
    RETURN:
          Quality assurance for 5 cloud mask tests
    '''
    data_bits_3 = get_bits(data_SD_Quality_Assurance, 2, \
                           cMask_or_QualityAssur=False)
    data_bits_4 = get_bits(data_SD_Quality_Assurance, 3, \
                           cMask_or_QualityAssur=False)

    QA_High_Cloud_Flag_1380nm         = data_bits_3[:,:, 0]
    QA_Cloud_Flag_Visible_Reflectance = data_bits_3[:,:, 4]
    QA_Cloud_Flag_Visible_Ratio       = data_bits_3[:,:, 5]
    QA_Near_IR_Reflectance            = data_bits_3[:,:, 6]
    QA_Cloud_Flag_Spatial_Variability = data_bits_4[:,:, 1]

    return QA_High_Cloud_Flag_1380nm,\
           QA_Cloud_Flag_Visible_Reflectance,\
           QA_Cloud_Flag_Visible_Ratio,\
           QA_Near_IR_Reflectance,\
           QA_Cloud_Flag_Spatial_Variability

def decode_tests(data_SD, filename_MOD_35):
    '''
    INPUT:
          data_SD         - numpy array (6,2030,1354) - SD from HDF of cloud
                                                        mask
          filename_MOD_35 - str                       - path to mod 35 file
    RETURN:
          5 cloud mask tests that are quality assured - numpy arrays
                                                        (2030, 1354)
          High_Cloud_Flag_1380nm,
          Cloud_Flag_Visible_Reflectance,
          Cloud_Flag_Visible_Ratio,
          Near_IR_Reflectance,
          Cloud_Flag_Spatial_Variability

    '''
    data_bits_3_ = get_bits(data_SD, 2)
    data_bits_4_ = get_bits(data_SD, 3)

    data_SD_Quality_Assurance = get_data(filename_MOD_35, 'Quality_Assurance',2)
    #for bytes 3&4
    data_bits_QA = decode_Quality_Assurance(data_SD_Quality_Assurance)

    High_Cloud_Flag_1380nm         = data_bits_3_[:,:, 0]
    Cloud_Flag_Visible_Reflectance = data_bits_3_[:,:, 4]
    Cloud_Flag_Visible_Ratio       = data_bits_3_[:,:, 5]
    Near_IR_Reflectance            = data_bits_3_[:,:, 6]
    Cloud_Flag_Spatial_Variability = data_bits_4_[:,:, 1]

    # find indicies where test is not applied; set to 9
    High_Cloud_Flag_1380nm[np.where(data_bits_QA[0]==0)]         = 9
    Cloud_Flag_Visible_Reflectance[np.where(data_bits_QA[1]==0)] = 9
    Cloud_Flag_Visible_Ratio[np.where(data_bits_QA[2]==0)]       = 9
    Near_IR_Reflectance[np.where(data_bits_QA[3]==0)]            = 9
    Cloud_Flag_Spatial_Variability[np.where(data_bits_QA[4]==0)] = 9

    return High_Cloud_Flag_1380nm,\
           Cloud_Flag_Visible_Reflectance,\
           Cloud_Flag_Visible_Ratio,\
           Near_IR_Reflectance,\
           Cloud_Flag_Spatial_Variability




if __name__ == '__main__':

    ############################################################################
    #speed test
    filename_MOD_35 = '/Users/vllgsbr2/Desktop/MODIS_Training/Data/toronto_09_05_18/MOD35_L2.A2018248.1630.061.2018249014414.hdf'
    data_SD         = get_data(filename_MOD_35, 'Cloud_Mask', 2)

    #fast bit
    start = time.time()
    fast_bits = get_bits(data_SD, 0)
    fast_bits = decode_byte_1(fast_bits)
    print('---------- ', time.time()-start,' ---------')

    #plot
    import matplotlib.colors as matCol
    from matplotlib.colors import ListedColormap
    cmap=plt.cm.PiYG
    cmap = ListedColormap(['white', 'green', 'blue','black'])
    norm = matCol.BoundaryNorm(np.arange(0,5,1), cmap.N)
    plt.imshow(fast_bits[1], cmap=cmap, norm=norm)
    #plt.imsave('test.png', fast_bits[1], cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_ticks([0.5,1.5,2.5,3.5])
    cbar.set_ticklabels(['cloudy', 'uncertain\nclear', \
                         'probably\nclear', 'confident\nclear'])
    plt.title('MODIS Cloud Mask')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # #slow bit
    # start = time.time()
    # slow_bits = get_bits_old(data_SD, 0)
    # print('---------- ', time.time()-start,' ---------')
    #
    # #check results
    # fields = ['Cloud_Mask_Flag              :',\
    #           'Unobstructed_FOV_Quality_Flag:',\
    #           'Day_Night_Flag               :',\
    #           'Sun_glint_Flag               :',\
    #           'Snow_Ice_Background_Flag     :',\
    #           'Land_Water_Flag              :']
    #
    # for i, field in enumerate(fields):
    #     print(field, np.all(slow_bits[i]==fast_bits[i]))

    # ############################################################################
    # #plot
    # import matplotlib.colors as matCol
    # from matplotlib.colors import ListedColormap
    #
    # f, ax = plt.subplots(ncols=2)
    #
    # cmap=plt.cm.PiYG
    # cmap = ListedColormap(['white', 'green', 'blue','black'])
    # norm = matCol.BoundaryNorm(np.arange(0,5,1), cmap.N)
    #
    # im = ax[0].imshow(slow_bits[1], cmap=cmap, norm=norm)
    # im = ax[1].imshow(fast_bits[1], cmap=cmap, norm=norm)
    #
    # cbar_ax = f.add_axes([0.9, 0.15, 0.02, 0.7])
    # cbar = plt.colorbar(im, cax=cbar_ax)
    # cbar.set_ticks([0.5,1.5,2.5,3.5])
    # cbar.set_ticklabels(['cloudy', 'uncertain\nclear', 'probably\nclear',\
    #                      'confident\nclear'])
    #
    # ax[0].set_title('slow bits')
    # ax[1].set_title('fast bits')
    # plt.suptitle('MODIS Cloud Mask')
    #
    # plt.show()


    #############################################
