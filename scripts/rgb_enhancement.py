import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
from plt_MODIS_03 import get_solarZenith
from plt_MODIS_02 import prepare_data

def get_enhanced_RGB(RGB):
    def scale_image(image):
        along_track = image.shape[0]
        cross_track = image.shape[1]

        x = np.array([0,  30,  60, 120, 190, 255], dtype=np.uint8)
        y = np.array([0, 110, 160, 210, 240, 255], dtype=np.uint8)

        scaled = np.zeros((along_track, cross_track), dtype=np.uint8)
        for i in range(len(x)-1):
            x1 = x[i]
            x2 = x[i+1]
            y1 = y[i]
            y2 = y[i+1]
            m = (y2 - y1) / float(x2 - x1)
            b = y2 - (m *x2)
            mask = ((image >= x1) & (image < x2))
            scaled = scaled + mask * np.asarray(m * image + b, dtype=np.uint8)

        mask = image >= x2
        scaled = scaled + (mask * 255)
        return scaled


    # case = ['/home/javi/MODIS_Training/BRF_RGB_Toronto.npz',
    #         '/home/javi/MODIS_Training/BRF_RGB_Aerosol.npz' ]
    # rgb = np.load(case[1])['arr_0'][:]
    enhanced_RGB = np.zeros_like(RGB, dtype=np.uint8)
    for i in range(3):
        enhanced_RGB[:, :, i] = scale_image(scipy.misc.bytescale(RGB[:, :, i]))

    return enhanced_RGB

def get_BRF_RGB(filename_MOD_02, filename_MOD_03):

    #get Ref RGB to compare by eye
    cos_sza = np.cos(np.deg2rad(get_solarZenith(filename_MOD_03)))
    fieldnames_list  = ['EV_500_Aggr1km_RefSB', 'EV_250_Aggr1km_RefSB']
    rad_or_ref = False #True for radiance, False for reflectance
    #make channels for RGB photo (index 01234 -> band 34567)
    image_blue  = prepare_data(filename_MOD_02, fieldnames_list[0],rad_or_ref)[0,:,:] #band 3 from 500 meter res
    image_green = prepare_data(filename_MOD_02, fieldnames_list[0],rad_or_ref)[1,:,:] #band 4 from 500 meter res
    image_red   = prepare_data(filename_MOD_02, fieldnames_list[1],rad_or_ref)[0,:,:] #band 1 from 250 meter res
    #convert to BRF
    image_blue  /= cos_sza
    image_green /= cos_sza
    image_red   /= cos_sza

    RGB = np.dstack((image_red, image_green, image_blue))

    return RGB

if __name__ == '__main__':

    import matplotlib.pyplot as plt


    filename_MOD_03 ='/Users/vllgsbr2/Desktop/MODIS_Training/Data/toronto_09_05_18/MOD03.A2018248.1630.061.2018248230625.hdf'
    filename_MOD_02 = '/Users/vllgsbr2/Desktop/MODIS_Training/Data/toronto_09_05_18/MOD021KM.A2018248.1630.061.2018249014250.hdf'
    RGB = get_BRF_RGB(filename_MOD_02, filename_MOD_03)
    RGB_enhanced  = get_enhanced_RGB(RGB)

    plt.imshow(RGB_enhanced)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./clouds_over_ice.png', dpi=300)
    plt.show()
