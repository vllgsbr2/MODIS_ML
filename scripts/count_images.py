'''
This script will read in output from processed_data.py and look in the path
MODIS_MLData_Shape_64x64_YYYYDDD.HHMM_.hdf/image_xxx/ClassificationAccuracy
and look for a -1, 0, or 1.
-1 -> not classified
0  -> not fit for training data
1  -> fit for training data

And will return the number of images per file and the total number of classified
images in each file
'''

import h5py

def count_images(file_path):
    '''
    TO-DO ;)
    '''

    image_file = h5py.File(file_path, 'r')

    label_count = 0
    num_images  = len(list(image_file.keys()))
    num_valid_images = 0
    for i in range(num_images):
        path = 'Image_{0:03d}/ClassificationAccuracy/'.format(i)
        if image_file[path][()] != -1:
            label_count += 1
        elif image_file[path][()] == 1:
            num_valid_images += 1
        else:
            pass

    return label_count, num_images, num_valid_images

if __name__ == '__main__':
    import os

    home = '/Users/vllgsbr2/Desktop/MODIS_ML_data_sample/'
    files = os.listdir('/Users/vllgsbr2/Desktop/MODIS_ML_data_sample/')
    files = [home + x for x in files]

    total_labeled = 0
    total_images  = 0
    total_valid_images = 0
    for file in files:
        label_count, num_images, num_valid_images = count_images(file)

        total_labeled += label_count
        total_images  += num_images
        total_valid_images += num_valid_images

        print('file: {}\nlabel_count: {}\nnum_images: {}'\
              .format(file[-17:-5], label_count, num_images))

    total_invalid_images = total_images - total_valid_images
    print('total_labeled: {} total_images: {} total_valid_images: {} total_invalid_images: {}'\
          .format(total_labeled, total_images, total_valid_images, total_invalid_images))
