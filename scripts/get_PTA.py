'''
Download MODIS 02,03,35 products from LAADS DAAC csv file
and check the integrity of the file
'''
import pandas as pd
import urllib.request
import h5py
import os
import sys
from plt_MODIS_02 import get_data
#sys.argv[0] is always the script filepath

def check_file_integrity(url, filename, real_file_size, fieldnames, save_path, directory, output_file):
    #check that file downloaded and has some size and datafields are not corrupt

    filename   = save_path + directory + filename
    statinfo   = os.stat(filename)
    downloaded_file_size  = statinfo.st_size

    if url[23:25]=='02':
        if downloaded_file_size == real_file_size :
            try:
                get_data(filename, fieldnames[2], 2)
                get_data(filename, fieldnames[3], 2)
                get_data(filename, fieldnames[4], 2)
            except:
                output_file.writelines(filename+' is corrupt'+'\n')

        else:
            output_file.writelines(filename+' failed to download properly'+'\n')

    elif url[23:25]=='03':
        if downloaded_file_size == real_file_size:
            try:
                get_data(filename, fieldnames[5], 2)
                get_data(filename, fieldnames[6], 2)
                get_data(filename, fieldnames[7], 2)
                get_data(filename, fieldnames[8], 2)
                get_data(filename, fieldnames[9], 2)
                get_data(filename, fieldnames[10], 2)
            except:
                output_file.writelines(filename+' is corrupt'+'\n')

        else:
            output_file.writelines(filename+' failed to download properly'+'\n')

    else:
        if downloaded_file_size == real_file_size:
            try:
                get_data(filename, fieldnames[0], 2)
                get_data(filename, fieldnames[1], 2)
            except:
                output_file.writelines(filename+' is corrupt'+'\n')
        else:
            output_file.writelines(filename+' failed to download properly'+'\n')

def download_granule(url, url_base, save_path, output_file, download_check=True, filenum=None):
    #check product to put into corresponding directory
    if url[23:25]=='03':
        n = 35
        directory = 'MOD_03/'
    elif url[23:25]=='02':
        n = 38
        directory = 'MOD_02/'
    else:
        n = 38
        directory = 'MOD_35/'

    #give file its full name without url path
    filename = '{}'.format(url[n:])

    #arg1: Download the file and  arg2: save it locally
    if download_check:
        if not os.path.isfile(save_path + directory + filename):
            try:
                urllib.request.urlretrieve(url_base+url, save_path + directory + filename)
                print(url_base+url)
                output_file.writelines('file --- {} ---  {} --- downloaded\n '.format(filenum+1, filename))
            except Exception as e:
                print(e)
                sys.exit()
                output_file.writelines('unable to download: '+filename+'\n')
        else:
            output_file.writelines('file --- {} ---  {} --- exists\n '.format(filenum+1, filename))

    return filename, directory

if __name__ == '__main__':
#MOD021KM_QUERY.csv  MOD03_+_MOD35_QUERY.csv
#/data/keeling/a/vllgsbr2/MODIS_ML/data

    #sys.argv -> [script path, LAADS DAAC csv, MOD_02/03, file name column]
    filepath          = sys.argv[1]
    filenames_archive = pd.read_csv(filepath, header=0)
    url_base          = 'https://ladsweb.modaps.eosdis.nasa.gov'
    save_path         = '/data/keeling/a/vllgsbr2/MODIS_ML/data/'
    file_name_column  = sys.argv[2]
    #'fileUrls from query MOD021KM--61 MOD03--61 MOD35_L2'\
    #'--61 2002-01-01..2019-10-15 x-124.4y39.8 x-112.8y30.7[5]'
    #print(filenames_archive.keys())

    output_file = open("get_PTA_stats.txt","w")
    filenames = filenames_archive[file_name_column]
    file_sizes = filenames_archive['size']

    fieldnames = ['Cloud_Mask', 'Quality_Assurance',\
                  'EV_1KM_RefSB', 'EV_250_Aggr1km_RefSB', 'EV_500_Aggr1km_RefSB',\
                  'SolarZenith', 'SensorZenith', 'SolarAzimuth','SensorAzimuth', \
                  'Latitude', 'Longitude']

    #loop to check pre downloaded files and to download files and check all their integrity
    for idx, (url, file_size) in enumerate(zip(filenames, file_sizes)):
        print(url)
        filename, directory  = download_granule(url, url_base, save_path, output_file, True, idx)
        check_file_integrity(url, filename, file_size, fieldnames, save_path, directory, output_file)
       # print('.', end="")
    print('')
    output_file.close()

