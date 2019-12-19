'''
MOD03: geolocation data
- Produce latitude/longitude data set ('Latitude', 'Longitude')
- Produce the Land Sea Mask
- Show color bar for all graphs
- Sun field geometry (imshow the values over an area)
    - Viewing zenith angle ('SensorZenith')
    - Relative azimuthal   ('SensorAzimuth'-'SolarAzimuth')
    - Solar zenith angle   ('SolarZenith')
- Function to crop area out of modis file from lat lon
'''

from plt_MODIS_02 import * #includes matplotlib and numpy

# #used by get functions(filename can be modified by choose_file function)
# geo_files       = {'high_lat'     :'/Users/vllgsbr2/Desktop/MODIS_Training/Data/high_latitude/MOD03.A2018248.0450.061.2018248114733.hdf',
#                    'toronto'      :'/Users/vllgsbr2/Desktop/MODIS_Training/Data/toronto_09_05_18/MOD03.A2018248.1630.061.2018248230625.hdf',
#                    'maracaibo'    :'/Users/vllgsbr2/Desktop/MODIS_Training/Data/venezuela_08_21_18/MOD03.A2018233.1545.061.2018233214936.hdf',
#                    'twhs'         :'/Users/vllgsbr2/Desktop/MODIS_Training/Data/03032015TWHS/MOD03.A2015062.1645.061.2017319034323.hdf'}
# filename   = geo_files['high_lat']

fieldnames_list  = ['SolarZenith', 'SensorZenith', 'SolarAzimuth',\
                    'SensorAzimuth', 'Latitude', 'Longitude']

#create dictionaries for angles (used by get functions)
solar_zenith  = {}
sensor_zenith = {}
solar_azimuth = {}
sensor_azimuth = {}

def get_solarZenith(filename):
    #obtain field information to grab scales/offsets
    SD_field_rawData = 1 #0 SD, 1 field & 2 returns raw data
    solar_zenith['scale_factor']   = get_data(filename, fieldnames_list[0], SD_field_rawData).attributes()['scale_factor']

    #correct values by scales/offsets
    SD_field_rawData = 2 #0 SD, 1 field & 2 returns raw data
    solar_zenith['corrected_raw_data']   = get_data(filename, fieldnames_list[0], SD_field_rawData) * solar_zenith['scale_factor']

    return solar_zenith['corrected_raw_data']

def get_sensorZenith(filename):
    #obtain field information to grab scales/offsets
    SD_field_rawData = 1 #0 SD, 1 field & 2 returns raw data
    sensor_zenith['scale_factor']  = get_data(filename, fieldnames_list[1], SD_field_rawData).attributes()['scale_factor']

    #correct values by scales/offsets
    SD_field_rawData = 2 #0 SD, 1 field & 2 returns raw data
    sensor_zenith['corrected_raw_data']  = get_data(filename, fieldnames_list[1], SD_field_rawData) * sensor_zenith['scale_factor']

    return sensor_zenith['corrected_raw_data']

def get_solarAzimuth(filename):
    #obtain field information to grab scales/offsets
    SD_field_rawData = 1 #0 SD, 1 field & 2 returns raw data
    solar_azimuth['scale_factor']  = get_data(filename, fieldnames_list[2], SD_field_rawData).attributes()['scale_factor']

    #correct values by scales/offsets
    SD_field_rawData = 2 #0 SD, 1 field & 2 returns raw data
    solar_azimuth['corrected_raw_data']  = get_data(filename, fieldnames_list[2], SD_field_rawData) * solar_azimuth['scale_factor']

    return solar_azimuth['corrected_raw_data']

def get_sensorAzimuth(filename):
    #obtain field information to grab scales/offsets
    SD_field_rawData = 1 #0 SD, 1 field & 2 returns raw data
    sensor_azimuth['scale_factor'] = get_data(filename, fieldnames_list[3], SD_field_rawData).attributes()['scale_factor']

    #correct values by scales/offsets
    SD_field_rawData = 2 #0 SD, 1 field & 2 returns raw data
    sensor_azimuth['corrected_raw_data'] = get_data(filename, fieldnames_list[3], SD_field_rawData) * sensor_azimuth['scale_factor']

    return sensor_azimuth['corrected_raw_data']

def get_relativeAzimuth(filename):
    relative_azimuth = get_sensorAzimuth(filename) - get_solarAzimuth(filename)
    return relative_azimuth

def get_lat(filename):
    SD_field_rawData = 2
    lat = get_data(filename, fieldnames_list[4], SD_field_rawData)

    return lat

def get_lon(filename):
    SD_field_rawData = 2
    lon = get_data(filename, fieldnames_list[5], SD_field_rawData)

    return lon




def get_LandSeaMask(filename):

    SD_field_rawData = 2
    land_sea_mask = get_data(filename, 'Land/SeaMask', SD_field_rawData)

    return land_sea_mask



if __name__ =='__main__':

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.colors as matCol
    import cartopy.crs as ccrs

    filename_MOD_03 = '/Users/vllgsbr2/Desktop/MODIS_Training/Data/toronto_09_05_18/MOD03.A2018248.1630.061.2018248230625.hdf'
    filename_MOD_03 = '/Users/vllgsbr2/Desktop/LA_PTA_2017_12_16_1805/2017_09_03_1855/MOD03.A2017246.1855.061.2017257170030.hdf'
    filename_MOD_35 = '/Users/vllgsbr2/Desktop/LA_PTA_2017_12_16_1805/2017_09_03_1855/MOD35_L2.A2017246.1855.061.2017258203025.hdf'

    land_sea_mask = get_LandSeaMask(filename_MOD_03)
    lat, lon = get_lat(filename_MOD_03), get_lon(filename_MOD_03)

    # cmap = ListedColormap(['white', 'green', 'blue','black'])
    plt.rcParams.update({'font.size': 18})
    cmap = plt.cm.get_cmap('jet')
    norm = matCol.BoundaryNorm(np.arange(0,9,1), cmap.N)
    f = plt.figure()# plt.axes(projection=ccrs.AlbersEqualArea(), frame_on=False)
    im = plt.pcolormesh(lon, lat, land_sea_mask, cmap=cmap, norm=norm)

    cbar = plt.colorbar(im)
    cbar.set_ticks(np.arange(0.5,8,1.))#[0.5,1.5,2.5,3.5, 4.5, 5.5, 6.5, 7.5])
    cbar.set_ticklabels(['Shallow Ocean', 'Land', 'Ocean Coastlines and\nLake Shorelines',\
                         'Shallow Inland\nWater', 'Ephemeral\nWater', 'Deep Inland\nWater',\
                         'Moderate or\nContinental Ocean', 'Deep Ocean'])

    plt.axis('off')

    plt.title('MOD03 Land Sea Mask')

    plt.show()

    # print(get_sensorZenith(filename_MOD_03).max())
    # print(get_sensorZenith(filename_MOD_03).min())



    # #plot
    # fig, axes = plt.subplots(ncols=3)
    # cmap = 'jet'
    #
    # plot_1 = axes[0].imshow(get_solarZenith(filename_MOD_03), cmap = cmap)
    # axes[0].set_title('Solar Zenith Angle\n[degrees]')
    #
    # plot_2 = axes[1].imshow(get_sensorZenith(filename_MOD_03), cmap = cmap)
    # axes[1].set_title('Sensor Zenith Angle\n[degrees]')
    #
    # plot_3 = axes[2].imshow(get_relativeAzimuth(filename_MOD_03), cmap = cmap, vmin=-260, vmax=-210)
    # axes[2].set_title('Relative Azimuthal Angle\n[degrees]')
    #
    # fig.colorbar(plot_1, ax=axes[0])
    # fig.colorbar(plot_2, ax=axes[1])
    # fig.colorbar(plot_3, ax=axes[2])
    #
    # fig1, axes1 = plt.subplots(ncols=2)
    #
    # plot_11  = axes1[0].imshow(get_lon(), cmap = cmap)
    # axes1[0].set_title('Longitude\n[degrees]')
    # plot_22 = axes1[1].imshow(get_lat(), cmap = cmap)
    # axes1[1].set_title('Latitude\n[degrees]')
    #
    # fig1.colorbar(plot_1, ax=axes1[0])
    # fig1.colorbar(plot_2, ax=axes1[1])
    #
    # plt.show()

    # #debugging tools
    # file = SD('/Users/vllgsbr2/Desktop/MODIS_Training/Data/03032015TWHS/MOD03.A2015062.1645.061.2017319034323.hdf')
    # data = file.select('EV_500_Aggr1km_RefSB')
    # pprint.pprint(data.attributes()) #tells me scales, offsets and bands
    # pprint.pprint(file.datasets()) # shows data fields in file from SD('filename')
