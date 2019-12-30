import numpy as np

def tesselate(data, height, width):
    '''
    Objective:
        chop image into n square peices and delete the excess
    Arguments:
        data {2D narray} -- array to chop into pieces for classification
        height, width {int} -- height and width of subsection to make
    Return:
        data_stacked {3D narray} -- Stack of images that form the orignal in row
                                    row major order along the third axis
        num_cols, num_rows {int} -- if you multiply these two you get the number
                                    of subsections produced
    '''
    center = data.shape[1]//2
    cut  = 400//2

    #data has shape 2030x1354
    #first take middle section such that we have a 2030x800 section
    data = data[:, (center-cut) : (center+cut)]

    #now cut into n equal square peices and delete the left overs
    data_stack_rowwise = np.zeros((height,width))

    num_cols = 400//width
    num_rows = data.shape[0]//height

    data_stacked = np.empty((height, width, num_cols*num_rows))
    counter = 0

    #first cut data row wise
    for j in range(1, num_rows+1):
        data_stack_rowwise = data[height * (j-1):height * j ,:]

        for i in range(1, num_cols+1):
            data_stacked[:,:,counter] = data_stack_rowwise[:, width * (i-1):width * i]
            counter +=1

    return data_stacked, num_cols, num_rows


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    fake_data = np.arange(2030*1354).reshape(2030,1354)
    tesselated_data, num_cols, num_rows = tesselate(fake_data, 60,80)

    f1 = plt.figure(1)
    im1 = plt.imshow(fake_data, cmap='jet')
    plt.colorbar()

    f = plt.figure(2)

    vmin = fake_data.min()
    vmax = fake_data.max()
    counter = 0

    for i in range(num_rows):
        for j in range(num_cols):
            ax = plt.subplot2grid((num_rows,num_cols), (i,j))
            im = ax.imshow(tesselated_data[:,:,counter], cmap='jet', vmin=vmin, vmax=vmax)
            #ax.set_title('section '+str(i))
            ax.set_yticks([])
            ax.set_xticks([])
            counter +=1

    cbar_ax = f.add_axes([0.9, 0.12, 0.025, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)

    f.tight_layout()

    plt.show()

