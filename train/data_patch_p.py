from torch.nn.modules.loss import _Loss
import numpy as np
from torch.utils.data import Dataset
import torch
patch_size, stride = 30, 30
aug_times = 1
scales = [0.5,1,1.5,2]
batch_size = 32



def PepperandSalt(src,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=np.random.randint(0,src.shape[0]-1)
        randY=np.random.randint(0,src.shape[1]-1)
        if np.random.randint(0,2)<=0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=1
    return NoiseImg


class DenoisingDataset(Dataset):#数据加上噪声

    def __init__(self, data_patches, sigma,peplevel):
        super(DenoisingDataset, self).__init__()
        self.xs =data_patches

        self.sigma = sigma
        self.pep=peplevel
    def __getitem__(self, index):
        #print(index)
        batch_x = self.xs[index]#[960,1,20,20]

        batch_x=batch_x.float()/255.0
       # batch_x_real=batch_x+0
        noise_level=np.random.randint(0, self.sigma)
        noise_level_p = np.random.randint(0, self.pep)
        noise_x = torch.randn(batch_x.size()).mul_( noise_level / 255.0)  # add Gaus noise
        batch_x_noise = batch_x + noise_x

        #batch_x_noise[:,:,:] = PepperandSalt(batch_x_noise.squeeze(), noise_level_p / 100)
        stds=(batch_x_noise-batch_x).std()
        #noise_map = torch.from_numpy(np.ones(batch_x.size()) * stds.item()).float()
        noise_map = torch.from_numpy(np.ones(batch_x.size()) * noise_level/255.0)
        #print(stds,noise_level / 255.0,noise_level_p)
        return batch_x,noise_map

    def __len__(self):
        return self.xs.size(0)


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.rot90(img,k=1,axes=(1,2))
    elif mode == 2:
        return np.rot90(img,k=2,axes=(1,2))
    elif mode == 3:
        return np.rot90(img,k=3,axes=(1,2))







def gen_patches(numpy_data,channel_is):
    # get multiscale patches from a single image
    channels=numpy_data.shape[0]

    print(channels)
    h, w = numpy_data.shape[1],numpy_data.shape[2]
    patches = []

    for channel_i in range(channel_is):
        for i in range(0, h-patch_size+1, stride):
            for j in range(0, w-patch_size+1, stride):
                x = numpy_data[channel_i,i:i+patch_size, j:j+patch_size]
                patches.append(x)


    #print(len(patches),len(cubic_paches))
    return patches


def datagenerator(numpy_data,channel_is):

    # generate patches
    patches= gen_patches(numpy_data,channel_is)
    #print(len(patches))
    print(len(patches),)

    data_patches = np.array(patches)

    print(data_patches.shape)
    data = np.expand_dims(data_patches, axis=3)

    discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)

    print('^_^-training data finished-^_^')
    return data

