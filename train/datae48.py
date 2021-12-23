import glob

import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch

import os
import torchvision


from torch.utils.data import DataLoader

from scipy.io import savemat
patch_size, stride = 30, 30
aug_times = 1
scales = [0.5,1,1.5,2]
batch_size = 32
k=18



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


class DenoisingDataset(Dataset):

    def __init__(self, data_patches,data_cubic_patches, sigma,peplevel):
        super(DenoisingDataset, self).__init__()
        self.xs =data_patches
        self.cubic=data_cubic_patches
        self.sigma = sigma
        self.pep = peplevel

    def __getitem__(self, index):
        #print(index)
        batch_x = self.xs[index]#[1,20,20]
        batch_y=self.cubic[index]#[1,24,20,20]
        batch_x=batch_x.float()/255.0
        batch_y=batch_y.float()/255.0


        noise_level_x=np.random.randint(0, self.sigma)
        noise_level_xp = np.random.randint(0, self.pep)
        noise_x = torch.randn(batch_x.size()).mul_(noise_level_x/255.0)#
        #noise_map_x = torch.from_numpy(np.ones(batch_x.size()) * noise_level_x).float()/255.0
        batch_x_noise = batch_x + noise_x
        batch_x_noise[:, :, :] = PepperandSalt(batch_x_noise.squeeze(), noise_level_xp / 100)
        stds = (batch_x_noise - batch_x).std()
        noise_map_x = torch.from_numpy(np.ones(batch_x.size()) * stds.item()).float()
        #print(stds,noise_level_x/255,noise_level_xp,'x')
        noise_map_y=torch.from_numpy(np.ones(batch_y.size()) ).float()
        batch_y_noise=torch.from_numpy(np.ones(batch_y.size()) ).float()
        for channels in range(36):
            noise_level_y=np.random.randint(0, self.sigma)
            noise_level_yp = np.random.randint(0, self.pep)
            #noise_map_y[:, channels, :, :]=noise_map_y[:,channels, :, :]*noise_level_y/255.0

            batch_y_noise[:,channels, :, :] = batch_y[:,channels, :, :] + torch.randn(batch_y[:,channels, :, :].squeeze().size()).mul_(noise_level_y/ 255.0)
            batch_y_noise[:,channels, :, :] = PepperandSalt(batch_y_noise[:,channels, :, :].squeeze(), noise_level_yp / 100)
            stds_y = (batch_y_noise[:,channels, :, :] - batch_y[:,channels, :, :]).std()
            noise_map_y[:, channels, :, :] = noise_map_y[:, channels, :, :] * stds_y.item()
        #print(batch_y_noise, batch_y_noise, batch_x, noise_map_x, noise_map_y)
        return batch_x_noise, batch_y_noise, batch_x, noise_map_x, noise_map_y

    def __len__(self):
        return self.xs.size(0)






def data_aug(img, mode=0):#
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
    cubic_paches=[]
    for channel_i in range(channel_is):
        for i in range(0, h-patch_size+1, stride):
            for j in range(0, w-patch_size+1, stride):
                x = numpy_data[channel_i,i:i+patch_size, j:j+patch_size]
                patches.append(x)
                #print(x.shape)
                if channel_i < k:
                    # print(channel_i)
                    y = numpy_data[0:36, i:i + patch_size, j:j + patch_size]
                    # print(y.shape)
                    cubic_paches.append(y)
                elif channel_i < channels - k:
                    # print(channel_i)
                    y = np.concatenate((numpy_data[channel_i - k:channel_i, i:i + patch_size, j:j + patch_size],
                                        numpy_data[channel_i + 1:channel_i + k + 1, i:i + patch_size,
                                        j:j + patch_size]))
                    cubic_paches.append(y)
                    # print(y.shape)
                else:
                    # print(channel_i)
                    y = numpy_data[channel_is - 36:channel_is, i:i + patch_size, j:j + patch_size]
                    cubic_paches.append(y)
                    #print(y.shape)

    #print(len(patches),len(cubic_paches))
    return patches,cubic_paches



def datagenerator(numpy_data,channel_is):

    # generate patches
    patches,cubic_paches= gen_patches(numpy_data,channel_is)
    #print(len(patches))
    print(len(patches),len(cubic_paches))

    data_patches = np.array(patches)
    data_cubic_patches=np.array(cubic_paches)

    print(data_patches.shape,data_cubic_patches.shape)
    data = np.expand_dims(data_patches, axis=3)
    data_cubic = np.expand_dims(data_cubic_patches, axis=4)
    discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)
    data_cubic=np.delete(data_cubic, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data, data_cubic

