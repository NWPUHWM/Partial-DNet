import torch
from torch.nn.modules.loss import _Loss

import numpy as np
import torch.nn.functional as F

from datae48 import datagenerator
from datae48 import DenoisingDataset
from torch.utils.data import DataLoader
import os, glob, datetime, time
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
from datae48 import data_aug
from models import PartialDNet

#super parameters
torch.cuda.set_device(0)
LEARNING_RATE=0.001
EPOCH=70
SIGMA=75
PEPLEVEL=5
BATCH_SIZE=64

#train datas
train2=np.load('train_pavia8.npy')
train3=np.load('train_washington8.npy')



train2=train2.transpose((2,1,0))
train3=train3.transpose((2,1,0))

#test datas
test=np.load('test_center.npy')
# test=test[:400,:400,:]
test=np.uint8(test)
test=test.transpose((2,1,0))
root='./'

#define denoising model
def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)
class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

def loss_fuction(x,y):
    MSEloss=sum_squared_error()
    loss=MSEloss(x,y)
    return loss

#test demo
net = PartialDNet()
net.cuda()

y=torch.randn(12,1,36,20,20).cuda()
c=torch.randn(12,1,36,20,20).cuda()
z=torch.randn(12,1,20,20).cuda()
x=torch.randn(12,1,20,20).cuda()
out=net(x,y,z,c)
print(out.size())

#begin train process
if __name__ == '__main__':
    # model selection
    print('===> Building model')
    # criterion = sum_squared_error()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE,)
    scheduler = MultiStepLR(optimizer, milestones=[5,35,50], gamma=0.25)

    for epoch in range(EPOCH):#
        scheduler.step(epoch)
        print("Decaying learning rate to %g" % scheduler.get_lr()[0])
        for tex in range(1):
            mode=np.random.randint(0,4)
            net.train()

            train2 = data_aug(train2, mode)
            train3 = data_aug(train3, mode)

            print('epochs:', epoch)

            channels2 = 72  # 93 channels
            channels3 = 72  # 191 channels

            data_patches2, data_cubic_patches2 = datagenerator(train2, channels2)
            data_patches3, data_cubic_patches3 = datagenerator(train3, channels3)

            data_patches = np.concatenate((data_patches2, data_patches3,), axis=0)
            data_cubic_patches = np.concatenate((data_cubic_patches2, data_cubic_patches3), axis=0)

            data_patches = torch.from_numpy(data_patches.transpose((0, 3, 1, 2, )))
            data_cubic_patches = torch.from_numpy(data_cubic_patches.transpose((0, 4, 1, 2, 3)))

            DDataset = DenoisingDataset(data_patches, data_cubic_patches, SIGMA,PEPLEVEL)

            print('yes, train process begin')
            DLoader = DataLoader(dataset=DDataset, batch_size=BATCH_SIZE, shuffle=True) 

            epoch_loss = 0
            start_time = time.time()
            for step, x_y in enumerate(DLoader):
                batch_x_noise, batch_y_noise, batch_x ,noise_map_x,noise_map_y= x_y[0].cuda(), x_y[1].cuda(), x_y[2].cuda(),x_y[3].cuda(),x_y[4].cuda()

                optimizer.zero_grad()

                loss = loss_fuction(net(batch_x_noise, batch_y_noise,noise_map_x,noise_map_y), batch_x)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                if step % 10 == 0:
                    print('%4d %4d / %4d loss = %2.4f' % (
                        epoch + 1, step, data_patches.size(0) // BATCH_SIZE, loss.item() / BATCH_SIZE))

            elapsed_time = time.time() - start_time
            log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / step, elapsed_time))






        start_time = time.time()
        net.eval()
        channel_s = 93  
        data_patches, data_cubic_patches = datagenerator(test, channel_s)

        data_patches = torch.from_numpy(data_patches.transpose((0, 3, 1, 2,)))
        data_cubic_patches = torch.from_numpy(data_cubic_patches.transpose((0, 4, 1, 2, 3)))

        DDataset = DenoisingDataset(data_patches, data_cubic_patches, SIGMA,PEPLEVEL)
        DLoader = DataLoader(dataset=DDataset, batch_size=BATCH_SIZE, shuffle=True)
        epoch_loss = 0
        for step, x_y in enumerate(DLoader):
            batch_x_noise, batch_y_noise, batch_x, noise_map_x, noise_map_y = x_y[0].cuda(), x_y[1].cuda(), x_y[
                2].cuda(), x_y[3].cuda(), x_y[4].cuda()
            loss = loss_fuction(net(batch_x_noise, batch_y_noise,noise_map_x,noise_map_y), batch_x)
            epoch_loss += loss.item()

            if step % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (
                    epoch + 1, step, data_patches.size(0) // BATCH_SIZE, loss.item() / BATCH_SIZE))

        elapsed_time = time.time() - start_time
        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / step, elapsed_time))
        torch.save(net.state_dict(), 'PartialDNet_%03dSIGMA%03dpep%03d.pth' % (epoch + 1, SIGMA,PEPLEVEL))

