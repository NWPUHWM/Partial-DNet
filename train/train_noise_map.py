import torch
import torchvision
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.utils.data.dataloader as dataloader
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from data_patch_p import datagenerator
from data_patch_p import DenoisingDataset
from torch.utils.data import DataLoader
import os, glob, datetime, time
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
from data_patch_p import data_aug

#super parameters
torch.cuda.set_device(0)
LEARNING_RATE=0.001
EPOCH=70
SIGMA=75
BATCH_SIZE=64
PEPLEVEL=5

#train datas
train1=np.load('train_washington8.npy')
train1=train1.transpose((2,1,0))


#test datas
test=np.load('test_center.npy')

# test=test[:400,:400,:]*255
test=np.uint8(test)

test=test.transpose((2,1,0))
root='./'

def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]



class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):

        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def loss_fuction(x,y):
    MSEloss=sum_squared_error()
    loss=MSEloss(x,y)
    return loss

class predict_noise_map(nn.Module):
    def __init__(self):
        super(predict_noise_map, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,32,3,1,1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 48, 3, 1, 1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(48, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.ReLU()
        )
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x



net = predict_noise_map()
net.cuda()

if __name__ == '__main__':
    # model selection
    print('===> Building model')
    # criterion = sum_squared_error()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE,)
    scheduler = MultiStepLR(optimizer, milestones=[25,40,55], gamma=0.25)
    for epoch in range(EPOCH):
        scheduler.step(epoch)
        print("Decaying learning rate to %g" % scheduler.get_lr()[0])
        for tex in range(1):
            mode = np.random.randint(0, 4)
            net.train()
            train1 = data_aug(train1, mode)


            print('epochs:', epoch)
            channels1 = 191 # 191 bands

            data_patches1 = datagenerator(train1, channels1)
            data_patches = data_patches1
            data_patches = torch.from_numpy(data_patches.transpose((0, 3, 1, 2,)))
            DDataset = DenoisingDataset(data_patches, SIGMA,PEPLEVEL)

            print('yes')
            DLoader = DataLoader(dataset=DDataset, batch_size=BATCH_SIZE, shuffle=True)  
            epoch_loss = 0
            start_time = time.time()
            for step, x_y in enumerate(DLoader):
                batch_x_noise, noise_map= x_y[0].cuda(), x_y[1].cuda().float()

                optimizer.zero_grad()

                loss = loss_fuction(net(batch_x_noise), noise_map)
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
        data_patches= datagenerator(test, channel_s)

        data_patches = torch.from_numpy(data_patches.transpose((0, 3, 1, 2,)))


        DDataset = DenoisingDataset(data_patches, SIGMA,PEPLEVEL)
        DLoader = DataLoader(dataset=DDataset, batch_size=BATCH_SIZE, shuffle=True)
        epoch_loss = 0
        for step, x_y in enumerate(DLoader):
            batch_x_noise,  noise_map= x_y[0].cuda(), x_y[1].cuda()
            loss = loss_fuction(net(batch_x_noise), noise_map)
            epoch_loss += loss.item()

            if step % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (
                    epoch + 1, step, data_patches.size(0) // BATCH_SIZE, loss.item() / BATCH_SIZE))

        elapsed_time = time.time() - start_time
        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / step, elapsed_time))
        torch.save(net.state_dict(), 'predict_%03dSIGMA%03dpep%03d.pth' % (epoch + 1, SIGMA,PEPLEVEL))