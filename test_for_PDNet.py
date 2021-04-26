
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init

from psnr import mpsnr
from psnr import mssim
torch.cuda.set_device(0)
#The Partial-DNet model and prediction model
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
class conv_relu(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(conv_relu, self).__init__()
        self.channel = out_channels // 2
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, self.channel, 3, stride, 1),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, self.channel, 3, stride,2,dilation=2),
            nn.ReLU(inplace=True)
        )



    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(x)

        concat = torch.cat((layer1, layer2), dim=1)

        return concat
class ChannelAttention_GP(nn.Module):
    def __init__(self, in_planes, ):
        super(ChannelAttention_GP, self).__init__()
        self.avg_pool_feature = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_noises = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()



    def forward(self, x,y):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool_feature(x))))
        noise_out = self.fc2(self.relu(self.fc1(self.avg_pool_noises(y))))



        out = avg_out + noise_out
        return self.sigmoid(out)
class SDblock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(SDblock, self).__init__()

        self.layer1 = conv_relu(in_channels, out_channels)

        self.layer2 = conv_relu(out_channels+in_channels, out_channels)

        self.layer3 = conv_relu(out_channels+in_channels, out_channels)

        self.layer4 = conv_relu(out_channels+in_channels, out_channels)

        self.layer5 = conv_relu(out_channels+in_channels, out_channels)
        self.concat = nn.Sequential(
            nn.Conv2d(in_channels+out_channels*5,out_channels,1,1,0),

        )

    def forward(self, x):
        layer1=self.layer1(x)
        layer2=self.layer2(torch.cat((x,layer1),dim=1))
        layer3=self.layer3(torch.cat((x,layer2),dim=1))
        layer4=self.layer4(torch.cat((x,layer3),dim=1))
        layer5=self.layer5(torch.cat((x,layer4),dim=1))

        concat = torch.cat((x,layer1,layer2,layer3,layer4,layer5), dim=1)
        out= self.concat(concat)

        return out
class PartialDNet(nn.Module):
    def __init__(self):
        super(PartialDNet, self).__init__()
        self.f_3 = nn.Conv2d(2, 32, 3, 1, 1)
        self.f3_2 = nn.Conv3d(1, 32, (72, 3, 3), 1, (0, 1, 1))

        self.block1 = SDblock(64, 48)
        self.block2 = SDblock(112, 48)
        self.block3 = SDblock(112, 48)
        self.block4 = SDblock(112, 48)

        self.out = nn.Sequential(
            nn.Conv2d(48 * 4 + 64, 1, 3, 1, 1),

        )
        self._initialize_weights()

        self.ca = ChannelAttention_GP(36)
    def forward(self, x, y,noise_map_x,noise_map_y):
        y_map = y.squeeze(1)
        noise_map = noise_map_y.squeeze(1)

        y_map = y_map * self.ca(y_map, noise_map)
        y = y_map.unsqueeze(1)
        noise_map_y = noise_map.unsqueeze(1)


        f3_2 = self.f3_2(torch.cat((y,noise_map_y),dim=2)).squeeze(2)
        f3 = self.f_3(torch.cat((x, noise_map_x), dim=1))
        out3 = F.relu(torch.cat((f3, f3_2), dim=1))

        block1 = self.block1(out3)
        block2 = self.block2(torch.cat((block1, out3), dim=1))
        block3 = self.block3(torch.cat((block2, out3), dim=1))
        block4 = self.block4(torch.cat((block3, out3), dim=1))

        concat = torch.cat(
            (block1, block2, block3, block4, out3), dim=1)

        out = self.out(concat)
        # out2_1 = self.out2(out1_1)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
# load the trained model of prediction and Partial-DNet
def predict():
    models = predict_noise_map()
    models.load_state_dict(torch.load('predict.pth', map_location='cuda:0'))
    models.cuda()
    models.eval()
    return models
def model_rand():
    models=PartialDNet()
    models.load_state_dict(torch.load('PartialDNet.pth',map_location='cuda:0'))
    models.cuda()
    models.eval()
    return models
# the denoising function
def teste(tests,model,predict,test,noise_map,cs=191):


    k = 18
    tests = tests.astype(np.float32)
    noise_map_predict = np.ones(tests.shape).astype(np.float32)
    # prediction process
    for i in range(cs):

        predict_image = tests[i, :, :]
        predict_image = np.expand_dims(predict_image, axis=0)
        predict_image = np.expand_dims(predict_image, axis=0)
        predict_image = torch.from_numpy(predict_image)
        out = predict(predict_image.cuda()).data.cpu().numpy()
        stds = (tests[i, :, :] - test[i, :, :]).std()
        # noise_map_predict[i, :, :] = noise_map_predict[i, :, :] * stds.item()#use Partial-DNet-I
        noise_map_predict[i, :, :] = noise_map_predict[i, :, :] *out.mean()#use Partial-DNet





    test_out = np.zeros(tests.shape).astype(np.float32)

    noise_map = noise_map_predict
    #denoising process
    for channel_i in range(cs):
        x = tests[channel_i, :, :]
        x_1 = noise_map[channel_i, :, :]

        x_1 = np.expand_dims(x_1, axis=0)
        x_1 = np.expand_dims(x_1, axis=0)
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=0)
        if channel_i < k:

            y = tests[0:36, :, :]
            y_1 = noise_map[0:36, :, :]

        elif channel_i < cs - k:

            y = np.concatenate((tests[channel_i - k:channel_i, :, :],
                                tests[channel_i + 1:channel_i + k + 1, :, :]))
            y_1 = np.concatenate((noise_map[channel_i - k:channel_i, :, :],
                                  noise_map[channel_i + 1:channel_i + k + 1, :, :]))

        else:
            y = tests[cs - 36:cs, :, :]
            y_1 = noise_map[cs - 36:cs, :, :]
        y = np.expand_dims(y, axis=0)
        y = np.expand_dims(y, axis=0)

        y_1 = np.expand_dims(y_1, axis=0)
        y_1 = np.expand_dims(y_1, axis=0)

        x = torch.from_numpy(x)
        x_1 = torch.from_numpy(x_1)

        y = torch.from_numpy(y)
        y_1 = torch.from_numpy(y_1)


        with torch.no_grad():

            out = model(x.cuda(), y.cuda(), x_1.cuda(),y_1.cuda())

        out = out.squeeze(0)

        out = out.data.cpu().numpy()

        test_out[channel_i, :, :] = out

    test_out=test_out.astype(np.double)

    return test_out

# add  PepperandSalt noise function
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

def main():
    cs=191
    SIGMA=75
    test = np.load("./test_washington.npy")
    test = test.transpose((2, 1, 0))
    tests = np.zeros(test.shape).astype(np.float32)
    noise_map=np.ones(test.shape).astype(np.float32)
    #add Gaussian noise
    # for i in range(cs):
    #     # +(np.random.randint(0, 26) / 255.0) * np.random.randn(200, 200)
    #     # +SIGMA/255.0
    #     tests[i, :, :] = test[i, :, :] +(np.random.randint(10, 70) / 255.0) * np.random.randn(test.shape[2], test.shape[1])

    # add mix noise
    for i in range(cs):

        noise_level_x = np.random.randint(1, 70)
        noise_level_xp = np.random.randint(0, 5)
        tests[i, :, :] = test[i, :, :] + noise_level_x * np.random.randn(test.shape[1], test.shape[2]) / 255.0

        tests[i, :, :] = PepperandSalt(tests[i, :, :].squeeze(), noise_level_xp / 100)


    pre = predict()
    model = model_rand()
    test_out = teste(tests, model, pre, test, noise_map,cs)



    test_out = test_out.transpose((2, 1, 0))
    test = test.transpose((2, 1, 0))
    print("test_shape:",test.shape,"test_out_shape:", test_out.shape)

    ssim = mssim(test, test_out)
    psnr = mpsnr(test, test_out)
    print("PSNR:",psnr,"SSIM:", ssim)

main()