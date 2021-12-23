import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


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
