import numpy as np
import torch
from torch import nn
from torchvision import transforms as tsf
from torch.utils import model_zoo
from collections import OrderedDict
import torch.nn.functional as F

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
} #only resnet18 and resnet34 work for now.



#block of 2 conv layers
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, stride=1, dilation=1, padding=1):
        super(DoubleConv, self).__init__()

        if mid_channels==None:
            mid_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation), #change padding?
            nn.BatchNorm2d(out_channels),
            nn.PReLU(), #(P)relu?
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.PReLU()
    
    def forward(self, x):
        x = self.block(x)
        return self.act(x)


#2D UNET
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        
        self.block = DoubleConv(in_channels, out_channels)
        self.downblock =  nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(0.5) #dropout to less, or make optional?
        )

    def forward(self, x):
        x = self.downblock(x)
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels):
        super(UpBlock, self).__init__()
        out_chan = in_channels//2
        self.upsamp = nn.ConvTranspose2d(in_channels, out_chan, kernel_size=2, stride=2)
        self.block = DoubleConv(in_channels, out_chan)


    def forward(self, x, x_skip):
        x = self.upsamp(x)

        diffY = x_skip.size()[2] - x.size()[2]
        diffX = x_skip.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        #x_skip = tsf.CenterCrop((h,w))(x_skip)
        x = torch.cat([x_skip, x], dim=1)
        return self.block(x)





class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, dummy=None, dummyNet=None):
        super(UNet, self).__init__()

        self.n_class = n_classes
        self.n_chan = in_channels

        self.firsttwo = DoubleConv(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)
        self.up4 = UpBlock(1024)
        self.up3 = UpBlock(512)
        self.up2 = UpBlock(256)
        self.up1 = UpBlock(128)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x_in):
        x1 = self.firsttwo(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        return self.final(x)


#2D DeepMedic
class Pathway(nn.Module):
    def __init__(self, in_channels):
        super(Pathway, self).__init__()
        self.block1 = DoubleConv(in_channels, 30)
        self.block2 = DoubleConv(30, 40)
        self.block3 = DoubleConv(40, 40)
        self.block4 = DoubleConv(40, 50)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.block4(x)


class DeepMedic(nn.Module):
    def __init__(self, in_channels_upper, n_classes, in_channels_lower, dummy=None):
        super(DeepMedic, self).__init__()
        self.in_channels_u = in_channels_upper
        self.n_classes = n_classes
        self.in_channels_l = in_channels_lower

        self.upperPath = Pathway(in_channels_upper)
        self.lowerPath = Pathway(in_channels_lower)

        #self.upsample = nn.ConvTranspose2d(50, 50, kernel_size=2, stride=2)
        self.final = nn.Sequential(
            DoubleConv(100, 150, kernel_size=1),
            nn.Conv2d(150, n_classes, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x_upper, x_lower):
        x_u = self.upperPath(x_upper)
        x_l = self.lowerPath(x_lower)
        h, w = x_u.size(2), x_u.size(3)
        #x_l = self.upsample(x_l) #could we use transpConv instead of simple upsampling??
        x_l = nn.functional.interpolate(x_l, size=(h,w), mode='nearest') #size=x_u.shape
        x = torch.cat([x_u, x_l], dim=1)
        return self.final(x)


#PSP-Net
class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PyramidPooling, self).__init__()

        self.channels_per_pool = in_channels//4
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.level1 = self.get_level(1)
        self.level2 = self.get_level(2)
        self.level3 = self.get_level(3)
        self.level4 = self.get_level(6)

        self.final = nn.Conv2d(in_channels+channels_per_pool*4, out_channels, kernel_size=1)
    
    def get_level(self, size):
        level = nn.Sequential(
            nn.AdaptiveAvgPool2d((size, size)),
            nn.Conv2d(self.in_channels, self.channels_per_pool, kernel_size=1, bias=False)
        )
        return level
    
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x1 = nn.functional.upsample(self.level1(x),  size=(h, w), mode='bilinear')
        x2 = nn.functional.upsample(self.level2(x),  size=(h, w), mode='bilinear')
        x3 = nn.functional.upsample(self.level3(x),  size=(h, w), mode='bilinear')
        x4 = nn.functional.upsample(self.level4(x),  size=(h, w), mode='bilinear')
        x = torch.cat([x,x4, x3, x2, x1], dim=1)
        return self.final(x)


class UpsamplingConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=2):
        super(UpsamplingConv, self).__init__()

        self.f = factor
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = self.f*x.size(2), self.f*x.size(3)
        x = nn.functional.upsample(input=x, size=(h, w), mode='bilinear') #should we try with learning instead? ie with TransposeConv?
        return self.block(x)



class PSPNet(nn.Module):
    def __init__(self, in_channels, n_classes, dummy=None, extractor_net='resnet34'):
        super(PSPNet, self).__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.get_features = self.get_pretrained(extractor_net)

        self.PSPmodule = PyramidPooling(in_channels, in_channels)
        self.PSPdrop = nn.Dropout2d(p=0.3)
        self.to_orig_size = nn.Sequential(
            UpsamplingConv(in_channels, 512),
            nn.Dropout2d(p=0.15),
            UpsamplingConv(512, 256),
            nn.Dropout2d(p=0.15),
            UpsamplingConv(256, 64),
            nn.Dropout2d(p=0.15)   #do we need all these dropouts?
        )
        self.final = nn.Sequential( 
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def get_pretrained(self, network, pretrained=True):
        return globals()[network](pretrained)

    def forward(self, x):
        feats, for_deep_loss = self.get_features(x) #here we assume the output size will be 1/8 of original img size
        #for now we don't use the auxiliary loss to improve training. so ignore for_deep_loss. 
        x = self.PSPmodule(feats)
        x = self.PSPdrop(x)
        x = self.to_orig_size(x)
        return self.final(x)


class ResidualConvBlock(DoubleConv):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None, dilation=1):
        super(ResidualConvBlock, self).__init__(in_channels, out_channels, stride=stride, dilation=dilation, padding=dilation, kernel_size=kernel_size)

        self.expansion = 1
        self.downsample = downsample
        self.stride = stride 
        self.dilation = dilation

    def forward(self, x):
        y = self.block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        y += x
        return self.act(y)



class DilatedResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 6, 3)):
        super(DilatedResNet, self).__init__()

        self.in_channels = 64
        self.startblock = nn.Sequential( 
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(), #PRELU?
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=1, dilation=4)

    def make_layer(self, block, in_channels, howmany, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.in_channels != in_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, in_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(in_channels * block.expansion),
            )

        layers = [block(self.in_channels, in_channels, stride, downsample)]
        self.in_channels = in_channels * block.expansion
        for i in range(1, howmany):
            layers.append(block(self.in_channels, in_channels, dilation=dilation))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.startblock(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        y = self.layer3(x)
        x = self.layer4(y)

        return x, y #return both for deep loss

    


#################loading pretrained nets

def load_weights_sequential(target, source_state):
    new_dict = OrderedDict()
    for (k1, v1), (k2, v2) in zip(target.state_dict().items(), source_state.items()):
        new_dict[k1] = v2
    target.load_state_dict(new_dict)


def resnet18(pretrained=True):
    model = DilatedResNet(ResidualConvBlock, [2, 2, 2, 2])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=True):
    model = DilatedResNet(ResidualConvBlock, [3, 4, 6, 3])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet34']))
    return model

#################weight initialization
def weights_init(network):

    def PSPinit(m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def generalInit(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
    if network=='PSPNet':
        return PSPinit
    return generalInit