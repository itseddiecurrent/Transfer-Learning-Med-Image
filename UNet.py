import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision 




class UNet(torch.nn.Module):
    def __init__(self):

        super(UNet, self).__init__()

        self.contract1 =  torch.nn.Sequential(
                    torch.nn.Conv2d(1, 64, kernel_size=(3,3), stride=1, padding=0), 
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=0),
                    torch.nn.ReLU())

        self.crop1 = torchvision.transforms.CenterCrop(size=392)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2)
        
        self.contract2 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=0), 
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=(2,2), stride=2))

        self.crop2 = torchvision.transforms.CenterCrop(size=200)
        
        self.contract3 = torch.nn.Sequential(torch.nn.Conv2d(128, 256, kernel_size=(3,3), stride=1, padding=2),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=(2,2), stride=2))

        self.crop3 = torchvision.transforms.CenterCrop(size=104)

        self.contract4 = torch.nn.Sequential(torch.nn.Conv2d(256, 512, kernel_size=(3,3), stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(512, 512, kernel_size=(3,3), stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=(2,2), stride=2))

        self.crop4 = torchvision.transforms.CenterCrop(size=56)
        
        self.contract5 = torch.nn.Sequential(torch.nn.Conv2d(512, 1024, kernel_size=(3,3), stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(1024, 1024, kernel_size=(3,3), stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.Upsample(scale_factor=4))

        self.upsample = torch.nn.Upsample(scale_factor=4)

        self.expand1 = torch.nn.Sequential(
                torch.nn.Conv2d(1024, 512, kernel_size=(3,3), stride=1, padding=0),
                torch.nn.ReLU(), 
                torch.nn.Conv2d(512, 256, kernel_size=(3,3), stride=1, padding=0),
                torch.nn.ReLU())

        self.expand2 = torch.nn.Sequential(
                torch.nn.Conv2d(512, 256, kernel_size=(3,3), stride=1, padding=0),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=0),
                torch.nn.ReLU())

        self.expand3 = torch.nn.Sequential(
                torch.nn.Conv2d(256, 128, kernel_size=(3,3), stride=1, padding=0),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=0),
                torch.nn.ReLU())

        self.expand4 = torch.nn.Sequential(
                torch.nn.Conv2d(128, 64, kernel_size=(3,3), stride=1, padding=0),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=0),
                torch.nn.ReLU())

        self.out = torch.nn.Conv2d(64, 2, kernel_size=(1,1), stride=1, padding=0)


        def forward_pass(self, input):
            '''in: input training sample'''
            
            down1 = self.contract1(input)
            cropped1 = self.crop1(down1)

            down1 = self.maxpool(down1)
            down2 = self.contract2(down1)
            cropped2 = self.crop2(down2)

            down2 = self.maxpool(down2)
            down3 = self.contract3(down2)
            cropped3 = self.crop3(down3)

            down3 = self.maxpool(down3)
            down4 = self.contract4(down3)
            cropped4 = self.crop4(down4)

            down5 = self.contract5(down4)

            upsampled1 = self.upsample(down5)

            concat1 = torch.cat((cropped4, upsampled1), dim=2)
            up1 = self.expand1(concat1)

            upsampled2 = self.upsample(up1)
            concat2 = torch.cat((cropped3, upsampled2))

            up2 = self.expand2(concat2)

            upsampled3 = self.upsample(up2)
            concat3 = torch.cat((cropped2, upsampled3), dim=2)

            up3 = self.expand3(concat3)

            upsampled4 = self.upsample(up3)
            concat4 = torch.cat((cropped1, upsampled4), dim=2)

            up4 = self.expand4(concat4)

            out = self.out(up4)

        def train_step(self, minibatch):
            input, target = minibatch
            input = input.to(device)
            target = target.to(device)
            prediction = self.forward_pass(input)
            ##TODO: compute loss, update SummaryWriter, backwardpass



if __name__== "main":
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device = 'cuda:0'
    else:
        device = 'cpu'


