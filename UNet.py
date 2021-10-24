import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm




class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        contract1 =  torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=(3,3), stride=1, padding=0), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2,2), stride=2))
        
        contract2 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=0), 
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=(2,2), stride=2))
        
        contract3 = torch.nn.Sequential(torch.nn.Conv2d(128, 256, kernel_size=(3,3), stride=1, padding=2),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=(2,2), stride=2))


        contract4 = torch.nn.Sequential(torch.nn.Conv2d(256, 512, kernel_size=(3,3), stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(512, 512, kernel_size=(3,3), stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=(2,2), stride=2))
        
        contract5 = torch.nn.Sequential(torch.nn.Conv2d(512, 1024, kernel_size=(3,3), stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(1024, 1024, kernel_size=(3,3), stride=1, padding=0),
                    torch.nn.ReLU())

        expand1 = torch.nn.Sequential(torch.nn.Upsample(size=(56, 56)),
                torch.nn.Conv2d(1024, 512, kernel_size=(3,3), stride=1, padding=0)
                ##continue)


