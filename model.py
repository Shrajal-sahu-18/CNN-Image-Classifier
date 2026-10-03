import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv_layers = nn.Sequential(
            # First Convonutional Layer
            nn.Conv2d(3,32,kernel_size =3,padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # kernel = 2 ,stride = 2

            # Second Convonutional layer
            nn.Conv2d(32,64,kernel_size = 3,padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # kernel = 2 ,stride = 2

            # Third Layer
            nn.Conv2d(64,128,kernel_size = 3,padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # kernel = 2 ,stride = 2
            
            
            
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(4*4*128,256),
            nn.ReLU(),

            nn.Linear(256,10),
        )

    def forward(self,x):  #upper constructor hai call niche say hoga
        x = self.conv_layers(x)
        x = x.view(x.size(0),-1) # Flattening
        x = self.fc_layers(x)
        return x