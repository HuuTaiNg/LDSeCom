import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch

class senderModel(nn.Module):
    def __init__(self, num_classes, num_loops=5):
        super(senderModel, self).__init__()
        self.num_loops = num_loops
        self.maxPool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=7, padding=6, stride=1, dilation=2)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=7, padding=6, stride=1, dilation=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(3, 64, kernel_size=5, padding=4, stride=1, dilation=2)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=4, stride=1, dilation=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3_1 = nn.Conv2d(3, 64, kernel_size=3, padding=2, stride=1, dilation=2)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=2, stride=1, dilation=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4_1 = nn.Conv2d(3, 64, kernel_size=1, padding=0, stride=1)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn = nn.BatchNorm2d(256)
        self.conv_out1 = nn.Conv2d(256, 64, kernel_size=3, padding=2, stride=1, dilation=2)
        self.conv_out2 = nn.Conv2d(64, num_classes, kernel_size=3, padding=2, stride=1, dilation=2)

    def forward(self, x):
        x1 =  F.relu(self.conv1_1(x))
        x1 = self.maxPool(x1)
        x2 =  F.relu(self.conv2_1(x))
        x2 = self.maxPool(x2)
        x3 =  F.relu(self.conv3_1(x))
        x3 = self.maxPool(x3)
        x_flipped = torch.flip(x, dims=[3])
        x4 = F.relu(self.conv4_1(x_flipped))
        x4 = self.maxPool(x4)
        for i in range(self.num_loops):
            x1 =  F.relu(self.conv1_2(x1))
            x2 =  F.relu(self.conv2_2(x2))
            x3 =  F.relu(self.conv3_2(x3))
            x4 =  F.relu(self.conv4_2(x4))
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.bn(self.upsample(x))
        x = F.relu(self.conv_out1(x))
        x =  self.conv_out2(x) 
        return F.softmax(x, dim=1)
    
class receiverModel(nn.Module):
    def __init__(self):
        super(receiverModel, self).__init__()
        self.linear_t1 = nn.Linear(1, 256) 
        self.linear_t2 = nn.Linear(256, 256) 
        self.conv_img = nn.Conv2d(3, 256, kernel_size=3, padding=2, stride=1, dilation=2)
        self.conv_segment = nn.Conv2d(1, 256, kernel_size=3, padding=2, stride=1, dilation=2)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.gap_layer = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(512)
        self.flatten = nn.Flatten()

        self.conv1_1 = nn.Conv2d(512, 256, kernel_size=5, padding=4, stride=1, dilation=2)
        self.conv1_2 = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1)
        self.fc1_3 = nn.Linear(512, 256) 

        self.conv2_1 = nn.Conv2d(256, 64, kernel_size=5, padding=4, stride=1, dilation=2)
        self.conv2_2 = nn.Conv2d(256, 64, kernel_size=1, padding=0, stride=1)
        self.fc2_3 = nn.Linear(256, 64) 

        self.conv3_1 = nn.Conv2d(64, 3, kernel_size=5, padding=4, stride=1, dilation=2)
        self.conv3_2 = nn.Conv2d(64, 3, kernel_size=1, padding=0, stride=1)
        self.fc3_3 = nn.Linear(64, 3) 


    def forward(self, image_orginal, segmented_image, t, total_time):
        t = t.view(-1, 1).float() / total_time
        t = F.relu(self.linear_t1(t))
        t = F.relu(self.linear_t2(t))
        t = t.view(-1, 256, 1, 1)

        x1 = F.relu(self.conv_img(image_orginal))
        x1 = self.maxPool(x1) 
        x2 = F.relu(self.conv_segment(segmented_image.float()))
        x2 = self.maxPool(x2) + t

        x = self.bn(torch.cat([x1, x2], dim=1)) 
        x1_1 = self.conv1_1(x)
        x1_2 = self.conv1_2(x)
        x1_3 = self.gap_layer(x)
        x1_3 = self.flatten(x1_3)
        x1_3 = self.fc1_3(x1_3)
        x1_3 = x1_3.view(x1_3.size(0), 256, 1, 1)
        x = F.relu(x1_1 * x1_3 + x1_2)

        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.gap_layer(x)
        x2_3 = self.flatten(x2_3)
        x2_3 = self.fc2_3(x2_3)
        x2_3 = x2_3.view(x2_3.size(0), 64, 1, 1)
        x = F.relu(x2_1 * x2_3 + x2_2)

        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.gap_layer(x)
        x3_3 = self.flatten(x3_3)
        x3_3 = self.fc3_3(x3_3)
        x3_3 = x3_3.view(x3_3.size(0), 3, 1, 1)
        x = F.relu(x3_1 * x3_3 + x3_2)
        x = self.upsample(x)
        return x