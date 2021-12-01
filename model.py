import torch
from torch import nn
import torch.nn.functional as F

# class VGGnetwork(torch.nn.Module):
#     def __init__(self, class_num = 2):
#         super(VGGnetwork, self).__init__()
#         self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
#         self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv6 = nn.Linear(7*7*512, 4096)
#         self.conv7 = nn.Linear(4096, 4096)
#         self.conv8 = nn.Linear(4096, class_num)
#
#         self.convNet = nn.Sequential(
#             self.conv1_1, nn.ReLU(), self.conv1_2, nn.ReLU(), self.pool1,
#             self.conv2_1, nn.ReLU(), self.conv2_2, nn.ReLU(), self.pool2,
#             self.conv3_1, nn.ReLU(), self.conv3_2, nn.ReLU(), self.conv3_3, nn.ReLU(), self.pool3,
#             self.conv4_1, nn.ReLU(), self.conv4_2, nn.ReLU(), self.conv4_3, nn.ReLU(), self.pool4,
#             self.conv5_1, nn.ReLU(), self.conv5_2, nn.ReLU(), self.conv5_3, nn.ReLU(), self.pool5)
#
#         self.output = nn.Sequential(
#             self.conv6,
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             self.conv7,
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             self.conv8
#         )
#
#         # for i in self.modules():
#         #     if isinstance(i, nn.Conv2d):
#         #         nn.init.kaiming_normal_(i.weight, mode='fan_out', nonlinearity='relu')
#         #     elif isinstance(i, nn.Linear):
#         #         nn.init.normal_(i.weight, 0, 0.01)
#         #         nn.init.constant_(i.bias, 0)
#
#
#     def forward(self, input):
#         x = self.convNet(input)
#         x = x.view(x.size(0),-1)
#         x = self.output(x)
#         return x


class VGGnetwork(torch.nn.Module):
    def __init__(self, class_num = 2):
        super(VGGnetwork, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Linear(8*8*512, 1024)
        self.conv7 = nn.Linear(1024, 256)
        self.conv8 = nn.Linear(256, class_num)

        self.convNet = nn.Sequential(
            self.conv1_1, nn.BatchNorm2d(64), nn.ReLU(), self.conv1_2, nn.BatchNorm2d(64), nn.ReLU(), self.pool1,
            self.conv2_1, nn.BatchNorm2d(128), nn.ReLU(), self.conv2_2, nn.BatchNorm2d(128),  nn.ReLU(), self.pool2,

            self.conv3_1, nn.BatchNorm2d(256), nn.ReLU(), self.conv3_2, nn.BatchNorm2d(256), nn.ReLU(),
            self.conv3_3, nn.BatchNorm2d(64), nn.ReLU(), self.pool3,

            self.conv4_1, nn.BatchNorm2d(512), nn.ReLU(), self.conv4_2, nn.ReLU(),
            self.conv4_3, nn.BatchNorm2d(512), nn.ReLU(), self.pool4)
            # self.conv5_1, nn.ReLU(), self.conv5_2, nn.ReLU(), self.conv5_3, nn.ReLU(), self.pool5)

        self.output = nn.Sequential(
            self.conv6,
            nn.ReLU(),
            nn.Dropout(p=0.5),
            self.conv7,
            nn.ReLU(),
            nn.Dropout(p=0.5),
            self.conv8
        )

        # for i in self.modules():
        #     if isinstance(i, nn.Conv2d):
        #         nn.init.kaiming_normal_(i.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(i, nn.Linear):
        #         nn.init.normal_(i.weight, 0, 0.01)
        #         nn.init.constant_(i.bias, 0)


    def forward(self, input):
        x = self.convNet(input)
        x = x.view(x.size(0),-1)
        x = self.output(x)
        return x

