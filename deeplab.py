import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import AttentionBranch as AB

class deeplab(nn.Module):

    def __init__(self):
        super(deeplab, self).__init__()

        self.conv_flag = True
        self.use_residual = False

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,ceil_mode=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,ceil_mode=True),
        )
        self.conv3 = nn.Sequential(

            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1 ,ceil_mode=True),
        )

        self.conv4 = nn.Sequential(

            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels= 512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1,padding=1,ceil_mode=True),
        )

        self.conv5 = nn.Sequential(

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, dilation=2 ,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, dilation=2 ,stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, dilation=2, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3 ,stride=1, padding=1,ceil_mode=True),
            nn.AvgPool2d(kernel_size=3 , stride=1, padding=1,ceil_mode=True),
        )

        self.fc6 = nn.Sequential(

            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,dilation=12,padding=12),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.fc7 = nn.Sequential(

            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )
        self.score = nn.Conv2d(in_channels=1024,out_channels=21,kernel_size=1)

        self.AttentionBranch = AB.AttentionBranch(self.conv_flag,self.use_residual)

        self.WeightedContextEmedding = AB.Attention_Weighted_Context_Generation()

        self.Adaptive_Scale_Feature_Embedding = AB.Adaptive_Scale_Feature_Embedding(embedding_dim=1024,out_feature_dim=1024)

    def forward(self,input):

        x= self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        upsampling_shape = x.size()
        self.attention_weights = self.AttentionBranch(x,upsampling_shape)
        self.cnn_features = self.fc6(x)
        self.cnn_features = F.max_pool2d(self.cnn_features,kernel_size=2,stride=2,ceil_mode=True)
        self.weighted_cnn_feature = self.WeightedContextEmedding(self.attention_weights,self.cnn_features)
        self.weighted_cnn_feature = torch.squeeze(self.weighted_cnn_feature).transpose(1,0).contiguous().view(self.cnn_features.size())
        self.adptive_scale_feature = self.Adaptive_Scale_Feature_Embedding(self.cnn_features,self.weighted_cnn_feature)
        if self.use_residual:

            self.features = self.cnn_features + self.weighted_cnn_feature

        #self.features = F.upsample_bilinear(self.features,upsampling_shape[2:])
        x = self.fc7(self.adptive_scale_feature)
        x = self.score(x)
        out = F.upsample_bilinear(x, input.size()[2:])
        return out,self.attention_weights

    def init_parameters(self,pretrain_vgg16_1024):

        ##### init parameter using pretrain vgg16 model ###########
        conv_blocks = [self.conv1,
                       self.conv2,
                       self.conv3,
                       self.conv4,
                       self.conv5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(pretrain_vgg16_1024.features.children())

        for idx, conv_block in enumerate(conv_blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    # print idx, l1, l2
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data

        ####### init fc parameters (transplant) ##############
        self.fc6[0].weight.data = pretrain_vgg16_1024.classifier[0].weight.data.view(self.fc6[0].weight.size())
        self.fc6[0].bias.data = pretrain_vgg16_1024.classifier[0].bias.data.view(self.fc6[0].bias.size())
        '''''''''
        self.fc7[0].weight.data = pretrain_vgg16_1024.classifier[3].weight.data.view(self.fc7[0].weight.size())
        self.fc7[0].bias.data = pretrain_vgg16_1024.classifier[3].bias.data.view(self.fc7[0].bias.size())
        '''''''''
        '''''''''
        self.score.weight.data = pretrain_vgg16_1024.classifier[6].weight.data.view(self.score.weight.size())
        self.score.bias.data = pretrain_vgg16_1024.classifier[6].bias.data.view(self.score.bias.size())
        '''''''''
