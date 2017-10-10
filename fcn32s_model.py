import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class AttentionBranch(nn.Module):

    def __init__(self,conv_flag=True):
        super(AttentionBranch, self).__init__()
        if conv_flag:
           self.basicMode = nn.Sequential(

               nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1),
               nn.ReLU(inplace=True),
               nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
               nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,dilation=1),
               nn.ReLU(inplace=True),
           )

    def build_Gram_Matrix(self,x):

        a, b, c, d = x.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        Gram_Matrix = []
        x_reshape = x.view(c*d, a*b)
        for loc in range(c*d):

            loc_feat = x_reshape[loc]
            loc_gram_matrix = torch.mm(loc_feat,loc_feat.t())
            Gram_Matrix.append(loc_gram_matrix)

        return Gram_Matrix
        '''''''''
        features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)
        '''''''''

    def rebuild_features(self,gram_matrix,input):

        batch_size,height,width,nchannel = input.size()
        input_resize = input.view(height,width,nchannel)
        for loc in range(height*width):
            pass

    def forward(self,input):

        x = self.basicMode(input)
        gram_matrix = self.build_Gram_Matrix(x)
        features = self.rebuild_features(gram_matrix,input)
        return features

###### source code from tensorflow version ########
def whole_img_attention(self, bottom):
    #### assun batch_size == 1
    #### l2_normalization
    bottom_norm = tf.nn.l2_normalize(bottom, dim=3, epsilon=1e-12)
    assert len(bottom_norm.get_shape().as_list()) == 4
    batch_size, height, width, nchannel = bottom.get_shape().as_list()
    bottom_seq = 100 * tf.reshape(bottom_norm, [height * width, nchannel])
    time_step = height * width
    alpha = []
    ctx_attention = []
    for _step in range(time_step):
        hidden_step = bottom_seq[_step]
        alpha_ = self.attention_simple_dot(hidden_step, bottom_seq)
        # alpha_ = self.attention_model_generate_(hidden_i,out_seq)
        ctx_attention_step = tf.reduce_sum(tf.expand_dims(alpha_, 1) * bottom_seq, 0)
        ctx_attention.append(ctx_attention_step)
        alpha.append(alpha_)
    context_att_tensor_ = tf.pack(ctx_attention, axis=0)
    shape_ = [batch_size, height, width, nchannel]
    context_att_ = tf.reshape(context_att_tensor_, shape_)
    alphas = tf.pack(alpha, axis=0)
    a_shape = [time_step, height, width]
    alphas_ = tf.reshape(alphas, a_shape)
    print(context_att_.get_shape())
    return context_att_, alphas_

class fcn32s(nn.Module):

    def __init__(self):

        super(fcn32s, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
        )
        self.conv3 = nn.Sequential(

            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
        )

        self.conv4 = nn.Sequential(

            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels= 512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,ceil_mode=True),
        )

        self.conv5 = nn.Sequential(

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
        )

        self.fc6 = nn.Sequential(

            nn.Conv2d(in_channels=512,out_channels=4096,kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        self.fc7 = nn.Sequential(

            nn.Conv2d(in_channels=4096,out_channels=4096,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        self.score = nn.Conv2d(in_channels=4096,out_channels=21,kernel_size=1)

        self.upscore = nn.ConvTranspose2d(in_channels=21,out_channels=21,kernel_size=64,stride=32,bias=False)

    def forward(self,input):

        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        self.conv_features = self.conv5(x)
        x = self.fc6(self.conv_features)
        x = self.fc7(x)
        score = self.score(x)
        #out = F.upsample_bilinear(score, input.size()[2:])
        upscore = self.upscore(score)
        upscore = upscore[:, :,19:19+input.size()[2], 19:19+input.size()[3]].contiguous()
        #n_size = upscore.size()
        return upscore,self.conv_features

    def init_parameters(self,pretrain_vgg16):

        ##### init parameter using pretrain vgg16 model ###########
        conv_blocks = [self.conv1,
                       self.conv2,
                       self.conv3,
                       self.conv4,
                       self.conv5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(pretrain_vgg16.features.children())

        for idx, conv_block in enumerate(conv_blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    # print idx, l1, l2
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data

        ####### init fc parameters (transplant) ##############

        self.fc6[0].weight.data = pretrain_vgg16.classifier[0].weight.data.view(self.fc6[0].weight.size())
        self.fc6[0].bias.data = pretrain_vgg16.classifier[0].bias.data.view(self.fc6[0].bias.size())
        self.fc7[0].weight.data = pretrain_vgg16.classifier[3].weight.data.view(self.fc7[0].weight.size())
        self.fc7[0].bias.data = pretrain_vgg16.classifier[3].bias.data.view(self.fc7[0].bias.size())

        ###### random init socore layer parameters ###########
        assert  self.upscore.kernel_size[0] == self.upscore.kernel_size[1]
        initial_weight = get_upsampling_weight(self.upscore.in_channels, self.upscore.out_channels, self.upscore.kernel_size[0])
        self.upscore.weight.data.copy_(initial_weight)

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()
