import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import math
import utils as utils

class AttentionBranch(nn.Module):

    def __init__(self, conv_flag,residual_connection_flag):
        super(AttentionBranch, self).__init__()

        self.conv_flag = conv_flag
        self.residual_connection_flag = residual_connection_flag
        if self.conv_flag:

            self.attention_branch_base_unit = nn.Sequential(

                nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
                nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1, padding=1),
                nn.BatchNorm2d(512), ####### we meed to confirm that how to use batch normlization ######
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            self.attention_mask = AttentionMask('dot')

        else:  ##### we use recurrent neural network to catch spatial context

            self.attention_branch_base_unit = nn.Sequential(

                nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
            )

    def forward(self,input,upsampling_shape):
        
        if self.conv_flag:
            self.basic_unit = self.attention_branch_base_unit(input)
        else:
            self.basic_unit = self.attention_branch_base_unit(input)

        self.base_unit = F.upsample_bilinear(self.basic_unit,size=upsampling_shape[2:])

        self.attention_weights = self.attention_mask(self.basic_unit,upsampling_shape)

        return self.attention_weights

class AttentionMask(nn.Module):

    def __init__(self,context_match_med):
        super(AttentionMask, self).__init__()

        self.context_match_med = context_match_med
        self.attention_weights = []

    '''''''''
    core idea comes from 'Effective Approaches to Attention-based Neural Machine Translation'
    Minh-Thang Luong
    There are three models defined in paper, which are simple dot,general and concat
               for each alpha = transpose(h_s)*(for each h_j in bottom)
    '''''''''
    def attention_model_simple_dot(self, hidden_unit, bottom,c):

        #hidden_unit = hidden_unit.view(1 ,1 ,c)
        #hidden_unit_numpy = hidden_unit.data.cpu().numpy()
        ### make test
        #bottom = bottom.transpose(1, 2)
        #bottom_numpy = bottom.data.cpu().numpy()
        #alpha = torch.bmm(hidden_unit,bottom)
        alpha = torch.mv(torch.squeeze(bottom),hidden_unit)
        #alpha_numpy = alpha.data.cpu().numpy()
        #alpha_prob = F.softmax(torch.squeeze(alpha))
        alpha_prob = F.softmax(alpha)
        #alpha_prob_numpy = alpha_prob.data.cpu().numpy()
        return alpha_prob

    def forward(self,cnn_features,upsampling_shape):

        b ,c ,h, w = cnn_features.size()
        ###### we set batch_size =1 for simplify #######
        locations ,attention_weights,spatial_contxt= h * w,[],[]
        #cnn_features_flatten = cnn_features.view(1, h * w, c)
        cnn_features_flatten = cnn_features.view(1, c, h * w).transpose(2,1)
        for loc_element in range(locations):

            loc_feature = torch.squeeze(cnn_features_flatten)[loc_element]
            if self.context_match_med == 'dot':

                loc_attention_weights = self.attention_model_simple_dot(loc_feature,cnn_features_flatten,c)
                loc_attention_weights_align = loc_attention_weights.view(1, 1, h, w)
                #loc_attention_weights_align = F.upsample_bilinear(loc_attention_weights,upsampling_shape[2:])

            if self.context_match_med == 'gerenate':
                pass

            attention_weights.append(torch.squeeze(loc_attention_weights_align))
        attention_weights = torch.stack(attention_weights,0)

        return attention_weights

class Attention_Weighted_Context_Generation(nn.Module):

    def __init__(self):
        super(Attention_Weighted_Context_Generation, self).__init__()

    def forward(self,weights,cnn_feature):

        b ,c ,h, w = cnn_feature.size()
        locations,spatial_contxt= h * w,[]
        #cnn_features_flatten = cnn_feature.view(1, h * w, c)
        cnn_features_flatten = cnn_feature.view(1, c, h * w).transpose(2,1)
        for loc_element in range(locations):

            loc_attention_weights = weights[loc_element]
            loc_attention_weights_flatten = loc_attention_weights.view(1,1,h*w)
            ctx_attention = torch.bmm(loc_attention_weights_flatten,cnn_features_flatten)
            spatial_contxt.append(ctx_attention)

        spatial_contxt = torch.stack(spatial_contxt,dim=0)
        return spatial_contxt

class Adaptive_Scale_Feature_Embedding(nn.Module):

    def __init__(self,embedding_dim,out_feature_dim):
        super(Adaptive_Scale_Feature_Embedding, self).__init__()
        self.attention_context_embedding_dim = embedding_dim
        self.original_feature_embedding_dim = embedding_dim
        self.out_feature_dim = out_feature_dim
        self.contxt_embedding_matrix = Parameter(torch.FloatTensor(self.attention_context_embedding_dim,self.out_feature_dim))
        self.context_embedding_bias = Parameter(torch.FloatTensor(self.out_feature_dim))
        self.cnn_feature_embedding_matrix = Parameter(torch.FloatTensor(self.original_feature_embedding_dim,self.out_feature_dim))
        self.cnn_feature_embedding_bias = Parameter(torch.FloatTensor(self.out_feature_dim))
        self.init_parameter()
        #self.contxt_embedding = nn.Linear(self.attention_context_embedding_dim,self.attention_context_embedding_dim,bias=False)
        #self.cnn_feature_embedding = nn.Linear(self.orginal_feature_embedding_dim,self.orginal_feature_embedding_dim,bias=False)

    def init_parameter(self):

        stdv = 1. / math.sqrt(self.contxt_embedding_matrix.size(1))
        self.contxt_embedding_matrix.data.uniform_(-stdv, stdv)
        self.context_embedding_bias.data.uniform_(-stdv, stdv)
        stdv_f = 1. / math.sqrt(self.cnn_feature_embedding_matrix.size(1))
        self.cnn_feature_embedding_matrix.data.uniform_(-stdv_f,stdv_f)
        self.cnn_feature_embedding_bias.data.uniform_(-stdv_f, stdv_f)

    def forward(self,cnn_feature,attentive_context):

        assert cnn_feature.size() == attentive_context.size()
        b, c, h, w = cnn_feature.size()
        assert b == 1

        bias_embedding = Parameter(torch.ones(1,h * w).cuda(),requires_grad=False)
        #bias_embedding.data.fill_(1.0)

        cnn_feature_embedding_bias = self.cnn_feature_embedding_bias.view(c, 1)

        cnn_feature_embedding_bias = torch.mm(cnn_feature_embedding_bias, bias_embedding)

        #cnn_feature_embedding_bias = torch.mm(self.cnn_feature_embedding_bias.view(c,1),bias_embedding).transpose(1,0)
        context_embedding_bias = torch.mm(self.context_embedding_bias.view(c,1),bias_embedding).transpose(1,0)
        
        cnn_feature_vectorize = cnn_feature.view(c, h * w).transpose(1,0)
        attentive_context_vectorize = attentive_context.view(c, h * w).transpose(1,0)

        #attentive_context_feature_embedding = torch.bmm(cnn_feature_vectorize,self.cnn_feature_embedding_matrix) + self.cnn_feature_embedding_bias
        attentive_context_feature_embedding = torch.mm(attentive_context_vectorize,
                                                   self.contxt_embedding_matrix)

        attentive_context_feature_embedding = torch.add(attentive_context_feature_embedding,context_embedding_bias)

        cnn_feature_embedding = torch.add(torch.mm(cnn_feature_vectorize,
                                      self.cnn_feature_embedding_matrix), cnn_feature_embedding_bias)
        #cnn_feature_embedding = torch.bmm(attentive_context_vectorize,self.contxt_embedding_matirx) + self.context_embedding_bias
        #attentive_context_feature_embedding = self.contxt_embedding(attentive_context_vectorize)
        #cnn_feature_embedding = self.cnn_feature_embedding(cnn_feature_vectorize)
        #adaptive_scale_features = attentive_context_feature_embedding + cnn_feature_embedding
        adaptive_scale_features = (attentive_context_feature_embedding + cnn_feature_embedding).transpose(1,0)
        '''''''''
        adaptive_scale_features = torch.bmm(contxt_embedding_matrix,attentive_context_vectorize) \
                                  + torch.bmm(cnn_feature_embedding_matrix,cnn_feature_vectorize) + bias
        '''''''''
        adaptive_scale_features = F.relu(adaptive_scale_features.contiguous().view(cnn_feature.size()))
        return adaptive_scale_features

 # todo
'''''''''
(1) spatial_lstm
(2) tensorboard
(3) different learning rate for various layers
(4) poly learning policy
'''''''''

class Spatial_Identity_RNN(nn.Module):

    def __init__(self,input_embedding_dim,hidden_embedding_dim,activation_med= 'RELU'):
        super(Spatial_Identity_RNN, self).__init__()

        self.hidden_transition_matrix = Parameter(torch.FloatTensor(input_embedding_dim,hidden_embedding_dim))
        self.activation_med = activation_med

    def forward(self,cnn_features):

        b,c,h,w = cnn_features.size()
        self.cnn_sequence = utils.convert_cnn_features2temporal_sequence(cnn_features)
        for i in range(w):
            pass

class SpatialContextEmbeddingWithRNN(nn.Module):

    def __init__(self,RNNCellUnit,hidden_embedding_dim,output_embedding_dim):
        super(SpatialContextEmbeddingWithRNN, self).__init__()

        self.rnn_cell_cat = ['LSTM','GRU','IRNN']
        self.rnn_cell_unit = RNNCellUnit
        self.hidden_units_embedding_dim = hidden_embedding_dim
        self.out_units_embedding_dim = output_embedding_dim

    def single_direction_spatial_context_embedding(self, input, size , cell_unit_type):

        height,width = size
        start_pt = 0
        for idx in range(width):

            input_sample = input[start_pt: start_pt + (idx + 1) * width]
            if cell_unit_type == 'LSTM':
                pass

            if cell_unit_type == 'GRU':
                pass

            if cell_unit_type == 'IRNN':

                pass

    def forward(self,input):

        assert len(input.size()) == 4
        b, c, h, w = input
        input_single_batch = input.view(h * w, c)