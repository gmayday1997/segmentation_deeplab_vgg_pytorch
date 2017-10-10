import torch
import torchvision 
import numpy as np
import torch.nn.functional as F
import json
import vgg1024 as vggs
import cv2

def load_pretrain_model(model_file):

   model = torchvision.models.vgg16(pretrained=False)
   state_dict = torch.load(model_file)
   model.load_state_dict(state_dict)
   print('model has been load')
   return model

def load_deeplab_pretrain_model(model_file):

   model = vggs.vgg1024()
   state_dict = torch.load(model_file)
   model.load_state_dict(state_dict)
   print('model has been load')
   return model

def convert_cnn_features2temporal_sequence(cnn_features):

    b,c,h,w = cnn_features.size()
    ###### we need
    cnn_features_flatten = cnn_features.view(c,h * w)
    cnn_sequence = torch.chunk(cnn_features_flatten,chunks=h)
    return cnn_sequence

###### source code from https://github.com/meetshah1995/pytorch-semseg #####
def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def adjust_learning_rate(learning_rate,optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 25))
    print(str(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save2json(metric_dict,save_path):
    file_ = open(save_path,'w')
    file_.write(json.dumps(metric_dict,ensure_ascii=False,indent=2))
    file_.close()

def attention_weights_collection(attention_weights):


    loc_weights_dict = {}
    locs,height,width = attention_weights.shape
    for idx in range(locs):
       loc_weights = attention_weights[idx,:,:]
       loc_attention_vec = np.reshape(loc_weights,(height * width))
       max_ = np.max(loc_attention_vec,axis=0)
       if max_ != 0:
           loc_attention_vec = loc_attention_vec/max_
           loc_attention = loc_attention_vec.reshape(height,width)
       loc_weights_dict.setdefault(idx,loc_attention)

    return loc_weights_dict

def attention_weights_visulize(weights_dict,original_img,save_base_path):

    for idx,loc_attention_weight_vec in weights_dict.iteritems():

        height, width, channel = original_img.shape
        alpha_att_map = cv2.resize(loc_attention_weight_vec, (width,height), interpolation=cv2.INTER_LINEAR)
        alpha_att_map_ = cv2.applyColorMap(np.uint8(255 * alpha_att_map), cv2.COLORMAP_JET)
        fuse_heat_map = 0.6 * alpha_att_map_ + 0.4 * original_img
        cv2.imwrite(save_base_path + '_' + str(idx) + '.jpg',fuse_heat_map)
        print idx