import torch
import torch.nn as nn
import torch.nn.functional as F
import voc_config as config
import voc_dataset as dates
import deeplab_pure as models
#import deeplab as models
import utils as utils
import metrics as metric
import cv2
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import transforms as trans
from torch.autograd import Variable
import numpy as np
import os
import l2norm as norm

def untransform(transform_img):

    transform_img = transform_img.transpose(1,2,0)
    #transform_img *= [0.229, 0.224, 0.225]
    transform_img += (104.00698793, 116.66876762, 122.67891434)
    transform_img = transform_img.astype(np.uint8)
    transform_img = transform_img[:,:,::-1]
    return transform_img

def attention_model_simple_dot(hidden_unit, bottom, c):

    hidden_unit_numpy = hidden_unit.data.cpu().numpy()
    #hidden_unit = hidden_unit.contiguous().view(1, 1, c)
    ### make test
    #bottom = bottom.transpose(2, 1)
    bottom_numpy = bottom.data.cpu().numpy()
    alpha_prob = torch.mv(torch.squeeze(bottom),hidden_unit)
    #alpha = torch.bmm(hidden_unit, bottom)
    #alpha_numpy = alpha.data.cpu().numpy()
    #alpha_prob = F.softmax(alpha_prob)
    alpha_prob_numpy = alpha_prob.data.cpu().numpy()
    return alpha_prob

def weights_generation(cnn_features,context_match_med='dot'):
    b, c, h, w = cnn_features.size()
    ###### we set batch_size =1 for simplify #######
    locations, attention_weights, spatial_contxt = h * w, [], []
    #cnn_features_flatten = cnn_features.view(1, h * w, c)
    cnn_features_flatten = cnn_features.view(1, c, h * w).transpose(2,1)
    cnn_features_flatten_numyp = cnn_features_flatten.data.cpu().numpy()
    for loc_element in range(locations):

        loc_feature = torch.squeeze(cnn_features_flatten)[loc_element]
        loc_feature_numpy = loc_feature.data.cpu().numpy()
        if context_match_med == 'dot':
            loc_attention_weights = attention_model_simple_dot(loc_feature, cnn_features_flatten, c)
            loc_attention_weights_numpy = loc_attention_weights.data.cpu().numpy()
            loc_attention_weights_align = loc_attention_weights.view(1, 1, h, w)
            loc_attention_weights_align_numpy = loc_attention_weights_align.data.cpu().numpy()

        attention_weights.append(torch.squeeze(loc_attention_weights_align))
    attention_weights = torch.stack(attention_weights, 0)

    return attention_weights

def Attention_Weighted_Context_Generation(weights, cnn_feature):

    b, c, h, w = cnn_feature.size()
    locations, spatial_contxt = h * w, []
    cnn_features_numpy = cnn_feature.data.cpu().numpy()
    cnn_features_flatten = cnn_feature.view(1,c, h*w)
    cnn_features_flatten_numpy = cnn_features_flatten.data.cpu().numpy()
    #cnn_features_flatten = cnn_feature.view(1, h * w, c).transpose(2,1)
    cnn_features_flatten = cnn_feature.view(1,c, h*w).transpose(2,1)
    cnn_features_flatten_numpy = cnn_features_flatten.data.cpu().numpy()
    for loc_element in range(locations):
        loc_attention_weights = weights[loc_element]
        loc_attention_weights_numpy = loc_attention_weights.data.cpu().numpy()
        loc_attention_weights_flatten = loc_attention_weights.view(1, 1, h * w)
        loc_attention_weights_flatten_numpy = loc_attention_weights_flatten.data.cpu().numpy()
        ctx_attention = torch.bmm(loc_attention_weights_flatten, cnn_features_flatten)
        ctx_attention_numpy = ctx_attention.data.cpu().numpy()
        spatial_contxt.append(ctx_attention)

    spatial_contxt = torch.stack(spatial_contxt, dim=0)
    spatial_contxt_numpy = spatial_contxt.data.cpu().numpy()
    return spatial_contxt

def Adaptive_Scale_Feature_Embedding(cnn_feature, attentive_context, contxt_embedding_matrix, cnn_feature_embedding_matrix, cnn_feature_embedding_bias,context_embedding_bias):

    assert cnn_feature.size() == attentive_context.size()
    b, c, h, w = cnn_feature.size()
    assert b == 1

    cnn_feature_numpy = cnn_feature.data.cpu().numpy()
    attentive_context_numpy = attentive_context.data.cpu().numpy()

    bias_embedding = Variable(torch.ones(1,h*w))
    cnn_feature_embedding_bias = torch.mm(cnn_feature_embedding_bias.view(c,1),bias_embedding).transpose(1,0)
    context_embedding_bias = torch.mm(context_embedding_bias.view(c,1),bias_embedding).transpose(1,0)

    cnn_feature_vectorize = cnn_feature.view(c, h * w).transpose(1, 0)
    cnn_feature_vectorize_numpy = cnn_feature_vectorize.data.cpu().numpy()

    attentive_context_vectorize = attentive_context.view(c, h * w).transpose(1, 0)
    attentive_context_vectorize_numpy = attentive_context_vectorize.data.cpu().numpy()

    attentive_context_feature_embedding = torch.mm(attentive_context_vectorize,
                                                   contxt_embedding_matrix)
                                          #+ cnn_feature_embedding_bias
    attentive_context_feature_embedding_numpy = attentive_context_feature_embedding.data.cpu().numpy()

    attentive_context_feature_embedding = torch.add(attentive_context_feature_embedding,context_embedding_bias)

    attentive_context_feature_embedding_numpy = attentive_context_feature_embedding.data.cpu().numpy()

    cnn_feature_embedding = torch.mm(cnn_feature_vectorize,
                                      cnn_feature_embedding_matrix)
    cnn_feature_embedding_numpy = cnn_feature_embedding.data.cpu().numpy()

    cnn_feature_embedding = torch.add(cnn_feature_embedding,cnn_feature_embedding_bias)
    cnn_feature_embedding_numpy = cnn_feature_embedding.data.cpu().numpy()
    # attentive_context_feature_embedding = self.contxt_embedding(attentive_context_vectorize)
    # cnn_feature_embedding = self.cnn_feature_embedding(cnn_feature_vectorize)
    adaptive_scale_features = (attentive_context_feature_embedding + cnn_feature_embedding).transpose(1,0)
    adaptive_scale_features_numpy = adaptive_scale_features.data.cpu().numpy()
    '''''''''
    adaptive_scale_features = torch.bmm(contxt_embedding_matrix,attentive_context_vectorize) \
                              + torch.bmm(cnn_feature_embedding_matrix,cnn_feature_vectorize) + bias
    '''''''''
    adaptive_scale_features = F.relu(adaptive_scale_features.contiguous().view(cnn_feature.size()))
    adaptive_scale_features_numpy = adaptive_scale_features.data.cpu().numpy()

    return adaptive_scale_features

def test_weights_generation():

    #matrix = Variable(torch.IntTensor(3, 3,10).zero_())
    #matrix = Variable(torch.ones(1,10,3,3))
    configs = config.VOC_config()
    img_path = configs.val_img_dir + '/2007_000033.jpg'
    save_pred_dir = os.path.join(configs.save_pred_dir, 'weights')
    save_weights_dir = os.path.join(save_pred_dir, '2007_000033')
    transforms_rgb_rescal = cv2.imread(img_path)
    noise = Variable(torch.FloatTensor([[[[0.5, 0.3,],[0.70 , 0.8]], [[0.6, 0.9],[1,2]], [[3, 0.7],[0.1, 1]],[[1, 0.4],[0.2, 0.3]],[[1, 1],[1, 1]]]]))

    cnn_feature_embedding_matrix = Variable(torch.FloatTensor([[0.1,0,0.3,0.4,0],[0,0.2,0.7,0,0],[0,1,1,0.5,0],[0,0.5,1,0,1],[0,0,1,1,1]]))
    cnn_feature_embedding_matrix_numpy = cnn_feature_embedding_matrix.data.cpu().numpy()
    cnn_feature_embedding_bias = Variable(torch.FloatTensor([0.1,0.1,0.3,0.4,0]))
    cnn_feature_embedding_bias_numpy = cnn_feature_embedding_bias.data.cpu().numpy()
    context_embedding_matrix = Variable(torch.FloatTensor([[0.1,0,0.3,0.4,0],[0,0.2,0.7,0,0],[0,1,1,0.5,0],[0,0.5,1,0,1],[0,0,1,1,1]]))
    context_embedding_matrix_numpy = context_embedding_matrix.data.cpu().numpy()
    context_embedding_bias = Variable(torch.FloatTensor([1,1,1,1,1]))
    context_embedding_bias_numpy = context_embedding_bias.data.cpu().numpy()

    #noise = Variable(torch.FloatTensor(1, 4, 2, 2))
    #noise.data.normal_(0,0.5)
    matrxi_norm_numpy = noise.data.cpu().numpy()
    #torch.renorm()
    weights = weights_generation(noise)
    weights_numpy = weights.data.cpu().numpy()

    #loc_weights = utils.attention_weights_collection(weights_numpy)
    #utils.attention_weights_visulize(loc_weights, transforms_rgb_rescal, save_weights_dir)

    weighted_features = Attention_Weighted_Context_Generation(weights,noise)

    weighted_features_numpy = weighted_features.data.cpu().numpy()
    sq_weighted_features = torch.squeeze(weighted_features)
    sq_weighted_features_numpy = sq_weighted_features.data.cpu().numpy()
    weighted_cnn_feature = torch.squeeze(weighted_features).transpose(1,0).contiguous().view(noise.size())
    weighted_cnn_feature_numpy = weighted_cnn_feature.data.cpu().numpy()

    attentive_feature = Adaptive_Scale_Feature_Embedding(noise,weighted_cnn_feature,context_embedding_matrix,
                                                         cnn_feature_embedding_matrix,cnn_feature_embedding_bias,
                                                         context_embedding_bias)
    attentive_feature_numpy = attentive_feature.data.cpu().numpy()

    return attentive_feature

def main():

    configs = config.VOC_config()
    transform_det = trans.Compose([
        trans.Scale((321, 321)),
    ])

    pretrain_model = os.path.join(configs.save_ckpt_dir,'deeplab_model_best_iu0.57.pth')

    test_data = dates.VOCDataset(configs.val_img_dir, configs.val_label_dir, configs.val_txt_dir, 'val', transform=True,
                                 transform_med=transform_det)

    test_loader = Data.DataLoader(test_data, batch_size=configs.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)
    ######### build vgg model ##########
    deeplab = models.deeplab()
    deeplab = deeplab.cuda()

    checkpoint = torch.load(pretrain_model)
    deeplab.load_state_dict(checkpoint['state_dict'])
    params = list(deeplab.parameters())

    save_pred_dir = os.path.join(configs.save_pred_dir, 'weights')

    if not os.path.exists(save_pred_dir):
        os.mkdir(save_pred_dir)

    deeplab.eval()

    # hook the feature extractor
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    finalconv_name = 'conv5'  # this is the last conv layer of the network
    deeplab._modules.get(finalconv_name).register_forward_hook(hook_feature)

    for batch_idx, batch in enumerate(test_loader):
        inputs, targets, filename, height, width = batch
        inputs, targets = inputs.cuda(), targets.cuda()
        height, width, filename = height.numpy()[0], width.numpy()[0], filename[0]
        transformed_img = inputs.cpu().numpy()[0]
        transformed_img = untransform(transformed_img)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs,cnn_features = deeplab(inputs)
        transforms_rgb_rescal = cv2.resize(transformed_img, (width, height))
        #conv5_features = features_blobs[-1]
        #cnn_features_numpy = cnn_features.data.cpu().numpy()
        weights = weights_generation(cnn_features)

        save_weights_dir = os.path.join(save_pred_dir, filename)
        if batch_idx < 10:
            weights_numpy = weights.data.cpu().numpy()
            loc_weights = utils.attention_weights_collection(weights_numpy)
            utils.attention_weights_visulize(loc_weights, transforms_rgb_rescal, save_weights_dir)
        else:
            break

if __name__ == '__main__':

    #main()
    test_weights_generation()