import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.nn import functional as F
import  voc_config as config
import voc_dataset as dates
import fcn32s_model as models
import utils as utils
import cv2
import json
import matplotlib.pyplot as plt

configs = config.VOC_config()
resume = 1

def untransform(transform_img):

    transform_img = transform_img.transpose(1,2,0)
    #transform_img *= [0.229, 0.224, 0.225]
    transform_img += (104.00698793, 116.66876762, 122.67891434)
    transform_img = transform_img.astype(np.uint8)
    transform_img = transform_img[:,:,::-1]
    return transform_img

def test(net, testloader):

    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):

        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs, _ = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print(100.* correct/total)
    return 100.*correct/total

def main():

   ########  load training data ########

   test_data = dates.VOCDataset(configs.val_img_dir,configs.val_label_dir,configs.val_txt_dir,'val',transform=True)

   test_loader = Data.DataLoader(test_data,batch_size=configs.batch_size,
                                shuffle= False, num_workers= 4, pin_memory= True)
   ######### build vgg model ##########

   fcn32s = models.fcn32s()
   fcn32s = fcn32s.cuda()
   checkpoint = torch.load(configs.best_ckpt_dir)
   fcn32s.load_state_dict(checkpoint['state_dict'])

   save_pred_dir = os.path.join(configs.save_pred_dir,'prediction')

   for batch_idx, (inputs, targets) in enumerate(test_loader):

       inputs, targets = inputs.cuda(), targets.cuda()
       transformed_img = inputs.cpu().numpy()[0]
       transformed_img = untransform(transformed_img)
       inputs, targets = Variable(inputs, volatile=True), Variable(targets)
       outputs  = fcn32s(inputs)
       pred = outputs.data.max(1)[1].cpu().numpy()[:, 0, :, :]
       lbl_true = targets.data.cpu().numpy()
       pred_rgb = dates.decode_segmap(pred[0],plot=False)
       label_rgb = dates.decode_segmap(lbl_true[0],plot=False)

       save_fig_dir = os.path.join(save_pred_dir, 'seg_' + str(batch_idx) + '.jpg')
       plt.figure(1, figsize=(8, 6))
       ax =  plt.subplot(131)
       img1 = transformed_img
       ax.set_title(('{}').format('Image'),fontsize=14)
       ax.imshow(img1)

       ax =  plt.subplot(132)
       img2 = label_rgb
       ax.set_title(('{}').format('GTruth'),fontsize=14)
       ax.imshow(img2[:,:,::-1])

       ax = plt.subplot(133)
       img3 = pred_rgb
       ax.set_title(('{}').format('Prediction'), fontsize=14)
       ax.imshow(img3[:,:,::-1])

       plt.savefig(save_fig_dir)

       print batch_idx

if __name__ == '__main__':

    main()