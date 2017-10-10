import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transform
import torchvision.datasets as dates
from torch.autograd import Variable
from torch.nn import functional as F
import voc_config as config
import voc_dataset as voc_dates
import fcn32s_model as models
import utils as utils
import metrics as metric
import shutil

configs = config.VOC_config()
resume = 1

def get_parameters(model, bias=False):
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight

def validate(net, val_dataloader,epoch):

    net.eval()
    gts, preds = [], []
    for batch_idx, (inputs, targets) in enumerate(val_dataloader):

        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        #loss = utils.cross_entropy2d(outputs, targets)
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=1)
        gt = targets.data.cpu().numpy()
        for gt_, pred_ in zip(gt, pred):
            gts.append(gt_)
            preds.append(pred_)

    score, class_iou = metric.scores(gts, preds, n_class=21)
    for k, v in score.items():
        print k, v

    for i in range(21):
        print i, class_iou[i]

    metric_json_path = os.path.join(configs.py_dir,'results/metric' + str(epoch) + '.json')
    class_iou_json_path = os.path.join(configs.py_dir,'results/class_iu' + str(epoch) + '.json')
    utils.save2json(score,metric_json_path)
    utils.save2json(class_iou,class_iou_json_path)
    return score['Mean IoU : \t']

def main():

  #########  configs ###########
  best_metric = 0
  pretrain_vgg16_path = os.path.join(configs.py_dir, 'model/vgg16_from_caffe.pth')

  ######  load datasets ########

  train_data = voc_dates.VOCDataset(configs.train_img_dir,configs.train_label_dir,configs.train_txt_dir,'train',transform=True)
  train_loader = Data.DataLoader(train_data,batch_size=configs.batch_size,
                                 shuffle= True, num_workers= 4, pin_memory= True)

  val_data = voc_dates.VOCDataset(configs.val_img_dir,configs.val_label_dir,configs.val_txt_dir,'val',transform=True)
  val_loader = Data.DataLoader(val_data, batch_size= configs.batch_size,
                                shuffle= False, num_workers= 4, pin_memory= True)
  ######  build  models ########
  fcn32s = models.fcn32s()
  vgg_pretrain_model = utils.load_pretrain_model(pretrain_vgg16_path)
  fcn32s.init_parameters(vgg_pretrain_model)
  fcn32s = fcn32s.cuda()
  #########

  if resume:
      checkpoint = torch.load(configs.best_ckpt_dir)
      fcn32s.load_state_dict(checkpoint['state_dict'])
      print('resum sucess')

  ######### optimizer ##########
  ######## how to set different learning rate for differern layer #########
  optimizer = torch.optim.SGD(
      [
          {'params': get_parameters(fcn32s, bias=False)},
          {'params': get_parameters(fcn32s, bias=True),
           'lr': configs.learning_rate * 2, 'weight_decay': 0},
      ],lr=configs.learning_rate, momentum=configs.momentum,weight_decay=configs.weight_decay)

  ######## iter img_label pairs ###########

  for epoch in range(20):

      utils.adjust_learning_rate(configs.learning_rate,optimizer,epoch)
      for batch_idx, (img_idx,label_idx) in enumerate(train_loader):

          img,label = Variable(img_idx.cuda()),Variable(label_idx.cuda())
          prediction = fcn32s(img)
          loss = utils.cross_entropy2d(prediction,label,size_average=False)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          if (batch_idx) % 20 == 0:
              print("Epoch [%d/%d] Loss: %.4f" % (epoch, batch_idx, loss.data[0]))

      current_metric = validate(fcn32s, val_loader,epoch)

      if current_metric > best_metric:

         torch.save({'state_dict': fcn32s.state_dict()},
                     os.path.join(configs.save_ckpt_dir, 'fcn32s' + str(epoch) + '.pth'))

         shutil.copy(os.path.join(configs.save_ckpt_dir, 'fcn32s' + str(epoch) + '.pth'),
                     os.path.join(configs.save_ckpt_dir, 'model_best.pth'))
         best_metric = current_metric

      if epoch % 5 == 0:
          torch.save({'state_dict': fcn32s.state_dict()},
                       os.path.join(configs.save_ckpt_dir, 'fcn32s' + str(epoch) + '.pth'))

if __name__ == '__main__':
   main()
