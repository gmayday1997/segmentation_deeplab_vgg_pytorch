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
import transforms as trans
import deeplab as models
import utils as utils
import metrics as metric
import shutil

configs = config.VOC_config()
resume = 0

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
    for batch_idx, batch in enumerate(val_dataloader):

        inputs, targets, filename, height, width = batch
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs,weights = net(inputs)
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
    return score['Mean IoU :']

def main():

  #########  configs ###########
  best_metric = 0

  pretrain_deeplab_path = os.path.join(configs.py_dir, 'model/deeplab_coco.pth')

  ######  load datasets ########
  train_transform_det = trans.Compose([
      trans.Scale((321, 321)),
  ])
  val_transform_det = trans.Compose([
      trans.Scale((321,321)),

  ])

  train_data = voc_dates.VOCDataset(configs.train_img_dir,configs.train_label_dir,
                                    configs.train_txt_dir,'train',transform=True,
                                    transform_med = train_transform_det)
  train_loader = Data.DataLoader(train_data,batch_size=configs.batch_size,
                                 shuffle= True, num_workers= 4, pin_memory= True)

  val_data = voc_dates.VOCDataset(configs.val_img_dir,configs.val_label_dir,
                                  configs.val_txt_dir,'val',transform=True,
                                  transform_med = val_transform_det)
  val_loader = Data.DataLoader(val_data, batch_size= configs.batch_size,
                                shuffle= False, num_workers= 4, pin_memory= True)
  ######  build  models ########
  deeplab = models.deeplab()
  deeplab_pretrain_model = utils.load_deeplab_pretrain_model(pretrain_deeplab_path)
  deeplab.init_parameters(deeplab_pretrain_model)
  deeplab = deeplab.cuda()

  params = list(deeplab.parameters())
  #########

  if resume:
      checkpoint = torch.load(configs.best_ckpt_dir)
      deeplab.load_state_dict(checkpoint['state_dict'])
      print('resum sucess')

  ######### optimizer ##########
  ######## how to set different learning rate for differern layer #########
  optimizer = torch.optim.SGD(
      [
          {'params': get_parameters(deeplab, bias=False)},
          {'params': get_parameters(deeplab, bias=True),
           'lr': configs.learning_rate * 2, 'weight_decay': 0},
      ],lr=configs.learning_rate, momentum=configs.momentum,weight_decay=configs.weight_decay)

  ######## iter img_label pairs ###########

  for epoch in range(20):

      utils.adjust_learning_rate(configs.learning_rate,optimizer,epoch)
      for batch_idx, batch in enumerate(train_loader):

          img_idx, label_idx, filename,height,width = batch
          img,label = Variable(img_idx.cuda()),Variable(label_idx.cuda())
          prediction,weights = deeplab(img)
          loss = utils.cross_entropy2d(prediction,label,size_average=False)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          if (batch_idx) % 20 == 0:
              print("Epoch [%d/%d] Loss: %.4f" % (epoch, batch_idx, loss.data[0]))

          if (batch_idx) % 4000 == 0:

              current_metric = validate(deeplab, val_loader, epoch)
              print current_metric

      current_metric = validate(deeplab, val_loader,epoch)

      if current_metric > best_metric:

         torch.save({'state_dict': deeplab.state_dict()},
                     os.path.join(configs.save_ckpt_dir, 'deeplab' + str(epoch) + '.pth'))

         shutil.copy(os.path.join(configs.save_ckpt_dir, 'deeplab' + str(epoch) + '.pth'),
                     os.path.join(configs.save_ckpt_dir, 'model_best.pth'))
         best_metric = current_metric

      if epoch % 5 == 0:
          torch.save({'state_dict': deeplab.state_dict()},
                       os.path.join(configs.save_ckpt_dir, 'deeplab' + str(epoch) + '.pth'))


if __name__ == '__main__':
   main()