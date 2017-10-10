import os

class VOC_config():

  def __init__(self):

    self.py_dir = os.getcwd()
    self.train_dataset_dir = '/media/cheer/2T/Datasets/ImageDatesets/Deconvnet/VOC2012/VOC2012_SEG_AUG'
    self.train_img_dir = os.path.join(self.train_dataset_dir)
    self.train_label_dir = os.path.join(self.train_dataset_dir)
    self.train_txt_dir = os.path.join(self.train_dataset_dir,'train.txt')
    self.val_dataset_dir = '/media/cheer/2T/Datasets/ImageDatesets/VOC2012/VOCdevkit2012/VOC2012'
    self.val_img_dir = os.path.join(self.val_dataset_dir,'seg11valImages')
    self.val_label_dir = os.path.join(self.val_dataset_dir,'Segmentation_grey_Label')
    self.val_txt_dir = os.path.join(self.val_dataset_dir,'segval11.txt')
    self.test_datset_dir = ''
    self.test_txt_dir = ''
    self.save_pred_dir = '/media/cheer/2T/train_pytorch/fcn/'
    self.save_ckpt_dir = '/media/cheer/2T/train_pytorch/fcn/ckpt'
    self.best_ckpt_dir = '/media/cheer/2T/train_pytorch/fcn/ckpt/model_best.pth'
    self.mean_value_vector = [104.00698793, 116.66876762, 122.67891434]
    self.init_random_fc8 = True
    self.learning_rate = 1e-10
    self.max_iter_number = 100000
    self.validate_iter_number = 500
    self.save_ckpoints_iter_number = 10000
    self.weight_decay = 5e-5
    self.momentum = 0.99
    self.train = True
    if self.train:
       self.batch_size = 1
    else:
       self.batch_size = 1
    

