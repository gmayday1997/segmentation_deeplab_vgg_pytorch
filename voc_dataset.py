import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import scipy.io
import scipy.misc as m
from PIL import Image
import matplotlib.pyplot as plt
import transforms as trans

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_pascal_labels():
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
                       [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                       [0, 192, 0], [128, 192, 0], [0, 64, 128]])


def decode_segmap(temp, plot=False):

    label_colours = get_pascal_labels()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 21):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

class VOCDataset(Dataset):

    def __init__(self,img_path,label_path,file_name_txt_path,split_flag, transform=True, transform_med = None):

        self.label_path = label_path
        self.img_path = img_path
        self.img_txt_path = file_name_txt_path
        self.imgs_path_list = np.loadtxt(self.img_txt_path,dtype=str)
        self.flag = split_flag
        self.transform = transform
        self.transform_med = transform_med
        self.img_label_path_pairs = self.get_img_label_path_pairs()

    def get_img_label_path_pairs(self):

        img_label_pair_list = {}
        if self.flag =='train':

            for idx , did in enumerate(open(self.img_txt_path)):
                try:
                    image_name, mask_name = did.strip("\n").split(' ')
                except ValueError:  # Adhoc for test.
                    image_name = mask_name = did.strip("\n")

                extract_name = image_name[image_name.rindex('/') +1: image_name.rindex('.')]
                img_file = self.img_path + image_name
                lbl_file = self.label_path + mask_name

                img_label_pair_list.setdefault(idx, [img_file, lbl_file,extract_name])

        if self.flag == 'val':
            self.label_ext = '.png'

            for idx, file_path in enumerate(self.imgs_path_list):

                full_img_path = os.path.join(self.img_path, file_path + '.jpg')
                full_label_path = os.path.join(self.label_path, file_path + self.label_ext)
                img_label_pair_list.setdefault(idx, [full_img_path, full_label_path, file_path])

        return img_label_pair_list

    def data_transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= (104.00698793, 116.66876762, 122.67891434)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def __getitem__(self, index):

        img_path, label_path, filename = self.img_label_path_pairs[index]
        ####### load images #############
        img = Image.open(img_path)
        height,width,_ = np.array(img,dtype= np.uint8).shape
        if self.transform_med != None:
           img = self.transform_med(img)
        img = np.array(img,dtype= np.uint8)
        ####### load labels ############
        if self.flag == 'train':

            label = Image.open(label_path)
            if self.transform_med != None:
                label = self.transform_med(label)
            label = np.array(label,dtype=np.int32)
        if self.flag == 'val':
            label = Image.open(label_path)
            if self.transform_med != None:
               label = self.transform_med(label)
            label = np.array(label,dtype=np.int32)

        if self.transform:
            img,label = self.data_transform(img,label)

        return img, label,str(filename),int(height),int(width)

    def __len__(self):

        return len(self.img_label_path_pairs)
