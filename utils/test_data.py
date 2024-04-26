import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

class test_dataset:
    def __init__(self, image_root, gt_root):
        self.img_list_1 = [os.path.splitext(f)[0] for f in os.listdir(image_root) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')]
        self.img_list_2 = [os.path.splitext(f)[0] for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')]
        self.img_list = list(set(self.img_list_1).intersection(set(self.img_list_2)))

        self.image_root = image_root
        self.gt_root = gt_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.img_list)
        self.index = 0

    def load_data(self):
        #image = self.rgb_loader(self.images[self.index])
        rgb_png_path = os.path.join(self.image_root,self.img_list[self.index]+ '.png')
        rgb_jpg_path = os.path.join(self.image_root,self.img_list[self.index]+ '.jpg')
        rgb_bmp_path = os.path.join(self.image_root,self.img_list[self.index]+ '.bmp')
        if os.path.exists(rgb_png_path):
            image = self.binary_loader(rgb_png_path)
        elif os.path.exists(rgb_jpg_path):
            image = self.binary_loader(rgb_jpg_path)
        else:
            image = self.binary_loader(rgb_bmp_path)
        if os.path.exists(os.path.join(self.gt_root,self.img_list[self.index] + '.png')):
            gt = self.binary_loader(os.path.join(self.gt_root,self.img_list[self.index] + '.png'))
        elif os.path.exists(os.path.join(self.gt_root,self.img_list[self.index] + '.jpg')):
            gt = self.binary_loader(os.path.join(self.gt_root,self.img_list[self.index] + '.jpg'))
        else:
            gt = self.binary_loader(os.path.join(self.gt_root, self.img_list[self.index] + '.bmp'))

        self.index += 1
        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

class val_dataset:
    def __init__(self, image_root, gt_root):
        self.img_list_1 = [os.path.splitext(f)[0] for f in os.listdir(image_root) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')]
        self.img_list_2 = [os.path.splitext(f)[0] for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')]
        self.img_list = list(set(self.img_list_1).intersection(set(self.img_list_2)))

        self.image_root = image_root
        self.gt_root = gt_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.img_list)
        self.index = 0

    def load_data(self):
        #image = self.rgb_loader(self.images[self.index])
        rgb_png_path = os.path.join(self.image_root,self.img_list[self.index]+ '.png')
        rgb_jpg_path = os.path.join(self.image_root,self.img_list[self.index]+ '.jpg')
        rgb_bmp_path = os.path.join(self.image_root,self.img_list[self.index]+ '.bmp')
        if os.path.exists(rgb_png_path):
            image = self.rgb_loader(rgb_png_path)
        elif os.path.exists(rgb_jpg_path):
            image = self.rgb_loader(rgb_jpg_path)
        else:
            image = self.rgb_loader(rgb_bmp_path)
        if os.path.exists(os.path.join(self.gt_root,self.img_list[self.index] + '.png')):
            gt = self.binary_loader(os.path.join(self.gt_root,self.img_list[self.index] + '.png'))
        elif os.path.exists(os.path.join(self.gt_root,self.img_list[self.index] + '.jpg')):
            gt = self.binary_loader(os.path.join(self.gt_root,self.img_list[self.index] + '.jpg'))
        else:
            gt = self.binary_loader(os.path.join(self.gt_root, self.img_list[self.index] + '.bmp'))

        self.index += 1
        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')