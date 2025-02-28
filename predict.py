import numpy as np
import os
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from utils.config import diste1,diste2,diste3,diste4,disvd
from utils.misc import check_mkdir
from model.MVANet import inf_MVANet
import ttach as tta

torch.cuda.set_device(0)
ckpt_path = './saved_model/'
args = {
    'save_results': True
}


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

depth_transform = transforms.ToTensor()
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

to_test ={
            'DIS-TE1':diste1,
           'DIS-TE2':diste2,
           'DIS-TE3':diste3,
           'DIS-TE4':diste4,
            'DIS-VD':disvd,
}

transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.75, 1, 1.125], interpolation='bilinear', align_corners=False),
    ]
)

def main(item):
    net = inf_MVANet().cuda()
    pretrained_dict = torch.load(os.path.join(ckpt_path, item + '.pth'), map_location='cuda')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            root1 = os.path.join(root, 'images')
            img_list = [os.path.splitext(f) for f in os.listdir(root1)]
            for idx, img_name in enumerate(img_list):
                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                rgb_png_path = os.path.join(root, 'images', img_name[0] + '.png')
                rgb_jpg_path = os.path.join(root, 'images', img_name[0] + '.jpg')
                if os.path.exists(rgb_png_path):
                    img = Image.open(rgb_png_path).convert('RGB')
                else:
                    img = Image.open(rgb_jpg_path).convert('RGB')
                w_,h_ = img.size
                img_resize = img.resize([1024,1024],Image.BILINEAR)  
                img_var = Variable(img_transform(img_resize).unsqueeze(0), volatile=True).cuda()
                mask = []
                for transformer in transforms:  
                    rgb_trans = transformer.augment_image(img_var)
                    model_output = net(rgb_trans)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    mask.append(deaug_mask)

                prediction = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction = prediction.sigmoid()
                prediction = to_pil(prediction.data.squeeze(0).cpu())
                prediction = prediction.resize((w_, h_), Image.BILINEAR)
                if args['save_results']:
                    check_mkdir(os.path.join(ckpt_path, item,  name))
                    prediction.save(os.path.join(ckpt_path, item,  name, img_name[0] + '.png'))



if __name__ == '__main__':
    files = os.listdir(ckpt_path)
    files.sort()
    for items in files:
        if '80.pth' in items:
            item = items.split('.')[0]
            main(item)




