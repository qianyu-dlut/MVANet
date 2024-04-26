import torch
import  os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] ='0'
from datetime import datetime
from model.MVANet import MVANet
from utils.dataset_strategy_fpn import get_loader
from utils.misc import adjust_lr, AvgMeter
import torch.nn.functional as F
from torch.autograd import Variable
from torch.backends import cudnn
from torchvision import transforms
import torch.nn as nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=80, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=1024, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')

opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()
generator = MVANet()
generator.cuda()

generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)

image_root = './data/DIS5K/DIS-TR/images/'
gt_root = './data/DIS5K/DIS-TR/masks/'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)
to_pil = transforms.ToPILImage()
## define loss

CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [1]  
criterion = nn.BCEWithLogitsLoss().cuda()
criterion_mae = nn.L1Loss().cuda() 
criterion_mse = nn.MSELoss().cuda()
use_fp16 = True
scaler = amp.GradScaler(enabled=use_fp16)

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))


    pred  = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))

    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou  = 1-(inter+1)/(union-inter+1)

    return (wbce+wiou).mean()



for epoch in range(1, opt.epoch+1):
    torch.cuda.empty_cache()
    generator.train()
    loss_record = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    for i, pack in enumerate(train_loader, start=1):
        torch.cuda.empty_cache()
        for rate in size_rates:
            torch.cuda.empty_cache()
            generator_optimizer.zero_grad()
            images, gts = pack
            images = Variable(images)
            gts = Variable(gts)
            images = images.cuda()
            gts = gts.cuda()
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear',
                                          align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            b, c, h, w = gts.size()
            target_1 = F.upsample(gts, size=h // 4, mode='nearest')
            target_2 = F.upsample(gts,  size=h // 8, mode='nearest').cuda()
            target_3 = F.upsample(gts,  size=h // 16, mode='nearest').cuda()
            target_4 = F.upsample(gts,  size=h // 32, mode='nearest').cuda()
            target_5 = F.upsample(gts, size=h // 64, mode='nearest').cuda()

            with amp.autocast(enabled=use_fp16):
                sideout5, sideout4, sideout3, sideout2, sideout1, final, glb5, glb4, glb3, glb2, glb1, tokenattmap4, tokenattmap3,tokenattmap2,tokenattmap1= generator.forward(images)
                loss1 = structure_loss(sideout5, target_4)
                loss2 = structure_loss(sideout4, target_3)
                loss3 = structure_loss(sideout3, target_2)
                loss4 = structure_loss(sideout2, target_1)
                loss5 = structure_loss(sideout1, target_1)
                loss6 = structure_loss(final, gts)
                loss7 = structure_loss(glb5, target_5)
                loss8 = structure_loss(glb4, target_4)
                loss9 = structure_loss(glb3, target_3)
                loss10 = structure_loss(glb2, target_2)
                loss11 = structure_loss(glb1, target_2)
                loss12 = structure_loss(tokenattmap4, target_3)
                loss13 = structure_loss(tokenattmap3, target_2)
                loss14 = structure_loss(tokenattmap2, target_1)
                loss15 = structure_loss(tokenattmap1, target_1)
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + 0.3*(loss7 + loss8 + loss9 + loss10 + loss11)+ 0.3*(loss12 + loss13 + loss14 + loss15)
                Loss_loc = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                Loss_glb = loss7 + loss8 + loss9 + loss10 + loss11
                Loss_map = loss12 + loss13 + loss14 + loss15
                writer.add_scalar('loss', loss.item(), epoch * len(train_loader) + i)

            generator_optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(generator_optimizer)
            scaler.update()

            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)


        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))

    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
    # save checkpoints every 20 epochs
    if epoch % 20== 0 :
        save_path = './saved_model/MVANet/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '.pth')


