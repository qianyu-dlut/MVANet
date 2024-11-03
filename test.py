import numpy as np
import os
from utils.test_data import test_dataset
from utils.saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm, cal_dice, cal_iou,cal_ber,cal_acc
from utils.config import diste1,diste2,diste3,diste4,disvd
from tqdm import tqdm

test_datasets = {
    'te1':diste1,
          'te2':diste2,
           'te3':diste3,
          'te4':diste4,
        'vd':disvd
                }


dir = '/home/vanessa/code/HRSOD/MVANet-main/saved_model/MVANet'
files = os.listdir(dir)
files.sort()
for items in files:
    item = items.split('.')[0]
    if '.pth' in items:
        continue
    print(items)
    for name, root in test_datasets.items():
        print(name)
        sal_root = os.path.join(dir, items, name)
        print(sal_root)
        gt_root = root + 'masks'
        print(gt_root)
        if os.path.exists(sal_root):
            test_loader = test_dataset(sal_root, gt_root)
            mae, fm, sm, em, wfm, m_dice, m_iou, ber, acc = cal_mae(), cal_fm(
                test_loader.size), cal_sm(), cal_em(), cal_wfm(), cal_dice(), cal_iou(), cal_ber(), cal_acc()
            for i in tqdm(range(test_loader.size)):
                # print ('predicting for %d / %d' % ( i + 1, test_loader.size))
                sal, gt = test_loader.load_data()
                if sal.size != gt.size:
                    x, y = gt.size
                    sal = sal.resize((x, y))
                gt = np.asarray(gt, np.float64)
                gt /= (gt.max() + 1e-8)
                gt[gt > 0.5] = 1
                gt[gt != 1] = 0
                res = sal
                res = np.array(res, np.float64)
                if res.max() == res.min():
                    res = res / 255
                else:
                    res = (res - res.min()) / (res.max() - res.min())

                mae.update(res, gt)
                sm.update(res, gt)
                fm.update(res, gt)
                em.update(res, gt)
                wfm.update(res, gt)
                m_dice.update(res, gt)
                m_iou.update(res, gt)
                ber.update(res, gt)
                acc.update(res, gt)

            MAE = mae.show()
            maxf, meanf, _, _ = fm.show()
            sm = sm.show()
            em = em.show()
            wfm = wfm.show()
            m_dice = m_dice.show()
            m_iou = m_iou.show()
            ber = ber.show()
            acc = acc.show()
            print(
                'dataset: {} MAE: {:.4f} Ber: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} adpEm: {:.4f} M_dice: {:.4f} M_iou: {:.4f} Acc: {:.4f}'.format(
                    name, MAE, ber, maxf, meanf, wfm, sm, em, m_dice, m_iou, acc))
