import numpy as np
import os
from utils.test_data import test_dataset
from utils.saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm, cal_dice, cal_iou,cal_ber,cal_acc, HCEMeasure
from utils.config import diste1,diste2,diste3,diste4,disvd
from tqdm import tqdm
import cv2
from skimage.morphology import skeletonize

test_datasets = {
    'DIS-TE1':diste1,
    'DIS-TE2':diste2,
    'DIS-TE3':diste3,
    'DIS-TE4':diste4,
    'DIS-VD':disvd,
                }


dir = [
    './saved_model/MVANet/Model_80/'
    ]
for d in dir:
    for name, root in test_datasets.items():
        print(name)
        sal_root = os.path.join(d, name)
        print(sal_root)
        gt_root = root + 'masks'
        print(gt_root)
        if os.path.exists(sal_root):
            test_loader = test_dataset(sal_root, gt_root)
            mae, fm, sm, em, wfm, m_dice, m_iou, ber, acc, hce = cal_mae(), cal_fm(
                test_loader.size), cal_sm(), cal_em(), cal_wfm(), cal_dice(), cal_iou(), cal_ber(), cal_acc(), HCEMeasure()
            for i in tqdm(range(test_loader.size)):
                # print ('predicting for %d / %d' % ( i + 1, test_loader.size))
                sal, gt, gt_path = test_loader.load_data()

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

                ske_path = gt_path.replace("/masks/", "/ske/")
                if os.path.exists(ske_path):
                    ske_ary = cv2.imread(ske_path, cv2.IMREAD_GRAYSCALE)
                    ske_ary = ske_ary > 128
                else:
                    ske_ary = skeletonize(gt > 0.5)
                    ske_save_dir = os.path.join(*ske_path.split(os.sep)[:-1])
                    if ske_path[0] == os.sep:
                        ske_save_dir = os.sep + ske_save_dir
                    os.makedirs(ske_save_dir, exist_ok=True)
                    cv2.imwrite(ske_path, ske_ary.astype(np.uint8) * 255)

                mae.update(res, gt)
                sm.update(res, gt)
                fm.update(res, gt)
                em.update(res, gt)
                wfm.update(res, gt)
                m_dice.update(res, gt)
                m_iou.update(res, gt)
                ber.update(res, gt)
                acc.update(res, gt)
                hce.step(pred=res, gt=gt, gt_ske=ske_ary)


            MAE = mae.show()
            maxf, meanf, _, _ = fm.show()
            sm = sm.show()
            em = em.show()
            wfm = wfm.show()
            m_dice = m_dice.show()
            m_iou = m_iou.show()
            ber = ber.show()
            acc = acc.show()
            hce = hce.get_results()["hce"]

            print(
                'dataset: {} MAE: {:.4f} Ber: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} adpEm: {:.4f} M_dice: {:.4f} M_iou: {:.4f} Acc: {:.4f} HCE:{}'.format(
                    name, MAE, ber, maxf, meanf, wfm, sm, em, m_dice, m_iou, acc, int(hce)))
