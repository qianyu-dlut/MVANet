import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve, distance_transform_edt as bwdist
import cv2
from skimage.morphology import skeletonize
from skimage.morphology import disk
from skimage.measure import label


class cal_fm(object):
    # Fmeasure(maxFm,meanFm)---Frequency-tuned salient region detection(CVPR 2009)
    def __init__(self, num, thds=255):
        self.num = num
        self.thds = thds
        self.precision = np.zeros((self.num, self.thds))
        self.recall = np.zeros((self.num, self.thds))
        self.meanF = np.zeros((self.num,1))
        self.changeable_fms = []
        self.idx = 0

    def update(self, pred, gt):
        if gt.max() != 0:
            # prediction, recall, Fmeasure_temp = self.cal(pred, gt)
            prediction, recall, Fmeasure_temp, changeable_fms = self.cal(pred, gt)
            self.precision[self.idx, :] = prediction
            self.recall[self.idx, :] = recall
            self.meanF[self.idx, :] = Fmeasure_temp
            self.changeable_fms.append(changeable_fms)
        self.idx += 1

    def cal(self, pred, gt):
########################meanF##############################
        th = 2 * pred.mean()
        if th > 1:
            th = 1
        
        binary = np.zeros_like(pred)
        binary[pred >= th] = 1

        hard_gt = np.zeros_like(gt)
        hard_gt[gt > 0.5] = 1
        tp = (binary * hard_gt).sum()
        if tp == 0:
            meanF = 0
        else:
            pre = tp / binary.sum()
            rec = tp / hard_gt.sum()
            meanF = 1.3 * pre * rec / (0.3 * pre + rec)
########################maxF##############################
        pred = np.uint8(pred * 255)
        target = pred[gt > 0.5]
        nontarget = pred[gt <= 0.5]
        targetHist, _ = np.histogram(target, bins=range(256))
        nontargetHist, _ = np.histogram(nontarget, bins=range(256))
        targetHist = np.cumsum(np.flip(targetHist), axis=0)
        nontargetHist = np.cumsum(np.flip(nontargetHist), axis=0)
        precision = targetHist / (targetHist + nontargetHist + 1e-8)
        recall = targetHist / np.sum(gt)
        numerator = 1.3 * precision * recall
        denominator = np.where(numerator == 0, 1, 0.3 * precision + recall)
        changeable_fms = numerator / denominator
        return precision, recall, meanF, changeable_fms
    
    
    def show(self):
        assert self.num == self.idx
        precision = self.precision.mean(axis=0)
        recall = self.recall.mean(axis=0)
        # fmeasure = 1.3 * precision * recall / (0.3 * precision + recall + 1e-8)
        changeable_fm = np.mean(np.array(self.changeable_fms), axis=0)
        fmeasure_avg = self.meanF.mean(axis=0)
        return changeable_fm.max(),fmeasure_avg[0],precision,recall
    


class cal_mae(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, pred, gt):
        return np.mean(np.abs(pred - gt))

    def show(self):
        return np.mean(self.prediction)

class cal_dice(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, y_pred, y_true):
        # smooth = 1
        smooth = 1e-5
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    def show(self):
        return np.mean(self.prediction)

class cal_ber(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, y_pred, y_true):
        binary = np.zeros_like(y_pred)
        binary[y_pred >= 0.5] = 1
        hard_gt = np.zeros_like(y_true)
        hard_gt[y_true > 0.5] = 1
        tp = (binary * hard_gt).sum()
        tn = ((1-binary) * (1-hard_gt)).sum()
        Np = hard_gt.sum()
        Nn = (1-hard_gt).sum()
        ber = (1-(tp/(Np+1e-8)+tn/(Nn+1e-8))/2)
        return ber

    def show(self):
        return np.mean(self.prediction)

class cal_acc(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, y_pred, y_true):
        binary = np.zeros_like(y_pred)
        binary[y_pred >= 0.5] = 1
        hard_gt = np.zeros_like(y_true)
        hard_gt[y_true > 0.5] = 1
        tp = (binary * hard_gt).sum()
        tn = ((1-binary) * (1-hard_gt)).sum()
        Np = hard_gt.sum()
        Nn = (1-hard_gt).sum()
        acc = ((tp+tn)/(Np+Nn))
        return acc

    def show(self):
        return np.mean(self.prediction)

class cal_iou(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    # def cal(self, input, target):
    #     classes = 1
    #     intersection = np.logical_and(target == classes, input == classes)
    #     # print(intersection.any())
    #     union = np.logical_or(target == classes, input == classes)
    #     return np.sum(intersection) / np.sum(union)

    def cal(self, input, target):
        smooth = 1e-5
        input = input > 0.5
        target_ = target > 0.5
        intersection = (input & target_).sum()
        union = (input | target_).sum()

        return (intersection + smooth) / (union + smooth)
    def show(self):
        return np.mean(self.prediction)

    # smooth = 1e-5
    #
    # if torch.is_tensor(output):
    #     output = torch.sigmoid(output).data.cpu().numpy()
    # if torch.is_tensor(target):
    #     target = target.data.cpu().numpy()
    # output_ = output > 0.5
    # target_ = target > 0.5
    # intersection = (output_ & target_).sum()
    # union = (output_ | target_).sum()

    # return (intersection + smooth) / (union + smooth)

class cal_sm(object):
    # Structure-measure: A new way to evaluate foreground maps (ICCV 2017)
    def __init__(self, alpha=0.5):
        self.prediction = []
        self.alpha = alpha

    def update(self, pred, gt):
        gt = gt > 0.5
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def show(self):
        return np.mean(self.prediction)

    def cal(self, pred, gt):
        y = np.mean(gt)
        if y == 0:
            score = 1 - np.mean(pred)
        elif y == 1:
            score = np.mean(pred)
        else:
            score = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
        return score

    def object(self, pred, gt):
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)

        u = np.mean(gt)
        return u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, np.logical_not(gt))

    def s_object(self, in1, in2):
        x = np.mean(in1[in2])
        sigma_x = np.std(in1[in2])
        return 2 * x / (pow(x, 2) + 1 + sigma_x + 1e-8)

    def region(self, pred, gt):
        [y, x] = ndimage.center_of_mass(gt)
        y = int(round(y)) + 1
        x = int(round(x)) + 1
        [gt1, gt2, gt3, gt4, w1, w2, w3, w4] = self.divideGT(gt, x, y)
        pred1, pred2, pred3, pred4 = self.dividePred(pred, x, y)

        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def divideGT(self, gt, x, y):
        h, w = gt.shape
        area = h * w
        LT = gt[0:y, 0:x]
        RT = gt[0:y, x:w]
        LB = gt[y:h, 0:x]
        RB = gt[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = (h - y) * (w - x) / area

        return LT, RT, LB, RB, w1, w2, w3, w4

    def dividePred(self, pred, x, y):
        h, w = pred.shape
        LT = pred[0:y, 0:x]
        RT = pred[0:y, x:w]
        LB = pred[y:h, 0:x]
        RB = pred[y:h, x:w]

        return LT, RT, LB, RB

    def ssim(self, in1, in2):
        in2 = np.float32(in2)
        h, w = in1.shape
        N = h * w

        x = np.mean(in1)
        y = np.mean(in2)
        sigma_x = np.var(in1)
        sigma_y = np.var(in2)
        sigma_xy = np.sum((in1 - x) * (in2 - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + 1e-8)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0

        return score

class cal_em(object):
    #Enhanced-alignment Measure for Binary Foreground Map Evaluation (IJCAI 2018)
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, pred, gt):
        th = 2 * pred.mean()
        if th > 1:
            th = 1
        FM = np.zeros(gt.shape)
        FM[pred >= th] = 1
        FM = np.array(FM,dtype=bool)
        GT = np.array(gt,dtype=bool)
        dFM = np.double(FM)
        if (sum(sum(np.double(GT)))==0):
            enhanced_matrix = 1.0-dFM
        elif (sum(sum(np.double(~GT)))==0):
            enhanced_matrix = dFM
        else:
            dGT = np.double(GT)
            align_matrix = self.AlignmentTerm(dFM, dGT)
            enhanced_matrix = self.EnhancedAlignmentTerm(align_matrix)
        [w, h] = np.shape(GT)
        score = sum(sum(enhanced_matrix))/ (w * h - 1 + 1e-8)
        return score
    def AlignmentTerm(self,dFM,dGT):
        mu_FM = np.mean(dFM)
        mu_GT = np.mean(dGT)
        align_FM = dFM - mu_FM
        align_GT = dGT - mu_GT
        align_Matrix = 2. * (align_GT * align_FM)/ (align_GT* align_GT + align_FM* align_FM + 1e-8)
        return align_Matrix
    def EnhancedAlignmentTerm(self,align_Matrix):
        enhanced = np.power(align_Matrix + 1,2) / 4
        return enhanced
    def show(self):
        return np.mean(self.prediction)
class cal_wfm(object):
    def __init__(self, beta=1):
        self.beta = beta
        self.eps = 1e-6
        self.scores_list = []

    def update(self, pred, gt):
        assert pred.ndim == gt.ndim and pred.shape == gt.shape
        assert pred.max() <= 1 and pred.min() >= 0
        assert gt.max() <= 1 and gt.min() >= 0

        gt = gt > 0.5
        if gt.max() == 0:
            score = 0
        else:
            score = self.cal(pred, gt)
        self.scores_list.append(score)

    def matlab_style_gauss2D(self, shape=(7, 7), sigma=5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def cal(self, pred, gt):
        # [Dst,IDXT] = bwdist(dGT);
        Dst, Idxt = bwdist(gt == 0, return_indices=True)

        # %Pixel dependency
        # E = abs(FG-dGT);
        E = np.abs(pred - gt)
        # Et = E;
        # Et(~GT)=Et(IDXT(~GT)); %To deal correctly with the edges of the foreground region
        Et = np.copy(E)
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

        # K = fspecial('gaussian',7,5);
        # EA = imfilter(Et,K);
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode='constant', cval=0)
        MIN_E_EA = np.where(gt & (EA < E), EA, E)

        # %Pixel importance
        # B = ones(size(GT));
        # B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
        # Ew = MIN_E_EA.*B;
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))
        Ew = MIN_E_EA * B

        # TPw = sum(dGT(:)) - sum(sum(Ew(GT)));
        # FPw = sum(sum(Ew(~GT)));
        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])

        # R = 1- mean2(Ew(GT)); %Weighed Recall
        # P = TPw./(eps+TPw+FPw); %Weighted Precision
        R = 1 - np.mean(Ew[gt])
        P = TPw / (self.eps + TPw + FPw)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = (1 + self.beta) * R * P / (self.eps + R + self.beta * P)

        return Q

    def show(self):
        return np.mean(self.scores_list)
    

class HCEMeasure(object):
    def __init__(self):
        self.hces = []

    def step(self, pred: np.ndarray, gt: np.ndarray, gt_ske):
        # pred, gt = _prepare_data(pred, gt)

        hce = self.cal_hce(pred, gt, gt_ske)
        self.hces.append(hce)

    def get_results(self) -> dict:
        hce = np.mean(np.array(self.hces))
        return dict(hce=hce)


    def cal_hce(self, pred: np.ndarray, gt: np.ndarray, gt_ske: np.ndarray, relax=5, epsilon=2.0) -> float:
        # Binarize gt
        if(len(gt.shape)>2):
            gt = gt[:, :, 0]

        epsilon_gt = 0.5#(np.amin(gt)+np.amax(gt))/2.0
        gt = (gt>epsilon_gt).astype(np.uint8)

        # Binarize pred
        if(len(pred.shape)>2):
            pred = pred[:, :, 0]
        epsilon_pred = 0.5#(np.amin(pred)+np.amax(pred))/2.0
        pred = (pred>epsilon_pred).astype(np.uint8)

        Union = np.logical_or(gt, pred)
        TP = np.logical_and(gt, pred)
        FP = pred - TP
        FN = gt - TP

        # relax the Union of gt and pred
        Union_erode = Union.copy()
        Union_erode = cv2.erode(Union_erode.astype(np.uint8), disk(1), iterations=relax)

        # --- get the relaxed False Positive regions for computing the human efforts in correcting them ---
        FP_ = np.logical_and(FP, Union_erode) # get the relaxed FP
        for i in range(0, relax):
            FP_ = cv2.dilate(FP_.astype(np.uint8), disk(1))
            FP_ = np.logical_and(FP_, 1-np.logical_or(TP, FN))
        FP_ = np.logical_and(FP, FP_)

        # --- get the relaxed False Negative regions for computing the human efforts in correcting them ---
        FN_ = np.logical_and(FN, Union_erode) # preserve the structural components of FN
        ## recover the FN, where pixels are not close to the TP borders
        for i in range(0, relax):
            FN_ = cv2.dilate(FN_.astype(np.uint8), disk(1))
            FN_ = np.logical_and(FN_, 1-np.logical_or(TP, FP))
        FN_ = np.logical_and(FN, FN_)
        FN_ = np.logical_or(FN_, np.logical_xor(gt_ske, np.logical_and(TP, gt_ske))) # preserve the structural components of FN

        ## 2. =============Find exact polygon control points and independent regions==============
        ## find contours from FP_
        ctrs_FP, hier_FP = cv2.findContours(FP_.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        ## find control points and independent regions for human correction
        bdies_FP, indep_cnt_FP = self.filter_bdy_cond(ctrs_FP, FP_, np.logical_or(TP,FN_))
        ## find contours from FN_
        ctrs_FN, hier_FN = cv2.findContours(FN_.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        ## find control points and independent regions for human correction
        bdies_FN, indep_cnt_FN = self.filter_bdy_cond(ctrs_FN, FN_, 1-np.logical_or(np.logical_or(TP, FP_), FN_))

        poly_FP, poly_FP_len, poly_FP_point_cnt = self.approximate_RDP(bdies_FP, epsilon=epsilon)
        poly_FN, poly_FN_len, poly_FN_point_cnt = self.approximate_RDP(bdies_FN, epsilon=epsilon)

        # FP_points+FP_indep+FN_points+FN_indep
        return poly_FP_point_cnt+indep_cnt_FP+poly_FN_point_cnt+indep_cnt_FN

    def filter_bdy_cond(self, bdy_, mask, cond):

        cond = cv2.dilate(cond.astype(np.uint8), disk(1))
        labels = label(mask) # find the connected regions
        lbls = np.unique(labels) # the indices of the connected regions
        indep = np.ones(lbls.shape[0]) # the label of each connected regions
        indep[0] = 0 # 0 indicate the background region

        boundaries = []
        h,w = cond.shape[0:2]
        ind_map = np.zeros((h, w))
        indep_cnt = 0

        for i in range(0, len(bdy_)):
            tmp_bdies = []
            tmp_bdy = []
            for j in range(0, bdy_[i].shape[0]):
                r, c = bdy_[i][j,0,1],bdy_[i][j,0,0]

                if(np.sum(cond[r, c])==0 or ind_map[r, c]!=0):
                    if(len(tmp_bdy)>0):
                        tmp_bdies.append(tmp_bdy)
                        tmp_bdy = []
                    continue
                tmp_bdy.append([c, r])
                ind_map[r, c] =  ind_map[r, c] + 1
                indep[labels[r, c]] = 0 # indicates part of the boundary of this region needs human correction
            if(len(tmp_bdy)>0):
                tmp_bdies.append(tmp_bdy)

            # check if the first and the last boundaries are connected
            # if yes, invert the first boundary and attach it after the last boundary
            if(len(tmp_bdies)>1):
                first_x, first_y = tmp_bdies[0][0]
                last_x, last_y = tmp_bdies[-1][-1]
                if((abs(first_x-last_x)==1 and first_y==last_y) or
                (first_x==last_x and abs(first_y-last_y)==1) or
                (abs(first_x-last_x)==1 and abs(first_y-last_y)==1)
                ):
                    tmp_bdies[-1].extend(tmp_bdies[0][::-1])
                    del tmp_bdies[0]

            for k in range(0, len(tmp_bdies)):
                tmp_bdies[k] =  np.array(tmp_bdies[k])[:, np.newaxis, :]
            if(len(tmp_bdies)>0):
                boundaries.extend(tmp_bdies)

        return boundaries, np.sum(indep)

    # this function approximate each boundary by DP algorithm
    # https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
    def approximate_RDP(self, boundaries, epsilon=1.0):

        boundaries_ = []
        boundaries_len_ = []
        pixel_cnt_ = 0

        # polygon approximate of each boundary
        for i in range(0, len(boundaries)):
            boundaries_.append(cv2.approxPolyDP(boundaries[i], epsilon, False))

        # count the control points number of each boundary and the total control points number of all the boundaries
        for i in range(0, len(boundaries_)):
            boundaries_len_.append(len(boundaries_[i]))
            pixel_cnt_ = pixel_cnt_ + len(boundaries_[i])

        return boundaries_, boundaries_len_, pixel_cnt_
