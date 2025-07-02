import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp, sqrt

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1) # [num,2,H*W]
            logit = logit.permute(0, 2, 1).contiguous()# [num,H*W,2]
            logit = logit.view(-1, logit.size(-1)) # [N*H*W, 2]
        target = torch.squeeze(target, 1) # [num,H,W]
        target = target.view(-1, 1)# [N*H*W, 2]
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss
    
def smooth(arr, lamda1):
    new_array = arr
    arr2 = torch.zeros_like(arr)
    arr2[:, :-1, :] = arr[:, 1:, :]
    arr2[:, -1, :] = arr[:, -1, :]

    new_array2 = torch.zeros_like(new_array)
    new_array2[:, :, :-1] = new_array[:, :, 1:]
    new_array2[:, :, -1] = new_array[:, :, -1]
    loss = (torch.sum((arr2 - arr) ** 2) + torch.sum((new_array2 - new_array) ** 2)) / 2
    return lamda1 * loss

def sparsity(arr, target, lamda2):
    if target == 0:
        loss = torch.mean(torch.norm(arr, dim=0))
    else:
        loss = torch.mean(torch.norm(1-arr, dim=0))
    return lamda2 * loss

# 图像特征和正负文本间的L2距离
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):

        pos_distance = torch.sum((anchor - positive).pow(2), dim=1)

        neg_distance = torch.sum((anchor - negative).pow(2), dim=1)
        # print(pos_distance, neg_distance)

        loss = torch.exp(pos_distance - neg_distance + self.margin)

        return torch.mean(loss)
    
# 余弦相似度
class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()

    def forward(self, anchor_n, anchor_a, positive, negative):
        assert not torch.isnan(anchor_n).any(), "anchor_n contains NaN"
        assert not torch.isnan(anchor_a).any(), "anchor_a contains NaN"
        assert not torch.isnan(positive).any(), "positive contains NaN"
        assert not torch.isnan(negative).any(), "negative contains NaN"

        pos_distance_n = anchor_n @ positive.t()
        neg_distance_n = anchor_n @ negative.t()

        pos_distance_a = anchor_a @ negative.t()
        neg_distance_a = anchor_a @ positive.t()

        loss_n_n = -torch.log(torch.exp(pos_distance_n)/(torch.exp(pos_distance_n) + torch.exp(neg_distance_n) + 1e-4))
        loss_a_a = -torch.log(torch.exp(pos_distance_a)/(torch.exp(pos_distance_a) + torch.exp(neg_distance_a) + 1e-4))
        loss_n_m = -torch.log(torch.exp(pos_distance_n)/(torch.exp(pos_distance_n) + torch.exp(neg_distance_a) + 1e-4))
        loss_a_m = -torch.log(torch.exp(pos_distance_a)/(torch.exp(pos_distance_a) + torch.exp(neg_distance_n) + 1e-4))
        # loss = (loss_n_n + 0.5 * loss_a_a + loss_n_m + 0.5 *loss_a_m)/2
        loss = (loss_n_n + loss_a_a + loss_n_m + loss_a_m)/2

        return torch.mean(loss)
    
# 同一类文本特征间的距离
class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, feats):
        num, dim = feats.shape
        # 计算每两个特征之间的L2距离
        sum = 0
        count = 0
        for i in range(num):
            for j in range(i+1, num):
                distance = torch.sum((feats[i] - feats[j]).pow(2), dim=0)
                sum += distance
                count += 1
        sum = torch.exp(-sum/count)
        return sum
    
        # sum = -sum/count
        # return sum 

# 不同类特征间的距离
class L2LossDif(nn.Module):
    def __init__(self):
        super(L2LossDif, self).__init__()

    def forward(self, nfeats, afeats):
        nnum, anum = nfeats.shape[0], afeats.shape[0], 
        loss = 0
        nsum =asum= ncount= acount= sum= count= 0
        # print('loss.py',nnum,anum)
        if nnum >1:
            for i in range(nnum):
                for j in range(i+1, nnum):
                    distance = torch.sum((nfeats[i] - nfeats[j]).pow(2), dim=0)
                    nsum += distance
                    ncount += 1
            loss_n = nsum/ncount
            # print('loss_n',loss_n)
        else:
            loss_n = 0
        if anum >1:
            for i in range(anum):
                for j in range(i+1, anum):
                    distance = torch.sum((afeats[i] - afeats[j]).pow(2), dim=0)
                    asum += distance
                    acount += 1
            loss_a = asum/acount
            # print('loss_a',loss_a)
        else:
            loss_a = 0
        if nnum!= 0 and anum !=0:
            # 计算每两个特征之间的L2距离
            for i in range(nnum):
                for j in range(anum):
                    distance = torch.sum((nfeats[i] - afeats[j]).pow(2), dim=0)
                    sum += distance
                    count += 1
            loss_dif = sum/count
            loss = -torch.log(loss_dif/(loss_dif + (asum + nsum)/(acount+ncount)))
            # print('loss_dif',loss_dif)
            # print('loss',loss)
        else:
            loss = torch.tensor(1e-6, device=nfeats.device)
        return loss
    
class TextImageMix(nn.Module):
    def __init__(self, dim):
        super(TextImageMix, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(1, 2, kernel_size=(3, 5), padding=(0, 2)) 
        self.linear2 = nn.Linear(int(2 * dim), dim)
        self.gelu = nn.GELU()
        
    def forward(self, ifeats, tfeat_n, tfeat_a):
        res =[]
        for i in range(ifeats.shape[0]):
            # print(ifeats[i].shape, tfeats.shape, ifeats.shape)
            # ifeat = self.linear1(ifeats[i])
            # ifeat = ifeat/(ifeat.norm(dim=-1) + 1e-6)
            # feats = torch.cat([ifeat.unsqueeze(0), tfeats], dim = 0)
            feats = torch.cat([tfeat_n, tfeat_a, ifeats[i].unsqueeze(0)], dim = 0)
            # 卷积
            x1 = feats.view(1,1,feats.shape[0],feats.shape[1])
            x2 = self.conv(x1)
            x2 = self.gelu(x2)
            x2 = x2.view(1, -1)
            # print(x2.shape)
            # 通过第二个线性层
            x3 = self.linear2(x2)
            res.append(ifeats[i] + x3)
        res = torch.cat(res, dim=0)
        return res
    
class Text2Image(nn.Module):
    def __init__(self, dim, num=1):
        super(Text2Image, self).__init__()
        self.linear_n = nn.ModuleList([nn.Linear(dim, dim) for i in range(num)])
        self.linear_a = nn.ModuleList([nn.Linear(dim, dim) for i in range(num)])
        with torch.no_grad():
            for linearn, lineara in zip(self.linear_n, self.linear_a):
                linearn.weight.copy_(torch.eye(linearn.in_features))
                linearn.bias.zero_()
                lineara.weight.copy_(torch.eye(lineara.in_features))
                lineara.bias.zero_()
        
    def forward(self, tfeat_n, tfeat_a):
        nf = []
        af = []
        for linearn, lineara in zip(self.linear_n, self.linear_a):
            nf.append(linearn(tfeat_n))
            af.append(lineara(tfeat_a))
        nf = torch.cat(nf, dim=0)
        af = torch.cat(af, dim=0)
        nf = nf/nf.norm(dim = -1, keepdim=True)
        af = af/af.norm(dim=-1, keepdim=True)
        return nf , af
    
# 输入[batch_size,  1(final_feat)+layers(the first patch), feat_dim]
# class ImgRes(nn.Module):
#     def __init__(self, dim, dropout = 0.01):
#         super(ImgRes, self).__init__()
#         self.lineari1 = nn.Linear(dim, dim)
#         self.lineari2 = nn.Linear(dim, dim)
#         self.lineart = nn.Linear(dim, dim)
#         self.query = nn.Linear(dim, dim)
#         self.key = nn.Linear(dim, dim)
#         self.value = nn.Linear(dim, dim)
#         self.dropout = nn.Dropout(dropout)
#         self.gelu = nn.GELU()
        
#     def forward(self, ifeat, tfeat):
#         x = self.attention(ifeat)
#         x = self.lineari1(x)
#         x = self.gelu(x)
#         x = self.lineari2(x)
#         x = self.gelu(x) # [8, 768]
#         ifeat = ifeat[:, 0, :]
#         i_f = (ifeat+x)/((ifeat+x).norm(dim=-1, keepdim=True) + 1e-6)

#         y = self.lineart(tfeat)
#         y = self.gelu(y)
#         t_f = (tfeat+y)/((tfeat+y).norm(dim=-1, keepdim=True) + 1e-6)
#         return i_f , t_f
        
#     def attention(self, x):
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)
#         attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / sqrt(x.size(-1))
#         attention_weights = F.softmax(attention_scores, dim=-1)
#         attention_weights = self.dropout(attention_weights)
#         output = torch.matmul(attention_weights, V)
#         return output[:, 0, :] #[b,768]

# [b, dim]
class featAdapter(nn.Module):
    def __init__(self, dim, num=1):
        super(featAdapter, self).__init__()
        self.linear_1 = nn.ModuleList([nn.Linear(dim, dim) for i in range(num)])
        self.linear_2 = nn.ModuleList([nn.Linear(dim, dim) for i in range(num)])
        self.gelu = nn.GELU()
        
    def forward(self, feat):
        f = []
        for ln1, ln2 in zip(self.linear_1, self.linear_2):
            f.append(ln2(self.gelu(ln1(feat))))
        f = torch.mean(torch.stack(f, dim=1), dim =1)
        f = f/f.norm(dim = -1, keepdim=True)
        return f

# 残差, [batch_size,  1(final_feat)+layers(the first patch), feat_dim]
class ImgRes(nn.Module):
    def __init__(self, dim, dropout = 0.01):
        super(ImgRes, self).__init__()
        self.lineari1 = nn.Linear(dim*4, dim*6)
        self.lineari2 = nn.Linear(dim*6, dim*3)
        self.lineari3 = nn.Linear(dim*3, dim*1)
        self.lineari4 = nn.Linear(dim*1, dim*1)
        # self.lineari1 = nn.Linear(dim*4, dim*2)
        # self.lineari2 = nn.Linear(dim*2, dim)
        # self.lineart1 = nn.Linear(dim, dim)
        # self.lineart2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        
    def forward(self, ifeat):
        img_num, fnum, _ = ifeat.shape
        ifeat_ =[]
        for i in range(2,fnum):
            res = ifeat[:,i,:] - ifeat[:,i-1,:]
            ifeat_.append(res)
        res = (ifeat[:,0,:] - ifeat[:,-1,:])
        ifeat_.append(res)
        ifeat_ = torch.stack(ifeat_, dim = 1) #[1,...]
        ifeat_ = ifeat_.reshape(img_num,-1)
        x = self.lineari1(ifeat_)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.lineari2(x)
        x = self.gelu(x)
        x = self.lineari3(x)
        x = self.gelu(x)
        x = self.lineari4(x)
        x = self.gelu(x) # [8, 768]
            
        ifeat = ifeat[:, 0, :]
        i_f = (ifeat+x)/((ifeat+x).norm(dim=-1, keepdim=True))
        return i_f

        # y1 = self.lineart1(tfeat[0])
        # y1 = self.gelu(y1)
        # y2 = self.lineart1(tfeat[1])
        # y2 = self.gelu(y2)
        # y = torch.stack([y1,y2], dim = 0) # [2,768]
        # t_f = (tfeat+y)/((tfeat+y).norm(dim=-1, keepdim=True) + 1e-6)
        # return i_f , t_f
        
    def attention(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / sqrt(x.size(-1))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output[:, 0, :] #[b,768]