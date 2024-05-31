import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

# from CIFAR_10_100.ModulatedAttLayer import ModulatedAttLayer

# 用于估计每个类别的样本均值和协方差矩阵，并计算每个类别的权重
class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num

        # 一个大小为(class_num, feature_num, feature_num)的全零张量CoVariance
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()

        # 一个大小为(class_num, feature_num)的全零张量Ave
        self.Ave = torch.zeros(class_num, feature_num).cuda()

        # 一个大小为class_num的全零张量Amount
        self.Amount = torch.zeros(class_num).cuda()

    # 这段代码的目的是根据输入的特征和标签数据，更新EstimatorCV类中的协方差矩阵、样本均值和样本数量，
    # 更新EstimatorCV类中的样本均值、协方差矩阵和样本数量的方法。
    def update_CV(self, features, labels):
        # 计算样本数量N
        N = features.size(0)
        # 类别数量C
        C = self.class_num
        # 特征数量A
        A = features.size(1)

        # 将特征数据重塑为大小为(N, 1, A)的三维张量，
        # 并在第二维度上复制C次，得到大小为(N, C, A)的张量。
        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

        # 创建一个大小为(N, C)的零张量onehot
        onehot = torch.zeros(N, C).cuda()
        # 根据标签labels将对应位置的元素设置为1，从而得到一个one - hot编码的标签张量。
        onehot.scatter_(1, labels.view(-1, 1), 1)
        # 将onehot张量重塑为大小为(N, C, 1)的三维张量，并在第三维度上复制A次，得到大小为(N, C, A)的张量。
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        # 将NxCxFeatures和NxCxA_onehot相乘，得到按类别筛选后的特征张量features_by_sort。
        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        # 计算每个类别在每个特征上的样本数量，保存在Amount_CxA中。
        # 如果某个类别在某个特征上的样本数量为0，则将其设置为1，以避免除以0的错误。
        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        # 计算每个类别在每个特征上的样本均值，保存在ave_CxA中。
        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        # 计算每个类别在每个特征上的样本方差，保存在var_temp中。这里使用了矩阵的乘法和除法运算。
        var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)
        var_temp = torch.bmm(var_temp.permute(1, 2, 0),
                             var_temp.permute(1, 0, 2)) \
            .div(Amount_CxA.view(C, A, 1)
                 .expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)
        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        # 计算每个类别在每个特征上的权重，保存在weight_CV中。
        weight_CV = sum_weight_CV.div(sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A))
        weight_CV[weight_CV != weight_CV] = 0

        # 计算每个类别在每个特征上的样本均值权重，保存在weight_AV中。
        weight_AV = sum_weight_AV.div(sum_weight_AV + self.Amount.view(C, 1).expand(C, A))
        weight_AV[weight_AV != weight_AV] = 0

        # 计算额外的协方差矩阵更新项additional_CV，
        # 它是由权重、样本均值和先前的均值之间的差异计算得到的。
        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )

        # 更新

        # 协方差矩阵CoVariance、
        self.CoVariance = (self.CoVariance.mul(1 - weight_CV)
                           + var_temp.mul(weight_CV)).detach() \
                          + additional_CV.detach()
        # 样本均值Ave
        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()
        # 样本数量Amount。
        self.Amount += onehot.sum(0)


class LDAM_meta(nn.Module):
    # feature_num（特征数量）
    # class_num（类别数量）
    # cls_num_list（每个类别的有效样本数量列表）
    # max_m（最大的m值，默认为0.5）
    # s（标量，默认为30）
    def __init__(self, feature_num, class_num, cls_num_list, max_m=0.5, s=30):
        super(LDAM_meta, self).__init__()
        # estimator用于估计每个类别的样本均值和协方差矩阵。
        self.estimator = EstimatorCV(feature_num, class_num)
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()

        # 使用给定的cls_num_list计算每个类别的权重m_list
        # 这里使用了指数衰减公式，通过将每个类别的有效样本数量开
        # 四次方根并归一化，得到了每个类别的权重。然后，将权重转
        # 换为PyTorch张量，并将其移动到GPU上。
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list

        assert s > 0
        self.s = s


    # 样本增强，根据全连接层，特征，输出结果，标签，类别权重，增强比例，返回输出结果output = y
    def IDASAug(self, fc, features, y_s, labels_s, s_cv_matrix, ratio):
        # 获取输入features的批次大小（样本数量）。
        N = features.size(0)
        # 获取类别的数量。
        C = self.class_num
        # 获取特征的维度。
        A = features.size(1)

        # 获取模型fc的权重参数。
        weight_m = list(fc.named_leaves())[0][1]

        # 将权重参数扩展为与输入特征相同的维
        NxW_ij = weight_m.expand(N, C, A)
        # 根据labels_s的值，从NxW_ij中选择相应的权重。
        NxW_kj = torch.gather(NxW_ij, 1, labels_s.view(N, 1, 1).expand(N, C, A))

        # 根据labels_s的值，从s_cv_matrix中选择相应的子集。
        s_CV_temp = s_cv_matrix[labels_s]

        # 计算sigma2，它是一个与输入特征相同形状的张量，用于调整特征增强的幅度。
        sigma2 = ratio * torch.bmm(torch.bmm(NxW_ij - NxW_kj, s_CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
        # 将sigma2乘以一个与类别数量相同的单位矩阵，并对第二个维度求和，得到每个样本的sigma2值。
        sigma2 = sigma2.mul(torch.eye(C).cuda().expand(N, C, C)).sum(2).view(N, C)

        # 将原始输出y_s与sigma2相加，得到增强后的输出。
        aug_result = y_s + 0.5 * sigma2
        # 创建一个与y_s相同形状的全零张量。
        index = torch.zeros_like(y_s, dtype=torch.uint8)
        # 根据labels_s的值，在index张量中设置对应位置为1。
        index.scatter_(1, labels_s.data.view(-1, 1), 1)

        # 根据labels_s的值，在index张量中设置对应位置为1。
        index_float = index.type(torch.cuda.FloatTensor)
        # 将self.m_list与index_float的转置相乘。
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        # 将batch_m张量的形状改变为(-1, 1)。
        batch_m = batch_m.view((-1, 1))
        # 将增强后的输出减去batch_m。
        aug_result_m = aug_result - batch_m

        # 根据index的值，在aug_result_m和aug_result之间进行选择，得到最终的输出。
        output = torch.where(index, aug_result_m, aug_result)
        # print(output)
        return output

    def forward(self, fc, features, y_s, labels, ratio, weights, cv, epoch, manner):
        # Add
        # print(epoch)
        # if epoch < 1:
        #     print('enter')
        #     features, _ = self.attention(features)
        # print("epoch:{}".format(epoch))
        aug_y = self.IDASAug(fc, features, y_s, labels, cv, ratio)
        if ratio>0.5:
            print("epoch:{}".format(epoch))
            print("y_s:{}".format(y_s))
            print("labels:{}".format(labels))
            print("ratio:{}".format(ratio))
            print("aug_y:{}".format(aug_y))

        # print("Enter aug_y")
        if manner == "update":  # 后40行main函数的执行
            self.estimator.update_CV(features.detach(), labels)
            loss = F.cross_entropy(aug_y, labels, weight=weights)
        else:  # 前160行 + 后40行meta函数的执行
            loss = F.cross_entropy(aug_y, labels, weight=weights)
        return loss

    # 获取当前估计的协方差矩阵
    def get_cv(self):
        return self.estimator.CoVariance

    # 更新协方差矩阵
    def update_cv(self, cv):
        self.estimator.CoVariance = cv
