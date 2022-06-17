# -*- coding: utf-8 -*-
""" 对比排序损失 ranking loss """
import torch
from torch import nn
import numpy as np

class RankingLoss(nn.Module):
    """ 对比排序损失 """
    
    def __init__(self, args):
        """ 对比排序损失
        :param args: 超参
        """
        super(RankingLoss, self).__init__()
        self.margin = args.margin
        self.target = torch.ones(args.batch_size)
        if args.cuda:
            self.target = self.target.cuda(args.gpu)

    def forward(self, image_p_batch, _):
        """ 相邻两张图片的rank loss
        :param image_p_batch: 一个batch图片的推理结果，奇偶数两张是对比图，奇数图质量 > 接下来的偶数图质量
        :param _: DMOS分数占位符，因为是ranking，所以不需要该值
        :return 左图（奇数图像）对右图（偶数图像）的margin_ranking_loss
        """
        #image_p_batch = image_p_batch.view(-1).sigmoid()
        #-------------------------------------------------------zhengruidi------------------------------#
        
        #target = torch.tensor([0.0],requires_grad=True,device = 'cuda')
        batch_size = image_p_batch.size()
        output_temp = []
        for i in range(batch_size[0]):
            output_temp.append(image_p_batch[i])
        output1_temp = output_temp[::2]
        output2_temp = output_temp[1::2]
        output1 = torch.stack(output1_temp)
        output2 = torch.stack(output2_temp)
        #-------------------------------------------------------zhengruidi------------------------------#
        #loss = nn.functional.margin_ranking_loss(image_p_batch[0::2], image_p_batch[1::2],
        #                                         self.target, margin=self.margin)
        
        loss = nn.functional.margin_ranking_loss(output2, output1,
                                                 self.target, margin=self.margin)
        return loss

    @staticmethod
    def accuracy(outputs: torch.FloatTensor, _) -> \
            (torch.float32, torch.BoolTensor, torch.FloatTensor):
        """ 计算准确率和预测结果
        :param outputs: 模型输出
        :param _: scores的占位符，ranking loss不需要
        :return 排序（左图>右图）的准确率 torch.float32，
                排序预测结果（左图>右图）(batch_size)，
                左右图的概率(batch_size, 2)
        """
        with torch.no_grad():
            #------------------------------zhengruidi---------------------------#
            #outputs = outputs.view(-1).sigmoid()
            
            batch_size = outputs.size()
            output_temp = np.zeros(batch_size[0], dtype = np.float)
            for i in range(batch_size[0]):
                output_temp[i] = outputs[i]
            output1_temp = output_temp[::2]
            output2_temp = output_temp[1::2]
            output1 = torch.from_numpy(output1_temp)
            output2 = torch.from_numpy(output2_temp)
            #------------------------------zhengruidi---------------------------#
            #probs = torch.stack(output1, output2, dim=1)
            probs = (output1 - output2) > 0
            #-------------------------------------------------------zhengruidi------------------------------#
            # outputs_1 = (outputs[0].softmax(dim=-1) * torch.arange(1,1001,device = 'cuda')).sum(dim=-1)/10
            # outputs_2 = (outputs[1].softmax(dim=-1) * torch.arange(1,1001,device = 'cuda')).sum(dim=-1)/10
            # if(outputs_1 > outputs_2):
            #     pred =1
            # else:
            #     pred =0
            # acc = pred/1*100.0
            #-------------------------------------------------------zhengruidi------------------------------#
            pred = (output2 - output1) > 0
            acc = pred.int().float().mean() * 100.0
            return acc, pred, probs
        
            # outputs = outputs.view(-1).sigmoid()
            # num_classes = len(outputs)
            # outputs_1 = (outputs[0::2].softmax(dim=-1) * torch.arange(1, (num_classes + 1),device = 'cuda')).sum(dim=-1)/10
            # outputs_2 = (outputs[1::2].softmax(dim=-1) * torch.arange(1, (num_classes + 1),device = 'cuda')).sum(dim=-1)/10
            # probs = torch.stack([outputs[0::2], outputs[1::2]], dim=1)
            # pred = (outputs_1 - outputs_2) > 0
            # acc = pred.int().float().mean() * 100.0
            # print(acc)
            # return acc, pred, probs


if __name__ == '__main__':
    ranking_loss = RankingLoss()
    p_batch = torch.rand(20, 1)
    my_loss = ranking_loss(p_batch, None)
    print(f'my_loss: {my_loss}')
    my_acc, my_pred, my_probs = ranking_loss.accuracy(p_batch, None)
    print(f'acc: {my_acc.detach().cpu().item()}, \n'
          f'pred: {my_pred.detach().cpu().numpy()}, \n'
          f'my_probs: {my_probs.detach().cpu().numpy()}')
