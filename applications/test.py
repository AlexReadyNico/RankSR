# -*- coding: utf-8 -*-
""" 模型测试脚本 """
import time
import logging
import argparse
import numpy

import torch
from torch.utils.data import DataLoader
from torch import nn

from utils import my_meters
from scipy import stats


def test(test_loader: DataLoader, model: nn.Module, criterion: nn.Module,
         args: argparse.Namespace):
    """ 验证集、测试集 评估
    :param test_loader: 测试集DataLoader对象
    :param model: 待测试模型
    :param criterion: 损失函数
    :param args: 测试参数
    """
    batch_time = my_meters.AverageMeter('Time', ':6.3f')
    losses = my_meters.AverageMeter('Loss', ':.4e')
    top1 = my_meters.AverageMeter('Acc@1', ':6.2f')
    progress = my_meters.ProgressMeter(
        len(test_loader), batch_time, losses, top1, prefix='Test: ')

    # 模型评估
    model.eval()
    #-------------------------------------------------------zhengruidi------------------------------#
    outputs_temp = numpy.zeros((len(test_loader),), dtype = numpy.float)
    targets_temp = numpy.zeros((len(test_loader),), dtype = numpy.float)
    #-------------------------------------------------------zhengruidi------------------------------#
    total_paths, total_preds, total_probs = list(), list(), list()  # 样本的路径
    with torch.no_grad():
        end_time = time.time()
        for i, (images, scores, paths) in enumerate(test_loader):
            if args.cuda:
                images = images.cuda(args.gpu, non_blocking=True)
                scores = scores.cuda(args.gpu, non_blocking=True)

            # 模型预测
            outputs = model(images)
            outputs = outputs.squeeze(1) 
            # logging.info(paths)
            # logging.info(outputs)
            #logging.info(outputs.view(-1).sigmoid())
            #logging.info(outputs)
            # print((outputs[0].softmax(dim=-1) * torch.arange(1,1001,device = 'cuda')).sum(dim=-1)/10)
            # print((outputs[1].softmax(dim=-1) * torch.arange(1,1001,device = 'cuda')).sum(dim=-1)/10)
            loss = criterion(outputs, scores)
            
            # 统计准确率和损失函数
            acc1, pred, probs = criterion.accuracy(outputs, scores)
            #---------------------------------Calculate-LCCSROCC---------------------------------20210319zhengruidi--------------------------------------------#
            # outputs_temp2 = (model(images).softmax(dim=1) * torch.arange(1, 1001 ,device = 'cuda')).sum(dim=1)/10
            # print(outputs_temp2.item())
            # print(paths)
            # outputs_temp[i]=round(outputs_temp2.item(),2)
            # targets_temp[i]=round(scores.item(),2)
            #-------------------------------------Calculate-LCCSROCC----------------------20210319zhengruidi--------------------------------------------#
            # 收集结果
            if args.evaluate:
                total_paths.extend(zip(paths[0::2], paths[1::2]) if args.criterion == 'rank' else paths)
                # total_preds.extend(pred.detach().cpu().numpy())
                #total_preds.extend(pred)
                #total_probs.extend(probs.detach().cpu().numpy())

            # 统计量
            #losses.update(loss.item(), images.size(0)/2)
            #top1.update(acc1.item(), images.size(0)/2)
            batch_size = images.size(0)/2 if args.criterion == 'rank' else images.size(0)
            losses.update(loss.detach().cpu().item(), batch_size)
            top1.update(acc1, batch_size)
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            if i % args.print_freq == 0:
                progress.print(i)
        #---------------------------------Calculate-LCCSROCC-----------------------20210319zhengruidi--------------------------------------------#
        # print(outputs_temp)
        # print(targets_temp)
        # srocc = stats.spearmanr(outputs_temp,targets_temp)[0]
        # lcc = stats.pearsonr(outputs_temp,targets_temp)[0]
        # print ('%   LCC of mean : {}'.format(lcc))
        # print ('% SROCC of mean: {}'.format(srocc))
        #------------------------------------Calculate-LCCSROCC------------------20210319zhengruidi--------------------------------------------#
    logging.info(f'* Acc@1 {top1.avg:.3f} and loss {losses.avg:.3f} with time {batch_time.avg:.3f}')
    return top1.avg, losses.avg, zip(total_paths, total_preds, total_probs)
