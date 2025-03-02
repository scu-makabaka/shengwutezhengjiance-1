import os
import argparse
import pandas as pd
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys
sys.path.append('./') # change as you need
from datasets import nirislDataset
from models import EfficientUNet
from location import get_edge

def mask_to_boundary(mask, dilation_ratio=0.02):
    mask = mask.astype(np.uint8)  # 先把 mask 转换成 uint8 类型
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = max(int(round(dilation_ratio * img_diag)), 1)
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    return mask - new_mask_erode[1:h+1, 1:w+1]

def boundary_iou(gt, dt, dilation_ratio=0.02):
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    return intersection / union

def get_args():
    parser = argparse.ArgumentParser(description='Test parameters')
    parser.add_argument('--dataset', required=True, type=str, dest='dataset_name')
    parser.add_argument('--ckpath', required=True, type=str, dest='checkpoints_path')
    return parser.parse_args()

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def test(test_loader, net, save_dir):
    print('Start testing...')
    names, pupil_bious = [], []
    
    state_dict = torch.load(os.path.join(test_args['checkpoints_path'], 'for_inner.pth'), map_location=device)
    state_dict["module.heatmap4.loc.0.weight"] = state_dict.pop('module.loc4.loc.0.weight')
    state_dict["module.heatmap3.loc.0.weight"] = state_dict.pop('module.loc3.loc.0.weight')
    state_dict["module.heatmap2.loc.0.weight"] = state_dict.pop('module.loc2.loc.0.weight')
    state_dict["module.heatmap4.loc.0.bias"] = state_dict.pop('module.loc4.loc.0.bias')
    state_dict["module.heatmap3.loc.0.bias"] = state_dict.pop('module.loc3.loc.0.bias')
    state_dict["module.heatmap2.loc.0.bias"] = state_dict.pop('module.loc2.loc.0.bias')

    net.load_state_dict(state_dict)
    net.eval()
    
    for i, data in enumerate(test_loader):
        image_name, image = data['image_name'][0], data['image']
        print(f'Testing {i+1}-th image: {image_name}')
        image = Variable(image).to(device)
        
        with torch.no_grad():
            outputs = net(image)
        
        pred_pupil_circle_mask, pred_pupil_egde, pupil_circles_param = get_edge(outputs['pred_pupil_mask'])
        pred_pupil_mask = outputs['pred_pupil_mask'][0,0].cpu().numpy() > 0
        gt_pupil_mask = pred_pupil_circle_mask[0,0].cpu().numpy() > 0
        b_iou = boundary_iou(gt_pupil_mask, pred_pupil_mask)
        pupil_bious.append(b_iou)
        names.append(image_name)
    
    results_path = os.path.join(save_dir, 'pupil_biou_results.xlsx')
    pd.DataFrame({'name': names, 'pupil_BIoU': pupil_bious}).to_excel(results_path)
    print('Test done!')

def main(test_args):
    net = EfficientUNet(num_classes=3).to(device)
    net = torch.nn.DataParallel(net)
    test_dataset = nirislDataset(test_args['dataset_name'], mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False)
    
    save_dir = os.path.join('test-result', test_args['dataset_name'])
    check_mkdir(save_dir)
    test(test_loader, net, save_dir)

if __name__ == '__main__':
    args = get_args()
    test_args = {'dataset_name': args.dataset_name, 'checkpoints_path': args.checkpoints_path}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(test_args)
