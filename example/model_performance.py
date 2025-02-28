import os
import argparse
import pandas as pd
import numpy as np
# from thop.profile import profile

import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# import albumentations as A

import sys
sys.path.append('./') # change as you need
from datasets import nirislDataset
from models import EfficientUNet
from location import get_edge

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser(description='Test paprmeters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default=None, type=str, required=True, dest='dataset_name')
    parser.add_argument('--ckpath', default=None, type=str, required=True, dest='checkpoints_path')
    return parser.parse_args()


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def main(test_args):

    assert test_args['dataset_name'] in ['CASIA-Iris-Africa','CASIA-distance', 'Occlusion', 'Off_angle', 'CASIA-Iris-Mobile-V1.0']
    ############################################# define a CNN #################################################
    # net = EfficientUNet(num_classes=3).cuda()
    # net = torch.nn.DataParallel(net, device_ids=[0])
    net = EfficientUNet(num_classes=3).to(device)  # 使用 to(device) 而不是 .cuda()
    net = torch.nn.DataParallel(net)  # 这里使用 DataParallel 适配多 GPU，如果有的话

    ########################################### dataset #############################################
    # test_augment = A.Compose([
    #     # A.Resize(320, 544) # for Africa dataset
    # ])
    test_dataset = nirislDataset(test_args['dataset_name'], mode='test')
    # test_dataset = nirislDataset(test_args['dataset_name'], mode='test', transform=test_augment)
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False)
    print('The dataset {} is ready!'.format(test_args['dataset_name']))

    ######################################### test ############################################
    test(test_loader, net)


def test(test_loader, net):
    print('start test......')
    names, iris_circles_params, pupil_circles_params = [], [], []

    # state_dict = torch.load(os.path.join(test_args['checkpoints_path'], 'for_mask.pth'))
    state_dict = torch.load(os.path.join(test_args['checkpoints_path'], 'for_mask.pth'), map_location=device)

    state_dict["module.heatmap4.loc.0.weight"] = state_dict.pop('module.loc4.loc.0.weight')
    state_dict["module.heatmap3.loc.0.weight"] = state_dict.pop('module.loc3.loc.0.weight')
    state_dict["module.heatmap2.loc.0.weight"] = state_dict.pop('module.loc2.loc.0.weight')
    state_dict["module.heatmap4.loc.0.bias"] = state_dict.pop('module.loc4.loc.0.bias')
    state_dict["module.heatmap3.loc.0.bias"] = state_dict.pop('module.loc3.loc.0.bias')
    state_dict["module.heatmap2.loc.0.bias"] = state_dict.pop('module.loc2.loc.0.bias')
    net.load_state_dict(state_dict)
    net.eval()
    for i, data in enumerate(test_loader):
        image_name, image = data['image_name'][0], data['image'] #BCHW
        print('testing the {}-th image: {}'.format(i+1, image_name))
        # image = Variable(image).cuda()
        image = Variable(image).to(device)  # 使用 .to(device) 来适配当前设备

        with torch.no_grad():
            outputs = net(image)

        pred_mask, pred_iris_mask, pred_pupil_mask = outputs['pred_mask'], outputs['pred_iris_mask'], outputs['pred_pupil_mask']
        pred_mask_pil = transforms.ToPILImage()((pred_mask[0]>0).to(dtype=torch.uint8)*255).convert('L')
        pred_mask_pil.save(os.path.join(SegmentationClass_save_dir, image_name+'.png'))


    # state_dict = torch.load(os.path.join(test_args['checkpoints_path'], 'for_outer.pth'))
    state_dict = torch.load(os.path.join(test_args['checkpoints_path'], 'for_outer.pth'), map_location=device)

    state_dict["module.heatmap4.loc.0.weight"] = state_dict.pop('module.loc4.loc.0.weight')
    state_dict["module.heatmap3.loc.0.weight"] = state_dict.pop('module.loc3.loc.0.weight')
    state_dict["module.heatmap2.loc.0.weight"] = state_dict.pop('module.loc2.loc.0.weight')
    state_dict["module.heatmap4.loc.0.bias"] = state_dict.pop('module.loc4.loc.0.bias')
    state_dict["module.heatmap3.loc.0.bias"] = state_dict.pop('module.loc3.loc.0.bias')
    state_dict["module.heatmap2.loc.0.bias"] = state_dict.pop('module.loc2.loc.0.bias')
    net.load_state_dict(state_dict)
    net.eval()
    for i, data in enumerate(test_loader):
        image_name, image = data['image_name'][0], data['image'] #BCHW
        print('testing the {}-th image: {}'.format(i+1, image_name))
        # image = Variable(image).cuda()
        image = Variable(image).to(device)  # 使用 .to(device) 来适配当前设备

        with torch.no_grad():
            outputs = net(image)

        pred_mask, pred_iris_mask, pred_pupil_mask = \
            outputs['pred_mask'], outputs['pred_iris_mask'], outputs['pred_pupil_mask']
        pred_iris_circle_mask, pred_iris_edge, iris_circles_param = get_edge(pred_iris_mask)

        pred_iris_mask_pil = transforms.ToPILImage()((pred_iris_mask[0]>0).to(dtype=torch.uint8)*255).convert('L')
        pred_iris_circle_mask_pil = transforms.ToPILImage()((pred_iris_circle_mask[0]>0).to(dtype=torch.uint8)*255).convert('L')
        pred_iris_edge_pil = transforms.ToPILImage()((pred_iris_edge[0]>0).to(dtype=torch.uint8)*255).convert('L')
        iris_circles_params.append(iris_circles_param.cpu().numpy()[0])

        pred_iris_mask_pil.save(os.path.join(iris_edge_mask_raw_save_dir, image_name+'.png'))
        pred_iris_circle_mask_pil.save(os.path.join(iris_edge_mask_save_dir, image_name+'.png'))
        pred_iris_edge_pil.save(os.path.join(Outer_Boundary_save_dir, image_name+'.png'))
  
        
    # state_dict = torch.load(os.path.join(test_args['checkpoints_path'], 'for_inner.pth'))
    state_dict = torch.load(os.path.join(test_args['checkpoints_path'], 'for_outer.pth'), map_location=device)

    state_dict["module.heatmap4.loc.0.weight"] = state_dict.pop('module.loc4.loc.0.weight')
    state_dict["module.heatmap3.loc.0.weight"] = state_dict.pop('module.loc3.loc.0.weight')
    state_dict["module.heatmap2.loc.0.weight"] = state_dict.pop('module.loc2.loc.0.weight')
    state_dict["module.heatmap4.loc.0.bias"] = state_dict.pop('module.loc4.loc.0.bias')
    state_dict["module.heatmap3.loc.0.bias"] = state_dict.pop('module.loc3.loc.0.bias')
    state_dict["module.heatmap2.loc.0.bias"] = state_dict.pop('module.loc2.loc.0.bias')
    net.load_state_dict(state_dict)
    net.eval()
    for i, data in enumerate(test_loader):

        image_name, image = data['image_name'][0], data['image'] #BCHW
        print('testing the {}-th image: {}'.format(i+1, image_name))
        # image = Variable(image).cuda()
        image = Variable(image).to(device)  # 使用 .to(device) 来适配当前设备

        with torch.no_grad():
            outputs = net(image)

        pred_mask, pred_iris_mask, pred_pupil_mask = outputs['pred_mask'], outputs['pred_iris_mask'], outputs['pred_pupil_mask']
        # post processing
        pred_pupil_circle_mask, pred_pupil_egde, pupil_circles_param = get_edge(pred_pupil_mask)
        
        names.append(image_name)
        pupil_circles_params.append(pupil_circles_param.cpu().numpy().tolist()[0])

        pred_pupil_mask_pil = transforms.ToPILImage()((pred_pupil_mask[0]>0).to(dtype=torch.uint8)*255).convert('L')
        pred_pupil_circle_mask_pil = transforms.ToPILImage()((pred_pupil_circle_mask[0]>0).to(dtype=torch.uint8)*255).convert('L')
        pred_pupil_egde_pil = transforms.ToPILImage()((pred_pupil_egde[0]>0).to(dtype=torch.uint8)*255).convert('L')

        pred_pupil_mask_pil.save(os.path.join(pupil_edge_mask_raw_save_dir, image_name+'.png'))
        pred_pupil_circle_mask_pil.save(os.path.join(pupil_edge_mask_save_dir, image_name+'.png'))
        pred_pupil_egde_pil.save(os.path.join(Inner_Boundary_save_dir, image_name+'.png'))


    iris_circles_params = np.asarray(iris_circles_params)
    pupil_circles_params =np.asarray(pupil_circles_params)
    params_path = save_dir + '/test_params.xlsx'
    params_data = pd.DataFrame({
        'name':names,
        'ix':iris_circles_params[:,0],
        'iy':iris_circles_params[:,1],
        'ih':iris_circles_params[:,2],
        'iw':iris_circles_params[:,3],
        'ir':iris_circles_params[:,4],
        'px':pupil_circles_params[:,0],
        'py':pupil_circles_params[:,1],
        'ph':pupil_circles_params[:,2],
        'pw':pupil_circles_params[:,3],
        'pr':pupil_circles_params[:,4]
        })
    params_data.to_excel(params_path)

    print('test done!')
    net.train()


if __name__ == '__main__':
    args = get_args()
    test_args = {
        'dataset_name': args.dataset_name,
        'checkpoints_path': args.checkpoints_path

    }
    
    check_mkdir('./test-result')
    save_dir = os.path.join('test-result', test_args['dataset_name'])
    check_mkdir(save_dir)


    SegmentationClass_save_dir = os.path.join(save_dir, 'SegmentationClass')
    check_mkdir(SegmentationClass_save_dir)
    Inner_Boundary_save_dir = os.path.join(save_dir, 'Inner_Boundary')
    check_mkdir(Inner_Boundary_save_dir)
    Outer_Boundary_save_dir = os.path.join(save_dir, 'Outer_Boundary')
    check_mkdir(Outer_Boundary_save_dir)
    iris_edge_mask_raw_save_dir = os.path.join(save_dir, 'iris_edge_mask_raw')
    check_mkdir(iris_edge_mask_raw_save_dir)
    iris_edge_mask_save_dir = os.path.join(save_dir, 'iris_edge_mask')
    check_mkdir(iris_edge_mask_save_dir)
    pupil_edge_mask_raw_save_dir = os.path.join(save_dir, 'pupil_edge_mask_raw')
    check_mkdir(pupil_edge_mask_raw_save_dir)
    pupil_edge_mask_save_dir = os.path.join(save_dir, 'pupil_edge_mask')
    check_mkdir(pupil_edge_mask_save_dir)

    main(test_args)
