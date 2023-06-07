import os
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0,1')
args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision import models
from utils.settings import img_size
from utils.preprocess import mean, std
from saliency_methods import RandomBaseline, Gradients, IntegratedGradients, ExpectedGradients, AGI, LPI
from DiffID import DiffID


def load_explainer(model, **kwargs):
    method_name = kwargs['method_name']
    if method_name == 'Random':
        # ------------------------------------ Random Baseline ----------------------------------------------
        print('================= Random Baseline ==================')
        random = RandomBaseline()
        return random
    elif method_name == 'InputGrad':
        # ------------------------------------ Input Gradients ----------------------------------------------
        print('================= Input Gradients ==================')
        input_grad = Gradients(model)
        return input_grad
    elif method_name == 'AGI':
        # ------------------------------------ AGI ----------------------------------------------
        print('================= AGI ==================')
        k = kwargs['k']
        top_k = kwargs['top_k']
        cls_num = kwargs['cls_num']
        agi = AGI(model, k=k, top_k=top_k, cls_num=cls_num)
        return agi
    elif method_name == 'ExpGrad' or method_name == 'ExpGrad_new':
        # ------------------------------------ Expected Gradients ----------------------------------------------
        print('============================ Expected Gradients ============================')
        k = kwargs['k']
        bg_size = kwargs['bg_size']
        train_dataset = kwargs['train_dataset']
        test_batch_size = kwargs['test_batch_size']
        random_alpha = kwargs['random_alpha']
        expected_grad = ExpectedGradients(model, k=k, bg_dataset=train_dataset, bg_size=bg_size, batch_size=test_batch_size, random_alpha=random_alpha)
        # expected_grad = ExpectedGradients(model, k=k, bg_dataset=train_dataset, bg_size=bg_size, batch_size=test_batch_size)
        return expected_grad
    elif method_name == 'IntGrad':
        # ------------------------------------ Integrated Gradients ----------------------------------------------
        print('============================ Integrated Gradients ============================')
        k = kwargs['k']
        integrated_grad = IntegratedGradients(model, k=k)
        return integrated_grad
    elif method_name == 'LPI':
        # ------------------------------------ Local Path Integration ----------------------------------------------
        print('================= Local Path Integration ==================')
        alpha = True
        k = kwargs['k']
        num_centers = kwargs['num_centers']
        bg_size = kwargs['bg_size']
        root_pth = kwargs['root_pth']
        # ------------------------------------------------------
        bg_datasets = []
        for c_ind in range(num_centers):
            data_pth = os.path.join(root_pth, 'c'+str(num_centers)+'r'+str(bg_size), 'kmeans_c'+str(c_ind))
            bg_datasets.append(datasets.ImageFolder(
                data_pth,
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ])))

        # density function
        density = np.load(os.path.join(root_pth, 'c'+str(num_centers)+'r'+str(bg_size), 'density.npy'))
        density_tensor = torch.from_numpy(density)

        # ----------------------------------------------------------------------------------
        completed_grad = LPI(model, k=k, density=density_tensor, random_alpha=alpha, datasets=bg_datasets)
        return completed_grad
    else:
        return None


def load_dataset(dataset_name, test_batch_size):
    # ---------------------------- imagenet train ---------------------------
    if 'ImageNet' in dataset_name:
        imagenet_train_dataset = datasets.ImageNet(
            root='datasets',
            split='train',
            transform=transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]))
        # ---------------------------- imagenet eval ---------------------------
        imagenet_val_dataset = datasets.ImageNet(
            root='datasets',
            split='val',
            transform=transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]))
        if 'subset' in dataset_name:
            # ---------------------------- subset of imagenet eval ---------------------------
            data_pth = 'datasets/subset'
            imagenet_val_dataset = datasets.ImageFolder(
                data_pth,
                transforms.Compose([
                    transforms.Resize(size=(img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]))
        imagenet_val_loader = torch.utils.data.DataLoader(
            imagenet_val_dataset, batch_size=test_batch_size,
            shuffle=False, num_workers=4, pin_memory=False)

        return imagenet_train_dataset, imagenet_val_loader


def quan_exp(method_name, model_name, dataset_name, k=None, bg_size=None):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
    model = model.to('cuda')
    model = torch.nn.DataParallel(model)
    model.eval()

    # =================== load train dataset & test loader ========================

    test_bth = 40
    train_dataset, test_loader = load_dataset(dataset_name=dataset_name, test_batch_size=test_bth)

    # =================== load explainer ========================
    explainer_args = {
        'Random': {'method_name': 'Random'},
        'InputGrad': {'method_name': 'InputGrad'},
        'IntGrad': {'method_name': 'IntGrad', 'k': 20},
        'AGI': {'method_name': 'AGI', 'k': 20, 'top_k': 1, 'cls_num': 1000},
        'ExpGrad': {'method_name': 'ExpGrad', 'k': 1, 'bg_size': 20, 'train_dataset':train_dataset,
                    'test_batch_size': test_bth, 'random_alpha': False},
        'LPI': {'method_name': 'LPI', 'k': 1, 'bg_size': 20, 'num_centers': 11,
                'root_pth': 'dataset_distribution/'+model_name+'/'},
    }

    if k is not None:
        explainer_args[method_name]['k'] = k
        explainer_args[method_name]['bg_size'] = bg_size

    # load criterion
    explainer = load_explainer(model=model, **explainer_args[method_name])
    diff_id = DiffID(model, explainer=explainer, dataloader=test_loader)

    # --------------------- perturb experiments ----------------------
    if method_name == 'LPI':
        cent_num = explainer_args['LPI']['num_centers']
        centers = None
        if cent_num > 1:
            centers = np.load(
                'dataset_distribution/' + model_name +
                '/kmeans_center_' + str(cent_num) + '.npy')
        diff_id.quantify(baseline_name='mean', q_ratio_lst=[step * 0.1 for step in range(1, 10)], centers=centers)
    else:
        diff_id.quantify(baseline_name='mean', q_ratio_lst=[step * 0.1 for step in range(1, 10)])


if __name__ == '__main__':
    for method in ['LPI', 'IntGrad']:
        quan_exp(method_name=method, model_name='vgg16', dataset_name='ImageNet_subset')
