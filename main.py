import os
import argparse
import numpy as np

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision import models
from utils.settings import img_size
from utils.preprocess import mean_std_dict
from saliency_methods import RandomBaseline, Gradients, IntegratedGradients, ExpectedGradients, AGI, LPI
from evaluator import Evaluator


from utils.settings import parser_choices, parser_default

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-attr_method', type=str, required=False,
                    choices=parser_choices['attr_method'],
                    default=parser_default['attr_method'])
parser.add_argument('-model', type=str, required=False,
                    choices=parser_choices['model'],
                    default=parser_default['model'])
parser.add_argument('-dataset', type=str, required=False,
                    choices=parser_choices['dataset'],
                    default=parser_default['dataset'])
parser.add_argument('-metric', type=str, required=False,
                    choices=parser_choices['metric'],
                    default=parser_default['metric'])
parser.add_argument('-k', type=int, required=False,
                    default=parser_default['k'])
parser.add_argument('-bg_size', type=int, required=False,
                    default=parser_default['bg_size'])
parser.add_argument('-num_centers', type=int, required=False,
                    default=parser_default['num_centers'])
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid


def load_explainer(model, **kwargs):
    method_name = kwargs['method_name']
    if method_name == 'Random':
        print('================= Random Baseline ==================')
        random = RandomBaseline()
        return random
    elif method_name == 'InputGrad':
        print('================= Input Gradients ==================')
        input_grad = Gradients(model)
        return input_grad
    elif method_name == 'AGI':
        print('================= AGI ==================')
        agi = AGI(model, k=kwargs['k'], top_k=kwargs['top_k'], cls_num=kwargs['cls_num'])
        return agi
    elif method_name == 'ExpGrad' or method_name == 'ExpGrad_new':
        print('============================ Expected Gradients ============================')
        expected_grad = ExpectedGradients(model, k=kwargs['k'], bg_dataset=kwargs['train_dataset'],
                                          bg_size=kwargs['bg_size'], batch_size=kwargs['test_batch_size'], random_alpha=kwargs['random_alpha'])
        return expected_grad
    elif method_name == 'IntGrad':
        print('============================ Integrated Gradients ============================')
        integrated_grad = IntegratedGradients(model, k=kwargs['k'], dataset_name=kwargs['dataset_name'])
        return integrated_grad
    elif method_name == 'LPI':
        print('================= Local Path Integration ==================')
        num_centers = kwargs['num_centers']
        bg_size = kwargs['bg_size']
        root_pth = kwargs['root_pth']
        # ------------------------------------------------------
        bg_datasets = []
        for c_ind in range(num_centers):
            data_pth = os.path.join(root_pth, 'c'+str(num_centers)+'r'+str(bg_size), 'kmeans_c'+str(c_ind))
            mean, std = mean_std_dict[kwargs['dataset_name']]
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
        completed_grad = LPI(model, k=kwargs['k'], density=density_tensor, random_alpha=True, datasets=bg_datasets)
        return completed_grad
    else:
        return None


def load_dataset(dataset_name, test_batch_size):

    # ---------------------------- imagenet train ---------------------------
    if 'imagenet' in dataset_name:
        mean, std = mean_std_dict['imagenet']

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

        imagenet_val_loader = torch.utils.data.DataLoader(
            imagenet_val_dataset, batch_size=test_batch_size,
            shuffle=False, num_workers=4, pin_memory=False)

        return imagenet_train_dataset, imagenet_val_loader


def attr_eval(method_name, model_name, dataset_name, metric, k=None, bg_size=None, num_centers=None):
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
        'IntGrad': {'method_name': 'IntGrad', 'k': 20, 'dataset_name': dataset_name},
        'AGI': {'method_name': 'AGI', 'k': 20, 'top_k': 1, 'cls_num': 1000},
        'ExpGrad': {'method_name': 'ExpGrad', 'k': 1, 'bg_size': 20, 'train_dataset': train_dataset,
                    'test_batch_size': test_bth, 'random_alpha': False},
        'LPI': {'method_name': 'LPI', 'k': 1, 'bg_size': 20, 'num_centers': num_centers,
                'root_pth': 'dataset_distribution/'+model_name+'/'},
    }

    if k is not None:
        explainer_args[method_name]['k'] = k
        explainer_args[method_name]['bg_size'] = bg_size

    # load criterion
    explainer = load_explainer(model=model, **explainer_args[method_name])
    evaluator = Evaluator(model, explainer=explainer, dataloader=test_loader)

    # --------------------- perturb experiments ----------------------
    if metric == 'DiffID':
        if method_name == 'LPI':
            cent_num = explainer_args['LPI']['num_centers']
            centers = None
            if cent_num > 1:
                centers = np.load(
                    'dataset_distribution/' + model_name +
                    '/kmeans_center_' + str(cent_num) + '.npy')
            evaluator.DiffID(ratio_lst=[step * 0.1 for step in range(1, 10)], centers=centers)
        else:
            evaluator.DiffID(ratio_lst=[step * 0.1 for step in range(1, 10)])

    if metric == 'visualize':

        num_vis = 50
        f_name = 'exp_fig/'+method_name+'_vis/'
        if method_name == 'LPI':
            cent_num = explainer_args['LPI']['num_centers']
            centers = None
            if cent_num > 1:
                centers = np.load(
                    'dataset_distribution/' + model_name +
                    '/kmeans_center_' + str(cent_num) + '.npy')
            evaluator.visual_inspection(file_name=f_name, num_vis=num_vis, method_name=method_name, centers=centers)
        else:
            evaluator.visual_inspection(file_name=f_name, num_vis=num_vis, method_name=method_name)


if __name__ == '__main__':

    attr_eval(method_name=args.attr_method, model_name=args.model, dataset_name=args.dataset, metric=args.metric,
              k=args.k, bg_size=args.bg_size, num_centers=args.num_centers)
