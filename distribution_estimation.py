import os.path
import cv2

import numpy as np
import numpy.linalg

import torch.utils.data
from torchvision import models

from utils import img_size
from utils import mean, std

from generation_by_optimization import render_representation, render_density, render_sub_icons
# import umap
import matplotlib.pyplot as plt

import argparse

#     dataset = 'ImageNet'
#     model = 'resnet34'
#     center_num_lst = [11]
#     ref_num_lst = [20, ]

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-dataset', type=str, required=False, default='ImageNet')
parser.add_argument('-model', type=str, required=False, default='resnet34')
parser.add_argument('-center_num', type=str, required=False, default='0,9')
parser.add_argument('-ref_num', type=str, required=False, default='20,10')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid


def computing_distribution(dataset_name, model_name, center_num_lst, ref_num_lst):
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    # ------------------------------ model and dataset preparation --------------------------------
    ####################################################
    ############### IMAGENET 2012 ######################
    ####################################################
    imagenet_train_dataset = datasets.ImageNet(
        root='datasets',
        split='train',
        transform=transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]))
    test_batch_size = 200
    imagenet_train_loader = torch.utils.data.DataLoader(
        imagenet_train_dataset, batch_size=test_batch_size,
        shuffle=False, num_workers=20, pin_memory=False)

    ####################################################
    #################### CIFAR-10 ######################
    ####################################################
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    cifar10_tr_dataset = datasets.CIFAR10('datasets/cifar10',
                                          train=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
                                          ]),
                                          download=True)
    cifar10_tr_loader = torch.utils.data.DataLoader(
        dataset=cifar10_tr_dataset, batch_size=test_batch_size,
        shuffle=True, num_workers=6, pin_memory=False, drop_last=False)

    ####################################################
    ################### CIFAR-100 ######################
    ####################################################
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    cifar100_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    cifar100_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    cifar100_tr_dataset = datasets.CIFAR100('datasets/cifar100',
                                            train=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
                                            ]),
                                            download=True)
    cifar100_tr_loader = torch.utils.data.DataLoader(
        dataset=cifar100_tr_dataset, batch_size=test_batch_size,
        shuffle=True, num_workers=6, pin_memory=False, drop_last=False)

    # choose dataset
    if dataset_name == 'ImageNet':
        imagenet_train_loader = imagenet_train_loader
    if dataset_name == 'CIFAR10':
        imagenet_train_loader = cifar10_tr_loader
    elif dataset_name == 'CIFAR100':
        imagenet_train_loader = cifar100_tr_loader

    # choose model
    root_pth = 'dataset_distribution/' + model_name + '/'

    # ---------------------------------- render representations ---------------------------------
    from networks.preact_resnet import PreActResNet18
    if model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    # elif model_name == 'paresnet18_cifar100':
    #     model = PreActResNet18(num_classes=100)
    # elif model_name == 'paresnet18':
    #     model = PreActResNet18()

    model = model.to('cuda')
    model = torch.nn.DataParallel(model)
    model.eval()

    # model_path = None
    # if model_name == 'paresnet18':
    #     model_pth = 'pretrained_models/model_PResNet_CIFAR10.pth'
    # elif model_name == 'paresnet18_cifar100':
    #     model_pth = 'pretrained_models/model_PResNet_CIFAR100.pth'
    # if model_path is not None:
    #     pretrained_model = torch.load(model_pth)
    #     model.load_state_dict(pretrained_model, strict=True)

    temp_pth = 'dataset_distribution/' + model_name
    if not os.path.exists(temp_pth):
        os.mkdir(temp_pth)
    temp_pth += '/representations'
    if not os.path.exists(temp_pth):
        os.mkdir(temp_pth)
        render_representation(model, imagenet_train_loader, f_name=temp_pth + '/rep_', test_batch_size=test_batch_size)

    # ----------------------------- reading representations ---------------------------
    print('-------------- loading representations ---------------')
    folder_pth = root_pth + 'representations'
    folders = os.listdir(folder_pth)
    folders.sort()
    rep_lst = []
    for f in folders:
        rep_lst.append(np.load(root_pth + 'representations/' + f))
    reps = np.concatenate(rep_lst)
    print(reps.shape)

    # # --------------------------- distribution visualization by dimension reduction ------------------------------
    # layout = umap.UMAP(n_components=2, verbose=True, n_neighbors=20, min_dist=0.01, metric="cosine").fit_transform(reps)
    # # layout = TSNE(n_components=2, verbose=True, metric="cosine", learning_rate=10, perplexity=50).fit_transform(d)
    # np.save('dimension_reduce.npy', layout)
    # # --------------------------- visualization ---------------------------------
    # layout = np.load('dimension_reduce.npy')
    # plt.figure(figsize=(10, 10))
    # plt.scatter(x=layout[:, 0], y=layout[:, 1], s=2)
    # plt.savefig('representation.png')

    # --------------------------------------- clustering ---------------------------------------------
    from sklearn.cluster import MiniBatchKMeans, KMeans

    # ------------------- mini-batch k-means -----------------------
    # mini_kmeans = MiniBatchKMeans(n_clusters=10, random_state=0, batch_size=10000, verbose=True, max_iter=100)
    # labels = mini_kmeans.fit_predict(reps)
    # centers = mini_kmeans.cluster_centers_
    # np.save('mini_kmeans_resnet34_n10_centers.npy', centers)
    # np.save('mini_kmeans_resnet34_n10_labels.npy', labels)
    # f_name = root_pth + 'mini_kmeans_resnet34_n10'
    # indicates = render_density(reps, centers, labels, f_name=f_name)
    # render_icons(indicates, f_name=f_name, imagenet_train_loader=imagenet_train_loader)

    # ------------------- k-means -----------------------
    print('--------------- clustering ----------------')
    ################
    num_centers_lst = center_num_lst  # LPI
    reference_num_lst = ref_num_lst
    ################
    for n_center in num_centers_lst:  # , 11

        if os.path.exists(root_pth + 'kmeans_center_'+str(n_center)+'.npy'):
            continue
        kmeans = KMeans(n_clusters=n_center, n_init=3, verbose=True, max_iter=600)
        labels = kmeans.fit_predict(reps)
        centers = kmeans.cluster_centers_
        # ------------------------------------ save clusters ---------------------------------------

        np.save(root_pth + 'kmeans_center_'+str(n_center)+'.npy', centers)
        np.save(root_pth + 'kmeans_center_'+str(n_center)+'_label.npy', labels)
        # render_icons
        # f_name = root_pth + 'kmeans_' + model_name + '_n' + str(num_clu)
        # indicates = render_density(reps, centers, labels, f_name=f_name)
        # render_icons(indicates, f_name=f_name, imagenet_train_loader=imagenet_train_loader)

    # ---------------------- further clustering ----------------------------
    print('--------------- second clustering ----------------')
    # the number of centers
    for n_center in num_centers_lst:
        labels = np.load(root_pth + 'kmeans_center_'+str(n_center)+'_label.npy')

        for ref_n in reference_num_lst:  # ref num [20, 30, 40, 50, 60]
            if os.path.exists(root_pth + 'c' + str(n_center) + 'r' + str(ref_n)):
                continue
            sub_clu_label_lst = []
            for c_ind in range(n_center):
                labels_ = np.where(labels == c_ind)[0]
                reps_ = reps[labels_]

                # =================== local centers ====================
                kmeans = KMeans(n_clusters=ref_n, n_init=3, verbose=True, max_iter=600)
                sub_labels = kmeans.fit_predict(reps_)
                sub_centers = kmeans.cluster_centers_
                sub_lst = []
                for i in range(ref_n):
                    dist = np.linalg.norm(reps_ - sub_centers[i], axis=1)
                    ind = np.argsort(dist)[:1]
                    ind = labels_[ind]
                    sub_lst.append(int(ind))

                sub_clu_label_lst.append(sub_lst)

                f_name = root_pth + 'c' + str(n_center) + 'r' + str(ref_n)
                if not os.path.exists(f_name):
                    os.mkdir(f_name)
                f_name = os.path.join(f_name, 'kmeans_density')
                _ = render_density(reps_, sub_centers, sub_labels, f_name=f_name)

            lbl2c_dict = {}
            for ind, lst in enumerate(sub_clu_label_lst):
                for val in lst:
                    lbl2c_dict[val] = ind
            ind_lst = []
            for lst in sub_clu_label_lst:
                for val in lst:
                    ind_lst.append(val)
            f_name = root_pth + 'c' + str(n_center) + 'r' + str(ref_n)
            if not os.path.exists(f_name):
                os.mkdir(f_name)

            print('---------- rendering icons -----------')
            render_sub_icons(ind_lst, f_name=f_name, imagenet_train_loader=imagenet_train_loader,
                             v2c_dict=lbl2c_dict, test_batch_size=test_batch_size)

    for c_n in num_centers_lst:
        for r_n in reference_num_lst:
            pth = root_pth + 'c' + str(c_n) + 'r' + str(r_n)
            dense_lst = []
            for ind in range(c_n):
                den = np.load(
                    os.path.join(pth, 'kmeans_density.npy'))
                dense_lst.append(den)
            dense_array = np.array(dense_lst)
            np.save(os.path.join(pth, 'density.npy'), dense_array)
            print('transfer density complete')


if __name__ == '__main__':
    center_num_lst = [int(num) for num in args.center_num.split(',')]
    ref_num_lst = [int(num) for num in args.ref_num.split(',')]
    computing_distribution(dataset_name=args.dataset, model_name=args.model, center_num_lst=center_num_lst, ref_num_lst=ref_num_lst)
