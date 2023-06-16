#!/usr/bin/env python
import functools
import operator

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader
import random
# DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F

from utils.preprocess import preprocess, undo_preprocess


def gather_nd(params, indices):
    """
    Args:
        params: Tensor to index
        indices: k-dimension tensor of integers.
    Returns:
        output: 1-dimensional tensor of elements of ``params``, where
            output[i] = params[i][indices[i]]

            params   indices   output

            1 2       1 1       4
            3 4       2 0 ----> 5
            5 6       0 0       1
    """
    max_value = functools.reduce(operator.mul, list(params.size())) - 1
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i]*m
        m *= params.size(i)

    idx[idx < 0] = 0
    idx[idx > max_value] = 0
    return torch.take(params, idx)

# count = 0


def fgsm_step(image, epsilon, data_grad_adv, data_grad_lab):
    # generate the perturbed image based on steepest descent
    # grad_lab_norm = torch.norm(data_grad_lab, p=2)
    delta = epsilon * data_grad_adv.sign()

    # + delta because we are ascending
    perturbed_image = image + delta
    perturbed_rect = torch.clamp(perturbed_image, min=0, max=1)

    delta = perturbed_rect - image
    delta = - data_grad_lab * delta

    return perturbed_rect, delta


def pgd_step(image, epsilon, model, init_pred, targeted, max_iter):
    """target here is the targeted class to be perturbed to"""
    perturbed_image = image.clone()
    c_delta = 0  # cumulative delta
    sign = 0
    for i in range(max_iter):
        # requires grads
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        # if attack is successful, then break
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if False not in (pred == targeted.view(-1, 1)):
            break

        output = -torch.log_softmax(output, 1)/output.shape[0]
        sample_indices = torch.arange(0, output.size(0)).cuda()
        indices_tensor = torch.cat([
            sample_indices.unsqueeze(1),
            targeted.unsqueeze(1)], dim=1)
        loss = gather_nd(output, indices_tensor)
        model.zero_grad()
        model_grads = grad(
            outputs=loss,
            inputs=perturbed_image,
            grad_outputs=torch.ones_like(loss).cuda(),
            create_graph=True)
        data_grad_adv = model_grads[0].detach().data

        sample_indices = torch.arange(0, output.size(0)).cuda()
        indices_tensor = torch.cat([
            sample_indices.unsqueeze(1),
            init_pred.unsqueeze(1)], dim=1)
        loss = gather_nd(output, indices_tensor)
        model.zero_grad()
        model_grads = grad(
            outputs=loss,
            inputs=perturbed_image,
            grad_outputs=torch.ones_like(loss).cuda(),
            create_graph=True)
        data_grad_lab = model_grads[0].detach().data

        perturbed_image, delta = fgsm_step(image, epsilon, data_grad_adv, data_grad_lab)
        c_delta += delta

    return c_delta, perturbed_image


class AGI(object):
    def __init__(self, model, k, top_k, cls_num, eps=0.05):
        self.model = model
        self.cls_num = cls_num - 1
        self.eps = eps
        self.k = k
        self.top_k = top_k

    def select_id(self, label):
        while True:
            top_ids = random.sample(list(range(0, self.cls_num - 1)), self.top_k)
            if label not in top_ids:
                break
            else:
                continue
        return torch.as_tensor(random.sample(list(range(0, self.cls_num - 1)), self.top_k)).view([1, -1])

    def shap_values(self, input_tensor, sparse_labels=None):

        # Forward pass the data through the model
        output = self.model(input_tensor)
        self.model.eval()
        init_pred = output.max(1, keepdim=True)[1].squeeze(1)  # get the index of the max log-probability

        # initialize the step_grad towards all target false classes
        step_grad = 0
        top_ids_lst = []
        for bth in range(input_tensor.shape[0]):
            top_ids_lst.append(self.select_id(sparse_labels[bth]))  # only for predefined ids
        top_ids = torch.cat(top_ids_lst, dim=0).cuda()

        for l in range(top_ids.shape[1]):
            targeted = top_ids[:, l].cuda()
            delta, perturbed_image = pgd_step(undo_preprocess(input_tensor), self.eps, self.model, init_pred, targeted, self.k)
            # delta, perturbed_image = pgd_step(input_tensor, self.eps, self.model, init_pred, targeted, self.k)

            step_grad += delta

        attribution = step_grad
        return attribution
