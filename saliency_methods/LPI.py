#!/usr/bin/env python
import functools
import operator

import numpy as np
import torch
from torch.autograd import grad
import torch.utils.data
import torch.nn.functional as F

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class LPI(object):
    def __init__(self, model, k, density, random_alpha=False, scale_by_inputs=True, datasets=None):
        self.model = model
        self.model.eval()
        self.k = k
        self.random_alpha = random_alpha
        self.scale_by_inputs = scale_by_inputs
        self.density = density

        self.bg_size = len(datasets[0].imgs)
        self.ref_samplers = []
        for baseline_set in datasets:
            self.ref_samplers.append(torch.utils.data.DataLoader(
                dataset=baseline_set,
                batch_size=self.bg_size,
                shuffle=False,
                pin_memory=False,
                drop_last=False))

        densities = self.density.reshape([1, -1, 1, 1, 1])
        self.density_tensor = densities.cuda()

        self.f_std = 0.
        self.b_std = 0.
        self.img_num = 0

    def _get_ref_batch(self, c_ind):
        return next(iter(self.ref_samplers[c_ind]))[0].float()

    def _get_samples_input(self, input_tensor, reference_tensor):
        '''
        calculate interpolation points
        Args:
            input_tensor: Tensor of shape (batch, ...), where ... indicates
                          the input dimensions.
            reference_tensor: A tensor of shape (batch, k, ...) where ...
                indicates dimensions, and k represents the number of background
                reference samples to draw per input in the batch.
        Returns:
            samples_input: A tensor of shape (batch, k, ...) with the
                interpolated points between input and ref.
        '''
        input_dims = list(input_tensor.size())[1:]
        num_input_dims = len(input_dims)

        batch_size = reference_tensor.size()[0]
        k_ = self.k

        # Grab a [batch_size, k]-sized interpolation sample
        if self.random_alpha:
            t_tensor = torch.FloatTensor(batch_size, k_*self.bg_size).uniform_(0.01, 0.99).to(DEFAULT_DEVICE)  # ratio for ref

        else:
            if k_ == 1:
                t_tensor = torch.cat([torch.Tensor([1.0]) for i in range(batch_size*k_*self.bg_size)]).to(DEFAULT_DEVICE)
            else:
                t_tensor = torch.cat([torch.linspace(0, 1, k_) for i in range(batch_size*self.bg_size)]).to(DEFAULT_DEVICE)

        # -------------------- evaluate the end points ----------------------
        shape = [batch_size, k_*self.bg_size] + [1] * num_input_dims
        interp_coef = t_tensor.view(*shape)
        end_point_ref = (1.0 - interp_coef) * reference_tensor
        input_expand_mult = input_tensor.unsqueeze(1)
        end_point_input = interp_coef * input_expand_mult
        # A fine Affine Combine
        samples_input = end_point_input + end_point_ref

        return samples_input

    def _get_samples_delta(self, input_tensor, reference_tensor):
        input_expand_mult = input_tensor  # .unsqueeze(1)
        sd = input_expand_mult - reference_tensor[:, ::self.k, :]
        return sd

    def _get_grads(self, samples_input, sparse_labels=None):
        samples_input.requires_grad = True
        shape = list(samples_input.shape)
        shape[1] = self.bg_size
        grad_tensor = torch.zeros(shape).float().to(DEFAULT_DEVICE)
        # shape[1] = self.k
        # grad_sub_tensor = torch.zeros(shape).float().to(DEFAULT_DEVICE)

        for bg_id in range(self.bg_size):
            for k_id in range(self.k):
                particular_slice = samples_input[:, bg_id*self.k+k_id]
                # output, _, proto_output, _ = model(particular_slice)

                output = self.model(particular_slice)
                # additional
                # output = torch.log_softmax(output, 1)
                # original
                batch_output = output

                # should check that users pass in sparse labels
                # Only look at the user-specified label
                if sparse_labels is not None and batch_output.size(1) > 1:
                    sample_indices = torch.arange(0, batch_output.size(0)).to(DEFAULT_DEVICE)
                    indices_tensor = torch.cat([
                            sample_indices.unsqueeze(1),
                            sparse_labels.unsqueeze(1)], dim=1)
                    batch_output = gather_nd(batch_output, indices_tensor)

                self.model.zero_grad()
                model_grads = grad(
                        outputs=batch_output,
                        inputs=particular_slice,
                        grad_outputs=torch.ones_like(batch_output).to(DEFAULT_DEVICE),
                        create_graph=True)

                grad_tensor[:, bg_id, :] += (model_grads[0].detach().data / self.k) * self.density_tensor[:, bg_id, :]

        return grad_tensor

    def shap_values(self, input_tensor, sparse_labels=None, centers=None):
        """
        Calculate expected gradients approximation of Shapley values for the
        sample ``input_tensor``.

        Args:
            model (torch.nn.Module): Pytorch neural network model for which the
                output should be explained.
            input_tensor (torch.Tensor): Pytorch tensor representing the input
                to be explained.
            sparse_labels (optional, default=None):
            inter (optional, default=None)
        """
        shape = list(input_tensor.shape)
        shape.insert(1, self.bg_size) # self.k*
        # reference_tensor = torch.zeros(shape).float().to(DEFAULT_DEVICE)

        # if LPI else LPI
        ref = []
        if self.density.shape[0] > 1:
            for c_ind in centers:
                ref.append(self._get_ref_batch(c_ind))

            density_lst = []
            for b_ind in range(shape[0]):
                center = centers[b_ind]
                density_lst.append(self.density[center])
            densities = torch.cat(density_lst)
            densities = densities.reshape([shape[0], -1, 1, 1, 1])
            density_tensor = densities.cuda()
            self.density_tensor = density_tensor
        else:
            ref = [self._get_ref_batch(0) for _ in range(shape[0])]

        ref = torch.cat(ref)
        reference_tensor = ref.view(*shape).cuda()
        reference_tensor = reference_tensor.repeat(1, self.k, 1, 1, 1)

        samples_input = self._get_samples_input(input_tensor, reference_tensor)
        samples_delta = self._get_samples_delta(samples_input, reference_tensor)
        grad_tensor = self._get_grads(reference_tensor, sparse_labels)
        grad_tensor = grad_tensor.view(shape)

        mult_grads = samples_delta * grad_tensor if self.scale_by_inputs else grad_tensor
        attribution = mult_grads.mean(1)

        return attribution
