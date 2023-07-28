import torch
import torch.nn.functional as F
import random


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

    for i in range(max_iter):
        # requires grads
        perturbed_image.requires_grad = True

        output = model(perturbed_image)

        # ---------------------- data_grad_label -----------------------
        batch_output = -1. * F.nll_loss(output, targeted.flatten(), reduction='sum')
        model.zero_grad()
        batch_output.backward(retain_graph=True)
        gradients = perturbed_image.grad.clone()
        perturbed_image.grad.zero_()
        gradients.detach()
        data_grad_label = gradients

        # ---------------------- data_grad_pred -----------------------
        batch_output = -1. * F.nll_loss(output, init_pred.flatten(), reduction='sum')
        model.zero_grad()
        batch_output.backward()
        gradients = perturbed_image.grad.clone()
        perturbed_image.grad.zero_()
        gradients.detach()

        data_grad_pred = gradients

        perturbed_image, delta = fgsm_step(image, epsilon, data_grad_label, data_grad_pred)
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
            delta, perturbed_image = pgd_step(input_tensor, self.eps, self.model, init_pred, targeted, self.k)
            step_grad += delta

        attribution = step_grad
        return attribution
