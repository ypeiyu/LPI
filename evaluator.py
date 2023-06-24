import time
import cv2
import torch
import numpy as np
import os

from utils import undo_preprocess_input_function
from utils import visualize


def normalize_saliency_map(saliency_map):
    saliency_map = torch.sum(torch.abs(saliency_map), dim=1, keepdim=True)
    # saliency_map = torch.abs(saliency_map)

    flat_s = saliency_map.view((saliency_map.size(0), -1))
    temp, _ = flat_s.min(1, keepdim=True)
    saliency_map = saliency_map - temp.unsqueeze(1).unsqueeze(1)
    flat_s = saliency_map.view((saliency_map.size(0), -1))
    temp, _ = flat_s.max(1, keepdim=True)
    saliency_map = saliency_map / (temp.unsqueeze(1).unsqueeze(1) + 1e-10)

    saliency_map = saliency_map.repeat(1, 3, 1, 1)

    return saliency_map


class Evaluator(object):
    def __init__(self, model, explainer, dataloader, log=print):
        self.model = model
        self.explainer = explainer
        self.dataloader = dataloader
        self.log = log

        self.model.eval()

    def DiffID(self, ratio_lst, centers=None):
        log = self.log
        n_examples = 0
        n_correct = 0

        loc_ind = 0
        ratio_len = len(ratio_lst)

        n_pert_correct_del_ins_lst = [[0 for _ in range(ratio_len)], [0 for _ in range(ratio_len)]]
        logit_change_del_ins_lst = [[torch.zeros(len(self.dataloader.dataset.samples)) for _ in range(ratio_len)],
                                    [torch.zeros(len(self.dataloader.dataset.samples)) for _ in range(ratio_len)]]

        start = time.time()
        for batch_num, (batch_image, label) in enumerate(self.dataloader):
            batch_image = batch_image.cuda()
            target = label.cuda()

            batch_size = batch_image.shape[0]

            output = self.model(batch_image).detach()
            _, predicted = torch.max(output.data, 1)
            n_correct += (predicted == target).sum().item()
            n_examples += batch_size

            # ------------------ attribution estimation -------------------------
            if centers is not None:
                output_array = output.cpu().numpy()
                clu_lst = [np.argmin(np.linalg.norm(centers - output_array[bth], axis=1)) for bth in range(output.shape[0])]
                saliency_map = self.explainer.shap_values(batch_image, sparse_labels=target, centers=clu_lst)
            else:
                saliency_map = self.explainer.shap_values(batch_image, sparse_labels=target)

            # -------------------------------- saliency map normalization -----------------------------------------
            saliency_map = normalize_saliency_map(saliency_map.detach())

            self.model.eval()
            perturb_img_batch = torch.zeros(*batch_image.shape).cuda()
            num_elements = batch_image[0].numel()

            for r_ind, ratio in enumerate(ratio_lst):
                for is_del in [False, True]:
                    del_ratio = ratio

                    for b_num in range(batch_size):
                        image = batch_image.detach()[b_num]
                        flat_s_map = saliency_map[b_num].view(-1)
                        flat_image = image.view(-1)
                        # order by attributions
                        sorted_ind = torch.argsort(flat_s_map, descending=is_del)
                        # preserve pixels
                        num_delete = int(num_elements * del_ratio)
                        preserve_ind = sorted_ind[num_delete:]
                        mask = torch.zeros_like(flat_image, dtype=torch.int)
                        mask[preserve_ind] = 1
                        mean_preserve = torch.mean(flat_image[preserve_ind])
                        perturb_img = flat_image * mask + mean_preserve * ~mask
                        perturb_img = perturb_img.view(image.size())

                        perturb_img_batch[b_num] = perturb_img

                    output_pert = self.model(perturb_img_batch).detach()

                    isd = int(is_del)
                    _, predicted_pert = torch.max(output_pert.data, 1)
                    n_pert_correct_del_ins_lst[isd][r_ind] += (predicted_pert == target).sum().item()
                    for bth in range(batch_size):
                        t = target[bth]
                        logit_change_del_ins_lst[isd][r_ind][loc_ind+bth:loc_ind+bth+1] = output_pert[bth, t] / output[bth, t]

            loc_ind += batch_size

        end = time.time()
        log('\ttime: \t{:.3f}'.format(end - start))
        insertion_logit = []
        insertion_acc = []

        deletion_logit = []
        deletion_acc = []

        DiffID_logit = []
        DiffID_acc = []

        for r_ind in range(ratio_len):
            mean_accu_del = n_pert_correct_del_ins_lst[1][r_ind] / n_examples
            var_del, mean_del = torch.var_mean(logit_change_del_ins_lst[1][r_ind], unbiased=False)
            mean_del = mean_del.item()
            deletion_logit.append(round(mean_del, 3))
            deletion_acc.append(round(mean_accu_del, 3))

            mean_accu_ins = n_pert_correct_del_ins_lst[0][r_ind] / n_examples
            var_ins, mean_ins = torch.var_mean(logit_change_del_ins_lst[0][r_ind], unbiased=False)
            mean_ins = mean_ins.item()
            insertion_logit.append(round(mean_ins, 3))
            insertion_acc.append(round(mean_accu_ins, 3))

            del_accu = mean_accu_ins - mean_accu_del
            del_logit = mean_ins - mean_del
            DiffID_logit.append(round(del_logit, 3))
            DiffID_acc.append(round(del_accu, 3))
        self.log('deletion logit scores')
        self.log(deletion_logit)
        self.log('deletion accu scores')
        self.log(deletion_acc)
        self.log('\n')
        self.log('insertion logit scores')
        self.log(insertion_logit)
        self.log('insertion accu scores')
        self.log(insertion_acc)
        self.log('\n')
        self.log('Diff logit scores')
        self.log(DiffID_logit)
        self.log('Diff accu scores')
        self.log(DiffID_acc)

    def visual_inspection(self, file_name, num_vis, method_name, centers=None):
        for batch_num, (image, label) in enumerate(self.dataloader):
            if (batch_num * image.shape[0]) >= num_vis:
                break
            image = image.cuda()
            target = label.cuda()

            if centers is not None:
                # ---------------- LPI ---------------------
                output = self.model(image).detach()
                output_array = output.cpu().numpy()
                clu_lst = [np.argmin(np.linalg.norm(centers - output_array[bth], axis=1)) for bth in
                           range(output.shape[0])]
                saliency_map = self.explainer.shap_values(image, sparse_labels=target, centers=clu_lst)
            else:
                saliency_map = self.explainer.shap_values(image, sparse_labels=target)

            image = undo_preprocess_input_function(image).detach().cpu().numpy()

            if not os.path.exists(file_name):
                os.mkdir(file_name)
            for bth in range(image.shape[0]):
                img = image[bth]
                f_name = file_name + 'img_' + str(int(batch_num * self.dataloader.batch_size) + bth)
                visualize(image=img, saliency_map=saliency_map[bth], filename=f_name, method_name=method_name)
