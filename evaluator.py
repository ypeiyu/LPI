import random
import time
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import undo_preprocess_input_function
from exp_fig import visualize


def normalize_saliency_map(saliency_map):
    saliency_map = torch.sum(torch.abs(saliency_map), dim=1, keepdim=True)

    flat_s = saliency_map.view((saliency_map.size(0), -1))
    temp, _ = flat_s.min(1, keepdim=True)
    saliency_map = saliency_map - temp.unsqueeze(1).unsqueeze(1)
    flat_s = saliency_map.view((saliency_map.size(0), -1))
    temp, _ = flat_s.max(1, keepdim=True)
    saliency_map = saliency_map / (temp.unsqueeze(1).unsqueeze(1) + 1e-10)
    return saliency_map


class Evaluator(object):
    def __init__(self, model, explainer, dataloader, log=print):
        self.model = model
        self.explainer = explainer
        # self.correlated = self.explainer.correlated

        self.dataloader = dataloader
        self.log = log
        self.n_examples = 0
        self.n_correct = 0
        self.n_pert_correct = 0
        self.pos_num = 0
        self.NL_difference = []
        self.model.eval()

    def DiffID(self, q_ratio_lst, centers=None):
        log = self.log
        self.n_examples = 0
        self.n_correct = 0
        self.pos_num = 0
        n_pert_correct_top_lst = [0 for _ in range(len(q_ratio_lst))]
        n_pert_correct_bot_lst = [0 for _ in range(len(q_ratio_lst))]
        start_loc_top = [0 for _ in range(len(q_ratio_lst))]
        start_loc_bot = [0 for _ in range(len(q_ratio_lst))]
        logit_change_top_lst = [torch.zeros(len(self.dataloader.dataset.samples)) for _ in range(len(q_ratio_lst))]
        logit_change_bot_lst = [torch.zeros(len(self.dataloader.dataset.samples)) for _ in range(len(q_ratio_lst))]

        start = time.time()
        for batch_num, (image, label) in enumerate(self.dataloader):
            image = image.cuda()
            target = label.cuda()

            batch_size = image.shape[0]

            output = self.model(image).detach()
            _, predicted = torch.max(output.data, 1)
            self.n_correct += (predicted == target).sum().item()
            self.n_examples += batch_size

            if centers is not None:
                # ------------------ LPI -------------------------
                output_array = output.cpu().numpy()
                clu_lst = [np.argmin(np.linalg.norm(centers - output_array[bth], axis=1)) for bth in range(output.shape[0])]
                saliency_map = self.explainer.shap_values(image, sparse_labels=target, centers=clu_lst)
            else:
                saliency_map = self.explainer.shap_values(image, sparse_labels=target)

            # -------------------------------- saliency map normalization -----------------------------------------
            saliency_map = normalize_saliency_map(saliency_map.detach())

            self.model.eval()
            zero_tensor = torch.zeros(*image[0].shape).cuda()
            perturb_img_batch = torch.zeros(*image.shape).cuda()
            for q_ind, q_ratio in enumerate(q_ratio_lst):
                # ========================================================================
                for perturb_top in [False, True]:

                    q_r = 1-q_ratio if perturb_top else q_ratio  # is_top 0.9 else 0.1
                    threshold = torch.quantile(saliency_map.reshape(saliency_map.shape[0], -1), q=q_r, dim=1, interpolation='midpoint')  # < top90

                    for b_num in range(batch_size):
                        sat = image.detach()[b_num] if perturb_top else zero_tensor
                        dis_sat = zero_tensor if perturb_top else image.detach()[b_num]

                        perturb = torch.where(saliency_map[b_num] < threshold[b_num], sat, dis_sat).unsqueeze(0)
                        mean_insertion = torch.sum(perturb) / (torch.count_nonzero(perturb) + 1e-10)
                        dis_sat_t = dis_sat+mean_insertion if perturb_top else dis_sat
                        sat_t = sat if perturb_top else sat+mean_insertion
                        perturb_img = torch.where(saliency_map[b_num] < threshold[b_num], sat_t, dis_sat_t).unsqueeze(0)

                        # flag = 'insertion' if perturb_top else 'deletion'
                        # single_img_inspection(perturb_img, file_name='exp_fig/img_inspection/img' + str(
                        #     self.n_examples + b_num) + '_' + flag + str(q_r) + '.jpg')

                        perturb_img_batch[b_num] = perturb_img

                    output_pert = self.model(perturb_img_batch).detach()

                    _, predicted_pert = torch.max(output_pert.data, 1)
                    if perturb_top:
                        n_pert_correct_top_lst[q_ind] += (predicted_pert == target).sum().item()
                        for bth in range(batch_size):
                            t = target[bth]
                            logit_change_top_lst[q_ind][start_loc_top[q_ind]:(start_loc_top[q_ind]+1)] = output_pert[bth, t]/output[bth, t]
                            start_loc_top[q_ind] += 1
                    else:
                        n_pert_correct_bot_lst[q_ind] += (predicted_pert == target).sum().item()
                        for bth in range(batch_size):
                            t = target[bth]
                            logit_change_bot_lst[q_ind][start_loc_bot[q_ind]:(start_loc_bot[q_ind]+1)] = output_pert[bth, t]/output[bth, t]
                            start_loc_bot[q_ind] += 1
        end = time.time()
        log('\ttime: \t{:.3f}'.format(end - start))
        insertion_logit = []
        insertion_acc = []

        deletion_logit = []
        deletion_acc = []

        DiffID_logit = []
        DiffID_acc = []

        for q_ind, q_ratio in enumerate(q_ratio_lst):
            # ========================================================================
            mean_accu_top = n_pert_correct_top_lst[q_ind]/self.n_examples
            var_top, mean_top = torch.var_mean(logit_change_top_lst[q_ind], unbiased=False)
            mean_top = mean_top.item()
            deletion_logit.append(round(mean_top, 3))
            deletion_acc.append(round(mean_accu_top, 3))

            mean_accu_bot = n_pert_correct_bot_lst[q_ind]/self.n_examples
            var_bot, mean_bot = torch.var_mean(logit_change_bot_lst[q_ind], unbiased=False)
            mean_bot = mean_bot.item()
            insertion_logit.append(round(mean_bot, 3))
            insertion_acc.append(round(mean_accu_bot, 3))

            # log('\tDiff criterion')
            del_accu = mean_accu_bot - mean_accu_top
            del_logit = mean_bot - mean_top
            DiffID_logit.append(round(del_logit, 3))
            DiffID_acc.append(round(del_accu, 3))
        self.log('deletion 10-90 logit scores')
        self.log(deletion_logit)
        # print(np.mean(np.array(deletion_logit)))

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
