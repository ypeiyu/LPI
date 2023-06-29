import cv2
import numpy as np
import os
from utils import undo_preprocess


def render_centroid(representation, inds):
    rep_mean = np.mean(representation, axis=0)
    dist = np.linalg.norm(representation - rep_mean, axis=1)
    ind = np.argmin(dist)
    return inds[ind]


def render_icons(icon_ids, f_name, imagenet_train_loader=None, icon_size=40, is_icon=False):
    save_pth = f_name
    # if not os.path.exists(save_pth):
    #     os.mkdir(save_pth)
    #     save_pth = os.path.join(save_pth, 'img')
    #     os.mkdir(save_pth)

    icon_batch = []
    icon_id_lst = icon_ids
    for batch_id, (img, lbl) in enumerate(imagenet_train_loader):
        batch_sta = (batch_id-1)*500
        for ind in range(img.shape[0]):
            img_id = batch_sta + ind
            if img_id in icon_id_lst:
                icon = img[ind].unsqueeze(0)
                icon = undo_preprocess(icon, d_name='imagenet')[0]
                icon = icon.data.numpy() * 255
                icon = icon.astype(np.uint8)
                icon = np.transpose(icon, [1, 2, 0])
                icon = cv2.cvyoutColor(icon, cv2.COLOR_BGR2RGB)
                print('batch size {}'.format(img_id))
                cv2.imwrite(os.path.join(save_pth, 'img_'+str(img_id).zfill(7)+'.png'), icon)
                if is_icon:
                    icon = cv2.resize(icon, (icon_size, icon_size), interpolation=cv2.INTER_AREA)
                    icon_batch.append(icon)

    icon_id_order = np.argsort(icon_ids)
    if is_icon:
        icons_lst = []
        for i in range(icon_id_order.size):
            icons_lst.append(icon_batch[icon_id_order[i]])
        return icons_lst
    print('render icons completed')


def render_sub_icons(icon_ids, f_name, imagenet_train_loader=None, v2c_dict=None, test_batch_size=None):
    save_pth = f_name
    if not os.path.exists(save_pth):
        os.mkdir(save_pth)
        save_pth = os.path.join(save_pth, 'img')
        os.mkdir(save_pth)

    icon_id_lst = icon_ids
    for batch_id, (img, lbl) in enumerate(imagenet_train_loader):
        batch_sta = (batch_id-1)*test_batch_size
        for ind in range(img.shape[0]):
            img_id = batch_sta + ind
            if img_id in icon_id_lst:
                c_id = v2c_dict[img_id]
                icon = img[ind].unsqueeze(0)
                icon = undo_preprocess(icon, d_name='imagenet')[0]
                icon = icon.data.numpy() * 255
                icon = icon.astype(np.uint8)
                icon = np.transpose(icon, [1, 2, 0])
                icon = cv2.cvtColor(icon, cv2.COLOR_BGR2RGB)
                print('batch size {}'.format(img_id))
                if not os.path.exists(os.path.join(save_pth, 'kmeans_c'+str(c_id))):
                    os.mkdir(os.path.join(save_pth, 'kmeans_c'+str(c_id)))
                    os.mkdir(os.path.join(save_pth, 'kmeans_c' + str(c_id), 'img'))

                cv2.imwrite(os.path.join(save_pth, 'kmeans_c' + str(c_id), 'img',
                                         'img_'+str(img_id).zfill(7)+'.png'), icon)

    print('render icons completed')
