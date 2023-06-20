import numpy as np
import torch


def render_representation(model, data_loader, f_name, test_batch_size):
    represent_lst = []
    # count = 4
    for batch_num, (image, _) in enumerate(data_loader):
        print(batch_num)

        image = image.cuda()
        output = model(image).data  # .cpu().detach().numpy()
        represent_lst.append(output)
        if ((batch_num + 1) % 500 == 0) or (image.shape[0] != test_batch_size):
            rep_array = np.array(torch.cat(represent_lst).cpu())
            # rep_array = np.concatenate(represent_lst)  # cat(represent_lst, axis=1)
            np.save(f_name + str(batch_num).zfill(4) + '.npy', rep_array)
            print(rep_array.shape)
            represent_lst = []
    if len(represent_lst) != 0:
        rep_array = np.array(torch.cat(represent_lst).cpu())
        # rep_array = np.concatenate(represent_lst)  # cat(represent_lst, axis=1)
        np.save(f_name + str(batch_num).zfill(4) + '.npy', rep_array)
        print(rep_array.shape)


def render_density(reps, centers, labels, f_name=None):
    # ------------------------------------ obtain density and centers ---------------------------------------
    sum_num = reps.shape[0]
    density = []
    indicates = []
    for c_id in range(centers.shape[0]):
        center_ = centers[c_id]
        label_loc = np.where(labels == c_id)[0]
        example_num = label_loc.shape[0]
        rep_ = reps[label_loc]

        dist = np.linalg.norm(rep_ - center_, axis=1)
        ind = np.argmin(dist)
        indicates.append(label_loc[ind])

        density.append(example_num/sum_num)

    # ------------------------------------ order and save density ---------------------------------------
    ind_order = np.argsort(np.array(indicates))
    density = np.asarray(density)
    density_lst = []
    for i in range(len(indicates)):
        density_lst.append(density[ind_order[i]])
    density_arr = np.asarray(density_lst)
    if f_name is not None:
        np.save(f_name+'.npy', density_arr)
    return indicates
