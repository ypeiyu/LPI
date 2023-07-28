import torch
import torch.nn.functional as F
import torch.utils.data

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
            self.ref_samplers.append(
                torch.utils.data.DataLoader(
                    dataset=baseline_set,
                    batch_size=self.bg_size,
                    shuffle=False,
                    pin_memory=False,
                    drop_last=False)
            )

        densities = self.density.reshape([1, -1, 1, 1, 1])
        self.density_tensor = densities.cuda()

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
                t_tensor = torch.cat([torch.Tensor([1.0]) for _ in range(batch_size*k_*self.bg_size)]).to(DEFAULT_DEVICE)
            else:
                t_tensor = torch.cat([torch.linspace(0, 1, k_) for _ in range(batch_size*self.bg_size)]).to(DEFAULT_DEVICE)

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
        input_expand_mult = input_tensor.unsqueeze(1)
        sd = input_expand_mult - reference_tensor
        return sd

    def _get_grads(self, samples_input, sparse_labels=None):
        shape = list(samples_input.shape)
        shape[1] = self.bg_size
        grad_tensor = torch.zeros(shape).float().to(DEFAULT_DEVICE)

        for bg_id in range(self.bg_size):
            for k_id in range(self.k):
                particular_slice = samples_input[:, bg_id*self.k+k_id]
                particular_slice.requires_grad = True

                output = self.model(particular_slice)

                # should check that users pass in sparse labels
                # Only look at the user-specified label

                if sparse_labels is None:
                    sparse_labels = output.max(1, keepdim=True)[1].squeeze(1)

                batch_output = -1 * F.nll_loss(output, sparse_labels.flatten(), reduction='sum')

                self.model.zero_grad()
                batch_output.backward()
                gradients = particular_slice.grad.clone()
                particular_slice.grad.zero_()
                gradients.detach()

                grad_tensor[:, bg_id, :] += (gradients / self.k) * self.density_tensor[:, bg_id, :]

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
        b_num = input_tensor.shape[0]

        ref = []
        if self.density.shape[0] > 1:
            for c_ind in centers:
                ref.append(self._get_ref_batch(c_ind))

            density_lst = []
            for b_ind in range(b_num):
                center = centers[b_ind]
                density_lst.append(self.density[center])
            densities = torch.cat(density_lst)
            densities = densities.reshape([b_num, -1, 1, 1, 1])
            density_tensor = densities.cuda()
            self.density_tensor = density_tensor
        else:
            ref = [self._get_ref_batch(0) for _ in range(b_num)]

        reference_tensor = torch.stack(ref, dim=0).cuda()
        multi_ref_tensor = reference_tensor.repeat(1, self.k, 1, 1, 1)

        samples_input = self._get_samples_input(input_tensor, multi_ref_tensor)
        samples_delta = self._get_samples_delta(input_tensor, reference_tensor)
        grad_tensor = self._get_grads(samples_input, sparse_labels)

        mult_grads = samples_delta * grad_tensor if self.scale_by_inputs else grad_tensor
        attribution = mult_grads.mean(1)

        return attribution
