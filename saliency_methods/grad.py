import torch
import torch.nn.functional as F

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Gradients(object):
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def shap_values(self, input_tensor, sparse_labels=None):
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
        input_tensor.requires_grad = True

        output = self.model(input_tensor)

        if sparse_labels is None:
            sparse_labels = output.max(1, keepdim=False)[1]

        batch_output = -1 * F.nll_loss(output, sparse_labels.flatten(), reduction='sum')

        self.model.zero_grad()
        batch_output.backward()
        gradients = input_tensor.grad.clone()
        input_tensor.grad.zero_()
        gradients.detach()

        return gradients
