import torch
from torch.autograd import Function as TorchFunc
import numpy as np
from PIL import Image


class CustomReLU(TorchFunc):

    @staticmethod
    def forward(self, x):
        output = torch.addcmul(torch.zeros(x.size()), x, (x > 0).type_as(x))
        self.save_for_backward(x, output)
        return output

    @staticmethod
    def backward(self, dout):
        #grad=None
        x, output = self.saved_tensors
        grad = torch.zeros_like(x)
        grad = torch.where((x > 0) & (dout > 0), dout, grad)
        return grad


class GradCam:
    def guided_backprop(self, X_tensor, y_tensor, gc_model):
        """

        Input:
        - X_tensor: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the guided backprop.

        Returns:
        - guided backprop: A numpy of shape (N, H, W, 3) giving the guided backprop for 
        the input images.
        """
        for param in gc_model.parameters():
            param.requires_grad = True

        for idx, module in gc_model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                gc_model.features._modules[idx] = CustomReLU.apply
            elif module.__class__.__name__ == 'Fire':
                for idx_c, child in gc_model.features[int(idx)].named_children():
                    if child.__class__.__name__ == 'ReLU':
                        gc_model.features[int(idx)]._modules[idx_c] = CustomReLU.apply

        X_tensor.requires_grad = True
        out = gc_model(X_tensor)

        output = torch.zeros_like(out)
        output[range(out.size(0)), y_tensor] = 1
        X_tensor.grad = None  

        out.backward(gradient=output)

        gradients = X_tensor.grad.data
        backprop = gradients.permute(0, 2, 3, 1).cpu().numpy() 
        return backprop


    def grad_cam(self, X_tensor, y_tensor, gc_model):
        """
        Input:
        - X_tensor: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the gradcam.
        """
        conv_module = gc_model.features[12]
        self.gradient_value = None  # Stores gradient of the module you chose above during a backwards pass.
        self.activation_value = None  # Stores the activation of the module you chose above during a forwards pass.

        def gradient_hook(a, b, gradient):
            self.gradient_value = gradient[0]

        def activation_hook(a, b, activation):
            self.activation_value = activation

        conv_module.register_forward_hook(activation_hook)
        conv_module.register_backward_hook(gradient_hook)

        out = gc_model(X_tensor)
        gc_model.zero_grad()

        output = torch.zeros_like(out)
        for i in range(out.size(0)):
            output[i][y_tensor[i]] = 1

        out.backward(gradient=output)
        weights = torch.mean(self.gradient_value, dim=[2, 3], keepdim=True)
        
        cam = torch.sum(weights * self.activation_value, dim=1)
        cam = torch.clamp(cam, min=0)
        cam = cam.detach().cpu().numpy()

        # Rescale GradCam output to fit image.
        cam_scaled = []
        for i in range(cam.shape[0]):
            cam_scaled.append(np.array(Image.fromarray(cam[i]).resize(X_tensor[i, 0, :, :].shape, Image.BICUBIC)))
        cam = np.array(cam_scaled)
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam
