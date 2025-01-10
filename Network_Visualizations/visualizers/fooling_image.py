import torch
from torch.autograd import Variable

class FoolingImage:
    def make_fooling_image(self, X, target_y, model):
        """

        Inputs:
        - X: Input image; Tensor of shape (1, 3, 224, 224)
        - target_y: An integer in the range [0, 1000)
        - model: A pretrained CNN

        Returns:
        - X_fooling: An image that is close to X, but that is classifed as target_y
        by the model.
        """

        model.eval()

        # Initialize fooling image to the input image, and wrap it in a Variable.
        X_fooling = X.clone()
        X_fooling_var = Variable(X_fooling, requires_grad=True)


        learning_rate = 10
        max_iter = 100  # maximum number of iterations

        for it in range(max_iter):

            out = model(X_fooling_var)
            score = out[0, target_y]

            _, pred = out.max(1)
            if pred.item() == target_y:
                break

            score.backward()
            g = X_fooling_var.grad.data
            if g.norm() != 0:
                X_fooling_var.data += learning_rate * g / g.norm()

            X_fooling_var.grad.data.zero_()
            

        X_fooling = X_fooling_var.data

        return X_fooling
