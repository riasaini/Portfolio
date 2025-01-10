import torch
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import IntegratedGradients, Saliency

from visualizers import SaliencyMap
from data_utils import *
from image_utils import *
from captum_utils import *


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Download and load the pretrained SqueezeNet model.
model = torchvision.models.squeezenet1_1(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

X, y, class_names = load_imagenet_val(num=5)
# manually compute saliency maps
sm = SaliencyMap()
sm.show_saliency_maps(X, y, class_names, model)



# Convert X and y from numpy arrays to Torch Tensors
X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).requires_grad_(True)
y_tensor = torch.LongTensor(y)

# Example with captum

# int_grads = IntegratedGradients(model)
# attr_ig = compute_attributions(int_grads, X_tensor, target=y_tensor, n_steps=10)
# visualize_attr_maps('visualization/int_grads_captum.png', X, y, class_names, [attr_ig], ['Integrated Gradients'])

saliency = Saliency(model)
attr_s = compute_attributions(saliency, X_tensor, target=y_tensor)
visualize_attr_maps('visualization/saliency_captum.png', X, y, class_names, [attr_s], ['Saliency Map'])
