import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import GuidedGradCam, GuidedBackprop
from captum.attr import LayerActivation, LayerConductance, LayerGradCam

from data_utils import *
from image_utils import *
from captum_utils import *
import numpy as np

from visualizers import GradCam


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

X, y, class_names = load_imagenet_val(num=5)

gc_model = torchvision.models.squeezenet1_1(pretrained=True)
gc = GradCam()

X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).requires_grad_(True)
y_tensor = torch.LongTensor(y)

# Guided Back-Propagation
gbp_result = gc.guided_backprop(X_tensor,y_tensor, gc_model)

plt.figure(figsize=(24, 24))
for i in range(gbp_result.shape[0]):
    plt.subplot(1, 5, i + 1)
    img = gbp_result[i]
    img = rescale(img)
    plt.imshow(img)
    plt.title(class_names[y[i]])
    plt.axis('off')
plt.gcf().tight_layout()
plt.savefig('visualization/guided_backprop.png', bbox_inches = 'tight')



# GradCam
gc_model = torchvision.models.squeezenet1_1(pretrained=True)
for param in gc_model.parameters():
    param.requires_grad = True

X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).requires_grad_(True)
y_tensor = torch.LongTensor(y)
gradcam_result = gc.grad_cam(X_tensor, y_tensor, gc_model)

plt.figure(figsize=(24, 24))
for i in range(gradcam_result.shape[0]):
    gradcam_val = gradcam_result[i]
    img = X[i] + (matplotlib.cm.jet(gradcam_val)[:,:,:3]*255)
    img = img / np.max(img)
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title(class_names[y[i]])
    plt.axis('off')
plt.gcf().tight_layout()
plt.savefig('visualization/gradcam.png', bbox_inches = 'tight')


# Combine GradCam and Guided Backprop to get Guided GradCam.
X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).requires_grad_(True)
y_tensor = torch.LongTensor(y)
gradcam_result = gc.grad_cam(X_tensor, y_tensor, gc_model)
gbp_result = gc.guided_backprop(X_tensor, y_tensor, gc_model)

plt.figure(figsize=(24, 24))
for i in range(gradcam_result.shape[0]):
    gbp_val = gbp_result[i]
    gradcam_val = np.expand_dims(gradcam_result[i], axis=2)

    # Pointwise multiplication and normalization of the gradcam and guided backprop results (2 lines)
    img = gradcam_val * gbp_val

    img = rescale(img)
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title(class_names[y[i]])
    plt.axis('off')
plt.gcf().tight_layout()
plt.savefig('visualization/guided_gradcam.png', bbox_inches = 'tight')


# Captum
model = torchvision.models.squeezenet1_1(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

# Convert X and y from numpy arrays to Torch Tensors
X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).requires_grad_(True)
y_tensor = torch.LongTensor(y)

conv_module = model.features[12]


gbp1 = GuidedBackprop(model)
gbp_result = compute_attributions(gbp1, X_tensor, target=y_tensor)

grad_cam = GuidedGradCam(model, conv_module)
gradcam_result = compute_attributions(grad_cam, X_tensor, target=y_tensor)

visualize_attr_maps('visualization/guided_gradcam_and_gbp.png', X, y, class_names, [gbp_result, gradcam_result], ['Guided BackProp', 'Guided GradCam'])


layer = model.features[3]

# Example visualization for using layer visualizations 
# layer_act = LayerActivation(model, layer)
# layer_act_attr = compute_attributions(layer_act, X_tensor)
# layer_act_attr_sum = layer_act_attr.mean(axis=1, keepdim=True)


layer_gradCam = LayerGradCam(model, layer)
layerGC_attr = compute_attributions(layer_gradCam, X_tensor, target=y_tensor, relu_attributions=True)
#visualize_attr_maps('visualization/layer_gradcam.png', X, y, class_names, [layer_gradcam_result], ['Layer GradCAM Maps'])

layer_cond = LayerConductance(model, layer)
layer_cond_attr = compute_attributions(layer_cond, X_tensor, target=y_tensor)
layerCSum_attr = layer_cond_attr.mean(axis=1, keepdim=True)
#visualize_attr_maps('visualization/layer_conductance.png', X, y, class_names, [layer_conductance_result_sum], ['Layer conductance'])

#print("Layer Conductance Result Shape:", layer_conductance_result.shape)
visualize_attr_maps('visualization/layer_captum.png', X, y, class_names, [layerGC_attr, layerCSum_attr], ['Layer GradCam' , 'Layer Conductance'])


