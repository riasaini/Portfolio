import torch
import torchvision
from visualizers import ClassVisualization
from data_utils import load_imagenet_val

# Download and load the pretrained SqueezeNet model.
model = torchvision.models.squeezenet1_1(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to use GPU
model.type(dtype)

cv = ClassVisualization()

# target_y = 76 # Tarantula
# target_y = 78 # Tick
# target_y = 187 # Yorkshire Terrier
# target_y = 683 # Oboe
# target_y = 366 # Gorilla
# target_y = 604 # Hourglass

_, _, class_names = load_imagenet_val(num=5)

targets = [366]

for target in targets:
    out = cv.create_class_visualization(target, class_names, model, dtype)