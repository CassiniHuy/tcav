import os
import logging
import torch
import torchvision
import torch.nn.functional as F
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import gc
import os
import sys
from pathlib import Path
from .utils import get_grads_key

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

labels_path = os.path.join(Path(__file__).parent.absolute(), 'gistfile1.txt')

class ModelWrapper(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.model = None
        self.model_name = None

    @abstractmethod
    def get_cutted_model(self, bottleneck):
        pass
        
    def get_gradient(self, acts, y, bottleneck_name):
        inputs = torch.tensor(acts).to(device)
        inputs.requires_grad = True

        cutted_model = self.get_cutted_model(bottleneck_name).to(device)
        cutted_model.eval()
        outputs = cutted_model(inputs)
        outputs = outputs[:, y[0]]

        grad_outputs = torch.ones_like(outputs)
        grads = -torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs)[0]
        grads = grads.detach().cpu().numpy()

        cutted_model = None
        gc.collect()
        return grads

    def reshape_activations(self, layer_acts):
        return np.asarray(layer_acts).squeeze()

    @abstractmethod
    def label_to_id(self, label):
        pass
    
    @abstractmethod
    def id_to_label(self, id):
        pass

    def run_examples(self, examples, bottleneck_name):

        global bn_activation
        bn_activation = None

        def save_activation_hook(mod, inp, out):
            global bn_activation
            bn_activation = out

        handle = self.model._modules[bottleneck_name].register_forward_hook(save_activation_hook)

        self.model.to(device)
        inputs = torch.FloatTensor(examples).permute(0, 3, 1, 2).to(device)
        self.model.eval()
        self.model(inputs)
        acts = bn_activation.detach().cpu().numpy()
        handle.remove()

        return acts

class ImageModelWrapper(ModelWrapper):
    """Wrapper base class for image models."""

    def __init__(self, image_shape):
        super(ModelWrapper, self).__init__()
        # shape of the input image in this model
        self.image_shape = image_shape

    def get_image_shape(self):
        """returns the shape of an input image."""
        return self.image_shape


class PublicImageModelWrapper(ImageModelWrapper):
    """Simple wrapper of the public image models with session object."""

    def __init__(self, image_shape):
        super(PublicImageModelWrapper, self).__init__(image_shape=image_shape)
        try:
            self.labels = dict(eval(open(labels_path).read()))
        except:
            self.labels = open(labels_path).read().splitlines()

    def label_to_id(self, label):
        if isinstance(self.labels, dict):
            return list(self.labels.keys())[list(self.labels.values()).index(label)]
        else:
            return self.labels.index(label)

    def id_to_label(self, id):
        if isinstance(self.labels, dict):
            return self.labels[id]
        else:
            return self.labels[id]

class InceptionV3_cutted(torch.nn.Module):
    def __init__(self, inception_v3, bottleneck):
        super(InceptionV3_cutted, self).__init__()
        names = list(inception_v3._modules.keys())
        layers = list(inception_v3._modules.values())

        self.layers = torch.nn.ModuleList()
        self.layers_names = []

        bottleneck_met = False
        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue  # because we already have the output of the bottleneck layer
            if not bottleneck_met:
                continue
            if name == 'AuxLogits':
                continue

            self.layers.append(layer)
            self.layers_names.append(name)

    def forward(self, x):
        y = x
        for i, layer in enumerate(self.layers):
            # pre-forward process
            if self.layers_names[i] == 'fc':
                y = y.view(y.size(0), -1)

            y = layer(y)
        return y
    

class InceptionV3Wrapper(PublicImageModelWrapper):

    def __init__(self):
        image_shape = [299, 299, 3]
        super(InceptionV3Wrapper, self).__init__(image_shape=image_shape)
        self.model = torchvision.models.inception_v3(pretrained=True, transform_input=True)
        self.model_name = 'InceptionV3_public'

    def forward(self, x):
        return self.model.forward(x)

    def get_cutted_model(self, bottleneck):
        return InceptionV3_cutted(self.model, bottleneck)

class GoogleNet_cutted(torch.nn.Module):
    def __init__(self, googlenet, bottleneck):
        super(GoogleNet_cutted, self).__init__()
        names = list(googlenet._modules.keys())
        layers = list(googlenet._modules.values())

        self.layers = torch.nn.ModuleList()
        self.layers_names = []

        bottleneck_met = False
        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue  # because we already have the output of the bottleneck layer
            if not bottleneck_met:
                continue
            if name == 'aux1':
                continue
            if name == 'aux2':
                continue

            self.layers.append(layer)
            self.layers_names.append(name)

    def forward(self, x):
        y = x
        # print(self.layers)
        for i, layer in enumerate(self.layers):
            # pre-forward process
            if self.layers_names[i] == 'dropout':
                y = y.view(y.size(0), -1)

            y = layer(y)
        return y
    

class GoogleNetWrapper(PublicImageModelWrapper):

    def __init__(self):
        image_shape = [299, 299, 3]
        super(GoogleNetWrapper, self).__init__(image_shape=image_shape)
        self.model = torchvision.models.googlenet(pretrained=True, transform_input=True)
        self.model_name = 'GoogleNet_public'

    def forward(self, x):
        return self.model.forward(x)

    def get_cutted_model(self, bottleneck):
        return GoogleNet_cutted(self.model, bottleneck)

def get_model_wrapper(name):
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name)

def get_or_load_gradients(model, acts, grads_dir, target_class, bottleneck):
    if grads_dir is None:
        grads = model.get_gradient(acts, [model.label_to_id(target_class)], bottleneck)
    else:
        path = os.path.join(grads_dir, get_grads_key(target_class, model.model_name, bottleneck))
        if os.path.exists(path):
            with open(path, 'rb') as f:
                grads = np.load(f, allow_pickle=False)
                logging.info(path + ' exists and loaded, shape={}.'.format(str(grads.shape)))
        else:
            grads = model.get_gradient(acts, [model.label_to_id(target_class)], bottleneck)
            with open(path, 'wb') as f:
                np.save(f, grads, allow_pickle=False)
                logging.info(path + ' created, shape={}.'.format(str(grads.shape)))
    return grads.reshape([grads.shape[0], -1])

