import os, sys, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from warnings import warn
from tqdm.auto import tqdm as tqdm
from collections import defaultdict, OrderedDict

from PIL import Image
import torch.nn as nn
import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from model_options import *

def get_prepped_model(model_string):
    model_options = get_model_options()
    model_call = model_options[model_string]['call']
    model = eval(model_call)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        
    return(model)

def check_model(model_string, model = None):
    if not isinstance(model_string, str):
        model = model_string
    model_options = get_model_options()
    if model_string not in model_options and model == None:
        raise ValueError('model_string not available in prepped models. Please supply model object.')
            
def prep_model_for_extraction(model):
    if model.training:
        model = model.eval()
    if not next(model.parameters()).is_cuda:
        if torch.cuda.is_available():
            model = model.cuda()

    return(model)

def convert_relu(parent):
    for child_name, child in parent.named_children():
        if isinstance(child, nn.ReLU):
            setattr(parent, child_name, nn.ReLU(inplace=False))
        elif len(list(child.children())) > 0:
            convert_relu(child)

def get_image_transforms():
    imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                      'std': [0.229, 0.224, 0.225]}
    
    options = {'imagenet': [transforms.Resize((224,224)), 
                            transforms.ToTensor(),
                            transforms.Normalize(**imagenet_stats)]}
    
    return {key:transforms.Compose(value) 
            for (key, value) in options.items()} 

# Method 1: Flatten model; extract features by layer

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.out = output.clone().detach().requires_grad_(True).cuda()
    def close(self):
        self.hook.remove()
    def extract(self):
        return self.out
    
def get_layer_names(layers):
    layer_names = []
    for layer in layers:
        layer_name = str(layer).split('(')[0]
        layer_names.append(layer_name + '-' + str(sum(layer_name in string for string in layer_names) + 1))
    return layer_names

def get_features_by_layer(model, target_layer, img_tensor):
    model = prep_model_for_extraction()
    features = SaveFeatures(target_layer)
    model(img_tensor)
    features.close()
    return features.extract()

# Method 2: Hook all layers simultaneously; remove duplicates


def get_weights_dtype(model):
    module = list(model.children())[0]
    if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
        return module.weight.dtype
    if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
        return get_weights_dtype(module)

def get_module_name(module, module_list):
    class_name = str(module.__class__).split(".")[-1].split("'")[0]
    class_count = str(sum(class_name in module for module in module_list) + 1)
    
    return '-'.join([class_name, class_count])

def get_inputs_sample(inputs):
    if isinstance(inputs, torch.Tensor):
        input_sample = inputs[:3]
        
    if isinstance(inputs, DataLoader):
        input_sample = next(iter(inputs))[:3]
        
    return input_sample
    
def get_feature_maps_(model, inputs):
    model = prep_model_for_extraction(model)
    
    def register_hook(module):
        def hook(module, input, output):
            module_name = get_module_name(module, feature_maps)
            feature_maps[module_name] = output
                    
        if not isinstance(module, nn.Sequential): 
            if not isinstance(module, nn.ModuleList):
                hooks.append(module.register_forward_hook(hook))
                
    if next(model.parameters()).is_cuda:
        inputs = inputs.cuda()
            
    feature_maps = OrderedDict()
    hooks = []
    
    model.apply(register_hook)
    with torch.no_grad():
        model(inputs)

    for hook in hooks:
        hook.remove()
        
    return(feature_maps)

def remove_duplicate_feature_maps(feature_maps, method = 'hashkey', return_matches = False, use_tqdm = False):
    matches, layer_names = [], list(feature_maps.keys())
        
    if method == 'iterative':
        
        target_iterator = tqdm(range(len(layer_names)), leave = False) if use_tqdm else range(len(layer_names))
        
        for i in target_iterator:
            for j in range(i+1,len(layer_names)):
                layer1 = feature_maps[layer_names[i]].flatten()
                layer2 = feature_maps[layer_names[j]].flatten()
                if layer1.shape == layer2.shape and torch.all(torch.eq(layer1,layer2)):
                    if layer_names[j] not in matches:
                        matches.append(layer_names[j])

        deduplicated_feature_maps = {key:value for (key,value) in feature_maps.items()
                                         if key not in matches}
        
    if method == 'hashkey':
        
        target_iterator = tqdm(layer_names, leave = False) if use_tqdm else layer_names
        layer_lengths = [len(tensor.flatten()) for tensor in feature_maps.values()]
        random_tensor = torch.rand(np.array(layer_lengths).max())
        
        tensor_dict = defaultdict(lambda:[])
        for layer_name in target_iterator:
            target_tensor = feature_maps[layer_name].flatten()
            tensor_dot = torch.dot(target_tensor, random_tensor[:len(target_tensor)])
            tensor_hash = np.array(tensor_dot).tobytes()
            tensor_dict[tensor_hash].append(layer_name)
            
        matches = [match for match in list(tensor_dict.values()) if len(match) > 1]
        layers_to_keep = [tensor_dict[tensor_hash][0] for tensor_hash in tensor_dict]
        
        deduplicated_feature_maps = {key:value for (key,value) in feature_maps.items()
                                         if key in layers_to_keep}
    
    if return_matches:
        return(deduplicated_feature_maps, matches)
    
    if not return_matches:
        return(deduplicated_feature_maps)
    
def check_for_input_axis(feature_map, input_size):
    axis_match = [dim for dim in feature_map.shape if dim == input_size]
    return True if len(axis_match) == 1 else False

def reset_input_axis(feature_map, input_size):
    input_axis = feature_map.shape.index(input_size)
    return torch.swapaxes(feature_map, 0, input_axis)

def get_feature_maps(model, inputs, layers_to_retain = None, remove_duplicates = True):
    model = prep_model_for_extraction(model)
    enforce_input_shape = True
    
    def register_hook(module):
        def hook(module, input, output):
            def process_output(output, module_name):
                if layers_to_retain is None or module_name in layers_to_retain:
                    if isinstance(output, torch.Tensor):
                        outputs = output.cpu().detach().type(torch.FloatTensor)
                        if enforce_input_shape:
                            if outputs.shape[0] == inputs.shape[0]:
                                feature_maps[module_name] = outputs
                            if outputs.shape[0] != inputs.shape[0]:
                                if check_for_input_axis(outputs, inputs.shape[0]):
                                    outputs = reset_input_axis(outputs, inputs.shape[0])
                                    feature_maps[module_name] = outputs
                                if not check_for_input_axis(outputs, inputs.shape[0]):
                                    feature_maps[module_name] = None
                                    warn('Ambiguous input axis in {}. Skipping...'.format(module_name))
                        if not enforce_input_shape:
                            feature_maps[module_name] = outputs
                if layers_to_retain is not None and module_name not in layers_to_retain:
                    feature_maps[module_name] = None
                            
            module_name = get_module_name(module, feature_maps)
            
            if not any([isinstance(output, type_) for type_ in (tuple,list)]):
                process_output(output, module_name)
            
            if any([isinstance(output, type_) for type_ in (tuple,list)]):
                for output_i, output_ in enumerate(output):
                    module_name_ = '-'.join([module_name, str(output_i+1)])
                    process_output(output_, module_name_)
                    
        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
            hooks.append(module.register_forward_hook(hook))
            
    if next(model.parameters()).is_cuda:
        inputs = inputs.cuda()
    
    feature_maps = OrderedDict()
    hooks = []
    
    model.apply(convert_relu)
    model.apply(register_hook)
    with torch.no_grad():
        model(inputs)

    for hook in hooks:
        hook.remove()
        
    feature_maps = {map:features for (map,features) in feature_maps.items()
                        if features is not None}
        
    if remove_duplicates == True:
        feature_maps = remove_duplicate_feature_maps(feature_maps)
        
    return(feature_maps)

def get_empty_feature_maps(model, inputs = None, input_size=(3,224,224), dataset_size=3,
        layers_to_retain = None, remove_duplicates = True, names_only=False):

    
    if inputs is not None:
        inputs = get_inputs_sample(inputs)
        
    if inputs is None:
        inputs = torch.rand(3, *input_size)
        
    if next(model.parameters()).is_cuda:
        inputs = inputs.cuda()
        
    empty_feature_maps = get_feature_maps(model, inputs, layers_to_retain, remove_duplicates)
    
    for map_key in empty_feature_maps:
        empty_feature_maps[map_key] = torch.empty(dataset_size, *empty_feature_maps[map_key].shape[1:])
        
    if names_only == True:
        return list(empty_feature_maps.keys())
    
    if names_only == False:
        return empty_feature_maps  
    
    
def get_feature_map_names(model, inputs = None, remove_duplicates = True):
    feature_map_names = get_empty_feature_maps(model, inputs, names_only = True,
                                                remove_duplicates = remove_duplicates)
    
    return(feature_map_names)

def get_feature_map_count(model, inputs = None, remove_duplicates = True):
    feature_map_names = get_feature_map_names(model, inputs, remove_duplicates)
    
    return(len(feature_map_names))

def get_all_feature_maps(model, inputs, layers_to_retain=None, remove_duplicates=True, 
                         include_input_space = False, flatten=True, numpy=True, use_tqdm = True):
    
    check_model(model)
    if isinstance(model, str):
        model = get_prepped_model(model)
    
    if isinstance(inputs, DataLoader):
        input_size, dataset_size, start_index = inputs.dataset[0].shape, len(inputs.dataset), 0
        feature_maps = get_empty_feature_maps(model, next(iter(inputs))[:3], input_size, 
                                              dataset_size, layers_to_retain, remove_duplicates)
        
        if include_input_space:
            input_map = {'Input': torch.empty(dataset_size, *input_size)}
            feature_maps = {**input_map, **feature_maps}
        
        
        for imgs in tqdm(inputs, desc = 'Feature Extraction (Batch)') if use_tqdm else inputs:
            imgs = imgs.cuda() if next(model.parameters()).is_cuda else imgs
            batch_feature_maps = get_feature_maps(model, imgs, layers_to_retain, remove_duplicates = False)
            
            if include_input_space:
                batch_feature_maps['Input'] = imgs.cpu()
            
            for map_i, map_key in enumerate(feature_maps):
                feature_maps[map_key][start_index:start_index+imgs.shape[0],...] = batch_feature_maps[map_key]
            start_index += imgs.shape[0]
                    
    if not isinstance(inputs, DataLoader):
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cuda() if next(model.parameters()).is_cuda else inputs
        feature_maps = get_feature_maps(model, inputs, layers_to_retain, remove_duplicates)
        
        if include_input_space:
            feature_maps = {**{'Input': inputs.cpu()}, **feature_maps}
    
    if remove_duplicates == True:
        feature_maps = remove_duplicate_feature_maps(feature_maps)
     
    if flatten == True:
        for map_key in feature_maps:
            incoming_map = feature_maps[map_key]
            feature_maps[map_key] = incoming_map.reshape(incoming_map.shape[0], -1)
            
    if numpy == True:
        for map_key in feature_maps:
            feature_maps[map_key] = feature_maps[map_key].numpy()
            
    return feature_maps

def get_feature_map_metadata(model, input_size=(3,224,224), remove_duplicates = False):
    model = prep_model_for_extraction(model)
    enforce_input_shape = True

    inputs = torch.rand(3, *input_size)
    if next(model.parameters()).is_cuda:
        inputs = inputs.cuda()
    
    def register_hook(module):
        def hook(module, input, output):
            def process_output(output, module_name):
                if isinstance(output, torch.Tensor):
                    outputs = output.cpu().detach().type(torch.FloatTensor)
                    if not enforce_input_shape:
                        map_data[module_name] = outputs
                    if enforce_input_shape:
                        if outputs.shape[0] == inputs.shape[0]:
                            map_data[module_name] = outputs
                        if outputs.shape[0] != inputs.shape[0]:
                            if check_for_input_axis(outputs, inputs.shape[0]):
                                outputs = reset_input_axis(outputs, inputs.shape[0])
                                map_data[module_name] = outputs
                            if not check_for_input_axis(outputs, inputs.shape[0]):
                                feature_maps[module_name] = None
                                warn('Ambiguous input axis in {}. Skipping...'.format(module_name))

                if module_name in map_data:
                    module_name = get_module_name(module, metadata)
                    feature_map = output.cpu().detach()
                    map_data[module_name] = feature_map
                    metadata[module_name] = {}

                    metadata[module_name]['feature_map_shape'] = feature_map.numpy().shape[1:]
                    metadata[module_name]['feature_count'] = feature_map.numpy().reshape(1, -1).shape[1]

                    params = 0
                    if hasattr(module, "weight") and hasattr(module.weight, "size"):
                        params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    if hasattr(module, "bias") and hasattr(module.bias, "size"):
                        params += torch.prod(torch.LongTensor(list(module.bias.size())))
                    if isinstance(params, torch.Tensor):
                        params = params.item()
                    metadata[module_name]['parameter_count'] = params
          
            module_name = get_module_name(module, metadata)
          
            if not any([isinstance(output, type_) for type_ in (tuple,list)]):
                process_output(output, module_name)
        
            if any([isinstance(output, type_) for type_ in (tuple,list)]):
                for output_i, output_ in enumerate(output):
                    module_name_ = '-'.join([module_name, str(output_i+1)])
                    process_output(output_, module_name_)
              
        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
            hooks.append(module.register_forward_hook(hook))
    
    map_data = OrderedDict()
    metadata = OrderedDict()
    hooks = []
    
    model.apply(register_hook)
    with torch.no_grad():
        model(inputs)

    for hook in hooks:
        hook.remove()
        
    if remove_duplicates:
        map_data = remove_duplicate_feature_maps(map_data)
        metadata = {k:v for (k,v) in metadata.items() if k in map_data}
        
    return(metadata)
        
# Helpers: Dataloaders and functions for facilitating feature extraction

class StimulusSet(Dataset):
    def __init__(self, image_paths, image_transforms=None):
        self.images = image_paths
        self.transforms = image_transforms

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img
    
    def __len__(self):
        return self.images.shape[0]

class CSV2StimulusSet(Dataset):
    def __init__(self, csv, root_dir, image_transforms=None):
        
        self.root = os.path.expanduser(root_dir)
        self.transforms = image_transforms
        
        if isinstance(csv, pd.DataFrame):
            self.df = csv
        if isinstance(csv, str):
            self.df = pd.read_csv(csv)
        
        self.images = self.df.image_name

    def __getitem__(self, index):
        filename = os.path.join(self.root, self.images.iloc[index])
        img = Image.open(filename).convert('RGB')
        
        if self.transforms:
            img = self.transforms(img)
            
        return img
    
    def __len__(self):
        return len(self.images)
        
class Array2StimulusSet(Dataset):
    def __init__(self, img_array, image_transforms=None):
        self.transforms = image_transforms
        if isinstance(img_array, np.ndarray):
            self.images = img_array
        if isinstance(img_array, str):
            self.images = np.load(img_array)

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index]).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img
    
    def __len__(self):
        return self.images.shape[0]
    
def get_feature_map_size(feature_maps, layer=None):
    total_size = 0
    if layer is None:
        for map_key in feature_maps:
            if isinstance(feature_maps[map_key], np.ndarray):
                total_size += feature_maps[map_key].nbytes / 1000000
            elif torch.is_tensor(feature_maps[map_key]):
                total_size += feature_maps[map_key].numpy().nbytes / 1000000
        return total_size
    
    if layer is not None:
        if isinstance(feature_maps, np.ndarray):
            return feature_maps[layer].nbytes / 1000000
        elif torch.is_tensor(feature_maps):
            return feature_maps[layer].nbytes / 1000000
        
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
# Dataloader Visualizations
        
def reverse_typical_transforms(img_array):
    if torch.is_tensor(img_array):
        img_array = img_array.numpy()
    if len(img_array.shape) == 3:
        img_array = img_array.transpose((1,2,0))
    if len(img_array.shape) == 4:
        img_array = img_array.transpose((0,2,3,1))
    
    return(img_array)

def reverse_imagenet_transforms(img_array):
    if torch.is_tensor(img_array):
        img_array = img_array.numpy()
    if len(img_array.shape) == 3:
        img_array = img_array.transpose((1,2,0))
    if len(img_array.shape) == 4:
        img_array = img_array.transpose((0,2,3,1))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = np.clip(std * img_array + mean, 0, 1)
    
    return(img_array)

def numpy_to_pil(img_array):
    img_dim = np.array(img_array.shape)
    if (img_dim[-1] not in (1,3)) & (len(img_dim) == 3):
        img_array = img_array.transpose(1,2,0)
    if (img_dim[-1] not in (1,3)) & (len(img_dim) == 4):
        img_array = img_array.transpose(0,2,3,1)
    if ((img_array >= 0) & (img_array <= 1)).all():
        img_array = img_array * 255
    if img_array.dtype != 'uint8':
        img_array = np.uint8(img_array)
    
    return (img_array)

from torchvision.utils import make_grid

def get_dataloader_sample(dataloader, nrow = 5, figsize = (5,5), title=None, 
                          reverse_transforms = reverse_imagenet_transforms):
    
    image_batch = next(iter(dataloader))
    batch_size = image_batch.shape[0]
    image_grid = make_grid(image_batch, nrow = batch_size // nrow)
    if reverse_transforms:
        image_grid = reverse_transforms(image_grid)
    plt.figure(figsize=figsize)
    plt.imshow(image_grid)
    plt.axis('off')
    if title is not None:
        plt.title(title)
        