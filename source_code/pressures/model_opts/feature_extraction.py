import pandas as pd
import numpy as np
import os, logging

from tqdm.auto import tqdm
from functools import lru_cache

@lru_cache(128)
def warn_once(msg: str):
    logging.warning(msg)

from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, OrderedDict

try: # to import in python 3.7 style
    from collections import Mapping
except ImportError:
    from collections.abc import Mapping

from .model_options import *

def get_prepped_model(model_string):
    model_options = get_model_options()
    model_call = model_options[model_string]['call']
    model = eval(model_call)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        
    return(model)

def check_model(model_string, model=None):
    if not isinstance(model_string, str):
        model = model_string
    model_options = get_model_options()
    if model_string not in model_options and model == None:
        raise ValueError('model_string not available in prepped models. Please supply model object.')
            
def prep_model_for_extraction(model, inputs=None):
    if model.training:
        model = model.eval()
    if not next(model.parameters()).is_cuda:
        if torch.cuda.is_available():
            model = model.cuda()

    if inputs == None:
        return(model)
    
    if inputs != None:
        if next(model.parameters()).is_cuda:
            inputs = ({k:v.cuda() for k,v in inputs.items()} 
                      if isinstance(inputs, Mapping) else inputs.cuda())
            
        return(model, inputs)

def convert_relu(parent):
    for child_name, child in parent.named_children():
        if isinstance(child, nn.ReLU):
            setattr(parent, child_name, nn.ReLU(inplace=False))
        elif len(list(child.children())) > 0:
            convert_relu(child)

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

def get_module_name(module, module_list):
    class_name = str(module.__class__).split(".")[-1].split("'")[0]
    class_count = str(sum(class_name in module for module in module_list) + 1)
    
    return '-'.join([class_name, class_count])
    
def _get_feature_maps(model, inputs):
    model, inputs = prep_model_for_extraction(model, inputs)
    
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
        model(**inputs) if isinstance(inputs, dict) else model(inputs)

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

def get_feature_maps(model, inputs, layers_to_retain = None, remove_duplicates = True, 
                     enforce_input_shape = True, verbose = False):
    
    model, inputs = prep_model_for_extraction(model, inputs)
    
    if layers_to_retain:
        if not isinstance(layers_to_retain, list):
            layers_to_retain = [layers_to_retain]
    
    def fix_outputs_shape(inputs, outputs, module_name):
        input_shape = inputs[next(iter(inputs))].shape[0] if isinstance(inputs, Mapping) else inputs.shape[0]
    
        if len(outputs.shape) == 0:
            warn_once('Output in {} is empty. Skipping...'.format(module_name))
            return None
        if enforce_input_shape:
            if outputs.shape[0] == input_shape:
                return outputs
            if outputs.shape[0] != input_shape:
                if check_for_input_axis(outputs, input_shape):
                    return reset_input_axis(outputs, input_shape)
                if not check_for_input_axis(outputs, input_shape):
                    warn_once('Ambiguous input axis in {}. Skipping...'.format(module_name))
                    return None
        if not enforce_input_shape:
            return outputs
    
    def register_hook(module):
        def hook(module, input, output):
            def process_output(output, module_name):
                if layers_to_retain is None or module_name in layers_to_retain:
                    if isinstance(output, torch.Tensor):
                        outputs = output.cpu().detach().type(torch.FloatTensor)
                        outputs = fix_outputs_shape(inputs, outputs, module_name)
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
    
    feature_maps = OrderedDict()
    hooks = []
    
    model.apply(convert_relu)
    model.apply(register_hook)
    with torch.no_grad():
        model(**inputs) if isinstance(inputs, Mapping) else model(inputs)

    for hook in hooks:
        hook.remove()
        
    feature_maps = {map:features for (map,features) in feature_maps.items()
                        if features is not None}
        
    if remove_duplicates == True:
        feature_maps = remove_duplicate_feature_maps(feature_maps)
        
    return(feature_maps)

def get_inputs_sample(inputs, n = 3):
    inputs = next(iter(inputs)) if isinstance(inputs, DataLoader) else inputs
    return ({k:v[:3] for k,v in inputs.items()} 
            if isinstance(inputs, Mapping) else inputs[:3])

def get_empty_feature_maps(model, inputs = None, input_shape = None, dataset_size=3,
        layers_to_retain = None, remove_duplicates = True, names_only=False):
    
    check_model(model)
    if isinstance(model, str):
        model = get_prepped_model(model)
        
    if not inputs and not input_shape:
        raise ValueError('Neither inputs nor input_shape is specified.')

    if inputs is not None:
        inputs = get_inputs_sample(inputs)
        
    if inputs is None:
        inputs = torch.rand(3, *input_shape)
        
    empty_feature_maps = get_feature_maps(model, inputs, layers_to_retain, remove_duplicates)
    
    for map_key in empty_feature_maps:
        empty_feature_maps[map_key] = torch.empty(dataset_size, *empty_feature_maps[map_key].shape[1:])
        
    if names_only == True:
        return list(empty_feature_maps.keys())
    
    if names_only == False:
        return empty_feature_maps  
    

def get_all_feature_maps(model, inputs, layers_to_retain=None, remove_duplicates=True, 
                         include_input_space=False, flatten=True, numpy=True, use_tqdm=False):
    
    check_model(model)
    if isinstance(model, str):
        model = get_prepped_model(model)
        
    if include_input_space and isinstance(inputs, Mapping):
        raise ValueError('inputs seem to be dictionary-like. include_input_space must be set to False.')
    
    if isinstance(inputs, DataLoader):
        batch_size, dataset_size, start_index = inputs.batch_size, len(inputs.dataset), 0
        feature_maps = get_empty_feature_maps(model, inputs, None, dataset_size, 
                                              layers_to_retain, remove_duplicates)
        
        if include_input_space:
            input_size = next(iter(inputs))[0].shape[1:]
            input_map = {'Input': torch.empty(dataset_size, *input_size)}
            feature_maps = {**input_map, **feature_maps}
        
        
        for batch in tqdm(inputs, desc = 'Feature Extraction (Batch)') if use_tqdm else inputs:
            batch_feature_maps = get_feature_maps(model, batch, layers_to_retain, remove_duplicates = False)
            
            if include_input_space:
                batch_feature_maps['Input'] = batch.cpu()
            
            for map_i, map_key in enumerate(feature_maps):
                feature_maps[map_key][start_index:start_index+batch_size,...] = batch_feature_maps[map_key]
            start_index += batch_size
                    
    if not isinstance(inputs, DataLoader):
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

def get_feature_map_metadata(model, inputs = None, input_size = None, remove_duplicates = False):
    
    check_model(model)
    if isinstance(model, str):
        model = get_prepped_model(model)
    
    if not inputs and not input_size:
        raise ValueError('Neither inputs nor input_size is specified.')

    if inputs is not None:
        inputs = get_inputs_sample(inputs)
        
    if inputs is None:
        inputs = torch.rand(3, *input_size)

    model, inputs = prep_model_for_extraction(model, inputs)
    
    def register_hook(module):
        def hook(module, input, output):
            def process_output(output, module_name):
                if isinstance(output, torch.Tensor):
                    outputs = output.cpu().detach().type(torch.FloatTensor)
                    map_data[module_name] = outputs

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
    
    model.apply(convert_relu)
    model.apply(register_hook)
    with torch.no_grad():
        model(**inputs) if isinstance(inputs, Mapping) else model(inputs)

    for hook in hooks:
        hook.remove()
        
    if remove_duplicates:
        map_data = remove_duplicate_feature_maps(map_data)
        metadata = {k:v for (k,v) in metadata.items() if k in map_data}
        
    return(metadata)

def get_feature_map_names(model, inputs, remove_duplicates = True):
    feature_map_names = get_empty_feature_maps(model, inputs, names_only = True,
                                                remove_duplicates = remove_duplicates)
    
    return(feature_map_names)

def get_feature_map_count(model, inputs, remove_duplicates = True):
    feature_map_names = get_feature_map_names(model, inputs, remove_duplicates)
    
    return(len(feature_map_names))
        
# Helpers: Dataloaders and functions for facilitating feature extraction

class StimulusSet(Dataset):
    def __init__(self, image_paths, transforms=None):
        self.images = image_paths
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img
    
    def __len__(self):
        return self.images.shape[0]

class CSV2StimulusSet(Dataset):
    def __init__(self, csv, root_dir, transforms=None):
        
        self.root = os.path.expanduser(root_dir)
        self.transforms = transforms
        
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
    def __init__(self, img_array, transforms=None):
        self.transforms = transforms
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
    
class Tokenizer2StimulusSet(Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(texts, padding = True, 
                                   return_tensors = 'pt')

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for 
                key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings[next(iter(self.encodings))])
    
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
        