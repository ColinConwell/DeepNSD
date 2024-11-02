import numpy as np
import pandas as pd

import os, torch
import importlib

filepath = os.path.dirname(os.path.abspath(__file__))
model_typology = pd.read_csv(filepath + '/model_typology.csv')
model_typology['model_name'] = model_typology['model']

def subset_typology(model_source):
    return model_typology[model_typology['model_source'] == model_source].copy()

# Torchvision Options ---------------------------------------------------------------------------

from torch.hub import load_state_dict_from_url

def get_torchvision_model(model_path, pretrained = True):
    from torchvision import models
    return eval(f'{model_path}(pretrained = {pretrained})')
    
def define_torchvision_options():
    torchvision_options = {}
    
    model_types = ['classification','segmentation', 'detection', 'video']
    torchvision_dirs = dict(zip(model_types, ['.','.segmentation.', '.detection.', '.video.']))
    
    def get_torchvision_directory(model_name):
        from torchvision import models
        for torchvision_dir in torchvision_dirs.values():
            if model_name in eval('models{}__dict__'.format(torchvision_dir)):
                return torchvision_dir

    torchvision_typology = model_typology[model_typology['model_source'] == 'torchvision'].copy()
    training_calls = {'random': False, 'pretrained': True}
    for index, row in torchvision_typology.iterrows():
        model_name = row['model_name']
        train_type = row['train_type']
        train_data = row['train_data']
        if train_type == 'random':
            train_data = 'None'
        model_source = 'torchvision'
        torchvision_dir = get_torchvision_directory(model_name)
        model_string = '_'.join([model_name, train_type])
        training = 'random' if train_type == 'random' else 'pretrained'
        model_path = 'models' + torchvision_dir + model_name
        model_call =  "get_torchvision_model('{}', {})".format(model_path, training_calls[training])
        torchvision_options[model_string] = {'model_name': model_name, 'train_type': train_type,
                                             'train_data': train_data, 'model_source': model_source, 'call': model_call}
            
    return torchvision_options

import torchvision.transforms as transforms

def get_torchvision_transforms(train_type, input_type = 'PIL'):
    imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std':  [0.229, 0.224, 0.225]}
    
    base_transforms = [transforms.Resize((224,224)), transforms.ToTensor()]
    
    if train_type == 'random': specific_transforms = base_transforms
    
    if train_type == 'classification' or train_type == 'imagenet':
        specific_transforms = base_transforms + [transforms.Normalize(**imagenet_stats)]
    
    if input_type == 'PIL':
        recommended_transforms = specific_transforms
    if input_type == 'numpy':
        recommended_transforms = [transforms.ToPILImage()] + specific_transforms
        
    return transforms.Compose(recommended_transforms)

# Timm Options ---------------------------------------------------------------------------

def get_timm_model(model_name, pretrained = True):
    from timm import create_model
    return create_model(model_name, pretrained)

def define_timm_options():
    timm_options = {}

    timm_typology = model_typology[model_typology['model_source'] == 'timm'].copy()
    for index, row in timm_typology.iterrows():
        model_name = row['model_name']
        train_type = row['train_type']
        train_data = row['train_data']
        if train_type == 'random':
            train_data = 'None'
        model_source = 'timm'
        model_string = '_'.join([model_name, train_type])
        train_bool = False if train_type == 'random' else True
        model_call = "get_timm_model('{}', pretrained = {})".format(model_name, train_bool)
        timm_options[model_string] = ({'model_name': model_name, 'train_type': train_type,
                                       'train_data': train_data, 'model_source': model_source, 'call': model_call})
            
    return timm_options

def modify_timm_transform(timm_transform):
    
    transform_list = timm_transform.transforms
    
    crop_index, crop_size = next((index, transform.size) for index, transform 
                             in enumerate(transform_list) if 'CenterCrop' in str(transform))
    resize_index, resize_size = next((index, transform.size) for index, transform 
                                     in enumerate(transform_list) if 'Resize' in str(transform))
    
    transform_list[resize_index].size = crop_size
    transform_list.pop(crop_index)
    return transforms.Compose(transform_list)
    
def get_timm_transforms(model_name, input_type = 'PIL'):
    from timm.data.transforms_factory import create_transform
    from timm.data import resolve_data_config
    
    config = resolve_data_config({}, model = model_name)
    timm_transforms = create_transform(**config)
    timm_transform = modify_timm_transform(timm_transforms)

    if input_type == 'PIL':
        recommended_transforms = timm_transform.transforms
    if input_type == 'numpy':
        recommended_transforms = [transforms.ToPILImage()] + timm_transform.transforms

    return transforms.Compose(recommended_transforms)


# Taskonomy Options ---------------------------------------------------------------------------
    
def get_taskonomy_encoder(model_name, pretrained = True, verbose = False):
    from visualpriors.taskonomy_network import TASKONOMY_PRETRAINED_URLS
    from visualpriors import taskonomy_network
    
    weights_url = TASKONOMY_PRETRAINED_URLS[model_name + '_encoder'] 
    weights = torch.utils.model_zoo.load_url(weights_url)
    if verbose: print('{} weights loaded succesfully.'.format(model_name))
    model = taskonomy_network.TaskonomyEncoder()
    model.load_state_dict(weights['state_dict'])
    
    return model

def random_taskonomy_encoder():
    from visualpriors import taskonomy_network
    return taskonomy_network.TaskonomyEncoder()

def define_taskonomy_options():
    taskonomy_options = {}

    task_typology = model_typology[model_typology['model_source'] == 'taskonomy'].copy()
    for index, row in task_typology.iterrows():
        model_name = row['model_name']
        train_type = row['train_type']
        train_data = row['train_data']
        model_source = 'taskonomy'
        model_string = model_name + '_' + train_type
        model_call = "get_taskonomy_encoder('{}')".format(model_name)
        taskonomy_options[model_string] = ({'model_name': model_name, 'train_type': train_type,
                                            'train_data': train_data, 'model_source': model_source, 'call': model_call})
        
    taskonomy_options['random_weights_taskonomy'] = {'model_name': 'random_weights', 'train_type': 'taskonomy', 
                                                     'train_data': 'None', 'model_source': 'taskonomy', 
                                                     'call': 'random_taskonomy_encoder()'}
            
    return taskonomy_options

import torchvision.transforms.functional as functional

def taskonomy_transform(image):
    return (functional.to_tensor(functional.resize(image, (256,256))) * 2 - 1)#.unsqueeze_(0)

def get_taskonomy_transforms(input_type = 'PIL'):
    recommended_transforms = taskonomy_transform
    if input_type == 'PIL':
        return recommended_transforms
    if input_type == 'numpy':
        def functional_from_numpy(image):
            image = functional.to_pil_image(image)
            return recommended_transforms(image)
        return functional_from_numpy

# CLIP Options ---------------------------------------------------------------------------

def get_clip_model(model_name):
    import clip; model, _ = clip.load(model_name, device='cpu')
    return model.visual
    
def define_clip_options():
    clip_options = {}

    clip_typology = model_typology[model_typology['model_source'] == 'clip'].copy()
    for index, row in clip_typology.iterrows():
        model_name = row['model_name']
        train_type = row['train_type']
        train_data = row['train_data']
        model_source = 'clip'
        model_string = '_'.join([model_name, train_type])
        model_call = "get_clip_model('{}')".format(model_name)
        clip_options[model_string] = ({'model_name': model_name, 'train_type': train_type,
                                       'train_data': train_data, 'model_source': model_source, 'call': model_call})
            
    return clip_options

def get_clip_transforms(model_name, input_type = 'PIL'):
    import clip; _, preprocess = clip.load(model_name, device = 'cpu')
    if input_type == 'PIL':
        recommended_transforms = preprocess.transforms
    if input_type == 'numpy':
        recommended_transforms = [transforms.ToPILImage()] + preprocess.transforms
    recommended_transforms = transforms.Compose(recommended_transforms)
    
    using_half_tensor_models = False
    if using_half_tensor_models:
        if 'ViT' in model_name:
            def transform_plus_retype(image_input):
                return recommended_transforms(image_input).type(torch.HalfTensor)
            return transform_plus_retype
        if 'ViT' not in model_name:
            return recommended_transforms
    
    if not using_half_tensor_models:
        return recommended_transforms

# VISSL Options ---------------------------------------------------------------------------

def get_vissl_model(model_name):
    vissl_data = (model_typology[model_typology['model_source'] == 'vissl']
                  .set_index('model_name').to_dict('index'))
    
    weights = load_state_dict_from_url(vissl_data[model_name]['weights_url'], map_location = torch.device('cpu'))
    
    def replace_module_prefix(state_dict, prefix, replace_with = ''):
        return {(key.replace(prefix, replace_with, 1) if key.startswith(prefix) else key): val
                      for (key, val) in state_dict.items()}

    def convert_model_weights(model):
        if "classy_state_dict" in model.keys():
            model_trunk = model["classy_state_dict"]["base_model"]["model"]["trunk"]
        elif "model_state_dict" in model.keys():
            model_trunk = model["model_state_dict"]
        else:
            model_trunk = model
        return replace_module_prefix(model_trunk, "_feature_blocks.")

    converted_weights = convert_model_weights(weights)
    excess_weights = ['fc','projection', 'prototypes']
    converted_weights = {key:value for (key,value) in converted_weights.items()
                             if not any([x in key for x in excess_weights])}
    
    if 'module' in next(iter(converted_weights)):
        converted_weights = {key.replace('module.',''):value for (key,value) in converted_weights.items()
                             if 'fc' not in key}
    
    from torchvision.models import resnet50
    import torch.nn as nn

    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x

    model = resnet50()
    model.fc = Identity()

    model.load_state_dict(converted_weights)
    
    return model
    

def define_vissl_options():
    vissl_options = {}

    vissl_typology = model_typology[model_typology['model_source'] == 'vissl'].copy()
    for index, row in vissl_typology.iterrows():
        model_name = row['model_name']
        train_type = row['train_type']
        train_data = row['train_data']
        model_source = 'vissl'
        model_string = '_'.join([model_name, train_type])
        model_call = "get_vissl_model('{}')".format(model_name)
        vissl_options[model_string] = ({'model_name': model_name, 'train_type': train_type,
                                        'train_data': train_data, 'model_source': model_source, 'call': model_call})
            
    return vissl_options


def get_vissl_transforms(input_type = 'PIL'):
    imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std':  [0.229, 0.224, 0.225]}
    
    base_transforms = [transforms.Resize((224,224)), transforms.ToTensor()]
    specific_transforms = base_transforms + [transforms.Normalize(**imagenet_stats)]
    
    if input_type == 'PIL':
        recommended_transforms = specific_transforms
    if input_type == 'numpy':
        recommended_transforms = [transforms.ToPILImage()] + specific_transforms
        
    return transforms.Compose(recommended_transforms)

# DINO Options ---------------------------------------------------------------------------

def define_dino_options():
    dino_options = {}

    dino_typology = model_typology[model_typology['model_source'] == 'dino'].copy()
    for index, row in dino_typology.iterrows():
        model_name = row['model_name']
        train_type = row['train_type']
        train_data = row['train_data']
        model_source = 'dino'
        model_string = '_'.join([model_name, train_type])
        model_call = "torch.hub.load('facebookresearch/dino:main', '{}')".format(model_name)
        dino_options[model_string] = ({'model_name': model_name, 'train_type': train_type,
                                       'train_data': train_data, 'model_source': model_source, 'call': model_call})
            
    return dino_options

def get_dino_transforms(input_type = 'PIL'):
    imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std':  [0.229, 0.224, 0.225]}
    
    base_transforms = [transforms.Resize((224,224)), transforms.ToTensor()]
    
    specific_transforms = base_transforms + [transforms.Normalize(**imagenet_stats)]
    
    if input_type == 'PIL':
        recommended_transforms = specific_transforms
    if input_type == 'numpy':
        recommended_transforms = [transforms.ToPILImage()] + specific_transforms
        
    return transforms.Compose(recommended_transforms)


# MiDaS Options ---------------------------------------------------------------------------

def define_midas_options():
    midas_options = {}

    midas_typology = model_typology[model_typology['model_source'] == 'midas'].copy()
    for index, row in midas_typology.iterrows():
        model_name = row['model']
        train_type = row['train_type']
        train_data = row['train_data']
        model_source = 'midas'
        model_string = '_'.join([model_name, train_type])
        model_call = "torch.hub.load('intel-isl/MiDaS','{}')".format(model_name)
        midas_options[model_string] = ({'model_name': model_name, 'train_type': train_type,
                                        'train_data': train_data, 'model_source': model_source, 'call': model_call})
            
    return midas_options

def get_midas_transforms(model_name, input_type = 'PIL'):
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_name in ['DPT_Large', 'DPT_Hybrid']:
        transform = midas_transforms.dpt_transform
    if model_name not in ['DPT_Large', 'DPT_Hybrid']:
        transform = midas_transforms.small_transform
        
    transforms_lambda = [lambda img: np.array(img)] + transform.transforms 
    transforms_lambda += [lambda tensor: tensor.squeeze()]
    
    if input_type == 'PIL':
        recommended_transforms = transforms_lambda
    if input_type == 'numpy':
        recommended_transforms = [transforms.ToPILImage()] + transforms_lambda
        
    return transforms.Compose(recommended_transforms)

# YOLO Options ---------------------------------------------------------------------------

def define_yolo_options():
    yolo_options = {}

    yolo_typology = model_typology[model_typology['model_source'] == 'yolo'].copy()
    for index, row in yolo_typology.iterrows():
        model_name = row['model']
        train_type = row['train_type']
        train_data = row['train_data']
        model_source = 'yolo'
        model_string = '_'.join([model_name, train_type])
        model_call = "torch.hub.load('ultralytics/yolov5','{}', autoshape = False, force_reload = True)".format(model_name)
        yolo_options[model_string] = ({'model_name': model_name, 'train_type': train_type,
                                       'train_data': train_data, 'model_source': model_source, 'call': model_call})
            
    return yolo_options

def get_yolo_transforms(model_name, input_type = 'PIL'):
    assert input_type == 'PIL', "YoloV5 models requires input_type == 'PIL'"
    from PIL import Image
    
    def yolo_transforms(pil_image, size = (256,256)):
        img = np.asarray(pil_image.resize(size, Image.BICUBIC))
        if img.shape[0] < 5:  # image in CHW
            img = img.transpose((1, 2, 0))
        img = img[:, :, :3] if img.ndim == 3 else np.tile(img[:, :, None], 3)
        img = img if img.data.contiguous else np.ascontiguousarray(img)
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img_tensor = torch.from_numpy(img) / 255.
        
        return img_tensor
    
    return yolo_transforms

# Detectron2 Options ---------------------------------------------------------------------------

def get_detectron_model(model_name, downsize = True, backbone_only = True):
    from detectron2.modeling import build_model
    from detectron2 import model_zoo
    from detectron2.checkpoint import DetectionCheckpointer
    
    detectron_data = subset_typology('detectron')
    detectron_dict = (detectron_data.set_index('model').to_dict('index'))
    weights_path = detectron_dict[model_name]['weights_url']
    
    cfg = model_zoo.get_config(weights_path)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_path)
    
    cfg_clone = cfg.clone()
    model = build_model(cfg_clone)
    model = model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    
    if backbone_only:
        return model.backbone
    if not backbone_only:
        return model
    

def define_detectron_options():
    detectron_options = {}

    detectron_typology = model_typology[model_typology['model_source'] == 'detectron'].copy()
    for index, row in detectron_typology.iterrows():
        model_name = row['model']
        train_type = row['train_type']
        train_data = row['train_data']
        model_source = 'detectron'
        model_string = '_'.join([model_name, train_type])
        model_call = "get_detectron_model('{}')".format(model_name)
        detectron_options[model_string] = ({'model_name': model_name, 'train_type': train_type,
                                            'train_data': train_data, 'model_source': model_source, 'call': model_call})
            
    return detectron_options
    
def get_detectron_transforms(model_name, input_type = 'PIL', downsize = True):
    import detectron2.data.transforms as detectron_transform
    from detectron2 import model_zoo
    
    detectron_data = subset_typology('detectron')
    detectron_dict = (detectron_data.set_index('model').to_dict('index'))
    weights_path = detectron_dict[model_name]['weights_url']
    
    cfg = model_zoo.get_config(weights_path)
    model = get_detectron_model(model_name, backbone_only = False)
    
    augment = detectron_transform.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, 
                                                      cfg.INPUT.MIN_SIZE_TEST], 
                                                      cfg.INPUT.MAX_SIZE_TEST)
    
    if downsize:
        augment = detectron_transform.ResizeShortestEdge([224,224], 256)
    
    def detectron_transforms(original_image):
        if input_type == 'PIL':
            original_image = np.asarray(original_image)
        original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = augment.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        return model.preprocess_image([inputs]).tensor.squeeze()
    
    return detectron_transforms    

# Aggregate Options ---------------------------------------------------------------------------

if importlib.util.find_spec('custom_models') is not None:
    from custom_models import *
    
def get_model_options(train_type=None, train_data = None, model_source=None):
    model_options = {**define_torchvision_options(), 
                     **define_taskonomy_options(),
                     **define_timm_options(),  
                     **define_clip_options(), 
                     **define_vissl_options(), 
                     **define_yolo_options(),
                     **define_dino_options(),
                     **define_midas_options(), 
                     **define_detectron_options()}
    
    if importlib.util.find_spec('custom_models') is not None:
        model_options = {**model_options, 
                         **get_custom_model_options()}
        
    if train_type is not None:
        model_options = {string: info for (string, info) in model_options.items() 
                         if model_options[string]['train_type'] == train_type}
        
    if train_data is not None:
        model_options = {string: info for (string, info) in model_options.items() 
                         if model_options[string]['train_data'] == train_data}
        
    if model_source is not None:
        model_options = {string: info for (string, info) in model_options.items() 
                         if model_options[string]['model_source'] == model_source}
        
    return model_options    
    

transform_options = {'torchvision': get_torchvision_transforms, 
                     'timm': get_timm_transforms,
                     'taskonomy': get_taskonomy_transforms,
                     'clip': get_clip_transforms,
                     'vissl': get_vissl_transforms,
                     'yolo': get_yolo_transforms,
                     'dino': get_dino_transforms,
                     'midas': get_midas_transforms,
                     'detectron': get_detectron_transforms}

def get_transform_options():
    return transform_options

def get_transform_types():
    return list(transform_options.keys())
    
def get_recommended_transforms(model_query, input_type = 'PIL'):
    cached_model_types = ['imagenet','taskonomy','vissl']
    model_types = model_typology['train_type'].unique()
    if model_query in get_model_options():
        model_option = get_model_options()[model_query]
        model_type = model_option['train_type']
        model_name = model_option['model_name']
        model_source = model_option['model_source']
    if model_query in model_types:
        model_type = model_query
    
    if model_type in cached_model_types:
        if model_type == 'imagenet':
            return get_torchvision_transforms('imagenet', input_type)
        if model_type == 'vissl':
            return get_vissl_transforms(input_type)
        if model_type == 'taskonomy':
            return get_taskonomy_transforms(input_type)
            
    if model_type not in cached_model_types:
        if model_source == 'torchvision':
            return transform_options[model_source](model_type, input_type)
        if model_source in ['timm', 'clip', 'detectron']:
            return transform_options[model_source](model_name, input_type)
        if model_source in ['taskonomy', 'vissl', 'dino', 'yolo', 'midas']:
            return transform_options[model_source](input_type)
        
    if importlib.util.find_spec('custom_models') is not None:
        if model_query in (list(get_custom_model_options()) + 
                           list(custom_transform_options)):
            return get_custom_transforms(model_query, input_type)
    
    if model_query not in list(get_model_options()) + list(model_types):
         raise ValueError('No reference available for this model query.')
