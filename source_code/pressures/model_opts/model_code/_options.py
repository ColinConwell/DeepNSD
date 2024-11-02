import torch, numpy as np
import os, sys, yaml

from torchvision.datasets.utils import download_url

MODEL_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_download_dir = MODEL_CODE_DIR
sys.path.append(MODEL_CODE_DIR)

# ImageNet Transforms ---------------------------------------------------------------------------

import torchvision.transforms as transforms

def get_imagenet_transforms(input_type = 'PIL'):
    imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std':  [0.229, 0.224, 0.225]}
    
    base_transforms = [transforms.Resize((224,224)), lambda x: x.convert('RGB'), transforms.ToTensor()]
    specific_transforms = base_transforms + [transforms.Normalize(**imagenet_stats)]
    
    if input_type == 'PIL':
        recommended_transforms = specific_transforms
    if input_type == 'numpy':
        recommended_transforms = [transforms.ToPILImage()] + specific_transforms
        
    return transforms.Compose(recommended_transforms)

# SLIP Models ---------------------------------------------------------------------------

slip_model_weights = {
    'ViT-S-SimCLR': 'https://dl.fbaipublicfiles.com/slip/simclr_small_25ep.pt',
    'ViT-S-CLIP': 'https://dl.fbaipublicfiles.com/slip/clip_small_25ep.pt',
    'ViT-S-SLIP': 'https://dl.fbaipublicfiles.com/slip/slip_small_25ep.pt',
    'ViT-S-SLIP-Ep100': 'https://dl.fbaipublicfiles.com/slip/slip_small_100ep.pt',
    'ViT-B-SimCLR': 'https://dl.fbaipublicfiles.com/slip/simclr_base_25ep.pt',
    'ViT-B-CLIP': 'https://dl.fbaipublicfiles.com/slip/clip_base_25ep.pt',
    'ViT-B-SLIP': 'https://dl.fbaipublicfiles.com/slip/slip_base_25ep.pt',
    'ViT-B-SLIP-Ep100': 'https://dl.fbaipublicfiles.com/slip/slip_base_100ep.pt',
    'ViT-L-SimCLR': 'https://dl.fbaipublicfiles.com/slip/simclr_large_25ep.pt',
    'ViT-L-CLIP': 'https://dl.fbaipublicfiles.com/slip/clip_large_25ep.pt',
    'ViT-L-SLIP': 'https://dl.fbaipublicfiles.com/slip/slip_large_25ep.pt',
    'ViT-L-SLIP-Ep100': 'https://dl.fbaipublicfiles.com/slip/slip_large_100ep.pt',
    'ViT-L-CLIP-CC12M': 'https://dl.fbaipublicfiles.com/slip/clip_base_cc12m_35ep.pt',
    'ViT-L-SLIP-CC12M': 'https://dl.fbaipublicfiles.com/slip/slip_base_cc12m_35ep.pt',
}

slip_weight_paths = {key: os.path.join('{}/slip_weights'.format(_download_dir), value.split('/')[-1])
                                       for (key, value) in slip_model_weights.items()}

def define_slip_options():
    slip_options = {}

    for model_name in slip_model_weights:
        model_name = model_name
        train_type = 'slip'
        train_data = 'YFCC15M'
        model_source = 'slip'
        model_string = '_'.join([model_name, train_type])
        model_call = "get_slip_model('{}')".format(model_name)
        slip_options[model_string] = {'model_name': model_name, 'train_type': train_type,
                                      'train_data': train_data, 'model_source': model_source, 'call': model_call}
            
    return slip_options

def get_slip_model(model_name, verbose = True):
    sys.path.append('{}/slip_codebase'.format(MODEL_CODE_DIR))
    import models
    import utils
    from collections import OrderedDict
    from tokenizer import SimpleTokenizer
    
    if not os.path.exists(slip_weight_paths[model_name]):
        download_url(slip_model_weights[model_name], '{}/slip_weights'.format(_download_dir))
    
    ckpt_path = slip_weight_paths[model_name]
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    # create model
    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    model = getattr(models, old_args.model)(rand_embed=False,
        ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
    model.load_state_dict(state_dict, strict=True)
    if verbose: print("=> loaded resume checkpoint '{}' (epoch {})".format(ckpt_path, ckpt['epoch']))
    
    return model.visual

get_slip_transforms = get_imagenet_transforms


# SEER Models ---------------------------------------------------------------------------

seer_model_weights_root = 'https://dl.fbaipublicfiles.com/vissl/model_zoo/'

seer_model_weights = {
    'RegNet-32Gf-SEER': 'seer_regnet32d/seer_regnet32gf_model_iteration244000.torch',
    'RegNet-32Gf-SEER-INFT': 'seer_finetuned/seer_regnet32_finetuned_in1k_model_final_checkpoint_phase78.torch',
    'RegNet-64Gf-SEER': 'seer_regnet64/seer_regnet64gf_model_final_checkpoint_phase0.torch',
    'RegNet-64Gf-SEER-INFT': 'seer_finetuned/seer_regnet64_finetuned_in1k_model_final_checkpoint_phase78.torch',
    'RegNet-128Gf-SEER': ('swav_ig1b_regnet128Gf_cnstant_bs32_node16_sinkhorn10_proto16k_syncBN64_warmup8k/' + 
                          'model_final_checkpoint_phase0.torch'),
    'RegNet-128Gf-SEER-INFT': 'seer_finetuned/seer_regnet128_finetuned_in1k_model_final_checkpoint_phase78.torch',
    'RegNet-256Gf-SEER': ('swav_ig1b_cosine_rg256gf_noBNhead_wd1e5_fairstore_bs16_node64_sinkhorn10_proto16k' +
                          '_apex_syncBN64_warmup8k/' + 'model_final_checkpoint_phase0.torch'),
    'RegNet-256Gf-SEER-INFT': 'seer_finetuned/seer_regnet256_finetuned_in1k_model_final_checkpoint_phase38.torch'
}

seer_model_weights = {key: seer_model_weights_root + value for (key, value) in seer_model_weights.items()}

seer_config_names = {
    'RegNet-32Gf-SEER': 'regnet32Gf',
    'RegNet-32Gf-SEER-INFT': 'regnet32Gf',
    'RegNet-64Gf-SEER': 'regnet64Gf',
    'RegNet-64Gf-SEER-INFT': 'regnet64Gf',
    'RegNet-128Gf-SEER': 'regnet128Gf',
    'RegNet-128Gf-SEER-INFT': 'regnet128Gf',
    'RegNet-256Gf-SEER': 'regnet256Gf_1',
    'RegNet-256Gf-SEER-INFT': 'regnet256Gf_1',
}

def define_seer_options():
    seer_options = {}

    for model_name in seer_model_weights:
        model_name = model_name
        train_type = 'seer'
        train_data = 'custom'
        model_source = 'seer'
        model_string = '_'.join([model_name, train_type])
        model_call = "get_seer_model('{}')".format(model_name)
        seer_options[model_string] = {'model_name': model_name, 'train_type': train_type,
                                      'train_data': train_data, 'model_source': model_source, 'call': model_call}
            
    return seer_options

def get_seer_model(model_name):
    from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
    from vissl.models import build_model
    from classy_vision.generic.util import load_checkpoint
    from vissl.utils.checkpoint import init_model_from_consolidated_weights
    
    weights_dir = '{}/vissl_weights/'.format(_download_dir)
    os.makedirs(weights_dir, exist_ok = True)
    
    weights_path = '{}/{}.torch'.format(weights_dir, model_name)
    if not os.path.exists(weights_path):
        download_url(seer_model_weights[model_name], _download_dir, weights_path)

    model_config = seer_config_names[model_name]

    cfg = ['config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear', 
           '+config/benchmark/linear_image_classification/imagenet1k/models={}'.format(model_config), 
           'config.MODEL.WEIGHTS_INIT.PARAMS_FILE={}'.format(weights_path)]

    cfg = compose_hydra_configuration(cfg)
    _, cfg = convert_to_attrdict(cfg)

    model = build_model(cfg.MODEL, cfg.OPTIMIZER)
    model = model.eval()

    weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)

    init_model_from_consolidated_weights(config=cfg, model=model, 
                                         state_dict=weights, 
                                         skip_layers = [],
                                         state_dict_key_name="classy_state_dict")
    
    return model
    
get_seer_transforms = get_imagenet_transforms

# IPCL Models --------------------------------------------------------------------------

ipcl_models = {'alexnet_gn_ipcl_imagenet': 'ipcl1',
               'alexnet_gn_ipcl_openimages': 'ipcl7',
               'alexnet_gn_ipcl_places256': 'ipcl8',
               'alexnet_gn_ipcl_vggface2': 'ipcl9',
               'alexnet_gn_ipcl_random': 'ipcl16'}

def define_ipcl_options():
    ipcl_options = {}

    for ipcl_model in ipcl_models:
        if 'random' not in ipcl_model:
            model_name = ipcl_model
            train_type = ipcl_model.split('_')[-2]
            train_data = ipcl_model.split('_')[-1]
            model_source = 'ipcl'
            model_string = ipcl_model
            model_call = "get_ipcl_model('{}')".format(model_name)
            ipcl_options[model_string] = {'model_name': model_name, 'train_type': train_type,
                                          'train_data': train_data, 'model_source': model_source, 'call': model_call}
            
    return ipcl_options

def get_ipcl_model(model_name):
    import ipcl_codebase
    return ipcl_codebase.__dict__[ipcl_models[model_name]]()[0]

def get_ipcl_transforms(model_name, input_type = 'PIL'):
    import ipcl_codebase
    
    specific_transforms = ipcl_codebase.__dict__[ipcl_models[model_name]]()[1]
    
    if input_type == 'PIL':
        recommended_transforms = specific_transforms
    if input_type == 'numpy':
        recommended_transforms = [transforms.ToPILImage()] + specific_transforms
    
    return recommended_transforms
    

# Expert Models -------------------------------------------------------------------------

bit_experts = ['foodstuff','object','bird','implement','solid','arthropod',
               'consumer_goods','living_thing','artifact', 'natural_object', 
               'clothing', 'matter', 'carnivore','conveyance','part',
               'substance','placental','relation','commodity','nutriment',
               'instrument', 'animal','entity','invertebrate',
               'spermatophyte','angiosperm','device','structure',
               'chordate','food','vertebrate','covering','flower',
               'container','plant','herb','woody_plant','vascular_plant',
               'mammal','equipment','nutrient', 'whole','instrumentality',
               'abstraction','organism','tree','vehicle','physical_entity']

bit_expert_subset = ['food','vehicle','instrument','flower','animal','object',
                     'bird','mammal','arthropod','relation','abstraction']

def define_bit_expert_options():
    bit_expert_options = {}

    for expertise in bit_expert_subset:
        model_name = 'BiT-Expert-ResNet-V2-{}'.format(expertise.title())
        train_type = 'bit_expert'
        train_data = 'imagenet21k'
        model_source = 'bit_expert'
        model_string = '_'.join([model_name, train_type])
        model_call = "get_bit_expert_model('{}')".format(expertise)
        bit_expert_options[model_string] = {'model_name': model_name, 'train_type': train_type,
                                            'train_data': train_data, 'model_source': model_source, 'call': model_call}
            
    return bit_expert_options

def get_bit_expert_model(expertise):
    from .bit_experts import ResNetV2
    import tensorflow_hub
    
    model_url = 'https://tfhub.dev/google/experts/bit/r50x1/in21k/{}/1'.format(expertise)
    expert_weights = tensorflow_hub.KerasLayer(model_url).weights

    expert_weights_dict = {expert_weights[i].name.replace(':0',''): np.array(expert_weights[i]) 
                           for i in range(len(expert_weights))}

    model = ResNetV2(ResNetV2.BLOCK_UNITS['r50'], width_factor=1, head_size=1000, 
                     zero_head = True, remove_head = True).load_from(expert_weights_dict);
    
    return model

get_bit_expert_transforms = get_imagenet_transforms

# VicReg Models ------------------------------------------------------------------------------

vicreg_models = ['resnet50','resnet50x2']

def define_vicreg_options():
    vicreg_options = {}

    for model_arch in vicreg_models:
        model_name = model_arch + '_vicreg'
        train_type = 'selfsupervised'
        train_data = 'imagenet'
        model_source = 'vicreg'
        model_string = '_'.join([model_name, train_type])
        model_call = "torch.hub.load('facebookresearch/vicreg:main', '{}')".format(model_arch)
        vicreg_options[model_string] = {'model_name': model_name, 'train_type': train_type,
                                        'train_data': train_data, 'model_source': model_source, 'call': model_call}
            
    return vicreg_options

get_vicreg_transforms = get_imagenet_transforms

# Dall-e Models ------------------------------------------------------------------------------

# VQGAN Models ------------------------------------------------------------------------------

vqgan_urls = {'vqgan_imagenet_f16_16384': 
              {'checkpoint': 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1', 
               'config': 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1'}}

vqgan_paths = {'vqgan_imagenet_f16_16384': 
               {'config': f'{MODEL_CODE_DIR}/vqgans_weights/vqgan_imagenet_f16_16384/config.yaml',
                'checkpoint': f'{MODEL_CODE_DIR}/vqgans_weights/vqgan_imagenet_f16_16384/model.ckpt'}}

def define_vqgan_options():
    vqgan_options = {}
    
    for model_name in vqgan_paths:
        train_type = 'generative'
        train_data = 'imagenet'
        model_source = 'vqgan'
        model_string = '_'.join([model_name, train_type])
        model_call = "get_vqgan_model('{}')".format(model_name)
        vqgan_options[model_string] = {'model_name': model_name, 'train_type': train_type,
                                        'train_data': train_data, 'model_source': model_source, 'call': model_call}
        
    return vqgan_options
        
def get_vqgan_model(model_name):
    sys.path.append('{}/vqgans_codebase'.format(MODEL_CODE_DIR))
    
    from omegaconf import OmegaConf
    from taming.models.vqgan import VQModel, GumbelVQ
    
    
    config_path = vqgan_paths[model_name]['config']
    ckpt_path = vqgan_paths[model_name]['checkpoint']
    
    def load_config(config_path, display=False):
        config = OmegaConf.load(config_path)
        if display:
            print(yaml.dump(OmegaConf.to_container(config)))
        return config

    def load_vqgan(config, ckpt_path=None, is_gumbel=False):
        if is_gumbel:
            model = GumbelVQ(**config.model.params)
        else:
            model = VQModel(**config.model.params)
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
        return model.eval()
    
    config = load_config(config_path)
    return load_vqgan(config, ckpt_path=ckpt_path)

def get_vqgan_transforms(input_type = 'PIL'):
    from PIL import Image
    import torch.nn.functional as F
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    
    def preprocess(img, target_image_size=256):
        s = min(img.size)

        if s < target_image_size:
            raise ValueError(f'min dim for image {s} < {target_image_size}')

        r = target_image_size / s
        s = (round(r * img.size[1]), round(r * img.size[0]))
        img = TF.resize(img, s, interpolation=Image.LANCZOS)
        img = TF.center_crop(img, output_size=2 * [target_image_size])
        img = T.ToTensor()(img)
        return 2. * img - 1.
    
    return preprocess

# Aggregate Options ---------------------------------------------------------------------------

def get_custom_model_options(train_type=None, train_data = None, model_source=None):
    model_options = {**define_slip_options(), 
                     **define_seer_options(),
                     **define_ipcl_options(),
                     **define_vqgan_options(),
                     **define_bit_expert_options(),
                     **define_vicreg_options()}
        
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

custom_transform_options = {'slip': get_slip_transforms,
                            'seer': get_seer_transforms,
                            'ipcl': get_ipcl_transforms,
                            'bit_expert': get_bit_expert_transforms,
                            'vqgan': get_vqgan_transforms,
                            'vicreg': get_vicreg_transforms}

def get_custom_transforms_options():
    return custom_transform_options

def get_custom_transform_types():
    return list(custom_transform_options.keys())

def get_custom_transforms(model_query, input_type = 'PIL'):
    custom_model_options = get_custom_model_options()
    if model_query in custom_transform_options:
        return custom_transform_options[model_query](input_type)
    
    nonspecific_options = ['slip', 'seer', 'bit_expert', 'vicreg', 'vqgan']
    if model_query in custom_model_options:
        model_type = custom_model_options[model_query]['model_source']
        if model_type not in nonspecific_options:
            return custom_transform_options[model_type](model_query, input_type)
        if model_type in nonspecific_options:
            return custom_transform_options[model_type](input_type)
        
    if model_query not in list(custom_model_options) + list(custom_transform_options):
        raise ValueError('No reference available for this model query.')