import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm
import os, sys, time, pickle, argparse
sys.path.append('..')

import torch as torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA

from feature_extraction import *
from model_options import *

def check_reduction_inputs(feature_maps = None, model_inputs = None):
    if feature_maps == None and model_inputs == None:
        raise ValueError('Neither feature_maps nor model_inputs are defined.')
        
    if model_inputs is not None and not isinstance(model_inputs, (DataLoader, torch.Tensor)):
        raise ValueError('model_inputs not supplied in recognizable format.')

def get_feature_map_filepaths(feature_map_names, output_dir):
    return {feature_map_name: os.path.join(output_dir, feature_map_name + '.npy')
                                for feature_map_name in feature_map_names}

#source: stackoverflow.com/questions/26774892
def recursive_delete_if_empty(path):
    if not os.path.isdir(path):
        return False
    
    recurse_list = [recursive_delete_if_empty(os.path.join(path, filename))
                    for filename in os.listdir(path)]
    
    if all(recurse_list):
        os.rmdir(path)
        return True
    if not all(recurse_list):
        return False

def delete_saved_output(output_filepaths, output_dir = None, remove_empty_output_dir = False):
    for file_path in output_filepaths:
        os.remove(output_filepaths[file_path])
    if output_dir is not None and remove_empty_output_dir:
        output_dir = output_dir.split('/')[0]
        recursive_delete_if_empty(output_dir)
        

def torch_corrcoef(m):
    #calculate the covariance matrix
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov_m = 1 / (x.size(1) - 1) * x.mm(x.t())
    
    #convert covariance to correlation
    d = torch.diag(cov_m)
    sigma = torch.pow(d, 0.5)
    cor_m = cov_m.div(sigma.expand_as(cov_m))
    cor_m = cor_m.div(sigma.expand_as(cor_m).t())
    cor_m = torch.clamp(cor_m, -1.0, 1.0)
    return cor_m


#### Sparse Random Projection -------------------------------------------------------------------

def get_feature_map_srps(feature_maps, n_projections = None, upsampling = True, eps=0.1, seed = 0,
                        save_outputs = False, output_dir = 'srp_arrays', 
                        delete_originals = False, delete_saved_outputs = True):
    
    if n_projections is None:
        if isinstance(feature_maps, np.ndarray):
            n_samples = feature_maps.shape[0]
        if isinstance(feature_maps, dict):
            n_samples = next(iter(feature_maps.values())).shape[0]
        n_projections = johnson_lindenstrauss_min_dim(n_samples, eps=eps)
        
    srp = SparseRandomProjection(n_projections, random_state=seed)
    
    def get_srps(feature_map):
        if feature_map.shape[1] <= n_projections and not upsampling:
            srp_feature_map = feature_map
        if feature_map.shape[1] >= n_projections or upsampling:
            srp_feature_map = srp.fit_transform(feature_map)
            
        return srp_feature_map
        
    if isinstance(feature_maps, np.ndarray) and save_outputs:
        raise ValueError('Please provide a dictionary of the form {feature_map_name: feature_map}' + 
                                 'in order to save_outputs.')
        
    if isinstance(feature_maps, np.ndarray) and not save_outputs:
        return srp.fit_transform(feature_maps)
         
    if isinstance(feature_maps, dict) and not save_outputs:
        srp_feature_maps = {}
        for feature_map_name in tqdm(list(feature_maps), desc = 'SRP Extraction (Layer)'):
            srp_feature_maps[feature_map_name] = get_srps(feature_maps[feature_map_name])
            
        if delete_originals:
            feature_maps.pop(feature_map_name)
            
        return srp_feature_maps
    
    if isinstance(feature_maps, dict) and save_outputs:
        output_dir = os.path.join(output_dir, '_'.join(['projections', str(n_projections), 'seed', str(seed)]))
        output_filepaths = get_feature_map_filepaths(feature_maps, output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        srp_feature_maps = {}
        for feature_map_name in tqdm(list(feature_maps), desc = 'SRP Extraction (Layer)'):
            output_filepath = output_filepaths[feature_map_name]
            if not os.path.exists(output_filepath):
                srp_feature_maps[feature_map_name] = get_srps(feature_maps[feature_map_name])
                np.save(output_filepath, srp_feature_maps[feature_map_name])
            if os.path.exists(output_filepath):
                srp_feature_maps[feature_map_name] = np.load(output_filepath, allow_pickle=True)
                
            if delete_originals:
                feature_maps.pop(feature_map_name)
                
        if delete_saved_outputs:
            delete_saved_output(output_filepaths, output_dir, remove_empty_output_dir = True)
                
        return srp_feature_maps      
    
def srp_extraction(model_string, model = None, inputs = None, feature_maps = None, 
                   n_projections = None, upsampling = True, eps=0.1, seed = 0, 
                   output_dir='srp_arrays', delete_saved_outputs = True,
                   delete_original_feature_maps = False, verbose = False):
    
    check_reduction_inputs(feature_maps, inputs)
    output_dir_stem = os.path.join(output_dir, model_string)
        
    device_name = 'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name()
    
    if n_projections is None:
        if feature_maps is None:
            if isinstance(inputs, torch.Tensor):
                n_samples = len(inputs)
            if isinstance(inputs, DataLoader):
                n_samples = len(inputs.dataset)
        if feature_maps is not None:
            n_samples = next(iter(feature_maps.values())).shape[0]
        n_projections = johnson_lindenstrauss_min_dim(n_samples, eps=eps)
        
    if verbose:
        print('Computing {} SRPs for {}; using {} for feature extraction...'
              .format(n_projections, model_string, device_name))
        
    output_dir_ext = '_'.join(['projections', str(n_projections), 'seed', str(seed)])
    output_dir = os.path.join(output_dir_stem, output_dir_ext)
        
    if feature_maps is None:
        check_model(model_string, model)
        
        if model == None:
            model = get_prepped_model(model_string)
        
        model = prep_model_for_extraction(model)
        
        feature_map_names = get_empty_feature_maps(model, names_only = True)
        output_filepaths = get_feature_map_filepaths(feature_map_names, output_dir)
        
        if not all([os.path.exists(file) for file in output_filepaths.values()]):
            feature_maps_remaining = [feature_map for (feature_map, file) 
                                      in output_filepaths.items() 
                                      if not os.path.exists(file)]
            
            feature_maps = get_all_feature_maps(model, inputs, layers_to_retain = feature_maps_remaining)
            
    srp_args = {'feature_maps': feature_maps, 'n_projections': n_projections,
                'upsampling': upsampling, 'eps': eps, 'seed': seed,
                'save_outputs': True, 'output_dir': output_dir_stem,
                'delete_saved_outputs': delete_saved_outputs,
                'delete_originals': delete_original_feature_maps}
            
    return get_feature_map_srps(**srp_args)


#### Principal Components Analysis -------------------------------------------------------------------

def get_feature_map_pcs(feature_maps, n_components = None, return_pca_object = False,
                        save_outputs = False, output_dir = 'pca_arrays', 
                        delete_originals = False, delete_saved_outputs = True):
    
    def get_pca(feature_map):
        n_samples, n_features = feature_map.shape
        n_components_ = n_components
        if n_components_ is not None:
            if n_components_ > n_samples:
                n_components_ = n_samples
                print('More components requested than samples. Reducing...')
        pca = PCA(n_components_, random_state=0)
        if return_pca_object:
            return pca.fit(feature_map)
        if not return_pca_object:
            return pca.fit_transform(feature_map)
        
    if return_pca_object and save_outputs:
        raise ValueError('Saving fitted PCA objects is not currently supported.')
    
    if isinstance(feature_maps, np.ndarray) and save_outputs:
        raise ValueError('Please provide a dictionary of the form {feature_map_name: feature_map}' + 
                                 'in order to save_outputs.')
        
    if isinstance(feature_maps, np.ndarray) and not save_outputs:
        return get_pca(feature_maps)
    
    if isinstance(feature_maps, dict) and not save_outputs:
        pca_feature_maps = {}
        for feature_map_name in tqdm(feature_maps, desc = 'PCA Extraction (Layer)'):
            pca_feature_maps[feature_map_name] = get_pca(feature_maps[feature_map_name])
            
        if delete_originals:
            feature_maps.pop(feature_map_name)
            
        return pca_feature_maps
    
    if isinstance(feature_maps, dict) and save_outputs:
        output_filepaths = get_feature_map_filepaths(feature_maps, output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        pca_feature_maps = {}
        for feature_map_name in tqdm(feature_maps, desc = 'PCA Extraction (Layer)'):
            output_filepath = output_filepaths[feature_map_name]
            if not os.path.exists(output_filepath):
                pca_feature_maps[feature_map_name] = get_pca(feature_maps[feature_map_name])
                np.save(output_filepath, srp_feature_maps[feature_map_name])
            if os.path.exists(output_filepath):
                pca_feature_maps[feature_map_name] = np.load(output_filepath, allow_pickle=True)
                
            if delete_originals:
                feature_maps.pop(feature_map_name)
                
        if delete_saved_outputs:
            delete_saved_output(output_filepaths, output_dir, remove_empty_output_dir = True)
            
    return pca_feature_maps


def pca_extraction(model_string, model = None, inputs = None, feature_maps = None, 
                   n_components = None, aux_inputs = None, aux_feature_maps = None,
                   output_dir='pca_arrays', delete_saved_outputs = True,
                   delete_original_feature_maps = False, verbose = False):
    
    check_reduction_inputs(feature_maps, inputs)
    
    use_aux_pca = aux_inputs is not None or aux_feature_maps is not None
    
    if feature_maps is None:
        if isinstance(inputs, torch.Tensor):
            n_samples = len(inputs)
        if isinstance(inputs, DataLoader):
            n_samples = len(inputs.dataset)
    if feature_maps is not None:
        n_samples = next(iter(feature_maps.values())).shape[0]
    
    if aux_feature_maps is None:
        if aux_inputs is not None:
            if isinstance(aux_inputs, torch.Tensor):
                n_aux_samples = len(inputs)
            if isinstance(aux_inputs, DataLoader):
                n_aux_samples = len(inputs.dataset)
    if aux_feature_maps is not None:
        n_aux_amples = next(iter(feature_maps.values())).shape[0]
    
    if n_components is not None:
        if n_components > n_aux_samples and use_aux_pca:
            raise ValueError('Requesting more components than are available with PCs from auxiliary sample.')
        if n_components > n_samples: 
            raise ValueError('Requesting more components than are available with stimulus set sample size.')
            
    if n_components is None:
        if use_aux_pca:
            n_components = n_aux_samples
        if not use_aux_pca:
            n_components = n_samples
    
    device_name = 'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name()
    
    pca_type = 'auxiliary_input_pcs' if use_aux_pca else 'stimulus_direct'
    pca_printout = '{} Independent PCs' if use_aux_pca else 'up to {} Stimulus PCs'.format(n_components)
    
    if verbose:
        print('Computing {} for {}; using {} for feature extraction...'
              .format(pca_printout, model_string, device_name))

    output_dir = os.path.join(output_dir, model_string, pca_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    aux_maps_remaining = 0
    if feature_maps is not None and aux_feature_maps is not None and use_aux_pca:
        aux_maps_remaining = [map_ for map_ in feature_maps if map_ not in aux_feature_maps]
        
    if (feature_maps is None) or (aux_feature_maps is None and use_aux_pca) or (aux_maps_remaining > 0):
        check_model(model_string, model)
    
        if model == None:
            model = get_prepped_model(model_string)
        
        model = prep_model_for_extraction(model)
        
        if feature_maps is None:
            feature_map_names = get_empty_feature_maps(model, names_only = True)
        if feature_maps is not None:
            feature_map_names = list(feature_maps.keys())
        output_filepaths = get_feature_map_filepaths(feature_map_names, output_dir)
        
        if not all([os.path.exists(file) for file in output_filepaths.values()]):
            feature_maps_remaining = [feature_map for (feature_map, file) 
                                      in output_filepaths.items() 
                                      if not os.path.exists(file)]
            
            if feature_maps is not None:
                feature_maps_remaining = [map_ for map_ in feature_maps_remaining if map_ not in feature_maps]
            
            if feature_maps_remaining > 0:
                if verbose: print('Now extracting features for the stimulus inputs...')
                feature_maps = get_all_feature_maps(model, inputs, layers_to_retain = feature_maps_remaining)
            
            aux_maps_remaining = [map_ for map_ in feature_maps_remaining]
            
            if aux_maps_remaining is not None:
                aux_maps_remaining = [map_ for map_ in aux_maps_remaining if map_ not in feature_maps]
            
            if aux_maps_remaining > 0 and use_aux_pca:
                if verbose: print('Now extracting features for the auxiliary inputs...')
                aux_feature_maps = get_all_feature_maps(model, aux_inputs, layers_to_retain = aux_maps_remaining)
                
    if use_aux_pca:
        pca_args = {'feature_maps': aux_feature_maps, 'n_components': n_components,
                    'return_pca_object': True, 'save_outputs': False,
                    'delete_originals': delete_original_feature_maps}
        
        if save_outputs:
            raise Warning('save_outputs incompatible with using auxiliary PCA. Ignoring (and not saving)...')
        
        aux_pcas = get_feature_map_pcs(**pca_args)
        
        pca_feature_maps = {}
        for feature_map in feature_maps:
            pca_feature_maps = aux_pcas[feature_map].transform(feature_maps[feature_map])
            
    if not use_aux_pca:
        pca_args = {'feature_maps': feature_maps, 'n_components': n_components,
                    'return_pca_object': False, 'save_outputs': False, 'output_dir': output_dir,
                    'delete_saved_outputs': delete_saved_outputs,
                    'delete_originals': delete_original_feature_maps}
        
        pca_feature_maps = get_feature_map_pcs(**pca_args)
        
    return(pca_feature_maps)


#### Representational Similarity Analysis -------------------------------------------------------------------


def rdm_extraction(model_string, model = None, model_inputs = None, feature_maps = None,
                   use_torch_corr = False, append_filename_suffix = False,
                   output_dir='rdm_arrays', delete_saved_outputs = True,
                   delete_original_feature_maps = False, verbose = True):
    
    check_reduction_inputs(feature_maps, model_inputs)
    
    device_name = 'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name()
    
    if verbose: print('Computing RDMS for {}; using {} for feature extraction...'
                      .format(model_string, device_name))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, model_string + '_rdms.pkl' 
                           if append_filename_suffix else model_string + '.pkl')
    
    if os.path.exists(output_file):
        model_rdms = pickle.load(open(output_file,'rb'))
        
    if not os.path.exists(output_file):
        if feature_maps is None:
            check_model(model_string, model)
            
            if model == None:
                model = get_prepped_model(model_string)
             
            model = prep_model_for_extraction(model)
            feature_maps = get_all_feature_maps(model, model_inputs, numpy = not use_torch_corr)
        
        model_rdms = {}
        for model_layer in tqdm(feature_maps, leave=False):
            if use_torch_corr:
                feature_map = feature_maps[model_layer]
                if torch.cuda.is_available():
                    feature_map = feature_map.cuda()
                model_rdm = 1 - torch_corrcoef(feature_map).cpu()
                model_rdms[model_layer] = model_rdm.numpy()
            if not use_torch_corr:
                model_rdms[model_layer] = 1 - np.corrcoef(feature_maps[model_layer])
            if delete_original_feature_maps:
                feature_maps.pop(feature_map)
        with open(output_file, 'wb') as file:
            pickle.dump(model_rdms, file)
            
    return(model_rdms)
