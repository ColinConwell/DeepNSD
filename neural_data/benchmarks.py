import os, sys, shutil
import numpy as np
import pandas as pd
from PIL import Image
import gdown, tarfile

def extract_tar(tar_file, dest_path, delete = True):
    tar = tarfile.open(tar_file)
    tar.extractall(dest_path)
    tar.close()
    os.remove(tar_file)

def iterative_subset(df, index, cat_list):
    out_dict = {}
    df = df[[index] + cat_list]
    for row in df.itertuples():
        current = out_dict
        for i, col in enumerate(cat_list):
            cat = getattr(row, col)
            if cat not in current:
                if i+1 < len(cat_list):
                    current[cat] = {}
                else:
                    current[cat] = []
            current = current[cat]
            if i+1 == len(cat_list):
                current += [getattr(row, index)]
    
    return out_dict

def get_rdm_by_subset(groups, responses):
    rdm_dict = {}
    def get_neural_rdm(rdm_dict, groups):
        for key, value in groups.items():
            if isinstance(value, dict):
                rdm_dict[key] = {}
                get_neural_rdm(rdm_dict[key], groups[key])
            if isinstance(value, list):
                rdm_dict[key] = 1 - np.corrcoef(responses.loc[value,:].to_numpy().transpose())
                
    get_neural_rdm(rdm_dict, groups)
            
    return rdm_dict

def fisherz(r, eps=1e-5):
    return np.arctanh(r-eps)

def fisherz_inv(z):
    return np.tanh(z)

def average_rdms(rdms):
    return 1 - fisherz_inv(fisherz(np.stack([1 - rdm for rdm in rdms])).mean(axis = 0, keepdims = True).squeeze())

class NeuralDataset:
    
    def view_sample_stimulus(self, image_index = None):
        if image_index is None:
            image_index = np.random.randint(self.n_stimuli)

        sample_image_path = self.stimulus_data.image_name[image_index]

        return Image.open(os.path.join(self.image_root, sample_image_path))

    def get_rdms(self, group_vars):
        metadata = self.metadata.reset_index()
        index = self.index_name

        groups = iterative_subset(metadata, index, group_vars)

        return get_rdm_by_subset(groups, self.response_data)
    
    def get_rdm_indices(self, group_vars):
        assert np.array_equal(self.metadata.index, self.response_data.index)
        
        metadata = self.metadata.reset_index()
        metadata['row_number'] = metadata.index
        
        return iterative_subset(metadata, 'row_number', group_vars)
        
class NaturalScenesDataset(NeuralDataset):
    def __init__(self, dataset = 'demo', subset_idx = None):
        
        path_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(path_dir, 'natural_scenes_demo')
        
        response_path = os.path.join(data_dir, 'response_data.parquet')
        metadata_path = os.path.join(data_dir, 'metadata.csv')
        image_key_path = os.path.join(data_dir, 'image_key.csv')
        self.image_root = os.path.join(data_dir, 'stimulus_set')
        
        path_set = [response_path, metadata_path, image_key_path]
        if not all([os.path.exists(path) for path in path_set]):
            print('Downloading data from Google Drive to {}'.format(data_dir))
            tar_file = '{}/natural_scenes_demo.tar.bz2'.format(path_dir)
            gdown.download('https://drive.google.com/uc?export=download&id=16Qkj5g9uUUHiyRayJ_fCLR_nLpKO6S4i',
                           quiet = False, output = tar_file)
            extract_tar(tar_file, path_dir)
            
        self.stimulus_data = pd.read_csv(image_key_path)
        self.n_stimuli = len(self.stimulus_data)
        
        self.response_data = pd.read_parquet(response_path).set_index('voxel_id')
        self.metadata = pd.read_csv(metadata_path).set_index('voxel_id')
        
        self.dataset = dataset
        self.index_name = 'voxel_id'
        if subset_idx is not None:
            self.response_data = self.response_data.loc[subset_idx,:]
            self.metadata = self.metadata.loc[subset_idx,:]
            
        self.stimulus_data = self.stimulus_data.set_index('key').loc[self.response_data.columns].reset_index()
        self.stimulus_data['image_path'] = self.image_root + '/' + self.stimulus_data.image_name
        
        self.rdms = self.get_rdms(['roi_name', 'subj_id'])
        self.rdm_indices = self.get_rdm_indices(['roi_name','subj_id'])