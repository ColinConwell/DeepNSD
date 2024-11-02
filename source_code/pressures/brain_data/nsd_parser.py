import pandas as pd
import numpy as np

import os, sys, json
import yaml, h5py
import nibabel as nib

from copy import deepcopy
from scipy import stats
from tqdm.auto import tqdm

from os import listdir, makedirs
from os.path import join, exists
from os.path import splitext, dirname

__all__ = ['get_subj_dims', 'load_voxel_info', 'load_NSD_voxel_metadata',
           'load_NSD_benchmark_ROI_metadata', 'load_NSD_brain_data']

######################

def get_nsd_path(config=None, key='NSD_PATH'):
    if config is None: # get from env
        config = os.environ.copy()
    
    # If config is file, parse it:
    elif os.path.isfile(config):
        file_extension = splitext(config)[1].lower()
        
        if file_extension == '.json':
            with open(config, 'r') as f:
                config = json.load(f)
        elif file_extension in ['.yaml', '.yml']:
            with open(config, 'r') as f:
                config = yaml.safe_load(f)

        else: # file extension not recognized
            raise ValueError("Unsupported config file type. Use JSON or YAML.")
        
    if isinstance(config, dict):
        return config.get(key, None)
    
def _nsd_path_prompt():
    print("NSD_PATH not found in config files or environment variables.")
    print("Enter the path to NSD or type 'cancel' to exit.")
    while True:
        user_input = input("NSD path (or 'cancel'): ").strip()
        if user_input.lower() == 'cancel':
            return None
        if os.path.exists(user_input):
            os.environ['NSD_PATH'] = user_input
            return user_input
        print("Invalid path. Please try again or type 'cancel' to exit.", file=sys.stderr)
        
BASE_CFG = join(dirname(__file__), '_config.yml')
NSD_PATH = get_nsd_path(BASE_CFG, 'NSD_PATH')

if NSD_PATH is None:
    NSD_PATH = _nsd_path_prompt()
    
if not NSD_PATH or not os.path.exists(NSD_PATH):
    raise ValueError('NSD_PATH not found. Please set NSD_PATH environment variable.')

######################

BETADIR = f'{NSD_PATH}/nsddata_betas/ppdata'
SURFDIR = f'{NSD_PATH}/nsddata/freesurfer'
BETA_VERSION = 'betas_fithrf_GLMdenoise_RR'

def _check_space(space):
    if not space in ['func1pt8mm', 'nativesurface']:
        raise NotImplementedError(f'{space} not currently available.')

def get_subj_dims(subj, space='func1pt8mm'):
    _check_space(space) # make sure space is available
    
    fn = f'{NSD_PATH}/nsddata/ppdata/{subj}/{space}/mean.nii.gz'
    return nib.load(fn).get_fdata().shape

#####################################

def load_voxel_info(subj, space):
    
    if space == 'func1pt8mm':
        roidir = f'{NSD_PATH}/nsddata/ppdata/{subj}/{space}/roi'
        # suffix = '.nii.gz'

    elif space == 'nativesurface':
        roidir = f'{SURFDIR}/{subj}/label'
        # suffix = '.mgz'

    elif space == 'fsaverage':
        roidir = f'{SURFDIR}/fsaverage/label'
        # suffix = '.mgz'

    if space == 'func1pt8mm':

        # get list of files that have roi info
        roifiles = np.sort([fn for fn in listdir(roidir) if not 'lh' in fn and not 'rh' in fn])

        # get list of files that are roi metadata
        annot_fns = []
        for fn in np.sort([fn for fn in listdir(f'{SURFDIR}/{subj}/label') if fn[-4:] == '.mgz' or '.ctab' in fn]):
            if '.mgz.ctab' in fn:
                annot_fns.append(fn)

        roi_dfs = []

        roidata = dict()

        roidata['ncsnr'] = nib.load(f'{BETADIR}/{subj}/{space}/{BETA_VERSION}/ncsnr.nii.gz').get_fdata().reshape(-1)
        
        for domain in ['faces','bodies','word','places']:
            tvals = nib.load(f'{NSD_PATH}/nsddata/ppdata/{subj}/{space}/floc_{domain}tval.nii.gz').get_fdata().reshape(-1)
            if 'word' in domain:
                domain_ = 'words'
            else:
                domain_ = domain
            roidata[f'floc-{domain_}.tval'] = tvals

        for roifile in roifiles:

            X = nib.load(join(roidir,roifile)).get_fdata().reshape(-1)

            if np.ndim(X) == 1:
                roidata[roifile.split('.')[0]] = X.astype(float)

            if roifile.split('.')[0] + '.mgz.ctab' in annot_fns:

                # parse
                annots = pd.read_csv(join(f'{SURFDIR}/{subj}/label',
                                          roifile.split('.')[0] + '.mgz.ctab')).iloc[:,0].values
                annots_dict = dict()
                for an in annots:
                    vals = an.split(' ')
                    annots_dict[int(vals[0])] = vals[1].split('\t')[0]

                # add new col
                X_annots = []
                for x in X:

                    if int(x) <= 0:
                        X_annots.append('n/a')
                    else:
                        X_annots.append(annots_dict[int(x)])

                roidata[roifile.split('.')[0] + '.label'] = np.array(X_annots)

        roi_df = pd.DataFrame(data=roidata)

        roi_dfs.append(roi_df)

    elif space == 'nativesurface':

        # get list of files that have roi info
        roifiles = np.sort([fn for fn in listdir(roidir) if fn[-4:] == '.mgz' or '.ctab' in fn])

        # get list of files that are roi metadata
        annot_fns = []
        for fn in roifiles:
            if '.mgz.ctab' in fn:
                annot_fns.append(fn)

        hemis = ['lh','rh']

        roi_dfs = []

        for hemi in hemis:

            roidata = dict()

            roidata[f'{hemi}.ncsnr'] = np.squeeze(np.array(nib.load(f'{BETADIR}/{subj}/{space}/{BETA_VERSION}/{hemi}.ncsnr.mgh').get_fdata()).T)

            for roifile in roifiles:

                if hemi in roifile:
                    X = np.squeeze(np.array(nib.load(join(roidir,roifile)).get_fdata()).T)

                    if np.ndim(X) == 1:
                        roidata['.'.join(roifile.split('.')[:2])] = X.astype(float)

                    if roifile[3:] + '.ctab' in annot_fns:

                        # parse
                        annots = pd.read_csv(join(roidir,roifile[3:] + '.ctab')).iloc[:,0].values
                        annots_dict = dict()
                        for an in annots:
                            vals = an.split(' ')
                            annots_dict[int(vals[0])] = vals[1].split('\t')[0]

                        # add new col
                        X_annots = []
                        for x in X:

                            if int(x) <= 0:
                                X_annots.append('n/a')
                            else:
                                X_annots.append(annots_dict[int(x)])

                        roidata['.'.join(roifile.split('.')[:2]) + '.label'] = np.array(X_annots)

            roi_df = pd.DataFrame(data=roidata)

            roi_dfs.append(roi_df)

    return roi_dfs


###################################################

def load_NSD_voxel_metadata(subjs, roi_group, space, voxels_to_include=None, 
                            savedir=None, overwrite=False):
    
    if not isinstance(subjs, list):
        subjs = [subjs]
    
    savefile_exists = False
    voxel_metadata = dict()
    
    for subj in subjs:
        
        if savedir:
        
            savefile = join(savedir, f'{subj}_{space}_{roi_group}_voxel_meta_df.npy')
            savefile_exists = exists(savefile)
            
            if not savefile_exists:
                makedirs(savefile, exist_ok = True)
            
        if savefile_exists and not overwrite:
            
            voxel_metadata[subj] = np.load(savefile, allow_pickle=True).item()
        
        else:
            
            roi_dfs = load_voxel_info(subj, space)
        
            voxel_metadata[subj] = dict()
            
            if voxels_to_include is None:
                
                included_voxels = []

                hemis = ['lh','rh']

                for h, hemi in enumerate(hemis):
                    
                    htag = '' if space == 'func1pt8mm' else f'{hemi}.'

                    if roi_group == 'categ-selective':

                        include = [roi_dfs[h][f'{htag}floc-faces'] > 0,
                                   roi_dfs[h][f'{htag}floc-places'] > 0,
                                   roi_dfs[h][f'{htag}floc-bodies'] > 0,
                                   roi_dfs[h][f'{htag}floc-words'] > 0]

                        include = np.sum(np.vstack(include),axis=0) > 0 

                    elif roi_group == 'face-selective':

                        include = roi_dfs[h][f'{htag}floc-faces'].values > 0

                    elif roi_group == 'body-selective':

                        include = roi_dfs[h][f'{htag}floc-bodies'].values > 0

                    elif roi_group == 'place-selective':

                        include = roi_dfs[h][f'{htag}floc-places'].values > 0

                    elif roi_group == 'word-selective':

                        include = roi_dfs[h][f'{htag}floc-words'].values > 0

                    elif roi_group == 'early-visual':

                        include = roi_dfs[h][f'{htag}prf-visualrois'].values > 0

                    elif roi_group == 'streams':

                        include = roi_dfs[h][f'{htag}streams'].values > 0

                    elif roi_group == 'nsdgeneral':

                        include = roi_dfs[h][f'{htag}nsdgeneral'].values > 0

                    elif roi_group == 'nsdgeneral-streams':

                        include = [roi_dfs[h][f'{htag}streams'] > 0,
                                   roi_dfs[h][f'{htag}nsdgeneral'] > 0]

                        include = np.sum(np.vstack(include),axis=0) > 0

                    elif roi_group == 'VTC':

                        VTC_rois = ['V8', 'FFC', 'PIT', 'PHA3', 'TE2p', 'PH', 'VMV3', 'VVC']

                        include = np.isin(roi_dfs[h][f'{htag}HCP_MMP1.label'].values.astype(str),VTC_rois)

                    else: 
                        raise ValueError('roi group not implemented yet')

                    included_voxels.append(include)

                    if space == 'func1pt8mm':
                        break

                all_included_voxels = np.concatenate(included_voxels)
                hemi_included_voxels = deepcopy(included_voxels)
                
            else:
                
                all_included_voxels = voxels_to_include
                hemi_included_voxels = deepcopy(voxels_to_include)

            ##############

            if space == 'func1pt8mm':

                subj_dims = get_subj_dims(subj)
                indices = np.empty(subj_dims, dtype=object)
                for i in range(subj_dims[0]):
                    for j in range(subj_dims[1]):
                        for k in range(subj_dims[2]):
                            indices[i,j,k] = f'({i},{j},{k})'

            elif space == 'nativesurface':

                indices = np.array([f'({i})' for i in np.arange(len(all_included_voxels))])

            indices = indices.reshape(-1)

            included_indices = indices[all_included_voxels]

            ###############

            if space == 'func1pt8mm':
                voxel_meta_df = roi_dfs[h].iloc[all_included_voxels]

            elif space == 'nativesurface':
                voxel_meta_df = [roi_dfs[h] for h in range(2)]
                voxel_meta_df = pd.concat(voxel_meta_df,sort=True).iloc[all_included_voxels]

            voxel_meta_df.index = included_indices

            voxel_metadata[subj]['metadata'] = voxel_meta_df
            voxel_metadata[subj]['all_included_voxels'] = all_included_voxels
            voxel_metadata[subj]['all_included_indices'] = included_indices
            voxel_metadata[subj]['hemi_included_voxels'] = hemi_included_voxels

            if savedir: np.save(savefile, voxel_metadata[subj], allow_pickle=True)      
    
    return voxel_metadata

##########################

def load_NSD_brain_data(subjs, space, roi_group, voxel_metadata, annotations, savedir, output=False):

    zaxis = 1 if space == 'func1pt8mm' else 0
    
    brain_data = dict()
    
    for subj in subjs:
        
        s = int(subj[-1])

        subsess = [40, 40, 32, 30, 40, 32, 40, 30]
        this_subsess = subsess[s-1]
        
        if not exists(join(savedir,f'{subj}_{space}_{roi_group}_brain_data_df.pkl')):

            allses_export_betas = []
    
            for ses in tqdm(range(1,this_subsess+1)):

                #print('\tloading from ses ' + str(ses))

                if space == 'func1pt8mm':
                    betafns = [join(BETADIR,f'{subj}',space,
                                    'betas_fithrf_GLMdenoise_RR',f'betas_session{ses:02d}.nii.gz')]
                elif space == 'nativesurface':
                    betafns = [join(BETADIR,f'{subj}',space,
                                    'betas_fithrf_GLMdenoise_RR',
                                    f'{hemi}.betas_session{ses:02d}.hdf5') for hemi in ['lh','rh']]
                elif space == 'fsaverage':
                    betafns = [join(BETADIR,f'{subj}',space,
                                    'betas_fithrf_GLMdenoise_RR',
                                    f'{hemi}.betas_session{ses:02d}.mgh') for hemi in ['lh','rh']]

                for betafn in betafns:

                    # load data 
                    if space == 'func1pt8mm':
                        betas = nib.load(betafn).get_fdata()        

                    elif space == 'nativesurface':

                        hf = [h5py.File(betafn, 'r') for betafn in betafns]

                        betas = []
                        x = 0
                        for h_ in hf:
                            
                            hemi_included_voxels = voxel_metadata[roi_group][subj]['hemi_included_voxels']
                            incl_idx = np.squeeze(np.argwhere(hemi_included_voxels[x])).astype(int)

                            betas.append(stats.zscore(np.array(h_['betas'])[:,incl_idx],axis=zaxis))
                            x+=1

                    elif space == 'fsaverage':

                        betas = [np.squeeze(np.array(nib.load(betafn).get_data()).T) for betafn in betafns]

                    if type(betas) is list:
                        betas = np.hstack(betas)

                if space == 'func1pt8mm':

                    included_voxels = voxel_metadata[roi_group][subj]['all_included_voxels']

                    included_voxels_3d = np.reshape(included_voxels, betas.shape[:3])

                    export_betas = np.squeeze(betas[included_voxels_3d])

                    export_betas = stats.zscore(export_betas, axis=zaxis)

                    allses_export_betas.append(export_betas)

                elif space == 'nativesurface':

                    export_betas = np.squeeze(betas.T)

                    allses_export_betas.append(export_betas)

            allses_export_betas_stacked = np.float32(np.hstack(allses_export_betas))

            brain_data_df = pd.DataFrame(data=allses_export_betas_stacked)

            brain_data_df.index = voxel_metadata[roi_group][subj]['all_included_indices']
            brain_data_df.columns = np.array(annotations[subj].index)[:brain_data_df.shape[1]]

            brain_data_df.to_pickle(join(savedir,f'{subj}_{space}_{roi_group}_brain_data_df.pkl'))
            
            brain_data[subj] = brain_data_df
            
        else:
            
            if output is True:
                brain_data[subj] = pd.read_pickle(join(savedir,f'{subj}_{space}_{roi_group}_brain_data_df.pkl'))
        

    return brain_data

##########################

def load_NSD_benchmark_ROI_metadata(subjs, space, ncsnr_threshold=0.2, t_threshold=1, 
                                    savedir=None, overwrite=False, **kwargs):
    
    roi_groups = ['nsdgeneral','streams','face-selective','place-selective',
                  'word-selective','body-selective', 'early-visual']

    if kwargs.get('roi_groups', None):
        roi_groups = kwargs.pop('roi_groups')
    
    def get_idx_by_roi_group(metadata, group_label, roi_names, logic='include', t_threshold=None):
        roi_names = np.array(roi_names)
        
        if logic == 'exclude':
            def logic(values, names):
                return np.isin(values, names)

        if logic == 'include':
            def logic(values, names):
                return np.logical_not(np.isin(values, names))
        
        if space == 'func1pt8mm':
            condition = logic(metadata[f'{group_label}.label'].values, roi_names)
            excl_idx = np.squeeze(np.argwhere(condition))
        else:
            condition = [logic(metadata[f'{h}.{group_label}.label'].values, roi_names) for h in ['lh','rh']]
            excl_idx = np.squeeze(np.argwhere(np.logical_or(*condition)))
        
        if t_threshold:
            if space == 'func1pt8mm':
                lowt_idx = np.squeeze(np.argwhere(metadata[f'{group_label}.tval'].values < t_threshold))
            else:
                lowt_idx = np.squeeze(np.argwhere(np.logical_or(
                            metadata[f'lh.{group_label}.tval'].values < t_threshold,
                            metadata[f'rh.{group_label}.tval'].values < t_threshold)))
                
            excl_idx = np.unique(np.concatenate((excl_idx, lowt_idx)))
                
        return excl_idx
        
    if not isinstance(subjs, list):
        subjs = [subjs]
    
    all_metadata = dict()

    voxels_to_include = None
        
    roi_group_iterator = tqdm(roi_groups)
    for roi_group in roi_group_iterator:
        
        roi_group_iterator.set_description(f'Loading voxel metadata for ROI group ({roi_group})')
        
        all_metadata[roi_group] = load_NSD_voxel_metadata(subjs, roi_group, space, voxels_to_include, savedir, overwrite)
        
    voxel_metadata = dict()

    subj_iterator = tqdm(subjs)
    for subj in subj_iterator:
        
        subj_iterator.set_description(f'Parsing voxel metadata for subject ({subj})')

        voxel_metadata[subj] = dict()

        roi_dfs = load_voxel_info(subj, space)

        all_incl_idx = []

        for i, roi_group in enumerate(roi_groups):

            # length of whole volume, ones where the ROI is
            this_included_voxels = deepcopy(all_metadata[roi_group][subj]
                                            ['all_included_voxels'])
            
            this_included_voxels = this_included_voxels.astype(int)

            #print(roi_group, this_included_voxels.sum())

            this_metadata = all_metadata[roi_group][subj]['metadata']

            # get numeric indices of where the ROI is
            roi_idx = np.argwhere(this_included_voxels)

            if 'nsdgeneral' in roi_group or 'early-visual' in roi_group:

                excl_idx = None

            else: # selectively in/exclude IDX

                if 'streams' in roi_group:
        
                    group_label, roi_names = 'streams', ['early','midparietal','parietal']
                    excl_idx = get_idx_by_roi_group(this_metadata, group_label, roi_names, 'exclude')

                elif 'face' in roi_group:
                    
                    group_label, roi_names = 'floc-faces', ['FFA-1','FFA-2','OFA']
                    excl_idx = get_idx_by_roi_group(this_metadata, group_label, roi_names, 'include', t_threshold)
                    
                elif 'body' in roi_group:
                    
                    group_label, roi_names = 'floc-bodies', ['EBA','FBA-1','FBA-2']
                    excl_idx = get_idx_by_roi_group(this_metadata, group_label, roi_names, 'include', t_threshold)

                elif 'place' in roi_group:
                    
                    group_label, roi_names = 'floc-places', ['PPA','OPA']
                    excl_idx = get_idx_by_roi_group(this_metadata, group_label, roi_names, 'include', t_threshold)

                elif 'word' in roi_group:
                    
                    group_label, roi_names = 'floc-words', ['OWFA','VWFA-1','VWFA-2']
                    excl_idx = get_idx_by_roi_group(this_metadata, group_label, roi_names, 'include', t_threshold)

                elif 'early-visual' in roi_group:
                
                    group_label, roi_names = 'prf-visualrois', ['V1v','V1d','V2v','V2d','V3v','V3d','hv4']
                    excl_idx = get_idx_by_roi_group(this_metadata, group_label, roi_names, 'include', t_threshold)
                    
                ################

                # get bad indices bc low reliability
                if ncsnr_threshold:
                    if space == 'func1pt8mm':
                        lowrel_idx = np.squeeze(np.argwhere(this_metadata['ncsnr'].values < ncsnr_threshold))
                    else:
                        or_filter = np.logical_or(this_metadata['lh.ncsnr'].values < ncsnr_threshold,
                                                  this_metadata['rh.ncsnr'].values < ncsnr_threshold)
                        lowrel_idx = np.squeeze(np.argwhere(or_filter))

                    if excl_idx is None:
                        excl_idx = lowrel_idx
                    else:
                        excl_idx = np.unique(np.concatenate((excl_idx, lowrel_idx)))

                # set the bad indices to zero
                if excl_idx is not None:
                    this_included_voxels[roi_idx[excl_idx]] = 0

            this_included_voxels[this_included_voxels == 1] += i

            all_incl_idx.append(this_included_voxels)

        if len(roi_groups) >= 2:

            roi_incl_idx = np.logical_and(all_incl_idx[0], all_incl_idx[1]) # all IDX in nsdgeneral + streams
            roi_incl_idx = np.logical_or(roi_incl_idx, np.sum(np.vstack(all_incl_idx[2:]),axis=0)>0) # or ROI

        else: # target ROI only
            
            roi_incl_idx = all_incl_idx[0].copy()

        print(f'#voxels for {subj}:', np.sum(roi_incl_idx), '(included) / ', len(roi_incl_idx), '(total)')

        ##############

        if space == 'func1pt8mm':

            subj_dims = get_subj_dims(subj)
            indices = np.empty(subj_dims, dtype=object)
            for i in range(subj_dims[0]):
                for j in range(subj_dims[1]):
                    for k in range(subj_dims[2]):
                        indices[i,j,k] = f'({i},{j},{k})'

        elif space == 'nativesurface':

            indices = np.array([f'({i})' for i in np.arange(len(roi_incl_idx))])

        indices = indices.reshape(-1)

        included_indices = indices[roi_incl_idx]

        ###############

        if space == 'func1pt8mm':
            voxel_meta_df = roi_dfs[0].iloc[roi_incl_idx]

        elif space == 'nativesurface':
            voxel_meta_df = [roi_dfs[h] for h in range(2)]
            voxel_meta_df = pd.concat(voxel_meta_df,sort=True).iloc[roi_incl_idx]

        voxel_meta_df.index = included_indices

        voxel_metadata[subj]['metadata'] = voxel_meta_df
        voxel_metadata[subj]['all_included_voxels'] = roi_incl_idx
        voxel_metadata[subj]['all_included_indices'] = included_indices

    return voxel_metadata
