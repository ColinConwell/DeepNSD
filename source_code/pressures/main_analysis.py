from warnings import filterwarnings
filterwarnings('ignore')

import os, sys, argparse

# analysis module imports
from model_opts import *
from brain_data import *

from ridge_gcv_mod import RidgeCVMod
from sklearn import preprocessing

def get_splithalf_xy(feature_map, response_data, scale = True):
    data_splits = {'train': {}, 'test': {}}
    
    scaler = preprocessing.StandardScaler()
    data_splits['train']['X'] = scaler.fit_transform(feature_map[::2,:])
    data_splits['test']['X'] = scaler.transform(feature_map[1::2,:])
    
    response_data = response_data.to_numpy()
    data_splits['train']['y'] = response_data[:,::2]
    data_splits['test']['y'] = response_data[:,1::2]
    
    return data_splits
    
def permute_benchmark(benchmark):
    benchmark.response_data = np.random.randn(*benchmark.response_data.shape)
    benchmark.rdms = benchmark.get_rdms()

def run_benchmarking(benchmark, model_option,
                     precomputed_feature_maps=None, 
                     layers_to_retain=None, 
                     metrics=['crsa','srpr','wrsa'], 
                     alpha_values=np.logspace(-1,5,7).tolist(), 
                     regression_means=True, 
                     precompute_rdms=True):
    
    model_name = model_option['model_name']
    train_type = model_option['train_type']
    
    if precomputed_feature_maps is None:
        model_string = model_option['model_string']
        image_path, image_transforms = benchmark.stimulus_data.image_path, model_option['transforms']
        stimulus_loader = DataLoader(StimulusSet(image_path, image_transforms), batch_size=64)
        feature_maps = get_all_feature_maps(model_option['model'], stimulus_loader, layers_to_retain)
        feature_maps_redux = srp_extraction(model_string, feature_maps=feature_maps, eps=0.1, seed=0,
                                            output_dir='temp_data/srp_arrays/{}'.format(benchmark.name))
        
    if precomputed_feature_maps is not None:
        feature_maps = precomputed_feature_maps['original'] 
        feature_maps_redux = precomputed_feature_maps['redux']
        
    if precompute_rdms:
        feature_map_rdms = {model_layer: {'train': 1 - np.corrcoef(feature_maps[model_layer][::2,:]),
                                          'test': 1 - np.corrcoef(feature_maps[model_layer][1::2,:])}
                            for model_layer in feature_maps}
            
        del feature_maps
    
    scoresheet_lists = {metric: [] for metric in metrics}
    
    if 'randomize' in model_option:
        permute_benchmark(benchmark)
    
    splithalf_rdms = benchmark.get_splithalf_rdms()
    rdm_indices = benchmark.get_rdm_indices(row_number=True)
        
    for model_layer_index, model_layer in enumerate(tqdm(feature_maps_redux, 'Mapping (Layer)')):
        xy = get_splithalf_xy(feature_maps_redux[model_layer], benchmark.response_data)

        regression = RidgeCVMod(alphas=alpha_values, store_cv_values=True, 
                                alpha_per_target=True, scoring='pearson_r')

        regression.fit(xy['train']['X'], xy['train']['y'].transpose())
        best_alpha_idx = np.array([alpha_values.index(alpha_) for alpha_ in regression.alpha_])
        predictions = {'train': np.take_along_axis(regression.cv_values_, best_alpha_idx[None,:,None], axis = 2)[:,:,0],
                       'test': xy['test']['X'].dot(regression.coef_.transpose()) + regression.intercept_}

        for metric in scoresheet_lists:

            if metric == 'crsa':       
                if precompute_rdms:
                    model_rdms = feature_map_rdms[model_layer]
                if not precompute_rdms:
                    model_rdms = {'train': 1 - np.corrcoef(feature_maps[model_layer][::2,:]),
                                  'test': 1 - np.corrcoef(feature_maps[model_layer][1::2,:])}

                for score_set in ['train', 'test']:
                    for region in benchmark.rdms:
                        for subj_id in benchmark.rdms[region]:
                            target_rdm = splithalf_rdms[region][subj_id][score_set]
                            score = compare_rdms(model_rdms[score_set], target_rdm)

                            scoresheet = {'score': score, 
                                          'score_set': score_set,
                                          'region': region, 
                                          'subj_id': subj_id,
                                          'model': model_name, 
                                          'train_type': train_type,
                                          'model_layer': model_layer,
                                          'model_layer_index': model_layer_index,
                                          'distance_1': 'pearson_r',
                                          'distance_2': 'pearson_r'}

                            scoresheet_lists['crsa'].append(scoresheet)

            if metric == 'srpr':
                for score_set in ['train','test']:
                    if score_set == 'train':
                        y = xy['train']['y'].transpose()
                        y_pred = predictions['train']
                    if score_set == 'test':
                        y = xy['test']['y'].transpose()
                        y_pred = predictions['test']
                    for score_type in ['pearson_r']:
                        scores = score_func(y, y_pred, score_type = score_type)

                        for region in benchmark.rdms:
                            for subj_id in benchmark.rdms[region]:
                                response_indices = rdm_indices[region][subj_id]
                                score_subset = scores[response_indices]
                                mean_score = score_subset.mean()

                                scoresheet = {'score': mean_score, 
                                              'score_set': score_set,
                                              'score_type': score_type,
                                              'region': region, 
                                              'subj_id': subj_id,
                                              'model': model_name, 
                                              'train_type': train_type, 
                                              'model_layer': model_layer,
                                              'model_layer_index': model_layer_index}

                                scoresheet_lists['srpr'].append(scoresheet)

            if metric == 'wrsa':
                for score_set in ['train', 'test']:
                    for region in benchmark.rdms:
                        for subj_id in benchmark.rdms[region]:
                            response_indices = rdm_indices[region][subj_id]
                            prediction_subset = predictions[score_set][:,response_indices]
                            model_rdm = 1 - np.corrcoef(prediction_subset)
                            target_rdm = splithalf_rdms[region][subj_id][score_set]
                            score = compare_rdms(model_rdm, target_rdm)

                            scoresheet = {'score': score, 
                                          'score_set': score_set,
                                          'region': region, 
                                          'subj_id': subj_id,
                                          'model': model_name, 
                                          'train_type': train_type,
                                          'model_layer': model_layer,
                                          'model_layer_index': model_layer_index,
                                          'distance_1': 'pearson_r',
                                          'distance_2': 'pearson_r'}

                            scoresheet_lists['wrsa'].append(scoresheet)

    results = {}

    for metric in scoresheet_lists:
        results[metric] = pd.DataFrame(scoresheet_lists[metric])
        
    return results

def get_results_max(results, average_over = None, average_when = 'after'):
    if average_over and not isinstance(average_over, list):
        average_over = [average_over]
        
    if average_over and average_when == 'before':
        group_vars = [var for var in results.columns if var not in average_over + ['score']]
        results = results.groupby(group_vars)['score'].mean().reset_index()
        
    index_cols = [col for col in results.columns if 'score' not in col]
    results = results.copy().pivot_table(index=index_cols, values='score',
                                         columns='score_set').reset_index()
    
    group_vars = [col for col in index_cols if 'model_layer' not in col]
    results = max_transform(results, measure_var='train', group_vars=group_vars)
    results = results.rename(columns={'test': 'score'})
    results.drop(columns = ['train'], inplace = True)

    if average_over and average_when == 'after':
        group_vars = [var for var in index_cols if var not in average_over]
        results = results.groupby(group_vars)['score'].mean().reset_index()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmarking Script')
    parser.add_argument('--model_string', required=True, type=str,
                        help='name of deep net model to load')
    parser.add_argument('--benchmark', required=True, type=str,
                       help='name of benchmark data to assess')
    parser.add_argument('--randomize', required=False, type=bool,
                        default=False, help='premute the benchmark')
    parser.add_argument('--output_ext', required=False, type=str,
                        default = 'parquet', help='format to use when saving results')
    parser.add_argument('--layer_splits', required=False, type=int,
                        default=0, help='number of times to split the layers')
    parser.add_argument('--cuda_device', required=False, type=str,
                        default = '', help='cuda device to use for extraction')
    
    args = parser.parse_args()
    model_string = args.model_string
    benchmark = args.benchmark
    randomize = args.randomize
    output_ext = args.output_ext
    layer_splits = args.layer_splits
    cuda_device = args.cuda_device
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    cuda_device = '(Standard)' if not torch.cuda.is_available() else cuda_device
    
    device_name = 'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name()
    print('Using {} ({}) for feature extraction...'.format(device_name, cuda_device))
    
    outputs = ['crsa','srpr','wrsa']
    
    output_files = {}
    for output in outputs:
        output_dir = 'results_fresh/{}/{}/'.format(benchmark, output)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_files[output] = os.path.join(output_dir, model_string.replace('/','-') + '.' + output_ext)
        
    if not all([os.path.exists(output_file) for output_file in output_files.values()]):
        print('Now scoring {} on the {} benchmark...'.format(model_string, benchmark))
        
        model_option = get_model_options()[model_string]
        model_option['model_string'] = model_string
        model_option['model'] = eval(model_option['call'])
        model_option['transforms'] = get_recommended_transforms(model_string)
        
        benchmark = NSDBenchmark(*benchmark.split('_'))
        
        if randomize:
            model_option['randomize'] = randomize
            
        alpha_values = np.logspace(-1,5,7).tolist()
        
        model_layer_names = get_empty_feature_maps(model_option['model'], 
                                                   names_only = True,
                                                   input_shape = (3,224,224))
        
        layer_subset = model_layer_names[int(len(model_layer_names) / 2):]
        
        results = run_benchmarking(benchmark, model_option, alpha_values = alpha_values,
                                   layers_to_retain = layer_subset)

        for metric in results:
            results[metric].to_parquet(output_files[metric], index = None)
    
    
                             
    
    
    
    