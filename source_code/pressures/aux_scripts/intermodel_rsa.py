from ..main_analysis import *

def make_rs_dataframe(rs):
    rs = pd.DataFrame(rs)
    rs.columns = ['X' + str(col) for col in rs.columns]
    return rs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Intermodel RSA Script')
    parser.add_argument('--model_string', required=True, type=str,
                        help='string / uid of deep net model to load')
    parser.add_argument('--benchmark', required=True, type=str,
                        help='name of benchmark data to analyze')
    parser.add_argument('--output_ext', required=False, type=str,
                        default='csv', help='format to use for saved results')
    parser.add_argument('--cuda_device', required=False, type=str,
                        default='0', help='cuda device to use for extraction')
    
    args = parser.parse_args()
    benchmark = args.benchmark
    model_string = args.model_string
    output_ext = args.output_ext
    cuda_device = args.cuda_device
    
    print('Now computing the most-brain-predictive RSM of {}...'.format(model_string))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    cuda_device = '(Standard)' if not torch.cuda.is_available() else cuda_device
    
    device_name = 'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name()
    print('Using {} ({}) for feature extraction...'.format(device_name, cuda_device))
    
    outputs = ['crsa','wrsa']
    output_files = {}
    for output in outputs:
        output_dir = '../results_fresh/intermodel_rsa/{}/{}'.format(benchmark, output)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_files[output] = os.path.join(output_dir, model_string.replace('/','-') + '.' + output_ext)
        
    if not all([os.path.exists(output_file) for output_file in output_files.values()]):
    
        model_option = get_model_options()[model_string]
        model_option['model_string'] = model_string
        model_option['model'] = eval(model_option['call'])
        model_option['transforms'] = get_recommended_transforms(model_string)

        benchmark = NSDBenchmark(*benchmark.split('_'))

        superlative_layers = (pd.read_csv('press_results/superlative_layers.csv')
                              .set_index('model_string').to_dict(orient='index'))

        model_layer = superlative_layers[model_string]['model_layer']
        target_region = superlative_layers[model_string]['region']

        image_path, image_transforms = benchmark.stimulus_data.image_path, model_option['transforms']
        stimulus_loader = DataLoader(StimulusSet(image_path, image_transforms), batch_size = 32)
        feature_maps = get_all_feature_maps(model_option['model'], stimulus_loader, model_layer)
        feature_maps_redux = srp_extraction(model_string, feature_maps = feature_maps,  eps = 0.1, seed = 0,
                                            output_dir = 'temp_data/srp_arrays/{}'.format(benchmark.name))

        crsm = np.corrcoef(feature_maps[model_layer][1::2,:])
        crs_dataframe = make_rs_dataframe(crsm)
        if output_ext == 'csv':
            crs_dataframe.to_csv(output_files['crsa'], index = None)
        if output_ext == 'parquet':
            crs_dataframe.to_parquet(output_files['crsa'], index = None)

        rdm_indices = benchmark.get_rdm_indices(row_number=True)

        xy = get_splithalf_xy(feature_maps_redux[model_layer], benchmark.response_data)

        alpha_values = np.logspace(-1,5,7).tolist()

        regression = RidgeCVMod(alphas=alpha_values, store_cv_values = True, 
                            alpha_per_target = True, scoring = 'pearson_r')

        regression.fit(xy['train']['X'], xy['train']['y'].transpose())
        best_alpha_idx = np.array([alpha_values.index(alpha_) for alpha_ in regression.alpha_])
        predictions = {'train': np.take_along_axis(regression.cv_values_, best_alpha_idx[None,:,None], axis = 2)[:,:,0],
                       'test': xy['test']['X'].dot(regression.coef_.transpose()) + regression.intercept_}

        subj_rsms = np.zeros((4,500,500))
        for subj_id_idx, subj_id in enumerate(benchmark.rdms[target_region]):
            response_indices = rdm_indices[target_region][subj_id]
            prediction_subset = predictions['test'][:,response_indices]
            subj_rsms[subj_id_idx] = np.corrcoef(prediction_subset)

        wrsm = fisherz_inv(fisherz(subj_rsms).mean(axis = 0))
        wrs_dataframe = make_rs_dataframe(wrsm)
        if output_ext == 'csv':
            wrs_dataframe.to_csv(output_files['wrsa'], index = None)
        if output_ext == 'parquet':
            wrs_dataframe.to_parquet(output_files['wrsa'], index = None)
    