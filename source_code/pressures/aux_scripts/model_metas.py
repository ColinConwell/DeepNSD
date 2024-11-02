from ..main_analysis import *

def get_model_metadata(model_string, sample_image_paths, convert_to_dataframe=True):

    model_option = get_model_options()[model_string]
    model_option['model_string'] = model_string

    model = eval(model_option['call'])
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    image_paths = sample_image_paths
    image_transforms = get_recommended_transforms(model_string)

    image_loader = DataLoader(StimulusSet(image_paths, image_transforms), 
                              batch_size = len(sample_image_paths))
    
    sample_inputs = next(iter(image_loader))
    
    feature_map_metadata = get_feature_map_metadata(model, sample_inputs)
    
    layer_count = len(feature_map_metadata)
    
    feature_counts = [data['feature_count'] for data in feature_map_metadata.values()]
    parameter_counts = [data['parameter_count'] for data in feature_map_metadata.values()]
    
    total_feature_count = int(np.array(feature_counts).sum())
    total_parameter_count = int(np.array(parameter_counts).sum())
    
    model_metadata = {'total_feature_count': total_feature_count,
                      'total_parameter_count': total_parameter_count,
                      'layer_count': layer_count,
                      'layer_metadata': feature_map_metadata}
    
    if not convert_to_dataframe:
        return(model_metadata)
        
    if convert_to_dataframe:

        model_metadata_dictlist = []
        
        for model_layer_index, (model_layer, metadata) in enumerate(feature_map_metadata.items()):
            model_metadata_dictlist.append({'model_string': model_string,
                                            'model_layer': model_layer,
                                            'model_layer_index': (model_layer_index + 1),
                                            'model_layer_depth': (model_layer_index + 1) / layer_count,
                                            'feature_map_shape': metadata['feature_map_shape'],
                                            'feature_count': metadata['feature_count'],
                                            'parameter_count': metadata['parameter_count'],
                                            'total_feature_count': total_feature_count,
                                            'total_parameter_count': total_parameter_count})

        return(pd.DataFrame(model_metadata_dictlist))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Model Metadata')
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
    
    print('Now calculating the parameters of {}...'.format(model_string))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    cuda_device = '(Standard)' if not torch.cuda.is_available() else cuda_device
    
    device_name = 'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name()
    print('Using {} ({}) for feature extraction...'.format(device_name, cuda_device))
    
    
    output_dir = '../results_fresh/model_metadata/{}'.format(benchmark)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
            
    output_file = os.path.join(output_dir, model_string.replace('/','-') + '.' + output_ext)
        
    if not os.path.exists(output_file):
    
        model_option = get_model_options()[model_string]
        model_option['model_string'] = model_string

        benchmark = NSDBenchmark(*benchmark.split('_'))
        sample_image_paths = benchmark.stimulus_data.image_path[:3]
        
        model_metadata = get_model_metadata(model_string, sample_image_paths)
        
        if output_ext == 'csv':
            model_metadata.to_csv(output_file, index=None)
        if output_ext == 'parquet':
             model_metadata.to_parquet(output_file, index=None)
        
        
    