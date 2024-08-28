from ..main_analysis import *
from .manifold_stats import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manifold Statistics Script')
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
    
    print('Now calculating the effective dimensionality of {}...'.format(model_string))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    cuda_device = '(Standard)' if not torch.cuda.is_available() else cuda_device
    
    device_name = 'CPU' if not torch.cuda.is_available() else torch.cuda.get_device_name()
    print('Using {} ({}) for feature extraction...'.format(device_name, cuda_device))
    
    
    output_dir = '../results_fresh/dimensionality/{}'.format(benchmark)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
            
    output_file = os.path.join(output_dir, model_string.replace('/','-') + '.' + output_ext)
        
    if not os.path.exists(output_file):
    
        model_option = get_model_options()[model_string]
        model_option['model_string'] = model_string
        model_option['model'] = eval(model_option['call'])
        model_option['transforms'] = get_recommended_transforms(model_string)

        benchmark = NSDBenchmark(*benchmark.split('_'))

        superlative_layers = (pd.read_csv('press_results/superlative_layers.csv')
                              .set_index('model_string').to_dict(orient='index'))

        model_layer = superlative_layers[model_string]['model_layer']

        image_path, image_transforms = benchmark.stimulus_data.image_path, model_option['transforms']
        stimulus_loader = DataLoader(StimulusSet(image_path, image_transforms), batch_size = 32)
        feature_maps = get_all_feature_maps(model_option['model'], stimulus_loader, model_layer)
        feature_maps_redux = srp_extraction(model_string, feature_maps = feature_maps,  eps = 0.1, seed = 0,
                                            output_dir = 'temp_data/srp_arrays/{}'.format(benchmark.name))

        geometry_dictlist = []
        for model_layer in tqdm(feature_maps, desc = 'Manifold Stats'):
            geometry = ManifoldGeometry(feature_maps[model_layer])
            geometry_dictlist.append({'model_layer': model_layer,
                                      'random_projection': False,
                                      'effective_dimensions': geometry.dimensionality})
            
        for model_layer in tqdm(feature_maps_redux, desc = 'Manifold Stats (Redux)'):
            geometry = ManifoldGeometry(feature_maps_redux[model_layer])
            geometry_dictlist.append({'model_layer': model_layer,
                                      'random_projection': True,
                                      'effective_dimensions': geometry.dimensionality})
            
        manifold_geometry = pd.DataFrame(geometry_dictlist)
        manifold_geometry.insert(0, 'model_string', model_string)
        
        if output_ext == 'csv':
            manifold_geometry.to_csv(output_file, index = None)
        if output_ext == 'parquet':
            manifold_geometry.to_parquet(output_file, index = None)
    