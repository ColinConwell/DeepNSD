### Auxiliary Functions: Feature Extraction ---------------------------------------------------------

from .feature_extraction import *

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_image_transforms():
    from torchvision import transforms
    imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                      'std': [0.229, 0.224, 0.225]}
    
    options = {'imagenet': [transforms.Resize((224,224)), 
                            transforms.ToTensor(),
                            transforms.Normalize(**imagenet_stats)]}
    
    return {key:transforms.Compose(value) 
            for (key, value) in options.items()} 

def get_weights_dtype(model):
    module = list(model.children())[0]
    if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
        return module.weight.dtype
    if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
        return get_weights_dtype(module)

def reverse_typical_transforms(img_array):
    if torch.is_tensor(img_array):
        img_array = img_array.numpy()
    if len(img_array.shape) == 3:
        img_array = img_array.transpose((1,2,0))
    if len(img_array.shape) == 4:
        img_array = img_array.transpose((0,2,3,1))
    
    return(img_array)

def reverse_imagenet_transforms(img_array):
    if torch.is_tensor(img_array):
        img_array = img_array.numpy()
    if len(img_array.shape) == 3:
        img_array = img_array.transpose((1,2,0))
    if len(img_array.shape) == 4:
        img_array = img_array.transpose((0,2,3,1))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = np.clip(std * img_array + mean, 0, 1)
    
    return(img_array)

def numpy_to_pil(img_array):
    img_dim = np.array(img_array.shape)
    if (img_dim[-1] not in (1,3)) & (len(img_dim) == 3):
        img_array = img_array.transpose(1,2,0)
    if (img_dim[-1] not in (1,3)) & (len(img_dim) == 4):
        img_array = img_array.transpose(0,2,3,1)
    if ((img_array >= 0) & (img_array <= 1)).all():
        img_array = img_array * 255
    if img_array.dtype != 'uint8':
        img_array = np.uint8(img_array)
    
    return (img_array)

def get_dataloader_sample(dataloader, nrow = 5, figsize = (5,5), title=None,  
                          reverse_transforms = reverse_imagenet_transforms):
    
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    
    image_batch = next(iter(dataloader))
    batch_size = image_batch.shape[0]
    image_grid = make_grid(image_batch, nrow = batch_size // nrow)
    if reverse_transforms:
        image_grid = reverse_transforms(image_grid)
    plt.figure(figsize=figsize)
    plt.imshow(image_grid)
    plt.axis('off')
    if title is not None:
        plt.title(title)
        
    plt.pause(0.001)
    
### Auxiliary Functions: Mapping Methods ---------------------------------------------------------

from .mapping_methods import *

def get_best_alpha_index(y_pred, regression):
    best_score = 0; best_alpha_index = 0
    for alpha_index, alpha_value in enumerate(regression.alphas):
        score = score_func(y_pred, regression.cv_values_[:, :, alpha_index].squeeze()).mean()
        if score >= best_score:
            best_alpha_index, best_score = alpha_index, score
            
    return best_alpha_index