models_to_run = [
 'alexnet_classification',
 'vgg16_classification',
 'resnet18_classification',
 'resnet50_classification',
 'resnet101_classification',
 'resnet152_classification',
 'squeezenet1_0_classification',
 'densenet121_classification',
 'googlenet_classification',
 'shufflenet_v2_x1_0_classification',
 'mobilenet_v2_classification',
 'mnasnet1_0_classification',
 'coat_lite_tiny_classification',
 'convit_base_classification',
 'convit_tiny_classification',
 'convmixer_768_32_classification',
 'convnext_base_classification',
 'convnext_large_classification',
 'crossvit_base_240_classification',
 'deit_base_patch16_224_classification',
 'cspresnet50_classification',
 'dla34_classification',
 'eca_nfnet_l0_classification',
 'efficientnet_b1_classification',
 'efficientnet_b3_classification',
 'ghostnet_100_classification',
 'gmixer_24_224_classification',
 'gmlp_s16_224_classification',
 'hardcorenas_a_classification',
 'hardcorenas_f_classification',
 'jx_nest_tiny_classification',
 'levit_128_classification',
 'inception_v3_classification',
 'mixer_b16_224_classification',
 'mixer_l16_224_classification',
 'mobilenetv3_large_100_classification',
 'nf_resnet50_classification',
 'nfnet_l0_classification',
 'pit_b_224_classification',
 'pit_ti_224_classification',
 'poolformer_s36_classification',
 'regnetx_064_classification',
 'regnety_064_classification',
 'resmlp_12_224_classification',
 'resmlp_24_224_classification',
 'resmlp_36_224_classification',
 'resmlp_big_24_224_classification',
 'semnasnet_100_classification',
 'seresnext50_32x4d_classification',
 'skresnext50_32x4d_classification',
 'swin_base_patch4_window7_224_classification',
 'swin_large_patch4_window7_224_classification',
 'swin_tiny_patch4_window7_224_classification',
 'tnt_s_patch16_224_classification',
 'visformer_small_classification',
 'vit_large_patch16_224_classification',
 'vit_small_patch16_224_classification',
 'vit_tiny_patch16_224_classification',
 'vit_base_patch16_224_classification',
 'vit_base_patch32_224_classification',
 'vit_small_patch32_224_classification',
 'volo_d1_classification',
 'volo_d3_classification',
 'xception_classification',
 'xcit_nano_12_p8_224_classification',
 'xcit_nano_12_p16_224_classification',
 'convnext_base_in22k_classification',
 'convnext_large_in22k_classification',
 'mixer_b16_224_in21k_classification',
 'mixer_l16_224_in21k_classification',
 'vit_large_patch16_224_in21k_classification',
 'vit_small_patch16_224_in21k_classification',
 'vit_tiny_patch16_224_in21k_classification',
 'vit_base_patch16_224_in21k_classification',
 'vit_base_patch32_224_in21k_classification',
 'vit_small_patch32_224_in21k_classification',
 'vit_base_patch16_224_in21k_classification',
 'swin_base_patch4_window7_224_in22k_classification',
 'swin_large_patch4_window7_224_in22k_classification',
 'vit_base_r50_s16_224_in21k_classification',
 'resmlp_big_24_224_in22ft1k_classification',
 'faster_rcnn_R_50_FPN_3x_detection',
 'retinanet_R_50_FPN_3x_detection',
 'mask_rcnn_R_50_FPN_3x_segmentation',
 'keypoint_rcnn_R_50_FPN_3x_segmentation',
 'autoencoding_taskonomy',
 'class_object_taskonomy',
 'class_scene_taskonomy',
 'curvature_taskonomy',
 'denoising_taskonomy',
 'depth_euclidean_taskonomy',
 'depth_zbuffer_taskonomy',
 'edge_occlusion_taskonomy',
 'edge_texture_taskonomy',
 'egomotion_taskonomy',
 'fixated_pose_taskonomy',
 'inpainting_taskonomy',
 'jigsaw_taskonomy',
 'keypoints2d_taskonomy',
 'keypoints3d_taskonomy',
 'nonfixated_pose_taskonomy',
 'normal_taskonomy',
 'point_matching_taskonomy',
 'reshading_taskonomy',
 'room_layout_taskonomy',
 'segment_semantic_taskonomy',
 'segment_unsup25d_taskonomy',
 'segment_unsup2d_taskonomy',
 'vanishing_point_taskonomy',
 'random_weights_taskonomy',
 'RN50_clip',
 'RN101_clip',
 'ViT-B/32_clip',
 'ViT-B/16_clip',
 'ViT-L/14_clip',
 'ViT-S-SimCLR_slip',
 'ViT-S-CLIP_slip',
 'ViT-S-SLIP_slip',
 'ViT-B-SimCLR_slip',
 'ViT-B-CLIP_slip',
 'ViT-B-SLIP_slip',
 'ViT-L-SimCLR_slip',
 'ViT-L-CLIP_slip',
 'ViT-L-SLIP_slip',
 'ViT-L-CLIP-CC12M_slip',
 'ViT-L-SLIP-CC12M_slip',
 'dino_vitb16_selfsupervised',
 'dino_resnet50_selfsupervised',
 'DPT_Hybrid_monoculardepth',
 'MiDaS_monoculardepth',
 'yolov5l_yolo',
 'yolov5m_yolo',
 'yolov5s_yolo',
 'ResNet50-JigSaw-P100_selfsupervised',
 'ResNet50-JigSaw-Goyal19_selfsupervised',
 'ResNet50-RotNet_selfsupervised',
 'ResNet50-ClusterFit-16K-RotNet_selfsupervised',
 'ResNet50-PIRL_selfsupervised',
 'ResNet50-SimCLR_selfsupervised',
 'ResNet50-DeepClusterV2-2x224+6x96_selfsupervised',
 'ResNet50-SwAV-BS4096-2x224+6x96_selfsupervised',
 'ResNet50-MoCoV2-BS256_selfsupervised',
 'ResNet50-BarlowTwins-BS2048_selfsupervised',
 'RegNet-32Gf-SEER_seer',
 'RegNet-32Gf-SEER-INFT_seer',
 'RegNet-64Gf-SEER_seer',
 'RegNet-64Gf-SEER-INFT_seer',
 'RegNet-128Gf-SEER_seer',
 'RegNet-128Gf-SEER-INFT_seer',
 'BiT-Expert-ResNet-V2-Food_bit_expert',
 'BiT-Expert-ResNet-V2-Vehicle_bit_expert',
 'BiT-Expert-ResNet-V2-Instrument_bit_expert',
 'BiT-Expert-ResNet-V2-Flower_bit_expert',
 'BiT-Expert-ResNet-V2-Animal_bit_expert',
 'BiT-Expert-ResNet-V2-Object_bit_expert',
 'BiT-Expert-ResNet-V2-Bird_bit_expert',
 'BiT-Expert-ResNet-V2-Mammal_bit_expert',
 'BiT-Expert-ResNet-V2-Arthropod_bit_expert',
 'BiT-Expert-ResNet-V2-Relation_bit_expert',
 'BiT-Expert-ResNet-V2-Abstraction_bit_expert',
 'alexnet_gn_ipcl_imagenet',
 'alexnet_gn_ipcl_openimages',
 'alexnet_gn_ipcl_places256',
 'alexnet_gn_ipcl_vggface2'
]
