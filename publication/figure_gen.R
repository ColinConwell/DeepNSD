if (!require(pacman)) {install.packages("pacman")}
pacman::p_load('this.path', 'glue', 'tidyverse')

# set path to current file
setwd(dirname(this.path()))

# source main results data
source('all_results.R')

# initialize figure list
figure_list <- list()

# overwrite saved figures?
overwrite_saved <- TRUE

# NOTE: When saving figures, you may have to play around with the sizing,
# as R will default to using your device's screen pixel ratio

# ··············································································
# Figure 2 Plot ----------------------------------------------------------------

temp_data <- results$max %>% filter(region == 'OTC') %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  rename(contrast = compare_architecture) %>%
  filter(contrast %in% c('Convolutional','Transformer')) %>%
  mutate(contrast = str_replace(contrast, 'Convolutional','CNN')) %>%
  select(architecture, contrast, metric, subj_id, score) %>%
  {left_join(., filter(., metric == 'wrsa') %>%
               group_by(architecture, contrast, metric) %>%
               summarise(score = mean(score)) %>%
               group_by(contrast, metric) %>%
               mutate(rank = dense_rank(score)) %>% ungroup %>%
               select(architecture, contrast, rank))} %>%
  left_join(results$max %>% filter(!is.na(compare_architecture)) %>%
              distinct(architecture, display_name))

temp_data %>% group_by(metric) %>%
  mutate(grand_mean = mean(score)) %>%
  group_by(metric, subj_id) %>%
  mutate(subj_mean = mean(score),
         score = score - subj_mean + grand_mean) %>%
  ggplot(aes(x = rank, y = score, color = contrast)) +
  geom_rect(aes(color = contrast, fill = contrast, 
                group = metric, alpha = metric,
                xmin = min_rank, xmax = max_rank,
                ymin = score$ymin, ymax = score$ymax), 
            inherit.aes = FALSE,  linetype = 3,
            data = . %>% group_by(metric, contrast) %>%
              summarise(min_rank = min(rank) - 0.75,
                        max_rank = max(rank) + 0.75,
                        score = mean_cl_boot(score))) +
  geom_hline(aes(yintercept = 0.7975), data = noise_ceilings$group_avg, 
             linetype = 1, size = 2, color = 'gray') +
  stat_summary(aes(fill = contrast, alpha = metric), 
               fun.data = mean_cl_boot, geom = 'crossbar') +
  stat_summary(aes(y = -0.025, label = display_name), fun = mean, 
               geom = 'text', size = 8 / .pt, angle = 45,
               label.padding = unit(c(0.15, 0.15, 0.15, 0.15), "lines"),
               data = . %>% filter(!metric %in% c()), hjust = 1) +
  facet_wrap(~contrast, scales = 'free_x',
             strip.position='bottom') +
  force_panelsizes(cols = c(0.625, 0.375)) +
  theme_minimal() +  ylim(c(0.0,1)) +
  scale_alpha_manual(values = c(0,0.5)) + 
  scale_fill_manual(values = palette[1:2]) + 
  scale_color_manual(values = palette[1:2]) + 
  labs(y = '*r<sub>Pearson</sub>* (Score)', x = element_blank(),
       color = element_blank(), shape = 'Metric') +
  facetted_pos_scales(
    x = list(scale_x_continuous(expand = expansion(add = c(0.5, 0))),
             scale_x_continuous(expand = expansion(add = c(0, 0.5))))) +
  scale_y_continuous(expand = c(0,0), breaks = seq(0,0.8,0.1),
                     sec.axis = add_ev_axis) + 
  coord_cartesian(ylim=c(0.0, 0.8), clip = 'off') +
  easy_remove_legend() + easy_remove_x_axis() +
  theme(text = element_text(size = 10, face = 'plain'),
        #panel.border = element_rect(fill=NA),
        plot.margin = margin(t=1,r=1,b=3,l=1, "cm"),
        axis.ticks.y = element_line(),
        axis.line.y = element_line(),
        axis.line.x = element_line(),
        axis.ticks.length = unit(0.1, "cm"),
        axis.title.y.left = element_markdown(),
        axis.title.y.right = element_markdown(),
        strip.background = element_blank(),
        strip.text = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank())

figure_list[['figure2']] <- last_plot()

output_file <- 'figure_svgs/figure2-plot.svg'
if (!file.exists(output_file) | overwrite_saved) {
  ggsave(output_file, width=810, height=464, units='px')
}

# ··············································································
# Figure 3 Plot ----------------------------------------------------------------

contrast_levels <- c('Taskonomy', 'Non-Contrastive','Contrastive',
                     'Category-Supervised','SimCLR','CLIP','SLIP')

temp_data <- results$max %>% filter(region == 'OTC') %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  filter(!is.na(compare_goal_taskonomy_tasks)) %>%
  mutate(contrast = 'Taskonomy') %>%
  bind_rows(results$max %>% filter(region == 'OTC') %>%
              filter(metric %in% c('crsa','wrsa')) %>%
              filter(!is.na(compare_goal_contrastive)) %>%
              mutate(display_name = str_replace(display_name, '-BS2048|-BS256|-BS4096',''),
                     display_name = str_replace(display_name, '-2x224\\+6x96',''),
                     display_name = str_replace(display_name, 'ResNet50-', '')) %>%
              rename(contrast = compare_goal_contrastive)) %>%
  bind_rows(results$max %>% filter(region == 'OTC') %>%
              filter(metric %in% c('crsa','wrsa')) %>%
              filter(model == 'resnet50', 
                     train_task == 'classification') %>%
              mutate(display_name = 'Supervised (Reference)') %>%
              mutate(contrast = 'Category-Supervised')) %>%
  bind_rows(results$max %>% filter(region == 'OTC') %>%
              filter(metric %in% c('crsa','wrsa')) %>%
              filter(!is.na(compare_goal_slip)) %>%
              mutate(display_name = str_replace(display_name, '-CLIP|-SLIP|-SimCLR', ''),
                     display_name = str_replace(display_name, 'ViT-', ''),
                     display_name = paste(compare_goal_slip, display_name, sep = '-')) %>%
              mutate(contrast = compare_goal_slip)) %>%
  select(model, display_name, train_task, contrast, subj_id, metric, score) %>%
  {left_join(., filter(., metric == 'wrsa') %>%
               group_by(model, train_task, contrast, metric) %>%
               summarise(score = mean(score)) %>%
               group_by(contrast, metric) %>%
               mutate(rank = dense_rank(score)) %>% ungroup() %>%
               select(model, train_task, contrast, rank))}

label_offset <- function(x) {mean(x) - (0.05 + mean(x) / 1.75)}
panel_sizes = c(0.5, 0.085, 0.1, 0.015, 0.06, 0.06, 0.06)

temp_data %>% group_by(metric) %>%
  mutate(grand_mean = mean(score)) %>%
  group_by(metric, subj_id) %>%
  mutate(subj_mean = mean(score),
         score = score - subj_mean + grand_mean) %>%
  mutate(contrast = factor(contrast, levels = contrast_levels)) %>% 
  mutate(display_name = str_replace(display_name, 'Unsupervised', '')) %>%
  ggplot(aes(x = rank, y = score, color = contrast)) +
  geom_hline(aes(yintercept = 0.7975), data = noise_ceilings$group_avg, 
             linetype = 1, size = 2, color = 'gray') +
  geom_rect(aes(color = contrast, fill = contrast, 
                group = metric, alpha = metric,
                xmin = min_rank, xmax = max_rank,
                ymin = score$ymin, ymax = score$ymax), 
            inherit.aes = FALSE,  linetype = 3,
            data = . %>% group_by(metric, contrast) %>%
              mutate(rank = ifelse(str_detect(contrast,'gory'), NA, rank)) %>%
              summarise(min_rank = min(rank) - 0.75,
                        max_rank = max(rank) + 0.75,
                        score = mean_cl_boot(score))) +
  stat_summary(aes(fill = contrast, alpha = metric), 
               fun.data = mean_cl_boot, geom = 'crossbar') +
  stat_summary(aes(y = 0.025, label = display_name), fun = label_offset,
               geom = 'text', size = 8 / .pt, angle = 45,
               data = . %>% filter(!metric %in% c()), hjust = 1) +
  facet_grid(~contrast, scales = 'free_x', space = 'free_x') +
  force_panelsizes(cols = panel_sizes) +
  theme_minimal() +  ylim(c(0.0,1)) +
  scale_alpha_manual(values = c(0,0.5)) + 
  scale_fill_manual(values = c(palette[3:5], palette[1], palette[6:9])) + 
  scale_color_manual(values = c(palette[3:5], palette[1], palette[6:9])) + 
  labs(y = '*r<sub>Pearson</sub>* (Score)', x = element_blank(),
       color = element_blank(), shape = 'Metric') +
  facetted_pos_scales(
    x = list(scale_x_continuous(expand = expansion(add = c(0.25, 0))),
             scale_x_continuous(expand = expansion(add = c(0, 0))),
             scale_x_continuous(expand = expansion(add = c(0, 0))),
             scale_x_continuous(expand = expansion(add = c(0, 0))),
             scale_x_continuous(expand = expansion(add = c(0, 0))),
             scale_x_continuous(expand = expansion(add = c(0, 0))),
             scale_x_continuous(expand = expansion(add = c(0, 0.25))))) +
  scale_y_continuous(expand = c(0,0), breaks = seq(0,0.8,0.1),
                     sec.axis = add_ev_axis) + 
  coord_cartesian(ylim = c(0,0.8), clip = 'off') +
  easy_remove_x_axis() + easy_move_legend('bottom') +
  theme(text = element_text(size = 12, face = 'plain'),
        #panel.border = element_rect(fill=NA),
        plot.margin = margin(t=1,r=1,b=6,l=1, "cm"),
        axis.ticks.y = element_line(),
        axis.line.y = element_line(),
        #axis.ticks.x = element_line(),
        axis.line.x = element_line(),
        axis.ticks.length = unit(0.1, "cm"),
        axis.title.y = element_markdown(),
        axis.title.y.right = element_markdown(),
        strip.background = element_blank(),
        strip.text = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank()) + 
  guides(color = 'none', alpha = 'none') + easy_remove_legend()

figure_list[['figure3']] <- last_plot()

output_file <- 'figure_svgs/figure3-plot.svg'
if (!file.exists(output_file) | overwrite_saved) {
  ggsave(output_file, width=675, height=464, units='px')
}

# ··············································································
# Figure 4 Plot ----------------------------------------------------------------

plot_list <- list()

contrast_levels <- c('imagenet' = 'ImageNet1K', 
                     'imagenet21k' = 'ImageNet21K') %>% list_reverse()

temp_data <- results$max %>% filter(region == 'OTC') %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  mutate_at(vars(compare_diet_imagenetsize), fct_recode, 
            !!!contrast_levels) %>%
  filter(!is.na(compare_diet_imagenetsize)) %>%
  rename(contrast = compare_diet_imagenetsize) %>%
  select(contrast, display_name, metric, subj_id, score) %>%
  mutate(display_name = str_replace(display_name, '-IN21K',''),
         display_name = str_replace(display_name, '-IN22K',''),
         display_name = str_replace(display_name, 'MLP-Mixer', 'Mixer'),
         display_name = str_replace(display_name, 'Mixer', 'MLP-Mixer'),
         display_name = str_replace(display_name, 'Base', 'B'),
         display_name = str_replace(display_name, 'Large', 'L')) %>%
  {left_join(., filter(., metric == 'wrsa',
                       contrast == 'ImageNet1K') %>%
               group_by(display_name, metric) %>%
               summarise(score = mean(score)) %>%
               group_by(metric) %>%
               mutate(rank = dense_rank(score)) %>% ungroup %>%
               select(display_name, rank))}

plot_data <- temp_data %>% group_by(metric) %>%
  mutate(grand_mean = mean(score)) %>%
  group_by(metric, subj_id) %>%
  mutate(subj_mean = mean(score),
         score = score - subj_mean + grand_mean)

ggplot(plot_data, aes(x = rank, y = score, color = contrast)) +
  stat_summary(aes(fill = contrast, alpha = metric), 
               position = position_dodge(width = 0.9),
               fun.data = mean_cl_boot, geom = 'crossbar',
               data = . %>% filter(metric == 'wrsa')) +
  stat_summary(aes(fill = contrast, alpha = metric),
               position = position_dodge(width = 0.9),
               fun.data = mean_cl_boot, geom = 'crossbar',
               data = . %>% filter(metric == 'crsa')) + 
  stat_summary(aes(y = -0.025, label = display_name), fun = mean, 
               geom = 'text', size = 8 / .pt, angle = 45,
               position = position_nudge(x = 0.25),
               data = . %>% filter(contrast == 'ImageNet21K'), hjust = 1) +
  geom_hline(aes(yintercept = 0.7975), data = noise_ceilings$group_avg, 
             linetype = 1, size = 2, color = 'gray') +
  geom_rect(aes(color = contrast, fill = contrast, 
                group = metric, alpha = metric,
                xmin = min_rank, xmax = max_rank,
                ymin = score$ymin, ymax = score$ymax), 
            inherit.aes = FALSE,  linetype = 3,
            data = . %>% group_by(metric, contrast) %>%
              filter(str_detect(contrast, 'K')) %>%
              summarise(min_rank = min(rank) - 0.75,
                        max_rank = max(rank) + 0.75,
                        score = mean_cl_boot(score))) +
  theme_minimal() +  ylim(c(0.0,1)) +
  scale_alpha_manual(values = c(0,0.5)) + 
  scale_fill_manual(values = c(palette[10], palette[12])) + 
  scale_color_manual(values = c(palette[10], palette[12])) + 
  labs(y = '*r<sub>Pearson</sub>* (Score)', x = element_blank(),
       color = element_blank(), shape = 'Metric') +
  scale_x_continuous(expand = c(0.025,0.025)) + 
  scale_y_continuous(expand = c(0,0)) + 
  coord_cartesian(ylim = c(0,0.8), clip = 'off') +
  easy_remove_legend() + easy_remove_x_axis() +
  theme(text = element_text(size = 10, face = 'plain'),
        #panel.border = element_rect(fill=NA),
        plot.margin = margin(t=1,r=1,b=5,l=1, "cm"),
        axis.ticks.y = element_line(),
        axis.line.y = element_line(),
        axis.line.x = element_line(),
        axis.ticks.length = unit(0.1, "cm"),
        axis.title.y = element_markdown(),
        strip.background = element_blank(),
        strip.text = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank()) -> plot_list[[1]]

temp_data <- results$max %>% filter(region == 'OTC') %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  filter(!is.na(compare_diet_ipcl)) %>%
  rename(contrast = compare_diet_ipcl) %>%
  select(contrast, display_name, metric, subj_id, score) %>%
  {left_join(., filter(., metric == 'wrsa') %>%
               group_by(contrast, metric) %>%
               summarise(score = mean(score)) %>%
               group_by(contrast, metric) %>%
               mutate(rank = dense_rank(score)) %>% ungroup %>%
               select(contrast, rank))} %>%
  mutate(display_name = str_replace(display_name, 'AlexNet-GN-IPCL',''))

plot_data <- temp_data %>% group_by(metric) %>%
  mutate(grand_mean = mean(score)) %>%
  group_by(metric, subj_id) %>%
  mutate(subj_mean = mean(score),
         score = score - subj_mean + grand_mean)

ggplot(plot_data, aes(x = rank, y = score, color = contrast)) +
  geom_hline(aes(yintercept = 0.7975), data = noise_ceilings$group_avg, 
             linetype = 1, size = 2, color = 'gray') +
  stat_summary(aes(fill = contrast, alpha = metric), 
               fun.data = mean_cl_boot, geom = 'crossbar') +
  stat_summary(aes(y = -0.025, label = display_name), fun = mean,
               geom = 'text', size = 8 / .pt, angle = 45,
               position = position_nudge(x = 1.0),
               data = . %>% filter(!metric %in% c('wrsa')), hjust = 1) +
  facet_wrap(~contrast, scales = 'free_x',
             strip.position='bottom', nrow = 1) +
  scale_alpha_manual(values = c(0,0.5)) + 
  scale_fill_manual(values = palette[13:17]) + 
  scale_color_manual(values = palette[13:17]) + 
  labs(y = '*r<sub>Pearson</sub>* (Score)', x = element_blank(),
       color = element_blank(), shape = 'Metric') +
  scale_x_continuous(expand = expansion(add = c(0.0, 1.0))) +
  scale_y_continuous(expand = c(0,0), breaks = seq(0,0.8,0.1),
                     sec.axis = add_ev_axis) +
  coord_cartesian(ylim = c(0,0.8), clip = 'off') +
  theme_minimal() + easy_remove_legend() + easy_remove_x_axis() +
  theme(text = element_text(size = 10, face = 'plain'),
        #panel.border = element_rect(fill=NA),
        plot.margin = margin(t=1,r=1,b=5,l=1, "cm"),
        axis.ticks.y = element_line(),
        axis.line.y = element_line(),
        axis.line.x = element_line(),
        axis.ticks.length = unit(0.1, "cm"),
        axis.title.y.left = element_markdown(),
        axis.title.y.right = element_markdown(),
        strip.background = element_blank(),
        strip.text = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank()) -> plot_list[[2]]

cowplot::plot_grid(plot_list[[1]] + theme(plot.margin = margin(1, 1, 2, 1, unit = 'cm')), 
                   plot_list[[2]] + #easy_remove_y_axis(teach=TRUE) + 
                     theme(axis.ticks.y.left = element_blank(), 
                           axis.title.y.left = element_blank(),
                           axis.text.y.left = element_blank(), 
                           axis.line.y.left = element_blank()) +
                     theme(plot.margin = margin(1, 1, 2, -0.75, unit = 'cm')),
                   NULL, rel_widths = c(0.5, 0.2, 0.3), nrow = 1)

figure_list[['figure4']] <- last_plot()

output_file <- 'figure_svgs/figure4-plot.svg'
if (!file.exists(output_file) | overwrite_saved) {
  ggsave(output_file, width=810, height=464, units='px')
}

# ··············································································
# Figure 5 Plot ----------------------------------------------------------------

plot_list <- list()

models_to_label = c('RN50_clip','RegNet-64Gf-SEER_seer',
                    #'hardcorenas_f_classification',
                    'resmlp_36_224_classification',
                    'convnext_base_classification',
                    'ResNet50-SimCLR_selfsupervised', 
                    'faster_rcnn_R_50_FPN_3x_detection',
                    #'retinanet_R_50_FPN_3x_detection',
                    #'ResNet50-JigSaw-P100_selfsupervised',
                    #'efficientnet_b1_classification',
                    'vit_base_patch16_224_classification',
                    #'ViT-B-SimCLR_slip', 'ViT-B-SLIP_slip',
                    'alexnet_classification', 'resnet50_classification')

temp_data <- results$summary %>% select(-rank) %>%
  left_join(results$summary %>% select(model_string, metric, score) %>%
              filter(metric == 'wrsa') %>%
              mutate(rank = dense_rank(-score)) %>% select(-metric, -score))

target_levels <- c('ImageNet+','OpenImages', 'Places256', 'VGGFace2','Taskonomy','Untrained')
plot_data <- temp_data %>% filter(metric %in% c('crsa','wrsa')) %>%
  mutate(metric = str_to_upper(metric),
         metric = factor(metric, levels = c('CRSA','WRSA'))) %>%
  mutate(display_name = str_replace(display_name, 'CLiP-ResNet50','ResNet50-CLIP'),
         display_name = str_replace(display_name, 'Faster-RCNN-',''),
         display_name = str_replace(display_name, 'FPN','RCNN-FPN')) %>%
  mutate(training = ifelse(str_detect(model_string, 'random'), 'Untrained', 'ImageNet+'),
         training = ifelse(str_detect(model_string, 'open'), 'OpenImages', training),
         training = ifelse(str_detect(model_string, 'places'), 'Places256', training),
         training = ifelse(str_detect(model_string, 'vggface'), 'VGGFace2', training),
         training = ifelse(str_detect(model_string, 'taskonomy'), 'Taskonomy', training)) %>%
  ungroup() %>% distinct(model_string, display_name, metric, score, rank, training) %>%
  mutate(metric = factor(metric, levels = c('WRSA', 'CRSA')),
         training = factor(training, levels = target_levels))

ggplot(plot_data, aes(x = rank, y = score, color = training, 
                      shape = metric, linetype = metric)) + 
  theme_classic() + geom_point(cex = 3, alpha = 0.5, stroke = 1) + 
  geom_label_repel(aes(label = display_name), show.legend = FALSE,
                   data = . %>% filter(metric == 'WRSA') %>%
                     filter(model_string %in% models_to_label),
                   force = 5, direction = 'y', nudge_y = -0.1,
                   max.overlaps = 10, min.segment.length = 0.1) +
  geom_hline(aes(yintercept = 0.7975), data = noise_ceilings$group_avg, 
             linetype = 1, size = 2, color = 'gray') +
  scale_shape_manual(values = c(19, 21)) +
  scale_color_manual(values = c('black',palette[13:15], palette[4], 'gray')) +
  labs(x = 'Models (Sorted by veRSA Score)', 
       y = '*r<sub>Pearson</sub>* (Score)',
       color = element_blank(), shape = element_blank()) +
  scale_x_continuous(expand = c(0.15, 0)) +
  scale_y_continuous(expand = c(0,0), #sec.axis = add_ev_axis,
                     breaks = seq(0,0.8,0.2)) +
  coord_cartesian(ylim=c(0.0, 0.8), clip = 'off') +
  guides(alpha = 'none') + guides(shape = 'none') +
  theme(text = element_text(size = 20), legend.position=c(.9,0.625),
        plot.margin = unit(c(t=10,r=10,b=10,l=10), "pt"),
        #panel.border = element_rect(fill=NA),
        panel.spacing = unit(2, "lines"),
        axis.title.x = element_text(vjust = -0.5),
        axis.text.x = element_text(vjust = -0.5),
        axis.ticks.y = element_line(),
        axis.line.y = element_line(),
        axis.line.x = element_line(),
        axis.ticks.length = unit(0.1, "cm"),
        axis.title.y.left = element_markdown(),
        axis.title.y.right = element_markdown(),
        strip.background = element_blank(),
        strip.text = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank()) -> plot_list[[1]]

manifold_stats <- read_csv('source_data/model_statistics/dimensionality.csv') %>%
  filter(!str_detect(model_string, 'NPID')) %>%
  mutate(trained = ifelse(str_detect(model_string, '_random'), 'No', 'Yes'))

plot_data <- results$summary %>% select(model_string, metric, score) %>%
  left_join(manifold_stats, multiple = 'all',
            relationship = 'many-to-many') %>%
  filter(metric %in% c('crsa','wrsa'), random_projection) %>%
  mutate(metric = str_to_upper(metric),
         metric = factor(metric, levels = c('CRSA','WRSA'))) %>%
  mutate(training = ifelse(str_detect(model_string, 'random'), 'Untrained', 'ImageNet+'),
         training = ifelse(str_detect(model_string, 'open'), 'OpenImages', training),
         training = ifelse(str_detect(model_string, 'places'), 'Places256', training),
         training = ifelse(str_detect(model_string, 'vggface'), 'VGGFace2', training),
         training = ifelse(str_detect(model_string, 'taskonomy'), 'Taskonomy', training)) %>%
  mutate(metric = factor(metric, levels = c('WRSA', 'CRSA')),
         training = factor(training, levels = target_levels)) %>%
  ungroup() %>% distinct(model_string, metric, score, effective_dimensions, training)

ggplot(plot_data, aes(x = effective_dimensions, y = score, color = training,
                      shape = metric, linetype = metric)) + theme_minimal() +
  facet_wrap2(~metric, axes = 'all', remove_labels = 'none', nrow = 2) + 
  guides(alpha = 'none') + guides(shape = 'none') +
  geom_point(cex = 3, alpha = 0.5, stroke = 1) + theme_classic() +
  geom_smooth(method = 'lm', linetype = 1, color = 'black', 
              data = . %>% filter(training != 'Untrained', metric == 'WRSA')) +
  geom_smooth(method = 'lm', linetype = 2, color = 'black', 
              data = . %>% filter(training != 'Untrained', metric == 'CRSA')) +
  geom_hline(aes(yintercept = 0.7975), data = noise_ceilings$group_avg, 
             linetype = 1, size = 2, color = 'gray') +
  scale_shape_manual(values = c(19, 21)) +
  scale_color_manual(values = c('black',palette[13:15], palette[4], 'gray')) +
  scale_x_log10(breaks = trans_breaks(n = 5, "log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) + 
  annotation_logticks(sides ='b', outside = TRUE) + 
  labs(x = 'Effective Dimensionality', 
       y = '*r<sub>Pearson</sub>* (Score)',
       color = element_blank(), shape = element_blank()) +
  scale_y_continuous(expand = c(0,0), #sec.axis = add_ev_axis,
                     breaks = seq(0,0.8,0.2)) +
  easy_remove_legend() +
  coord_cartesian(ylim=c(0.0, 0.8), clip = 'off') +
  theme(text = element_text(size = 20),
        plot.margin = unit(c(t=10,r=10,b=10,l=10), "pt"),
        #panel.border = element_rect(fill=NA),
        panel.spacing = unit(1, "lines"),
        axis.title.x = element_text(vjust = -0.5),
        axis.text.x = element_text(vjust = -0.5),
        axis.ticks.y = element_line(),
        axis.line.y = element_line(),
        axis.line.x = element_line(),
        axis.ticks.length = unit(0.1, "cm"),
        axis.title.y.left = element_markdown(),
        axis.title.y.right = element_markdown(),
        strip.background = element_blank(),
        strip.text = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank()) -> plot_list[[2]]

plot_data <- results$summary %>% select(model_string, metric, score) %>%
  left_join(imagenet_scores, multiple = 'all') %>%
  drop_na(imagenet_accuracy) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  mutate(metric = str_to_upper(metric),
         metric = factor(metric, levels = c('WRSA','CRSA'))) %>%
  ungroup() %>% distinct(model_string, metric, score, imagenet_accuracy)

ggplot(plot_data, aes(x = imagenet_accuracy, y = score,
                      shape = metric, linetype = metric)) + theme_minimal() +
  facet_wrap2(~metric, axes = 'all', remove_labels = 'none', nrow = 2) + 
  guides(alpha = 'none') + guides(shape = 'none') +
  geom_point(cex = 3, alpha = 0.5, stroke = 1) + theme_classic() +
  geom_smooth(method = 'lm', linetype = 1, color = 'black', 
              data = . %>% filter(metric == 'WRSA')) +
  geom_smooth(method = 'lm', linetype = 2, color = 'black', 
              data = . %>% filter(metric == 'CRSA')) +
  geom_hline(aes(yintercept = 0.7975), data = noise_ceilings$group_avg, 
             linetype = 1, size = 2, color = 'gray') +
  scale_shape_manual(values = c(19, 21)) +
  scale_color_manual(values = c('black')) +
  labs(x = 'ImageNet Accuracy (Top 1)', 
       y = '*r<sub>Pearson</sub>* (Score)',
       color = element_blank(), shape = element_blank()) +
  scale_y_continuous(expand = c(0,0), #sec.axis = add_ev_axis,
                     breaks = seq(0,0.8,0.2)) +
  easy_remove_legend() +
  coord_cartesian(ylim=c(0.0, 0.8), clip = 'off') +
  theme(text = element_text(size = 20),
        plot.margin = unit(c(t=10,r=10,b=10,l=10), "pt"),
        #panel.border = element_rect(fill=NA),
        panel.spacing = unit(1, "lines"),
        axis.title.x = element_text(vjust = -0.5),
        axis.text.x = element_text(vjust = -0.5),
        axis.ticks.y = element_line(),
        axis.line.y = element_line(),
        axis.line.x = element_line(),
        axis.ticks.length = unit(0.1, "cm"),
        axis.title.y.left = element_markdown(),
        axis.title.y.right = element_markdown(),
        strip.background = element_blank(),
        strip.text = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank()) -> plot_list[[3]]

plot_list[[4]] <- cowplot::plot_grid(plot_list[[2]] + labs(y = ''), 
                                     plot_list[[3]] + labs(y = ''))

cowplot::plot_grid(plot_list[[1]], plot_list[[4]],
                   rel_widths = c(0.5, 0.5), ncol = 2)

figure_list[['figure5']] <- last_plot()

output_file <- 'figure_svgs/figure5-plot.svg'
if (!file.exists(output_file) | overwrite_saved) {
  ggsave(output_file, width=1532, height=720, units='px')
}

# ··············································································
# Figure 6 Plot ----------------------------------------------------------------

plot_list <- list()

plot_data <- model_uber_rsa %>% 
  filter(model1 %in% model_sets$upper, 
         model2 %in% model_sets$upper)

# write.csv(plot_data, 'figure_data/Figure6A.csv', row.names=FALSE)

ggplot(plot_data, aes(x = correlation)) + theme_classic() +
  facet_wrap(~ metric, nrow = 2) + #easy_remove_y_axis() +
  geom_histogram(aes(y = ..density..), fill = 'gray', color = 'black',
                 position='identity', binwidth = 0.05) +
  #geom_density(aes(y = ..density..), alpha = 0.6) +
  scale_y_continuous(expand = c(0, 0.1), 
                     labels = ~.x / 0.5) +
  scale_x_continuous(expand = c(0, 0.1), limits = c(0,1), 
                     breaks = seq(0,1.0,0.5)) +
  guides(fill = 'none') + labs(x = element_blank()) +
  labs(x = 'Representational Similarity', y = 'Count') +
  theme(text = element_text(size = 20),
        strip.background = element_blank(),
        #strip.text.x = element_blank(),
        panel.spacing = unit(1, "lines"),
        plot.margin = margin(1,1,1,1, "cm"),
        axis.line.x = element_blank(), 
        axis.line.y = element_blank(),
        panel.border = element_rect(fill='transparent'))

plot_list[[1]] <- last_plot()

group_levels <- c('Convolutional','Transformer','Taskonomy','Non-Contrastive-SSL','Contrastive-SSL',
                  'SimCLR','CLIP','SLIP','ImageNet1K','ImageNet21K',
                  'IPCL-ImageNet1K','IPCL-OpenImages','IPCL-Places256','IPCL-VGGFace2')

group_levels <- c('Convolutional','Transformer','Taskonomy','NC-SSL','C-SSL',
                  'SimCLR','CLIP','SLIP','IN1K','IN21K',
                  'Objects(1)','Objects(2)','Places','Faces')

model_uber_mds$crsa %>% 
  summarise(v1_min = min(V1), v1_max = max(V1),
            v2_min = min(V2), v2_max = max(V2))

model_uber_mds$wrsa %>% 
  summarise(v1_min = min(V1), v1_max = max(V1),
            v2_min = min(V2), v2_max = max(V2))

group_levels <- c('Convolutional','Transformer','Taskonomy','NC-SSL','C-SSL',
                  'SimCLR','CLIP','SLIP','ImageNet1K','ImageNet21K',
                  'Objects(1)','Objects','Places','Faces')

plot_data <- model_uber_mds %>% names() %>%
  lapply(function(x) {
    bind_rows(
      model_uber_mds[[x]] %>%
        left_join(main_metadata %>%
                    mutate(group = as.character(compare_architecture)) %>% 
                    mutate(group = ifelse(str_detect(group, 'Convolutional|Transformer'), group, NA)) %>%
                    select(model_string, group)) %>% mutate(experiment = 'Architecture'),
      model_uber_mds[[x]] %>%
        left_join(main_metadata %>% 
                    mutate(group = case_when(
                      str_detect(model_string, 'taskonomy') ~ 'Taskonomy',
                      !is.na(compare_goal_slip) ~ compare_goal_slip,
                      #!is.na(compare_goal_contrastive) ~ paste0(compare_goal_contrastive, '-SSL'),
                      compare_goal_contrastive == 'Contrastive' ~ paste0('C-SSL'),
                      compare_goal_contrastive == 'Non-Contrastive' ~ paste0('NC-SSL'),
                      TRUE ~ NA
                    )) %>% select(model_string, group)) %>% mutate(experiment = 'Task'),
      model_uber_mds[[x]] %>%
        left_join(main_metadata %>%
                    mutate(group = case_when(
                      !is.na(compare_diet_ipcl) ~ paste0('IPCL-', compare_diet_ipcl),
                      !is.na(compare_diet_imagenetsize) ~ compare_diet_imagenetsize,
                      TRUE ~ NA,
                    )) %>% select(model_string, group)) %>% mutate(experiment = 'Input')) %>% 
      mutate(metric = str_to_upper(x)) %>%
      mutate(group = str_replace(group, 'imagenet', 'ImageNet1K'),
             group = str_replace(group, 'ImageNet1K21k', 'ImageNet21K'),
             group = str_replace(group, 'openimages', 'OpenImages'),
             group = str_replace(group, 'vggface2', 'VGGFace2'),
             group = str_replace(group, 'places256', 'Places256')) %>%
      mutate(group = str_replace(group, 'IPCL-ImageNet1K', 'Objects(1)'),
             group = str_replace(group, 'IPCL-OpenImages', 'Objects'),
             group = str_replace(group, 'IPCL-Places256', 'Places'),
             group = str_replace(group, 'IPCL-VGGFace2', 'Faces')) %>%
      #mutate(group = str_replace(group, 'ImageNet1K', 'IN1K'),
      #       group = str_replace(group, 'ImageNet21K', 'IN21K')) %>%
      mutate(group = factor(group, levels = group_levels)) %>%
      mutate(experiment = factor(experiment, levels = c('Architecture','Task','Input'))) %>%
      mutate(focus = ifelse(is.na(group), 'Out-of-Focus','In-Focus'))
  }) %>% bind_rows()

# write.csv(plot_data, 'figure_data/Figure6B.csv', row.names=FALSE)

ggplot(plot_data, aes(x = V1, y = V2, fill = group, color = group, alpha = focus)) +
  facet_grid(metric~experiment) + geom_point(cex = 3, shape = 21) + theme_classic() +
  scale_fill_manual(values = palette) + 
  scale_color_manual(values = palette) + 
  scale_alpha_manual(values = c(1.0, 0.2)) +
  geom_mark_hull(alpha = 0.2, concavity = 10, expand = unit(2.5, 'mm'),
                 data = . %>% filter(!is.na(group))) +
  geom_label_repel(aes(label = group), fill = 'white', alpha = 0.65,
                   direction = 'both', size = 7, seed = 0,
                   force = 100, force_pull = 10,
                   data = . %>% filter(!is.na(group)) %>%
                     filter(group != 'Objects(1)') %>%
                     group_by(metric, group, focus, experiment) %>% 
                     summarise(V1 = median(V1), V2 = median(V2))) +
  geom_label_repel(aes(label = group), fill = NA, alpha = 1,
                   direction = 'both', size = 7, seed = 0,
                   force = 100, force_pull = 10,
                   data = . %>% filter(!is.na(group)) %>%
                     filter(group != 'Objects(1)') %>%
                     group_by(metric, group, focus, experiment) %>% 
                     summarise(V1 = median(V1), V2 = median(V2))) +
  easy_remove_axes() + easy_remove_legend() +
  scale_x_continuous(expand = c(0.1, 0.1), limits = c(-0.3, 0.6)) + 
  scale_y_continuous(expand = c(0.1, 0.1), limits = c(-0.5, 0.4)) + 
  theme(text = element_text(size = 20),
        panel.border = element_rect(fill = 'transparent'),
        strip.background = element_blank(),
        #strip.text.x = element_blank(),
        panel.spacing = unit(3, "lines"),
        plot.margin = margin(1,1,1,1, "cm")) +
  labs(color = element_blank(), fill = element_blank(), alpha = element_blank()) 


plot_list[[2]] <- last_plot()

cowplot::plot_grid(plot_list[[1]] + theme(strip.text.x = element_blank()) +
                     theme(plot.margin = margin(t=0.1,l=0.1,r=0.35,b=0.1, 'cm')) +
                     easy_remove_y_axis(what = 'title'), 
                   plot_list[[2]] + theme(panel.spacing = unit(1, "lines")) +
                     theme(plot.margin = margin(t=0.1,l=0.1,r=0.1,b=0.1, 'cm')), 
                   ncol = 2, align = 'vh', axis = 'bt', rel_widths = c(0.25, 0.75))

figure_list[['figure6']] <- last_plot()

figure_dims <- list(width_inches = 8.5, 
                    height_inches = 11, 
                    aspect_ratio = 11 / 8.5,
                    width_units = 8.5 / 0.0254, 
                    height_units = 11 / 0.0254)

output_file <- 'figure_svgs/figure6-plot.svg'
output_file <- 'figure_svgs/figure5-plot.svg'
if (!file.exists(output_file) | overwrite_saved) {
  ggsave(output_file, width=1178, height=632, units='px')
}


