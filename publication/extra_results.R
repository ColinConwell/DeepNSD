# To run this code, first run the main script `main_results.R` (at least to the data load)

# ··············································································
## Extra Analyses --------------------------------------------------------------

# ··············································································
###* Effect of Mapping Method --------------------------------------------------

temp_data <- results$max %>% filter(region == 'OTC') %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  mutate(metric = factor(metric, levels = c('crsa','wrsa')))

m1 <- lmer(score ~ metric + (metric | model_string),
           data = temp_data)

m2 <- lmer(score ~ metric + (metric | model_string),
           data = temp_data %>% filter(!str_detect(model_string, '_random')))

tab_model(m1, m2, dv.labels = c('all_models', 'trained_models'))

# statistical visualization of effect of training:
results$summary %>% filter(!str_detect(model_string, '_random')) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  mutate(metric = str_to_title(metric)) %>%
  ggwithinstats(x = metric, y = score,
                ggplot.component = common_coords)

# ··············································································
###* Effect of Diet+Task: ResNet50 ---------------------------------------------

comparisons <- c('class_object_taskonomy','resnet50_classification')
results$max %>% filter(model_string %in% comparisons) %>%
  filter(region == 'OTC', metric == 'wrsa') %T>%
  {print(group_by(., model_string) %>% 
           summarise(n = n(), score = mean_cl_boot(score)))} %>%
  t_test(score ~ model_string, detailed = TRUE)

results$summary %>% filter(!is.na(compare_architecture)) %>%
  select(model_string, metric, score) %>% 
  group_by(metric) %>%
  mutate(rank = dense_rank(-score)) %>%
  filter(rank == min(rank) | rank == max(rank))

contrast_levels <- c('denoising_taskonomy', 
                     'class_object_taskonomy',
                     'ResNet50-RotNet_selfsupervised', 
                     'ResNet50-SimCLR_selfsupervised',
                     'resnet50_classification')

temp_data <- results$max %>% 
  filter(!is.na(compare_goal_taskonomy_tasks)) %>%
  mutate(task_variation = 'Taskonomy',
         input_variation = 'Taskonomy') %>%
  filter(str_detect(model_string, 'denoising|class_object')) %>%
  bind_rows(results$max %>%
              mutate(task_variation = 'Self-Supervised',
                     input_variation = 'ImageNet1K') %>%
              filter(str_detect(model_string, 'RotNet|SimCLR'),
                     !str_detect(model_string, 'ClusterFit|_slip'))) %>%
  bind_rows(results$max %>% 
              filter(model == 'resnet50', 
                     train_task == 'classification') %>%
              mutate(task_variation = 'Category-Supervised',
                     input_variation = 'ImageNet1K')) %>% 
  select(model_string, input_variation, task_variation, 
         region, subj_id, metric, score) %>%
  mutate_at(vars(task_variation, input_variation, model_string), as.factor) %>%
  mutate(model_string = factor(model_string, levels = contrast_levels)) %>%
  mutate(task_variation = relevel(task_variation, ref = 'Taskonomy'),
         input_variation = relevel(input_variation, ref = 'ImageNet1K'),
         model_string = relevel(model_string, ref = 'class_object_taskonomy')) %>%
  filter(metric %in% c('crsa','wrsa', 'srpr'), region == 'OTC') 

temp_data %>% distinct(model_string)

temp_data <- mutate_at(temp_data, vars(model_string),
                       relevel, ref = 'ResNet50-RotNet_selfsupervised') %>%
  left_join(noise_ceilings$subj_avg %>% select(subj_id, y) %>%
              mutate(subj_id = as.factor(subj_id)) %>% rename(noise_ceiling = y)) %>%
  mutate(ev_score = (score ** 2) / (noise_ceiling ** 2)) %>%
  mutate(score = score)

m1 <- lm(score ~ model_string + subj_id,
         data = temp_data %>% filter(metric == 'crsa'))

m2 <- lm(score ~ model_string + subj_id,
         data = temp_data %>% filter(metric == 'wrsa'))

m3 <- lm(score ~ model_string + subj_id,
         data = temp_data %>% filter(metric == 'srpr'))

tab_model(m1, m2, dv.labels = c('crsa','wrsa'))

# ··············································································
###* Effect of Diet+Task: BiT-Expert -------------------------------------------

results$summary %>% filter(!is.na(compare_goal_expertise)) %>%
  group_by(model_string, metric) %>%
  summarise(score = mean(score)) %>% 
  arrange(metric, -score) %>% print(n = 33)

results$summary %>% filter(!is.na(compare_goal_expertise)) %>%
  group_by(model_string, metric) %>%
  summarise(score = mean(score)) %>% 
  group_by(metric) %>%
  summarise(score = mean_cl_boot(score))

temp_data <- results$max %>% filter(region == 'OTC') %>%
  filter(!is.na(compare_goal_expertise))

temp_data %>% pull(model) %>% unique() %>% length()
temp_data %>% dim()
11 * 4 * 3

m1 <- lm(score ~ compare_goal_expertise + subj_id,
         data = temp_data %>% filter(metric == 'crsa') %>%
           mutate_at(vars(compare_goal_expertise), relevel, ref = 'Flower'))

m2 <- lm(score ~ compare_goal_expertise + subj_id,
         data = temp_data %>% filter(metric == 'wrsa') %>%
           mutate_at(vars(compare_goal_expertise), relevel, ref = 'Food'))

m3 <- lm(score ~ compare_goal_expertise + subj_id,
         data = temp_data %>% filter(metric == 'srpr') %>%
           mutate_at(vars(compare_goal_expertise), relevel, ref = 'Bird'))

tab_model(m1, m2, m3, digits = 3)

coef(m1) %>% mean_cl_boot()
coef(m2) %>% mean_cl_boot()
coef(m3) %>% mean_cl_boot()

comparisons <- c('BiT-Expert-ResNet-V2-Abstraction_bit_expert', 
                 'BiT-Expert-ResNet-V2-Flower_bit_expert')

temp_data %>% filter(model_string %in% comparisons) %>%
  select(model_string, metric, subj_id, score) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  pivot_wider(names_from = model_string, 
              values_from = score, values_fn = mean) %>%
  mutate(difference = !!sym(comparisons[[1]]) - !!sym(comparisons[[2]])) %>%
  group_by(metric) %>%
  summarise(difference = mean_cl_boot(difference))

# ··············································································
###* Effects Combined ----------------------------------------------------

results$max %>% pull(compare_architecture) %>% levels()
results$max %>% pull(compare_training) %>% levels()
results$max %>% pull(compare_goal_contrastive) %>% levels()
results$max %>% pull(compare_goal_taskonomy_tasks) %>% levels()
results$max %>% pull(compare_diet_ipcl) %>% levels()
results$max %>% pull(compare_diet_imagenetsize) %>% levels()
results$max %>% pull(compare_goal_selfsupervised) %>% levels()
results$max %>% pull(compare_goal_slip) %>% levels()

ref_factors <- c(architecture = 'Convolutional',
                 training = 'random',
                 diet_ipcl = 'imagenet',
                 resnet50 = 'imagenet1k',
                 goal_contrastive = 'Non-Contrastive',
                 goal_taskonomy_tasks = 'denoising',
                 diet_imagenetsize = 'imagenet',
                 goal_selfsupervised = 'RotNet',
                 goal_slip = 'SimCLR')

results$max %>% filter(metric %in% c('wrsa')) %>%
  filter(region == 'OTC') %>%
  pivot_longer(starts_with('compare'), names_to = 'contrast_set',
               values_to = 'contrast', names_prefix = 'compare_') %>%
  filter(str_detect(contrast_set, 'architecture')) %>% 
  filter(!is.na(contrast)) %$%
  lm(score ~ contrast + subj_id) %>% 
  {bind_cols(coef(.) %>% as.data.frame(),
             confint(.)) %>% rownames_to_column('effect') %>%
      set_names('effect', 'beta','lower_ci','upper_ci')}


results$max %>% filter(metric %in% c('wrsa')) %>%
  filter(region == 'OTC') %>%
  mutate(compare_resnet50 = 
           case_when(model_string == 'resnet50_classification' ~ 'imagenet1k',
                     model_string == 'class_object_taskonomy' ~ 'taskonomy', TRUE ~ NA)) %>%
  pivot_longer(starts_with('compare'), names_to = 'contrast_set',
               values_to = 'contrast', names_prefix = 'compare_') %>%
  filter(!is.na(contrast)) %>%
  select(contrast_set, contrast) %>% unique() %>% 
  arrange(contrast_set) %>% print(n = 72)

target_betas <- c('Convolutional--Transformer' = 'Architecture: CNN vs Transformer',
                  'Non-Contrastive--Contrastive' = 'Contrastive vs Non-Contrastive SSL',
                  'SimCLR--SLIP' = 'Language Alignment: SimCLR vs SLIP',
                  'denoising--class_object' = 'Taskonomy: Best vs Worst',
                  'imagenet1k--taskonomy' = 'ResNet50: Taskonomy vs ImageNet1K',
                  'taskonomy--average' = 'Taskonomy: Average',
                  'imagenet--imagenet21k' = 'Diet: ImageNet1K vs ImageNet21K',
                  'imagenet--places256' = 'Diet: ImageNet1K vs Places265',
                  'imagenet--vggface2' = 'Diet: ImageNet1K vs VGGFace2',
                  'random--classification' = 'Weights: Random vs Pretrained')

get_contrast_betas <- function(data, reference) {
  data$contrast <- relevel(factor(data$contrast), ref = reference)
  lm(score ~ contrast + subj_id, data = data) %>% 
    {bind_cols(coef(.) %>% as.data.frame(),
               confint(.)) %>% rownames_to_column('effect') %>%
        set_names('term', 'beta','lower_ci','upper_ci')}
}

results$max %>% filter(metric %in% c('crsa','wrsa')) %>%
  filter(region == 'OTC') %>%
  mutate(compare_resnet50 = 
           case_when(model_string == 'resnet50_classification' ~ 'imagenet1k',
                     model_string == 'class_object_taskonomy' ~ 'taskonomy', TRUE ~ NA)) %>%
  pivot_longer(starts_with('compare'), names_to = 'contrast_set',
               values_to = 'contrast', names_prefix = 'compare_') %>%
  filter(!str_detect(contrast_set, 'cluster|expertise|selfsupervised')) %>%
  #filter(str_detect(contrast_set, 'architecture')) %>% 
  filter(!is.na(contrast)) %>%
  nest_by(region, metric, contrast_set) %>%
  mutate(reference = ref_factors[contrast_set],
         model = list(get_contrast_betas(data, reference))) %>%
  select(-data) %>% unnest(model) %>%
  filter(!str_detect(term, 'Intercept|subj_id')) %>%
  mutate(term = str_replace(term, 'contrast', '')) %>%
  mutate(effect = paste(reference, term, sep = '--'),
         .before = beta) %>%
  select(-reference, -term) %>%
  bind_rows(., filter(., str_detect(contrast_set, 'taskonomy')) %>%
              group_by(region, metric, contrast_set) %>%
              summarise(beta = list(mean_cl_normal(beta))) %>%
              mutate(effect = 'taskonomy--average') %>%
              unnest(beta) %>% rename(beta = y, lower_ci = ymin, upper_ci = ymax)) %>%
  filter(effect %in% names(target_betas)) %>% ungroup() %>%
  mutate(effect = fct_recode(as.factor(effect), !!!list_reverse(target_betas))) %>%
  mutate(metric = factor(metric, levels = c('crsa','wrsa'))) %>%
  mutate_at(vars(beta, lower_ci, upper_ci), abs) %>%
  ggplot(aes(x = reorder(effect, beta), y = beta, fill = metric)) +
  facet_wrap(~ metric) +
  scale_fill_manual(values = c('white','grey')) +
  geom_col(position = position_dodge(width = 0.9), width = 0.9, color = 'black') +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0.01,
                position = position_dodge(width = 0.9)) +
  scale_y_continuous(breaks = seq(0.1, 0.5, 0.1)) + 
  labs(x = element_blank(), y = 'Effect Size (Beta)',
       fill = element_blank()) + coord_flip() + theme_bw() +
  theme(text = element_text(size = 20)) + guides(fill = 'none')

# ··············································································
###* Effects across Regions ----------------------------------------------------

otc_divisions <- c('OFA','FFA-1','FFA-2','OPA','PPA','RSC',
                   'FBA-1','FBA-2','OWFA','VWFA-1','VWFA-2')

# rank-order correlation between models in OTC subdivisions:
results$max %>% select(model_string, metric, subj_id, region, score) %>%
  filter(model_string %in% model_sets$upper) %>%
  filter(region %in% c('OTC', otc_divisions),
         metric %in% c('crsa', 'wrsa')) %>%
  pivot_wider(names_from = c(metric, region,subj_id),
              values_from = score) %>%
  select(-model_string) %>% drop_na() %>%
  cor(method = 'spearman') %>%
  set_rownames(colnames(.)) %>%
  replace_upper_triangle(NA) %>%
  pivot_longer(-rowname, names_to = 'var2') %>%
  rename(var1 = rowname) %>% drop_na() %>%
  separate(var1, into = c('metric1','region1','subj_id1'), sep = '_') %>%
  separate(var2, into = c('metric2','region2','subj_id2'), sep = '_') %>%
  filter(metric1 == metric2, subj_id1 == subj_id2) %>%
  filter(region1 == 'OTC' | region2 == 'OTC') %>%
  filter(region1 %in% c('OTC', otc_divisions),
         region2 %in% c('OTC', otc_divisions)) %>%
  group_by(metric1, subj_id1) %>%
  summarise(n = n(), value = mean(value)) %>%
  group_by(metric1) %>%
  summarise(n = n(), min = min(value),
            cor = mean_cl_boot(value))

###** Plot: All Effects, All ROIs ----------------------------------------------

ref_factors <- c(architecture = 'Convolutional',
                 training = 'random',
                 diet_ipcl = 'imagenet',
                 resnet50 = 'imagenet1k',
                 goal_contrastive = 'Non-Contrastive',
                 goal_taskonomy_tasks = 'denoising',
                 diet_imagenetsize = 'imagenet',
                 goal_selfsupervised = 'RotNet',
                 goal_slip = 'SimCLR')


target_betas <- c('Convolutional--Transformer' = 'Architecture: CNN to Transformer',
                  'Non-Contrastive--Contrastive' = 'Task: Non-Contrastive to Contrastive SSL',
                  'RotNet--BarlowTwins-BS2048' = 'Task (Self-Supervision): RotNet to BarlowTwins',
                  'SimCLR--SLIP' = 'Task (Language Alignment): SimCLR to SLIP',
                  'denoising--class_object' = 'Taskonomy: Denoising to Object Classification',
                  'imagenet--imagenet21k' = 'Diet: ImageNet1K to ImageNet21K',
                  'imagenet--places256' = 'Diet: Objects to Places',
                  'imagenet--vggface2' = 'Diet: Objects to Faces')

regions_to_test <- results$max %>% distinct(region, subj_id) %>% 
  filter(!str_detect(region, 'FBA|early|lateral|ventral')) %>%
  group_by(region) %>% tally() %>% filter(n == 4) %>% pull(region) %>% unique()

get_contrast_betas <- function(data, reference) {
  data$contrast <- relevel(data$contrast, ref = reference)
  data <- select(data, score, contrast, subj_id)
  lm(score ~ contrast + subj_id, data = data) %>% tidy()
}

contrast_betas <- results$max %>% filter(metric %in% c('crsa','wrsa')) %>%
  filter(region %in% regions_to_test) %>%
  pivot_longer(starts_with('compare'), names_to = 'contrast_set',
               values_to = 'contrast', names_prefix = 'compare_') %>%
  filter(!str_detect(contrast_set, 'cluster|expertise|selfsupervised')) %>%
  filter(!is.na(contrast)) %>%
  nest_by(region, metric, contrast_set) %>%
  mutate(reference = ref_factors[contrast_set],
         model = list(get_contrast_betas(data, reference))) %>%
  select(-data) %>% unnest(model) %>%
  filter(!str_detect(term, 'Intercept|subj_id')) %>%
  select(-std.error, -statistic) %>%
  mutate(term = str_replace(term, 'contrast', '')) %>%
  mutate(effect = paste(reference, term, sep = '--'),
         .before = estimate)

contrast_betas %>% 
  filter(metric %in% c('crsa','wrsa'),
         region %in% c('PPA','FFA-1'),
         effect %in% names(target_betas)) %>%
  mutate(estimate = format(estimate, scientific=F)) %>% print(n = 30)

contrast_betas %>% filter(effect %in% names(target_betas)) %>% ungroup() %>%
  mutate(effect = factor(effect, levels = names(target_betas)),
         effect = fct_recode(as.factor(effect), !!!list_reverse(target_betas))) %>%
  mutate(region = factor(region, levels = region_hierarchy)) %>%
  mutate(metric = str_to_upper(metric),
         metric = factor(metric, levels = c('WRSA','CRSA'))) %>%
  #filter(region %in% c('EVC','OTC')) %>%
  ggplot(aes(x = region, y = effect, color = estimate,
             size = abs(estimate), shape = metric)) +
  scale_shape_manual(values = c(21,19)) +
  scale_color_gradient2(low="red", mid="gray",
                        high="blue", space ="Lab" ) +
  #facet_wrap(~metric, ncol = 1, strip.position="right") + geom_point() +
  geom_point(color = '#EBEBEB', shape = 3, size = 2, alpha = 1) + 
  geom_point(data = . %>% filter(metric == 'CRSA'), 
             position = position_nudge(y = 0.2)) + 
  geom_point(data = . %>% filter(metric == 'WRSA'),
             position = position_nudge(y = -0.2)) + 
  labs(y = element_blank(), x = element_blank(),
       color = 'Beta', shape = element_blank()) + guides(size = 'none') + 
  scale_y_discrete(limits=rev) + theme_bw() +
  theme(text = element_text(size = 13),
        panel.grid.major.x = element_blank(),
        strip.background = element_rect(fill = 'white'))

# ··············································································
###* Hierarchy (Depth) Statistics ----------------------------------------------

region_hierarchy <- c('V1v','V1d','V2v','V2d','V3v','V3d','hV4','EVC',
                      'OFA', 'FFA-1', 'FFA-2', 'OPA', 'PPA', 'RSC',
                      'EBA','FBA-1','FBA-2', 'OWFA', 'VWFA-1','VWFA-2',
                      'OTC','early','lateral','ventral')

test_hierarchy <- c('V1v','V1d','V2v','V2d','V3v','V3d','hV4','OTC')
region_depths <- c(1, 1, 2, 2, 3, 3, 4, 5)
region_levels <- test_hierarchy %>% set_names(region_depths)

# A visualization of mean layer depth across models:
results$max %>% filter(region %in% test_hierarchy) %>%
  filter(!str_detect(model_string, 'random|taskonomy|ipcl')) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  mutate(metric = str_to_upper(metric)) %>%
  group_by(model, train_task, region, metric) %>%
  summarise(model_layer_depth = mean(model_layer_depth)) %>%
  mutate(region = factor(region, levels = test_hierarchy)) %>%
  ggplot(aes(x = region, y = model_layer_depth)) + theme_bw() + 
  facet_wrap(~metric, scales = 'free', ncol = 1,
             strip.position = 'right') + 
  geom_point(alpha = 0.4, size = 2, stroke = 0) +
  stat_summary(fun.data = median_cl_boot, geom = 'crossbar', 
               fill = 'white', alpha = 0.7, width = 0.275) +
  stat_summary(fun = median, geom = 'point', size = 3, color = 'darkred') +
  stat_summary(aes(group = 1), fun = median, geom = 'line', color = 'darkred') +
  stat_summary(aes(label = paste('widehat(mu)==', round(..y.., 2))),
               fun = median, geom = GeomLabelRepel, parse = TRUE, size = 3,
               position = position_nudge_repel(x = 0.5, y = 0),
               segment.color = 'black',
               min.segment.length = 0, segment.linetype = 3) +
  labs(x = element_blank(), y= 'Model Layer Depth') + easy_remove_legend() +
  scale_x_discrete(expand = c(0,1.5)) +
  theme(text = element_text(size = 20),
        #strip.text.x = element_blank(),
        strip.background = element_rect(fill = 'white'),
        panel.spacing = unit(1, "lines"))

# pairwise comparisons in depth (layer) between regions:
results$max %>% filter(region %in% test_hierarchy) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  mutate(metric = str_to_upper(metric)) %>%
  filter(compare_architecture %in% c('Convolutional','Transformer')) %>%
  group_by(model_string, compare_architecture, region, metric) %>%
  summarise(n = n(), model_layer_depth = median(model_layer_depth)) %>%
  mutate(region = factor(region, levels = test_hierarchy),
         region = fct_recode(region, !!!region_levels)) %>%
  group_by(metric, compare_architecture) %>%
  t_test(model_layer_depth ~ region, detailed = FALSE)

# Significance matrix of the mean comparisons between layers:
results$max %>% filter(region %in% test_hierarchy) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  mutate(metric = str_to_upper(metric)) %>%
  group_by(model_string, region, metric) %>%
  filter(!str_detect(model_string, 'random')) %>%
  summarise(n = n(), model_layer_depth = mean(model_layer_depth)) %>%
  mutate(region = factor(region, levels = test_hierarchy)) %>%
  group_by(metric) %>%
  t_test(model_layer_depth ~ region, detailed = TRUE) %>%
  select(metric, group1, group2, p.adj.signif) %>%
  filter(metric == 'WRSA') %>%
  pivot_wider(names_from = group2, values_from = p.adj.signif)

# Hierarchy across distinct model subsets (i.e. architecture):

contrast_levels <- c('Convolutional','Transformer',
                     'Language|Self-Supervised')
contrast_levels <- c('Convolutional','Transformer')

results$max %>% filter(region %in% test_hierarchy) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  mutate(metric = str_to_upper(metric)) %>%
  group_by(model, train_task, model_class, region, metric) %>%
  summarise(model_layer_depth = mean(model_layer_depth)) %>%
  mutate(region = factor(region, levels = test_hierarchy)) %>%
  filter(train_task %in% c('classification','selfsupervised',
                           'clip','SLIP','SimCLR')) %>%
  filter(model_class %in% c('Convolutional','Transformer')) %>%
  mutate(contrast = ifelse(str_detect(train_task, 'class'),
                           model_class, 'Language|Self-Supervised')) %>%
  filter(contrast %in% contrast_levels) %>%
  mutate(contrast = factor(contrast, levels = contrast_levels)) %>%
  ggplot(aes(x = region, y = model_layer_depth)) + theme_bw() + 
  facet_grid(metric~contrast) + 
  geom_point(alpha = 0.4, size = 2, stroke = 0) +
  stat_summary(fun.data = median_cl_boot, geom = 'crossbar', 
               fill = 'white', alpha = 0.7, width = 0.275) +
  stat_summary(fun = median, geom = 'point', size = 3, color = 'darkred') +
  stat_summary(aes(group = 1), fun = median, geom = 'line', color = 'darkred') +
  labs(x = element_blank(), y= 'Model Layer Depth') + easy_remove_legend() +
  theme(text = element_text(size = 20),
        #strip.text.x = element_blank(),
        strip.background = element_rect(fill = 'white'))

# Hierarchy across metrics (side by side):
results$max %>% filter(region %in% test_hierarchy) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  mutate(metric = str_to_upper(metric)) %>%
  group_by(model, train_task, model_class, region, metric) %>%
  summarise(model_layer_depth = mean(model_layer_depth)) %>%
  mutate(region = factor(region, levels = test_hierarchy)) %>%
  filter(train_task %in% c('classification','selfsupervised',
                           'clip','SLIP','SimCLR')) %>%
  filter(model_class %in% c('Convolutional','Transformer')) %>%
  mutate(contrast = ifelse(str_detect(train_task, 'class'),
                           model_class, 'Language|Self-Supervised')) %>%
  filter(contrast %in% contrast_levels) %>%
  mutate(contrast = factor(contrast, levels = contrast_levels)) %>%
  ggplot(aes(x = region, y = model_layer_depth, fill = metric)) + 
  theme_bw() + facet_grid(~contrast) + 
  geom_point(alpha = 0.4, size = 2, stroke = 0,
             position = position_dodge(width = 0.9)) +
  stat_summary(fun.data = median_cl_boot, geom = 'crossbar', 
               alpha = 0.7, width = 0.275, show.legend = FALSE,
               position = position_dodge(width = 0.9)) +
  stat_summary(fun = median, geom = 'point', size = 3, 
               show.legend = FALSE,
               position = position_dodge(width = 0.9)) +
  stat_summary(aes(group = metric, linetype = metric), 
               fun = median, geom = 'line', #show.legend = FALSE,
               position = position_dodge(width = 0.9)) +
  scale_fill_manual(values = c('white','gray')) + 
  labs(x = element_blank(), y= 'Model Layer Depth') + #easy_remove_legend() +
  theme(text = element_text(size = 20),
        #strip.text.x = element_blank(),
        strip.background = element_rect(fill = 'white'))