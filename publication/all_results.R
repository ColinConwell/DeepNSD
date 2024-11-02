if (!require(pacman)) {install.packages("pacman")}
pacman::p_load('this.path','arrow','magrittr','glue', 'scales', 'aqm',
               'lme4', 'MASS', 'segmented', 'umap', 'Rtsne', 'rstatix',
               'xtable','janitor','officer', 'gt', 'lemon', 'sjPlot',
               'colorspace', 'ggeasy', 'ggforce', 'ggh4x', 'ggpubr',
               'ggstance', 'ggrepel','ggstatsplot', 'ggtext', 'tidytext',
               'rsample','broom','broom.mixed','modelsummary','tidyverse')

# set path to current file
setwd(dirname(this.path()))

# ensure correct select
select <- dplyr::select

# Helper Functions -------------------------------------------------------------

get_package_bibtex <- function(x) {print(x, bibtex = TRUE)}

common_coords <- coord_cartesian(ylim = c(0,1))

list_reverse <- function(list) 
  {new_list <- names(list); names(new_list) = list; return(new_list)}

get_package_versions <- function(dataframe=TRUE) {
  session_info <- sessionInfo() # current
  
  packages <- session_info$otherPkgs
  versions <- sapply(packages, function(x) x$Version)
  print('Package Versions:'); print(versions)
  
  if (dataframe) {
    return(data.frame(version = versions) %>% 
             rownames_to_column('package'))}
}

pkg_versions <- get_package_versions() # all imported packages
write.csv(pkg_versions, 'environment/R-Packages.csv', row.names = FALSE)

# Load Primary Results ---------------------------------------------------------

results <- list(complete = read_parquet(glue('source_data/results_complete.parquet')),
                max = read_csv(glue('source_data/results_max.csv')),
                summary = read_csv(glue('source_data/results_summary.csv'))) %>%
  lapply(function(x) {x %<>% mutate_at(vars(contains('subj_id'), starts_with('compare')), as.factor)}) %>%
  lapply(function(x) {x %<>% mutate(model_string = ifelse(str_detect(model, 'ipcl'), model, model_string))})

# results check: all unique models should have 12 entries: 4 subjects x 3 metrics
# this filter should return empty if TRUE:
results$max %>% filter(region == 'OTC') %>% 
  drop_na(model_string) %>% group_by(model_string) %>% tally() %>% filter(n != 12)

# group models by rank for use in later analyses
results$summary %<>% group_by(metric) %>%
  mutate(rank = dense_rank(-score)) %>% ungroup()

results$summary %>% arrange(rank, metric) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  select(model_string, metric, score, rank)

voxel_metadata <- read_csv('source_data/voxel_metadata_EVC.csv') %>%
  bind_rows(read_csv('source_data/voxel_metadata_OTC.csv'))

noise_ceilings <- list(subj_avg = read.csv('source_data/noise_ceilings.csv') %>%
                         group_by(subj_id) %>%
                         summarise(score_ci = list(mean_cl_boot(value))) %>% 
                                     unnest(score_ci))

noise_ceilings$group_avg <- noise_ceilings$subj_avg %>% 
  summarise(score_ci = list(mean_cl_boot(y))) %>% unnest(score_ci)

main_metadata <- results$summary %>%
  select(model_string, architecture, train_task, train_data, 
         starts_with('compare')) %>% distinct()

# create sets of high-performing and low-performing models
# (see Rank Breakpoints subsection below for justification)

model_sets = list()

model_sets$upper <- results$summary %>% 
  filter(metric == 'wrsa') %>% filter(rank <= 124) %>% pull(model_string)

model_sets$lower <- results$summary %>% 
  filter(metric == 'wrsa') %>% filter(rank > 124) %>% pull(model_string)

# ··············································································
## Counts (Regressions + Models) -----------------------------------------------

# number of unique models
results$complete %>% select(model_string) %>% distinct() %>% nrow()

# number of unique trained models
results$complete %>% filter(train_task != 'random') %>%
  select(model_string) %>% distinct() %>% print(n = 200)

# number of unique regressions run
voxel_count <- voxel_metadata %>% nrow()
((results$complete %>% select(model, train_task, train_data, model_layer) %>% 
    distinct() %>% nrow()) * voxel_count * 2) %>% # Train / Test
  pluck(1) %>% {prettyNum(., big.mark = ',')} %>% paste('total regressions run')

((results$max %>% select(model, train_task, train_data, model_layer, subj_id, region) %>% 
    distinct() %>% nrow())) %>% 
  pluck(1) %>% {prettyNum(., big.mark = ',')} %>% paste('total RS analyses run')

# voxel counts by region + subject
voxel_metadata %>% 
  pivot_longer(-c(voxel_id, subj_id, ncsnr),
               names_to = 'region', values_to = 'present') %>%
  filter(present == 1) %>% select(-present) %>%
  group_by(region, subj_id) %>% summarise(count = n()) %>% print(n = 120)

# voxel counts by region
voxel_metadata %>% 
  pivot_longer(-c(voxel_id, subj_id, ncsnr),
               names_to = 'region', values_to = 'inclusion') %>%
  filter(inclusion == 1) %>% select(-inclusion) %>%
  group_by(region) %>% summarise(count = n()) %>% print(n = 120)

# ··············································································
## Opportunistic Experiments ---------------------------------------------------
###* Effect of Architecture ----------------------------------------------------

temp_data <- results$max %>% filter(region == 'OTC') %>%
  rename(contrast = compare_architecture) %>%
  filter(!is.na(contrast)) %>%
  filter(contrast %in% c('Convolutional','Transformer'))
  
temp_data %>% select(model, train_task, train_data, contrast) %>% distinct() %>%
  group_by(train_task, train_data, contrast) %>% tally()

temp_data %>% dim()
(34 + 21) * 4 * 3

m1 <- lm(score ~ contrast + subj_id, 
         data = temp_data %>% filter(metric == 'crsa'))

m2 <- lm(score ~ contrast + subj_id, 
         data = temp_data %>% filter(metric == 'wrsa'))

m3 <- lm(score ~ contrast + subj_id, 
         data = temp_data %>% filter(metric == 'srpr'))

parameters::model_parameters(m1)

tab_model(m1, m2, m3, dv.labels = c('crsa', 'wrsa', 'srpr'))

temp_data %>% group_by(model_string, metric, contrast) %>%
  summarise(score = mean(score)) %>%
  group_by(metric, contrast) %>%
  summarise(n = n(), score = mean_cl_boot(score))

temp_data %>% select(metric, subj_id, contrast, score) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  pivot_wider(names_from = contrast, 
              values_from = score, values_fn = mean) %>%
  mutate(Conv2ViT = Convolutional - Transformer) %>%
  group_by(metric) %>%
  summarise(Conv2ViT = mean_cl_boot(Conv2ViT))

# ··············································································
###* Effect of Task: Taskonomy -------------------------------------------------

temp_data <- results$max %>% filter(region == 'OTC') %>%
  filter(!is.na(compare_goal_taskonomy_tasks)) %>%
  mutate_at(vars(compare_goal_taskonomy_tasks), as.factor) %>%
  filter(compare_goal_taskonomy_tasks != 'random_weights') %>%
  mutate_at(vars(compare_goal_taskonomy_tasks), relevel, ref = 'denoising')

temp_data %>% dim()
24 * 4 * 3

m1 <- lm(score ~ compare_goal_taskonomy_tasks + subj_id,
         data = temp_data %>% filter(metric == 'crsa'))

m2 <- lm(score ~ compare_goal_taskonomy_tasks + subj_id,
         data = temp_data %>% filter(metric == 'wrsa'))

m3 <- lm(score ~ compare_goal_taskonomy_tasks + subj_id,
         data = temp_data %>% filter(metric == 'srpr'))

coef(m1) %>% mean_cl_boot()
coef(m2) %>% mean_cl_boot()
coef(m3) %>% mean_cl_boot()

tab_model(m1, m2, m3)

comparisons <- c('class_object_taskonomy', 'autoencoding_taskonomy')
temp_data %>% filter(model_string %in% comparisons) %>%
  select(model_string, metric, subj_id, score) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  group_by(model_string, metric) %>% 
  summarise(score = mean_cl_boot(score))

temp_data %>% filter(model_string %in% comparisons) %>%
  select(model_string, metric, subj_id, score) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  pivot_wider(names_from = model_string, 
              values_from = score, values_fn = mean) %>%
  mutate(difference = !!sym(comparisons[[1]]) - !!sym(comparisons[[2]])) %>%
  group_by(metric) %>%
  summarise(difference = mean_cl_boot(difference))

# ··············································································
###* Effect of Task: Self-Supervised -------------------------------------------

temp_data <- results$max %>% filter(region == 'OTC') %>% 
  rename(contrast = compare_goal_selfsupervised) %>%
  filter(!is.na(contrast) | 
           (model == 'resnet50' & train_task == 'classification')) %>%
  mutate_at(vars(model, contrast), function (x) 
  {ifelse(.$model == 'resnet50', 'Category-Supervised', as.character(x))}) %>%
  mutate_at(vars(contrast), as.factor) %>%
  mutate_at(vars(contrast), relevel, ref = 'Category-Supervised')

temp_data %>% group_by(contrast, metric) %>%
  summarise(score = mean_cl_boot(score))

temp_data %>% dim()
10 * 4 * 3

m1 <- lm(score ~ contrast + subj_id,
         data = temp_data %>% filter(metric == 'crsa'))

m2 <- lm(score ~ contrast + subj_id,
         data = temp_data %>% filter(metric == 'wrsa'))

m3 <- lm(score ~ contrast + subj_id,
         data = temp_data %>% filter(metric == 'srpr'))

coef(m1) %>% mean_cl_boot()
coef(m2) %>% mean_cl_boot()
coef(m3) %>% mean_cl_boot()

tab_model(m1,m2, m3, dv.labels = c('crsa','wrsa','srpr'))

temp_data %>% filter(metric == 'wrsa') %>%
  group_by(contrast) %>%
  summarise(n = n(), score = mean_cl_boot(score))

contrast_levels <- c('Non-Contrastive','Contrastive')
temp_data <- results$max %>% filter(region == 'OTC') %>%
  rename(contrast = compare_goal_contrastive) %>%
  filter(!is.na(contrast)) %>%
  mutate_at(vars(contrast), factor, levels = contrast_levels)

m1 <- lm(score ~ contrast + subj_id,
         data = temp_data %>% filter(metric == 'crsa'))

m2 <- lm(score ~ contrast + subj_id,
         data = temp_data %>% filter(metric == 'wrsa'))

m3 <- lm(score ~ contrast + subj_id,
         data = temp_data %>% filter(metric == 'srpr'))

tab_model(m1,m2, m3, dv.labels = c('crsa','wrsa','srpr'))

temp_data %>% select(metric, subj_id, contrast, score) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  pivot_wider(names_from = contrast, 
              values_from = score, values_fn = mean) %>%
  mutate(ContrastiveDiff = Contrastive - `Non-Contrastive`) %>%
  group_by(metric) %>%
  summarise(ContrastiveDiff = mean_cl_boot(ContrastiveDiff))

# ··············································································
###* Effect of Task: Language-Alignment ----------------------------------------

slip_levels <- c('SimCLR','CLIP','SLIP')
size_levels <- c('ViT-B','ViT-S','ViT-L')
temp_data <- results$max %>% filter(region == 'OTC') %>%
  filter(!is.na(compare_goal_slip)) %>%
  mutate(compare_goal_slip = factor(compare_goal_slip, levels = slip_levels),
         architecture = factor(architecture, levels = size_levels))

temp_data %>% dim()
9 * 4 * 3

m1 <- lm(score ~ compare_goal_slip*architecture  + subj_id,
   data = temp_data %>% filter(metric == 'crsa'))

m2 <- lm(score ~ compare_goal_slip*architecture  + subj_id,
         data = temp_data %>% filter(metric == 'wrsa'))

m3 <- lm(score ~ compare_goal_slip*architecture  + subj_id,
         data = temp_data %>% filter(metric == 'srpr'))

tab_model(m1,m2, m3, dv.labels = c('crsa','wrsa','srpr'))

m1b <- lm(score ~ subj_id,
         data = temp_data %>% filter(metric == 'crsa'))

m2b <- lm(score ~ subj_id,
         data = temp_data %>% filter(metric == 'wrsa'))

m3b <- lm(score ~ subj_id,
         data = temp_data %>% filter(metric == 'srpr'))

anova(m1b, m1)
anova(m2b, m2)
anova(m3b, m3)

temp_data %>% mutate(contrast = compare_goal_slip) %>%
  select(metric, subj_id, contrast, score) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  pivot_wider(names_from = contrast, 
              values_from = score, values_fn = mean) %>%
  mutate(SimCLR2SLIP = SLIP - SimCLR) %>%
  group_by(metric) %>%
  summarise(SimCLR2SLIP = mean_cl_boot(SimCLR2SLIP))

# ··············································································
###* Effect of Diet: Objects-Faces-Places -----------------------------------

ipcl_levels <- c('imagenet' = 'ImageNet', 'openimages' = 'OpenImages',
                 'places256' = 'Places256', 'vggface2' = 'VGGFace2') %>% list_reverse()

temp_data <- results$max %>% filter(region == 'OTC') %>% 
  filter(!is.na(compare_diet_ipcl)) %>%
  mutate_at(vars(compare_diet_ipcl), fct_recode, !!!ipcl_levels) %>%
  mutate_at(vars(compare_diet_ipcl), relevel, ref = 'ImageNet')
  
temp_data %>% dim()
4 * 4 * 3

m1 <- lm(score ~ compare_diet_ipcl + subj_id,
         data = temp_data %>% filter(metric == 'crsa'))

m2 <- lm(score ~ compare_diet_ipcl + subj_id,
         data = temp_data %>% filter(metric == 'wrsa'))

m3 <- lm(score ~ compare_diet_ipcl + subj_id,
         data = temp_data %>% filter(metric == 'srpr'))

tab_model(m1,m2, m3, dv.labels = c('crsa','wrsa','srpr'))

results$max %>% filter(str_detect(model_string, 'ipcl')) %>%
  pull(model_string) %>% unique()

comparisons <- c('alexnet_gn_ipcl_vggface2',
                 'alexnet_gn_ipcl_imagenet')
results$max %>% filter(model_string %in% comparisons) %>%
  select(model_string, metric, subj_id, score) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  pivot_wider(names_from = model_string, 
              values_from = score, values_fn = mean) %>%
  mutate(difference = !!sym(comparisons[[1]]) - !!sym(comparisons[[2]])) %>%
  group_by(metric) %>%
  summarise(difference = mean_cl_boot(difference))

# ··············································································
###* Effect of Diet: ImageNet1K vs 21K ------------------------------------------

temp_data <- results$max %>% filter(region == 'OTC') %>% 
  filter(!is.na(compare_diet_imagenetsize)) %>%
  mutate_at(vars(compare_diet_ipcl), relevel, ref = 'imagenet')

temp_data %>% pull(model) %>% unique() %>% length()
temp_data %>% dim()
24 * 4 * 3

m1 <- lmer(score ~ compare_diet_imagenetsize + subj_id + (1 | model),
          data = temp_data %>% filter(metric == 'crsa'))

m2 <- lmer(score ~ compare_diet_imagenetsize + subj_id + (1 | model),
          data = temp_data %>% filter(metric == 'wrsa'))

m3 <- lmer(score ~ compare_diet_imagenetsize + subj_id + (1 | model),
          data = temp_data %>% filter(metric == 'srpr')) 

parameters::model_parameters(m1)
tab_model(m1,m2, m3, dv.labels = c('crsa','wrsa','srpr'))

temp_data %>% mutate(contrast = compare_diet_imagenetsize) %>%
  select(metric, subj_id, contrast, score) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  pivot_wider(names_from = contrast, 
              values_from = score, values_fn = mean) %>%
  mutate(Imagenet1Kv21K = imagenet - imagenet21k) %>%
  group_by(metric) %>%
  summarise(Imagenet1Kv21K = mean_cl_boot(Imagenet1Kv21K))

###* Effect of Training --------------------------------------------------------

training_levels <-  c('random','classification')
temp_data <- results$max %>% filter(region == 'OTC') %>%
  filter(!is.na(compare_training)) %>%
  mutate(compare_training = relevel(compare_training, ref = 'random'))

temp_data %>% select(model, compare_training) %>% distinct() %>%
  group_by(compare_training) %>% tally()

temp_data %>% dim()
(64 * 2) * 4 * 3

m1 <- lmer(score ~ compare_training + subj_id + (compare_training | model), 
           data = temp_data %>% filter(metric == 'crsa'))

m2 <- lmer(score ~ compare_training + subj_id + (compare_training | model), 
           data = temp_data %>% filter(metric == 'wrsa'))

m3 <- lmer(score ~ compare_training + subj_id + (compare_training | model), 
           data = temp_data %>% filter(metric == 'srpr'))

tab_model(m1, m2, m3, dv.labels = c('crsa', 'wrsa', 'srpr'))

# extra: statistical visual of training effect:
results$summary %>% filter(!is.na(compare_training)) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  mutate(contrast = compare_training) %>%
  grouped_ggwithinstats(x = contrast, y = score,
                        grouping.var = metric,
                        ggplot.component = common_coords)

# ··············································································
## Experiment Figures: Setup ---------------------------------------------------

palette <- qualitative_hcl(16, palette = "Dark 3")

ev_score <- Vectorize(function(x) {round(x**2 / noise_ceilings$group_avg$y**2, 2)})
add_ev_axis <- sec_axis(~., breaks =  seq(0,0.8,0.1), 
                        labels = ev_score(seq(0,0.8,0.1)),
                        name = 'Variance Explained *<sub></sub>*')

zoom_plots <- list()

# ··············································································
###* Figure: Architecture ------------------------------------------------------

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

zoom_plots[[1]] <- last_plot()

# ggsave('saved_figures/architecture.jpg',
#        width = 10, height = 6.5, units = 'in')

# write.csv(temp_data, 'figure_data/Figure2.csv', row.names=FALSE)

# ··············································································
###* Figure: Task ------------------------------------------------------------

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

adjustments <- function(x) {mean(x) - (0.05 + mean(x) / 1.75)}
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
  stat_summary(aes(y = 0.025, label = display_name), fun = adjustments, 
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

zoom_plots[[2]] <- last_plot()

# ggsave('saved_figures/task.jpg',
#        width = 10, height = 6.25, units = 'in')

# write.csv(temp_data, 'figure_data/Figure3.csv', row.names=FALSE)

# ··············································································
###* Figure: Diet -------------------------------------------------------------

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

# write.csv(plot_data, 'figure_data/Figure4A.csv', row.names=FALSE)

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

# write.csv(plot_data, 'figure_data/Figure4B.csv', row.names=FALSE)

cowplot::plot_grid(plot_list[[1]] + theme(plot.margin = margin(1, 1, 2, 1, unit = 'cm')), 
                   plot_list[[2]] + #easy_remove_y_axis(teach=TRUE) + 
                     theme(axis.ticks.y.left = element_blank(), 
                           axis.title.y.left = element_blank(),
                           axis.text.y.left = element_blank(), 
                           axis.line.y.left = element_blank()) +
                     theme(plot.margin = margin(1, 1, 2, -0.75, unit = 'cm')),
                   NULL, rel_widths = c(0.5, 0.2, 0.3), nrow = 1)

zoom_plots[[3]] <- plot_list

# ggsave('saved_figures/diet.jpg',
#        width = 8, height = 4.25, units = 'in')

# ··············································································
###* Extra Figure: Combo ------------------------------------------------------------

cowplot::plot_grid(zoom_plots[[1]] + 
                     theme(text = element_text(size = 10),
                           axis.title.y.right = element_blank(),
                           axis.ticks.y.right = element_blank(),
                           axis.text.y.right = element_blank(),
                           plot.margin = margin(1,0.2,5,1, 'cm')), 
                   zoom_plots[[2]] + 
                     theme(text = element_text(size = 10),
                           axis.title.y = element_blank(),
                           axis.ticks.y = element_blank(),
                           axis.text.y = element_blank(),
                           axis.title.y.right = element_blank(),
                           plot.margin = margin(1,0.2,5,0.2, 'cm')), 
                   zoom_plots[[3]][[1]] +
                     theme(axis.title.y.left = element_blank(),
                           axis.ticks.y.left = element_blank(),
                           axis.text.y.left = element_blank(),
                           text = element_text(size = 10),
                           plot.margin = margin(1,0.2,5,0.2, 'cm')), 
                   zoom_plots[[3]][[2]] +
                     theme(axis.title.y.left = element_blank(),
                           axis.ticks.y.left = element_blank(),
                           axis.text.y.left = element_blank(),
                           axis.line.y.left = element_blank(),
                           text = element_text(size = 10),
                           plot.margin = margin(1,0.2,5,0, 'cm')),
                   ncol = 4, rel_widths = c(0.475, 0.425, 0.125, 0.075))

# ··············································································
## Overall Model Variation -----------------------------------------------------

# ··············································································
###* Rank Statistics -----------------------------------------------------------

# min and max ranked models
results$max %>% filter(region == 'OTC') %>%
  filter(!str_detect(model_string, '_random')) %>%
  group_by(model_string, metric) %>% 
  summarise(score = mean_cl_boot(score)) %>%
  group_by(metric) %>%
  filter(score$y == min(score$y) | 
           score$y == max(score$y)) %>%
  select(model_string, metric, score)
  
# average score of top 125 models
results$summary %>% group_by(metric) %>% 
  filter(model_string %in% model_sets$upper) %>%
  summarise(n = n(), min(score), max(score), 
            mean_cl_boot(score))

# average score of top 125 models
results$summary %>% group_by(metric) %>% filter(metric == 'wrsa') %>%
  filter(!str_detect(model_string, '_random')) %>%
  filter(model_string %in% model_sets$lower) %>% distinct(model_string)

# average difference between top 10 models
results$summary %>% group_by(metric) %>% 
  mutate(rank = dense_rank(score)) %>% 
  select(model_string, metric, 
         rank, score) %>%
  filter(rank >= 10) %>%
  arrange(metric, rank) %>%
  mutate(lag_score = lag(score), 
         change_score = lag_score - score) %>%
  group_by(metric) %>%
  summarise(change_score = mean_cl_boot(change_score))

### ············································································
###** Rank Breakpoints ---------------------------------------------------------

# model ranks + score differences
temp_data <- results$summary %>% 
  filter(metric %in% c('crsa','wrsa')) %>%
  arrange(rank, metric) %>%
  select(model_string, metric, rank, score) %>%
  group_by(metric) %>% 
  mutate(diff_from_max = score - max(score),
         diff_proximal = score - lead(score))

# the top models and their range
temp_data %>% filter(metric == 'wrsa') %>%
  filter(diff_from_max > -0.1)

rank_segmented_lms <- list() 

rank_segmented_lms$crsa <- 
  lm(score ~ rank, data = results$summary %>% 
       filter(metric == 'crsa')) %>% segmented(seg.Z = ~ rank, npsi = 2)

rank_segmented_lms$wrsa <- 
  lm(score ~ rank, data = results$summary %>% 
       filter(metric == 'wrsa')) %>% segmented(seg.Z = ~ rank, npsi = 2)

# The breakpoint rank of 124 comes from this stat:
(rank_segmented_lms$crsa %>% pluck('psi'))[,2]
(rank_segmented_lms$wrsa %>% pluck('psi'))[,2]

# score corresponding to the breakpoint rank
results$summary %>% filter(metric == 'wrsa') %>% 
  filter(rank == 124) %>% select(model_string, metric, rank, score)

# Compute the CIs of the breakpoint using the standard method:
# (Note in the paper, we opt for the bootstrap CIs below)
confint.segmented(rank_segmented_lms$crsa, method = 'gradient')
confint.segmented(rank_segmented_lms$wrsa, method = 'gradient')

# compute breakpoint bootstrap CIs:
# Note: this takes a while to run...

boot_breakpoint <- function(data)
{(analysis(data) %>% mutate(rank = min_rank(-score)) %$%
    lm(score ~ rank) %>% segmented(seg.Z = ~ rank, npsi = 2) %>% pluck('psi'))[1,2]}

boot_breakpoint <- results$summary %>% 
  filter(metric == 'wrsa') %>%
  select(model_string, train_task, 
         train_data, metric, score) %>%
  bootstraps(times = 1000) %>%
  mutate(breakpoint = map(splits, boot_breakpoint, 
                          .progress = TRUE)) %>%
  select(-splits) %>% unnest(breakpoint) 

boot_breakpoint %>% 
  summarise(mean = mean(breakpoint),
            lower = quantile(breakpoint, 0.025),
            upper = quantile(breakpoint, 0.975))

# Visualize the breakpoints from the segmented fits:
results$summary %>% filter(metric %in% c('crsa','wrsa')) %>%
  mutate(metric = str_to_upper(metric),
         metric = factor(metric, levels = c('CRSA','WRSA'))) %>%
  distinct(model_string, metric, score, rank) %>%
  ggplot(aes(x = rank, y = score, shape = metric)) + 
  guides(alpha = 'none') + guides(shape = 'none') +
  geom_point(cex = 7, alpha = 0.5, stroke = 1) + theme_classic() +
  geom_hline(aes(yintercept = y), linetype = 3, size = 1.25,
             data = noise_ceilings$group_avg) +
  geom_line(color = 'pink', size = 2, data = rank_segmented_lms$crsa %>%
              fitted() %>% data.frame() %>% rename(score = '.') %>%
              mutate(metric = 'CRSA', rank = dense_rank(-score))) +
  geom_line(color = 'cyan', size = 2, data = rank_segmented_lms$wrsa %>%
              fitted() %>% data.frame() %>% rename(score = '.') %>%
              mutate(metric = 'WRSA', rank = dense_rank(-score))) +
  scale_shape_manual(values = c(21, 19)) +
  scale_color_manual(values = c('black',palette[13:15], palette[4], 'gray')) +
  labs(x = 'Models (Sorted by Group Average)', y = '*r<sub>Pearson</sub>* (Score)',
       color = element_blank(), shape = element_blank()) +
  theme(text = element_text(size = 30), legend.position=c(.9,0.625),
        axis.title.y = element_markdown(),
        plot.margin = margin(t=1,r=2.0,b=1,l=1, "cm"))

# ··············································································
###* Effective Dimensionality --------------------------------------------------

manifold_stats <- read_csv('source_data/model_statistics/dimensionality.csv') %>%
  filter(!str_detect(model_string, 'NPID')) %>%
  mutate(trained = str_detect(model_string, '_random'))
  
# difference between effective dimensionality with/out random projection
manifold_stats %>% 
  mutate(random_projection = ifelse(random_projection, 'Yes', 'No')) %>%
  pivot_wider(names_from = random_projection, 
              values_from = effective_dimensions) %>%
  mutate(difference = No - Yes) %T>%
  {print(summarise(., difference = mean_cl_boot(difference)))} %>%
  pivot_longer(c(Yes, No), names_to = 'random_projection', 
               values_to = 'effective_dimensions') %>%
  {left_join(t_test(., effective_dimensions ~ random_projection, 
                    paired = TRUE, detailed = TRUE),
             cohens_d(., effective_dimensions ~ random_projection))}

get_boot_spearman <- function(data) 
  {analysis(data) %$% cor(score, effective_dimensions, method = 'spearman')}

# effective dimensionality vs. score across all models
results$summary %>% select(model_string, metric, score) %>%
  left_join(manifold_stats, multiple = 'all',
            relationship='many-to-many') %>%
  filter(metric == 'wrsa', random_projection) %>%
  bootstraps(times = 1000) %>%
  mutate(cor = map_dbl(splits, get_boot_spearman)) %>%
  mutate(above_zero = cor > 0) %>%
  summarise(above_zero = sum(above_zero),
            total = n(), y = mean(cor),
            ymin = quantile(cor, 0.025), 
            ymax = quantile(cor, 0.975),
            p_value = 1 - (above_zero / (total + 1)))

# effective dimensionality vs. score for trained / random models
results$summary %>% select(model_string, metric, score) %>%
  left_join(manifold_stats, multiple = 'all',
            relationship='many-to-many') %>%
  filter(metric == 'wrsa', random_projection, trained == TRUE) %>%
  bootstraps(times = 1000) %>%
  mutate(cor = map_dbl(splits, get_boot_spearman)) %>%
  mutate(above_zero = cor > 0) %>%
  summarise(above_zero = sum(above_zero),
            total = n(), y = mean(cor),
            ymin = quantile(cor, 0.025), 
            ymax = quantile(cor, 0.975),
            p_value = 1 - (above_zero / (total + 1)))

# effective dimensionality of the bottom-ranking models
results$summary %>% select(model_string, metric, score) %>%
  left_join(manifold_stats, multiple = 'all',
            relationship='many-to-many') %>%
  filter(metric == 'crsa', random_projection, trained == TRUE) %>%
  filter(model_string %in% model_sets$lower) %T>%
  {print(summarise(., model_count = n()))} %>%
  bootstraps(times = 1000) %>%
  mutate(cor = map_dbl(splits, get_boot_spearman)) %>%
  mutate(above_zero = cor > 0) %>%
  summarise(above_zero = sum(above_zero),
            total = n(), y = mean(cor),
            ymin = quantile(cor, 0.025), 
            ymax = quantile(cor, 0.975),
            p_value = 1 - (above_zero / (total + 1)))

# effective dimensionality vs. score across taskonomy models
results$summary %>% select(model_string, metric, score) %>%
  left_join(manifold_stats, multiple = 'all',
            relationship='many-to-many') %>%
  filter(metric == 'wrsa', random_projection) %>%
  filter(str_detect(model_string, 'taskonomy')) %T>%
  {print(summarise(., model_count = n()))} %>%
  bootstraps(times = 1000) %>%
  mutate(cor = map_dbl(splits, get_boot_spearman)) %>%
  mutate(above_zero = cor > 0) %>%
  summarise(above_zero = sum(above_zero),
            total = n(), y = mean(cor),
            ymin = quantile(cor, 0.025), 
            ymax = quantile(cor, 0.975),
            p_value = 1 - (above_zero / (total + 1)))

# ··············································································
###* Imagenet Accuracy ---------------------------------------------------------

imagenet_scores <- read_csv('source_data/model_statistics/imagenet1k.csv') %>%
  rename(imagenet_accuracy = imagnet1k_top1)

# stats and plot for imagenet accuracy vs. score
results$summary %>% select(model_string, metric, score) %>%
  left_join(imagenet_scores, multiple = 'all') %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  mutate(metric = str_to_upper(metric)) %>%
  drop_na(imagenet_accuracy) %>% group_by(metric) %T>%
  {print(rstatix::cor_test(., score, imagenet_accuracy, 
                    method = 'spearman'))} %>% ungroup() %>%
  ggplot(aes(x = imagenet_accuracy, y= score)) + 
  facet_wrap(~metric) + geom_point() +
  geom_smooth(method='lm', color='black') + 
  stat_cor(geom='label', method = 'spearman', 
           show.legend = FALSE) + theme_bw() +
  labs(x = 'Imagenet Accuracy', y = 'Score')

# ··············································································
###* Parameter Counts ----------------------------------------------------------
  
param_counts <- read_csv('source_data/model_statistics/parameter_count.csv')

# stats and plot for parameter count vs. score
results$summary %>% select(model_string, metric, score) %>%
  left_join(param_counts, multiple = 'all') %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  mutate(metric = str_to_upper(metric)) %>%
  filter(!str_detect(model_string, 'ResNet50-|taskonomy|ipcl')) %>%
  mutate(Trained = !str_detect(model_string, '_random')) %T>%
  {print(group_by(., metric, Trained) %>% 
           rstatix::cor_test(score, total_params, method = 'spearman'))} %>%
  ggplot(aes(x = total_params, y= score, color = Trained)) + 
  facet_wrap(~metric) + geom_point() + geom_smooth(method='lm') + 
  stat_cor(geom='label', method = 'spearman', show.legend = FALSE) + 
  theme_bw() + annotation_logticks(sides = 'b') +
  scale_x_log10(breaks = trans_breaks(n = 4, "log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) +
  labs(x = 'Total # of Parameters', y = 'Score')

# ··············································································
### Figure: Overall Variation -------------------------------------------------

ev_score <- Vectorize(function(x) {round(x**2 / noise_ceilings$group_avg$y**2, 2)})
add_ev_axis <- sec_axis(~., breaks =  seq(0,0.8,0.2), 
                        labels = ev_score(seq(0,0.8,0.2)),
                        name = 'Variance Explained *<sub></sub>*')

# list of models to choose from as labels:
results$summary %>% arrange(rank, metric) %>% 
  filter(metric %in% c('crsa', 'wrsa')) %>% 
  select(model_string, metric, score, rank) %>% print(n = 300)

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

# write.csv(plot_data, 'figure_data/Figure5A.csv', row.names=FALSE)

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

# write.csv(plot_data, 'figure_data/Figure5B.csv', row.names=FALSE)

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

imagenet_scores <- read_csv('source_data/model_statistics/imagenet1k.csv') %>%
  rename(imagenet_accuracy = imagnet1k_top1)

plot_data <- results$summary %>% select(model_string, metric, score) %>%
  left_join(imagenet_scores, multiple = 'all') %>%
  drop_na(imagenet_accuracy) %>%
  filter(metric %in% c('crsa','wrsa')) %>%
  mutate(metric = str_to_upper(metric),
         metric = factor(metric, levels = c('WRSA','CRSA'))) %>%
  ungroup() %>% distinct(model_string, metric, score, imagenet_accuracy)

# write.csv(plot_data, 'figure_data/Figure5C.csv', row.names=FALSE)

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

# ggsave('saved_figures/overall_variation.jpg')

# ··············································································
## Model-to-Model Analysis -----------------------------------------------------

model_uber_rsa <- read_csv('source_data/model_uber_rsa.csv') %>%
  rename(model1 = var1, model2 = var2) %>%
  # NPID is a repeat here, and must be filtered
  filter_all(all_vars(!str_detect(., 'NPID')))

model_uber_rsm <- read_parquet('source_data/model_uber_rsm.parquet') %>% 
  column_to_rownames('__index_level_0__') %>%
  # NPID is a repeat here, and must be filtered
  select(-contains(c('NPID','_random'))) %>%
  rownames_to_column('row') %>%
  filter(!str_detect(row, 'NPID|_random')) %>%
  column_to_rownames('row')

model_uber_rsa %>% filter_all(all_vars(!str_detect(., '_random'))) %>%
  #filter_all(all_vars(!str_detect(., 'BiT-Expert'))) %>%
  filter(model1 %in% model_sets$upper, model2 %in% model_sets$upper) %>%
  group_by(metric) %>% summarise(n = n(), mean = mean(correlation),
                                 sd = sd(correlation),
                                 min = min(correlation),
                                 max = max(correlation),
                                 correlation = mean_cl_boot(correlation))

model_uber_rsa %>% filter_all(all_vars(!str_detect(., '_random'))) %>%
  filter_all(all_vars(!str_detect(., 'BiT-Expert'))) %>%
  filter(str_detect(model1, '_taskonomy') | str_detect(model2, '_taskonomy')) %>%
  ungroup() %>% group_by(metric) %>% 
  summarise(n = n(), mean = mean(correlation),
            sd = sd(correlation),
            min = min(correlation),
            max = max(correlation),
            correlation = mean_cl_boot(correlation))

mds_method <- 'classical' # alt: umap or tsne

compute_mds <- function(metric, method) {
  apply_mds <- function(x, method = 'classical') {
    if (method == 'umap') {return(umap(x) %>% pluck('layout'))}
    if (method == 'classical') {return(cmdscale(x))}
    if (method == 'tsne') {return(Rtsne(x) %>% pluck('Y'))}
  }
  
  metric <- paste0(metric, '_')
  (1 - model_uber_rsm) %>%
    select(contains(metric)) %>%
    rownames_to_column('row') %>%
    filter(str_detect(row, metric)) %>%
    column_to_rownames('row') %>% as.matrix() %>% 
    apply_mds(method) %>% 
    as.data.frame() %>% 
    purrr::set_names('V1','V2') %>%
    mutate(model_string  = model_uber_rsm %>%
             filter(str_detect(rownames(.), metric)) %>% 
             rownames(), .before = everything()) %>%
    mutate(model_string = str_replace(model_string, metric, ''))
}

model_uber_mds <- c('crsa'='crsa','wrsa'='wrsa') %>%
  lapply(compute_mds, mds_method)

# ··············································································
#### Figure: Model-to-Model RSA ------------------------------------------------

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

figure_dims <- list(width_inches = 8.5, 
                    height_inches = 11, 
                    aspect_ratio = 11 / 8.5,
                    width_units = 8.5 / 0.0254, 
                    height_units = 11 / 0.0254)

# Save the plot as an SVG with the calculated dimensions
# ggsave("model-to-model_plot.png", plot = last_plot(), units = "in",
#        width = figure_dims$width_inches, 
#        height = figure_dims$height_inches / figure_dims$aspect_ratio)

# ··············································································
## Python-Generated ------------------------------------------------------------

temp_data <- read_csv('source_data/voxel_metadata_OTC.csv') %>%
  select('voxel_id', 'ncsnr', 'subj_id') %>%
  mutate(coordinate = str_split(voxel_id, '-', simplify = TRUE)[,2:4],
         x = as.numeric(coordinate[,1]),
         y = as.numeric(coordinate[,2]),
         z = as.numeric(coordinate[,3])) %>% select(-coordinate)

# write.csv(temp_data, 'figure_data/Figure1A.csv', row.names=FALSE)

temp_data <- read_csv('source_data/activation_pcspace.csv') %>%
  mutate(StimIndex = rownames(.), .after=Activity)

temp_data %>% mutate(index = rownames(.)) %>%
  ggplot(aes(x = X, y = Y)) + theme_bw() +
  facet_wrap(~Activity, scales='free') + 
  geom_point(aes(color = StimIndex), show.legend=FALSE) 

# write.csv(temp_data, 'figure_data/Figure1C.csv', row.names=FALSE)

## Stitch Source Data ----------------------------------------------------------

pacman::p_load('readr', 'writexl')
folder_path <- "figure_data"

# Get all csv files in target folder
file_paths <- list.files(folder_path, pattern = "\\.csv$", full.names = TRUE)

temp_data <- list()

for (file_path in file_paths) {
  key <- tools::file_path_sans_ext(basename(file_path))
  temp_data[[key]] <- read_csv(file_path)
}

# Write to excel file, each csv as separate sheet
write_xlsx(temp_data, path = "source_data/source_data.xlsx")

## Supplementary Data ----------------------------------------------------------

### ············································································
###* Table: All Models ---------------------------------------------------------

results$summary %>% filter(metric %in% c('crsa','wrsa')) %>%
  mutate(metric = str_to_upper(metric)) %>%
  rename(model_id = model_string, 
         architecture_kind = model_class) %>%
  mutate(train_task = str_replace(train_task, 'random', 'random_weights'),
         display_name = str_replace(display_name, 'ResNet-V2-',''),
         train_data = ifelse(str_detect(train_data, 'DIML'), 'MegaDepth+', train_data)) %>%
  select(display_name, architecture, 
         train_task, train_data, metric, score) %>%
  clean_names(case = 'title') %>%
  rename(`Model ID` = `Display Name`) %>%
  pivot_wider(names_from = Metric, values_from = Score) %>% arrange(-WRSA) %>%
  xtable(caption = 'List of All Models Tested', digits = 3) %>% 
  print(tabular.environment="longtable", caption.placement = 'top')

### ············································································
###* Effective Dimensionality --------------------------------------------------

read_csv('source_data/supplementary/dimensionality_check.csv') %>%
  mutate(dataset = str_replace(dataset, 'Val',''),
         dataset = str_replace(dataset, 'Shared','Shared-')) %>%
  mutate(model_layer_index = model_layer_index + 1) %>%
  filter(model_layer_index %% 2 == 0) %>%
  mutate(model_layer_index = model_layer_index / 2) %>%
  ggplot(aes(x = model_layer_index, y = effective_dimensions,
             color = dataset)) + geom_point() + geom_line() +
  theme_bw() + #easy_move_legend('bottom') +
  scale_x_continuous(breaks = seq(1,8,1)) +
  labs(x = 'ResNet18 ReLU Layer', y = 'Effective Dimensionality',
       color = 'Probe Image Set') +
  theme(text = element_text(size = 20),
        panel.grid = element_blank())

read_csv('source_data/supplementary/dimensionality_check.csv') %>%
  filter(model_layer_index %% 2 == 1) %>%
  pivot_wider(names_from = dataset, values_from = effective_dimensions) %>%
  cor_test(`NSD-Shared1000`, `ImageNet1K-Val10000`, method = 'pearson')

temp_data <- results$complete %>% 
  filter(region == 'OTC', 
         metric %in% c('crsa','wrsa')) %>%
  filter(!is.na(compare_training)) %>%
  mutate(contrast = relevel(compare_training, ref='random')) %>%
  select(-starts_with('compare')) %>%
  group_by(model, contrast, model_layer, 
           model_layer_depth, metric) %>%
  summarize(score = mean(score, na.rm=TRUE)) %>%
  pivot_wider(names_from=contrast, values_from=score) %>%
  mutate(difference = classification - random) %>%
  mutate(depth = model_layer_depth) %>%
  mutate(depth_bin = cut(depth, breaks = seq(0.0, 1.05, .05),
                        labels=FALSE, include.lowest=TRUE),
         depth_bin = (depth_bin) / 20)

### ············································································
###* Prediction across Depth --------------------------------------------------

temp_data <- results$complete %>% 
  filter(score_set == 'test') %>%
  filter(!is.na(compare_training)) %>%
  filter(region == 'OTC') %>% 
  filter(metric %in% c('crsa','wrsa')) %>%
  mutate(training = ifelse(str_detect(model_string, 'random'), 
                           'Untrained', 'Trained')) %>%
  group_by(model_string, training, model_layer, 
           model_layer_depth, metric) %>%
  summarise(count = n(), score = mean(score, na.rm=TRUE))


temp_data %>%
  mutate(depth = model_layer_depth) %>% filter(depth > 0.50) %>%
  mutate(depth_bin = cut(depth, breaks = seq(0.0, 1.05, .05),
                         labels=FALSE, include.lowest=TRUE),
         depth_bin = (depth_bin) / 20) %>%
  group_by(model_string, training, depth_bin, metric) %>%
  summarise(score = mean(score, na.rm=TRUE)) %>%
  mutate(metric = str_to_upper(metric)) %>%
  mutate(metric = str_to_upper(metric),
         metric = str_replace(metric, 'CRSA', 'cRSA'),
         metric = str_replace(metric, 'WRSA', 'veRSA')) %>%
  ggplot(aes(x=depth_bin, y=score, color=training)) +
  facet_wrap2(~metric, ncol=2, scales='free_x') +
  scale_color_manual(values=c('purple','gray')) +
  geom_line(aes(group=model_string), alpha=0.1) +
  geom_hline(aes(yintercept = 0.7975), 
             data = noise_ceilings$group_avg, 
             linetype = 1, size = 2, color = 'gray') +
  #stat_summary(fun.data = mean_cl_boot, geom='crossbar') +
  stat_summary(fun.y = mean, geom='line', size=3) +
  stat_summary(fun.y = mean, geom='point', size=3) +
  coord_cartesian(ylim=c(0.0, 0.805), clip = 'off') +
  labs(x = 'Relative Layer Depth (Binned)',
       y = '*r<sub>Pearson</sub>*', color = '') +
  theme_bw() + easy_move_legend('bottom') +
  scale_y_continuous(expand = c(0,0), sec.axis = add_ev_axis,
                     breaks = seq(0,0.8,0.2)) +
  theme(text = element_text(size = 10*(1/0.8), face = 'plain'),
        #legend.position=c(.78,0.55),
        #panel.border = element_rect(fill=NA),
        #plot.margin = margin(t=1,r=1,b=3,l=1, "cm"),
        plot.margin = margin(t=1,r=1,b=1,l=1, "cm"),
        axis.ticks.y = element_line(),
        axis.line.y = element_line(),
        axis.line.x = element_line(),
        axis.ticks.length = unit(0.1, "cm"),
        axis.title.y.left = element_markdown(),
        axis.title.y.right = element_markdown(),
        strip.background = element_blank(),
        #strip.text = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank())

#ggsave('_figure_drafts/depth_scores.png', width=8, height=4.5, dpi=300)

### ············································································
###* Inter-Subject Variation ---------------------------------------------------

# average movement of models across rankings
results$max %>% filter(region == 'OTC') %>%
  filter(!str_detect(model_string, 'random|taskonomy|ipcl')) %>%
  select(model_string, metric, subj_id, score) %>%
  group_by(metric, subj_id) %>%
  mutate(rank = dense_rank(-score)) %>%
  select(-score) %>%
  group_by(model_string, metric) %>%
  summarise(min_rank = min(rank),
            max_rank = max(rank),
            rank_range = max(rank) - min(rank)) %>%
  group_by(metric) %>%
  summarise(n = n(), max_rank_range = max(rank_range),
            rank = median_cl_boot(rank_range))

# spearman rank order correlation across subjects
results$max %>% filter(region == 'OTC') %>%
  filter(!str_detect(model_string, 'random|taskonomy|ipcl')) %>%
  select(model_string, metric, subj_id, score) %>%
  group_by(metric, subj_id) %>%
  mutate(rank = dense_rank(-score)) %>%
  select(-score) %>%
  pivot_wider(names_from = c(metric, subj_id),
              values_from = c(rank)) %>%
  select(-model_string) %>% 
  cor_mat(method = 'spearman') %>%
  pull_upper_triangle() %>%
  cor_gather() %>%
  separate(var1, into = c('metric1','subj_id1')) %>%
  separate(var2, into = c('metric2','subj_id2')) %>%
  filter(metric1 == metric2) %>%
  group_by(metric1) %>%
  summarise(n = n(), cor = mean_cl_boot(cor))

# subject-wise rank order correlation visual:
results$max %>% filter(region == 'OTC') %>%
  filter(metric %in% c('crsa','wrsa')) %>%
    mutate(metric = str_to_upper(metric)) %>%
  filter(subj_id %in% c(1,5)) %>%
  select(model_string, metric, subj_id, score) %>%
  filter(!str_detect(model_string, 'random|taskonomy|ipcl')) %>%
  group_by(metric, subj_id) %>%
  mutate(rank = dense_rank(-score)) %>%
  select(-rank) %>%
  pivot_wider(names_from = c(subj_id),
              values_from = c(score), names_prefix = 'subj') %>%
  ggplot(aes(x = subj1, y = subj5)) + 
  facet_wrap(~metric, scales = 'free') +
  geom_point() + theme_bw() + stat_cor(method = 'spearman')

# function for rank permutation tests:
permute_rank_stability <- function() {
  print('Running loop...')
  results$max %>% filter(region == 'OTC') %>%
    filter(!str_detect(model_string, 'random|taskonomy|ipcl')) %>%
    select(model_string, metric, subj_id, score) %>%
    group_by(metric, subj_id) %>% 
    mutate(score = sample(score)) %>%
    group_by(metric, subj_id) %>%
    mutate(rank = dense_rank(-score)) %>%
    select(-score) %>%
    pivot_wider(names_from = c(metric, subj_id),
                values_from = c(rank)) %>%
    select(-model_string) %>% 
    cor_mat(method = 'spearman') %>%
    pull_upper_triangle() %>%
    cor_gather() %>%
    separate(var1, into = c('metric1','subj_id1')) %>%
    separate(var2, into = c('metric2','subj_id2')) %>%
    filter(metric1 == metric2) %>%
    group_by(metric1) %>%
    rename(metric = metric1) %>%
    summarise(cor = mean(cor)) %>%
    mutate(timestamp = as.numeric(Sys.time()))
}

rank_permutations <- replicate(1000, permute_rank_stability(), 
                               simplify = FALSE) %>% bind_rows()

rank_permutations %>% group_by(metric) %>%
  summarise(mean_cor = mean(cor),
            lower_cor = quantile(cor, 0.025),
            upper_cor = quantile(cor, 0.975))

### ----- Supplement Figure: Variation in Subjects

left_join(results$summary %>% select(model_string, rank),
          relationship='many-to-many') %>%
  distinct(model_string, display_name, metric, 
           subj_id, score, rank) %>%
  mutate(metric = str_to_upper(metric),
         metric = str_replace(metric, 'CRSA', 'cRSA'),
         metric = str_replace(metric, 'WRSA', 'veRSA')) %>%
  ggplot(aes(x = rank, y = score, color = subj_id, 
             shape = metric, linetype = metric)) + 
  theme_classic() + geom_point(cex = 3, alpha = 0.5, stroke = 1) +
  geom_rect(aes(ymin=ymin, ymax=ymax, fill=subj_id,
                xmin=xmin, xmax=xmax), inherit.aes=FALSE,
            show.legend = FALSE, alpha = 0.3,
            data = noise_ceilings$subj_avg %>%
              mutate(subj_id = as.factor(subj_id)) %>%
              mutate(xmin=0, xmax=230, metric='wrsa')) +
  scale_shape_manual(values = c(21, 19)) +
  labs(x = 'Models (Sorted by veRSA Score)', 
       y = '*r<sub>Pearson</sub>* (Score)',
       color = ' SubjectID', shape = 'Metric') +
  scale_x_continuous(expand = c(0.05, 0)) +
  scale_y_continuous(expand = c(0,0), #sec.axis = add_ev_axis,
                     breaks = seq(0,0.9,0.1)) +
  coord_cartesian(ylim=c(0.0, 0.9), clip = 'off') +
  guides(alpha = 'none') + #guides(shape = 'none') +
  theme(text = element_text(size = 10 * (1/0.6)), legend.position=c(.82,0.5),
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
        panel.grid.major = element_blank())

# ··············································································
###* Early Visual Cortex (EVC) -------------------------------------------------

results$max %>% group_by(model_string, region, metric) %>%
  mutate(metric = str_to_upper(metric)) %>%
  summarise(score = mean(score)) %>%
  filter(region %in% c('EVC','OTC'),
         metric %in% c('CRSA','WRSA')) %>%
  filter(model_string %in% model_sets$upper) %>%
  pivot_wider(names_from = region, values_from = score) %T>%
  {print(group_by(.,metric) %>%
           cor_test(EVC, OTC, method = 'spearman'))} %>%
  ggplot(aes(x = OTC, y = EVC)) + theme_bw() +
  facet_wrap(~metric) + geom_point() +
  geom_abline(intercept = 0, slope = 1) + 
  ylim(c(0,0.75)) + xlim(c(0,0.75)) +
  geom_smooth(method = 'lm') + 
  theme(text = element_text(size = 20)) +
  stat_cor(label.x = 0.15, label.y = 0.5, method = 'spearman')

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
