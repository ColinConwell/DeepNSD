if (!require(pacman)) {install.packages("pacman")}
pacman::p_load('this.path', 'arrow', 'progress', 'magrittr', 'tidyverse')

setwd(dirname(this.path())) # set working directory file location

# Global Options + Functions --------------------------------------------------
options(dplyr.summarise.inform = FALSE)

custom_themes <- list()
theme_set(theme_bw())

region_hierarchy <- c('V1v','V1d','V2v','V2d','V3v','V3d','hV4',
                      'OFA', 'FFA-1', 'FFA-2', 'OPA', 'PPA', 'RSC',
                      'EBA','FBA-1','FBA-2', 'OWFA', 'VWFA-1','VWFA-2',
                      'OTC','EVC','early','lateral','ventral')

remove_degenerate_layers <- function(results, diff_z_threshold = -3.291) {
  results %>% pivot_wider(names_from = score_set, values_from = score) %>%
    mutate(train_test_diff_z = scale(test - train)) %>%
    filter(train_test_diff_z > diff_z_threshold) %>% 
    pivot_longer(c(train,test), names_to = 'score_set', values_to = 'score')
}

process_result_files <- function(results_dir) {
  file_list  <- dir(results_dir, pattern = '.parquet', full.names = TRUE)
  #progress <- progress_bar$new(total = length(file_list))
  #read_with_progress <- function(filename) 
  #  {progress$tick(); read_parquet(filename)}
  benchmark_name = str_split(results_dir, '/', simplify = TRUE) %>% nth(-2)
  metric_name = str_split(results_dir, '/', simplify = TRUE) %>% last()
  region_hierarchy = region_hierarchy
  file_list %>% map(read_parquet) %>% bind_rows() %>%
    select(model, train_type, model_layer, model_layer_index, subj_id, region, score_set, score) %>%
    mutate(train_type = ifelse(train_type == 'panoptics', 'segmentation', train_type)) %>%
    filter(region %in% region_hierarchy) %>%
    mutate(region = factor(region, levels = region_hierarchy)) %>%
    mutate(benchmark = benchmark_name, metric = metric_name) %>%
    select(!contains('index_level')) %>%
    group_by(model, train_type) %>%
    mutate(model_depth = n_distinct(model_layer_index),
           model_layer_depth = (model_layer_index + 1) / model_depth) %>% ungroup() %>%
    remove_degenerate_layers() %>%
    select(model, train_type, model_layer, model_layer_index, model_layer_depth,
           benchmark, metric, region, subj_id, score_set, score)
}

process_benchmark <- function(results_dir, metrics = c('crsa','srpr','wrsa')) {
  results <- lapply(paste(results_dir, metrics, sep = '/'), process_result_files) %>% 
    set_names(metrics)
  results$combo <- bind_rows(results)
  return(results$combo)
}

get_model_layer_max <- function(results) {
  results %>% pivot_wider(names_from = score_set, values_from = score) %>%
    group_by(model, train_type, subj_id, region, metric) %>%
    filter(train == max(train, na.rm = TRUE)) %>%
    mutate(score = test) %>% select(-train, -test) %>% ungroup() %>%
    distinct(model, train_type, region, subj_id, metric, .keep_all = TRUE)
}

# General Overview --------------------------------------------------

results <- list() # list to store complete, max, and summary results data

voxel_metadata <- bind_rows(read_csv('brain_datasets/voxel_sets/shared1000_EVC-only/voxel_metadata.csv'),
                            read_csv('brain_datasets/voxel_sets/shared1000_OTC-only/voxel_metadata.csv'))

results$complete <- bind_rows(process_benchmark('fresh_results/shared1000_EVC-only'),
                              process_benchmark('fresh_results/shared1000_OTC-only'))

target_voxel_set <- 'shared1000_OTC-only'
target_regions <- c('V1v','V1d','V2v','V2d','V3v','V3d','hV4','EVC',
                    'OFA','FFA-1','FFA-2','OPA','PPA','RSC',
                    'EBA','FBA-1','FBA-2','OWFA', 'VWFA-1','VWFA-2','OTC')

#results$complete <- process_benchmark(paste0('fresh_results/', target_voxel_set))
results$max <- get_model_layer_max(results$complete)

results$summary <- results$max %>% filter(region == 'OTC') %>%
  group_by(model, train_type, metric) %>%
  summarise(n = n(), score = mean(score)) %T>% 
  print() %>% ungroup() %>% group_by(metric) %>% 
  mutate(rank = dense_rank(-score)) %>%
  ungroup() %>% arrange(metric, -rank) %>%
  mutate(model_string = paste(model, train_type, sep = '_'),
         .before = everything())

results$summary %>% filter(metric == 'wrsa') %>% print(n = 300)

# number of models run:
results$max %>% select(model, train_type) %>% 
  distinct() %>% nrow()

# check benchmark count:
results$max %>% distinct(benchmark, model, train_type) %>%
  group_by(model, train_type) %>% count() %>% filter(n < 2)

# check region count:
length(region_hierarchy)
results$max %>% distinct(region, metric, model, train_type) %>%
  group_by(model, train_type) %>% count() %>% filter(n < (23 * 3))

((results$complete %>% select(model, train_type, model_layer_index) %>% 
    distinct() %>% nrow()) * (dim(voxel_metadata) - 1)[[1]]) %>% 
  pluck(1) %>% {prettyNum(., big.mark = ',')} %>% paste('total regressions run')

mislabeled_models <- c('vit_large_patch32_224',
                       'mobilevit_s',
                       'mobilenetv3_large_100')

remove_mislabeled_models <- function(data) {data %>% filter(!model %in% mislabeled_models)}

# Results Export --------------------------------------------------

model_typology <- read_csv('../model_opts/model_typology.csv')
model_accuracy <- read_csv('../model_opts/model_accuracy.csv')
model_contrasts <- read_csv('model_contrasts.csv') %>%
  mutate(model_string = paste(model, train_type, sep = '_'))

models_run <- results$summary %>% filter(metric == 'wrsa') %>% pull(model_string)
model_contrasts %>% filter(!model_string %in% models_run)

results$summary %>% filter(metric == 'wrsa') %>% dim()

all_metadata <- model_contrasts %>% 
  filter(model_string %in% models_run) %>%
  rename(display_name = model_display_name) %>%
  select(model_string, model, architecture, train_type, train_data, 
         model_class, display_name, description, starts_with('compare')) %>%
  left_join(model_accuracy) %>%
  distinct(model_string, .keep_all = TRUE) %>%
  select(!starts_with('compare'), starts_with('compare'))

results$summary %>% filter(!model_string %in% all_metadata$model_string) %>%
  filter(metric == 'wrsa') %>% select(model_string, model, train_type)

# results summary export:
export_columns <- c('model_string', 'model', 'train_type', 
                    'train_task', 'train_data', 
                    'architecture','model_class','model_depth',
                    'display_name','description', 'model_layer',
                    'metric','region','subj_id','score_set','score')

results$summary %>% dim()

results$summary %>% left_join(all_metadata) %>% 
  rename(train_task = train_type) %>%
  select(any_of(export_columns), starts_with('compare')) %T>%
  {print(dim(.))} %>% write.csv('press_results/results_summary.csv', row.names = FALSE)

distinct(results$complete, model, train_type) %>% dim()
distinct(results$complete, region, subj_id) %>% dim() 
224 * 90 * 3
results$max %>% dim()

results$max %>% select(!starts_with('model_layer_'), -benchmark) %>%
  left_join(all_metadata) %>% rename(train_task = train_type) %>%
  select(any_of(export_columns), starts_with('compare')) %T>%
  {print(dim(.))} %>% write.csv('press_results/results_max.csv', row.names = FALSE)

results$complete %>% dim()

results$complete %>% select(!starts_with('model_layer_'), -benchmark) %>%
  left_join(all_metadata) %>% rename(train_task = train_type) %>%
  select(any_of(export_columns), starts_with('compare')) %T>%
  {print(dim(.))} %>% write_parquet('press_results/results_complete.parquet')

results$complete %>% pivot_wider(names_from = score_set, values_from = score) %>%
  group_by(model, train_type, subj_id, region, metric) %>%
  filter(region == 'OTC', metric %in% c('wrsa')) %>%
  group_by(model, train_type, model_layer, benchmark, metric, region) %>%
  summarise(n = n(), score = mean(test, na.rm = TRUE)) %>%
  group_by(model, train_type, benchmark, metric) %>%
  filter(score == max(score)) %>%
  mutate(model_string = paste(model, train_type, sep = '_'),
         model_string = ifelse(str_detect(model,'ipcl'), model, model_string)) %>%
  select(model_string, model, train_type, model_layer, 
         benchmark, metric, region, score) %>%
  write.csv('press_results/superlative_layers.csv', row.names = FALSE)
  
  









