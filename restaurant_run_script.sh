#!/bin/bash
echo 'Running the Language model augmentation with no additional targets and no threshold'
for k in 2 3 5 10
do
echo "Running $k"
augmented_data_path="./augmented_data/restaurant/no_additional_targets/lm_"$k"_no_threshold.json"
log_fp="./log_dir/Restaurant_no_additional_targets_lm_"$k"_no_threshold.log"
$1 run_models.py 5 ./data/splits/ ./results/augmentation/no_additional_targets/no_threshold_lm$k ./model_configs/standard Restaurant $log_fp --augmented_data_fp $augmented_data_path
done
echo 'Finished Running the Language model augmentation with no additional targets'
echo 'Running the Language model augmentation with no additional targets with threshold'
for k in 2 3 5 10
do
echo "Running $k"
augmented_data_path="./augmented_data/restaurant/no_additional_targets/lm_"$k".json"
log_fp="./log_dir/Restaurant_no_additional_targets_lm_"$k".log"
$1 run_models.py 5 ./data/splits/ ./results/augmentation/no_additional_targets/threshold_lm$k ./model_configs/standard Restaurant $log_fp --augmented_data_fp $augmented_data_path
done
echo 'Finished Running the Language model augmentation with no additional targets with threshold'
echo '-------------------------'
echo 'Running the Embedding model augmentation with no additional targets and no threshold'
for k in 2 3 5 10
do
echo "Running $k"
augmented_data_path="./augmented_data/restaurant/no_additional_targets/embedding_"$k"_no_threshold.json"
log_fp="./log_dir/Restaurant_no_additional_targets_embedding_"$k"_no_threshold.log"
$1 run_models.py 5 ./data/splits/ ./results/augmentation/no_additional_targets/no_threshold_embedding$k ./model_configs/standard Restaurant $log_fp --augmented_data_fp $augmented_data_path
done
echo 'Finished Running the Embedding model augmentation with no additional targets'
echo 'Running the Embedding model augmentation with no additional targets with threshold'
for k in 2 3 5 10
do
echo "Running $k"
augmented_data_path="./augmented_data/restaurant/no_additional_targets/embedding_"$k".json"
log_fp="./log_dir/Restaurant_no_additional_targets_embedding_"$k".log"
$1 run_models.py 5 ./data/splits/ ./results/augmentation/no_additional_targets/threshold_embedding$k ./model_configs/standard Restaurant $log_fp --augmented_data_fp $augmented_data_path
done
echo 'Finished Running the Embedding model augmentation with no additional targets with threshold'
echo "Running the baseline Restaurant models"
$1 run_models.py 5 ./data/splits/ ./results/baseline ./model_configs/standard Restaurant ./log_dir/Restaurant_baseline.log
