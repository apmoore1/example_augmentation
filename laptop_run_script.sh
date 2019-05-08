#!/bin/bash
echo 'Running the Language model augmentation with no additional targets and no threshold'
for k in 2 3 5 10
do
echo "Running $k"
augmented_data_path="./augmented_data/laptop/no_additional_targets/lm_"$k"_no_threshold.json"
$1 run_models.py 5 ./data/splits/ ./results/augmentation/no_additional_targets/no_threshold_lm$k ./model_configs Laptop ./log_dir --augmented_data_fp $augmented_data_path
done
echo 'Finished Running the Language model augmentation with no additional targets'
echo 'Running the Language model augmentation with no additional targets with threshold'
for k in 2 3 5 10
do
echo "Running $k"
augmented_data_path="./augmented_data/laptop/no_additional_targets/lm_"$k".json"
$1 run_models.py 5 ./data/splits/ ./results/augmentation/no_additional_targets/threshold_lm$k ./model_configs Laptop ./log_dir --augmented_data_fp $augmented_data_path
done
echo 'Finished Running the Language model augmentation with no additional targets with threshold'
echo '-------------------------'
echo 'Running the Embedding model augmentation with no additional targets and no threshold'
for k in 2 3 5 10
do
echo "Running $k"
augmented_data_path="./augmented_data/laptop/no_additional_targets/embedding_"$k"_no_threshold.json"
$1 run_models.py 5 ./data/splits/ ./results/augmentation/no_additional_targets/no_threshold_embedding$k ./model_configs Laptop ./log_dir --augmented_data_fp $augmented_data_path
done
echo 'Finished Running the Embedding model augmentation with no additional targets'
echo 'Running the Embedding model augmentation with no additional targets with threshold'
for k in 2 3 5 10
do
echo "Running $k"
augmented_data_path="./augmented_data/laptop/no_additional_targets/embedding_"$k".json"
$1 run_models.py 5 ./data/splits/ ./results/augmentation/no_additional_targets/threshold_embedding$k ./model_configs Laptop ./log_dir --augmented_data_fp $augmented_data_path
done
echo 'Finished Running the Embedding model augmentation with no additional targets with threshold'
echo "Running the baseline Laptop models"
$1 run_models.py 5 ./data/splits/ ./results/baseline ./model_configs Laptop ./log_dir
