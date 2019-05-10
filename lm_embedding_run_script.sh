#!/bin/bash
echo "Running the baseline models for the $2 domain"
baseline_log_fp="./log_dir/"$2"_baseline.log"
$1 run_models.py 5 ./data/splits/ ./results/baseline ./model_configs/standard $2 $baseline_log_fp

echo "Running the language model baseline models for the $2 domain"
baseline_lm_log_fp="./log_dir/"$2"_baseline_lm.log"
$1 run_models.py 5 ./data/splits/ ./results/baseline_lm ./model_configs/standard_lm $2 $baseline_lm_log_fp

echo "Running the language model and embedding baseline models for the $2 domain"
baseline_lm_embedding_log_fp="./log_dir/"$2"_baseline_lm_embedding.log"
$1 run_models.py 5 ./data/splits/ ./results/baseline_lm_embedding ./model_configs/standard_lm_embedding $2 $baseline_lm_embedding_log_fp

echo "Running the domain specific embedding models for the $2 domain"
ds_embedding_log_fp="./log_dir/"$2"_ds_embedding.log"
ds_embedding_model_configs="./model_configs/"$2"_ds_embedding"
$1 run_models.py 5 ./data/splits/ ./results/ds_embedding $ds_embedding_model_configs $2 $ds_embedding_log_fp

echo "Running the domain specific language model models for the $2 domain"
ds_lm_log_fp="./log_dir/"$2"_ds_lm.log"
ds_lm_model_configs="./model_configs/"$2"_ds_lm"
$1 run_models.py 5 ./data/splits/ ./results/ds_lm $ds_lm_model_configs $2 $ds_lm_log_fp

echo "Running the domain specific language model with standard embedding models for the $2 domain"
ds_lm_embedding_log_fp="./log_dir/"$2"_ds_lm_embedding.log"
ds_lm_embedding_model_configs="./model_configs/"$2"_ds_lm_embedding"
$1 run_models.py 5 ./data/splits/ ./results/ds_lm_embedding $ds_lm_embedding_model_configs $2 $ds_lm_embedding_log_fp

echo "Running the domain specific language model with domain sepcific embedding models for the $2 domain"
ds_lm_ds_embedding_log_fp="./log_dir/"$2"_ds_lm_ds_embedding.log"
ds_lm_ds_embedding_model_configs="./model_configs/"$2"_ds_lm_ds_embedding"
$1 run_models.py 5 ./data/splits/ ./results/ds_lm_ds_embedding $ds_lm_ds_embedding_model_configs $2 $ds_lm_ds_embedding_log_fp
