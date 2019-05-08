#!/bin/bash
# find out if threshold is set
threshold=${3:-no}

if [ $2 = 'laptop' ]
then
  lm='amazon_lm.json'
elif [ $2 = 'restaurant' ] 
then
  lm='yelp_lm.json'
else
  echo 'CANNOT FIND THE CORRECT LANGUAGE MODEL'
fi
for k in 2 3 5 10
do

if [ $threshold = 'no' ]
then
  lm_augmented_data_path="./augmented_data/"$2"/no_additional_targets/lm_"$k"_no_threshold.json"
  $1 create_datasets.py original_augmentation_datasets/$2/$lm $lm_augmented_data_path $k --lm
  embedding_augmented_data_path="./augmented_data/"$2"/no_additional_targets/embedding_"$k"_no_threshold.json"
  $1 create_datasets.py original_augmentation_datasets/$2/embedding.json $embedding_augmented_data_path $k --embedding
else
  lm_augmented_data_path="./augmented_data/"$2"/no_additional_targets/lm_"$k".json"
  $1 create_datasets.py original_augmentation_datasets/$2/$lm $lm_augmented_data_path $k --lm --threshold 1
  embedding_augmented_data_path="./augmented_data/"$2"/no_additional_targets/embedding_"$k".json"
  $1 create_datasets.py original_augmentation_datasets/$2/embedding.json $embedding_augmented_data_path $k --embedding --threshold $threshold
fi
done
