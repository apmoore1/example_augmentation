## Requirements
1. python >= 3.6.1
2. `pip install -r requirements.txt`
3. If you would like to create augmented data and copy the exact way we did it in the paper it requires the Stanford tokeniser of which the open source [Bella package](https://github.com/apmoore1/Bella) requires you to have this as a docker service therefore **Docker** is required.
4. run the following to get the stanford tokeniser as a service for the Bella package `docker run -p 9000:9000 --rm mooreap/corenlp`
5. The 300 dimension 840B token Glove vector that can be downloaded from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip), we assume that this word vector is at the following location `./embeddings/glove.840B.300d.txt`

## Getting the data and converting it into Train, Validation, and Test sets

Download the following datasets:

1. SemEval Restaurant and Laptop 2014 [1] [train](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-train-data-v20-annotation-guidelines/683b709298b811e3a0e2842b2b6a04d7c7a19307f18a4940beef6a6143f937f0/) and [test] and put the `Laptop_Train_v2.xml` and `Restaurants_Train_v2.xml` training files into the following directory `./data` and do the same for the test files (`Laptops_Test_Gold.xml` and `Restaurants_Test_Gold.xml`)
2. Election dataset [2] from the following [link](https://ndownloader.figshare.com/articles/4479563/versions/1) and extract all of the data into the following folder `./data/election`, the `election` folder should now contain the following files `annotations.tar.gz`, `test_id.txt`, `train_id.txt`, and `tweets.tar.gz`. Extract both the `annotations.tar.gz` and the `tweets.tar.gz` files.

Then run the following command to create the relevant and determinstic train, validaion, and test splits of which these will be stored in the following directory `./data/splits`:

`python generate_datasets.py ./data ./data/splits`

This should create all of the splits that we will use throughout the normal experiments that are the baseline values for all of our augmentation experiments. This will also print out some statistics for each of the splits to ensure that they are relatively similar.

## Create the augmented data

As stated in the paper the data is augmented by exchanging target words within their *K* (5) nearest words for each sample in the dataset thus creating a dataset up to five times larger than the original. The reason it is upto 5 times is due to the embedding not containing all of the target words.

, these samples are then subsampled by randomly sampling from this augmented dataset until we have dataset of the same size as the original dataset. We do this by random sub-sampling for each epoch to allow the model to see more unique examples. Therefore an embedding is required of which we use a custom embeddings for each dataset that is domain specific and can be downloaded from the following links:
1. Restaurant
2. Laptop
3. Twitter Election

All of the embeddings are trained with n-gram phrases of up to 6 grams where the phrases are detected based on [normalised pointwise mutual information](https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf) this was used to find targets that are more than one word long this idea also comes from [Mikolov](https://arxiv.org/abs/1310.4546). These embeddings were trained using the Word2Vec Skip Gram algorthim where all words were tokenised using Stanford, lower cased, and found phrases of up to 6 n-grams and finally create a 300 dimension vector. None of these parameters were optimised e.g. we did not tune the dimensionality to create more semantically meaningful embeddings, window 5, minimum word count of 5, trained for 5 epochs, basically standard settings within [Gensim](https://radimrehurek.com/gensim/). The Restaurant and Laptop embeddings were train on the [Yelp dataset of 2018](https://www.yelp.com/dataset) and the [electronics amazon review dataset](http://jmcauley.ucsd.edu/data/amazon/)*[Related paper](http://cseweb.ucsd.edu/~jmcauley/pdfs/www16a.pdf)* respectively. The Twitter Election embeddings were trained on Tweets that we have collected from the 2nd of Febrauary to the 8th of March 2018 where the Tweets where collected based on content from the mention of a list of [MP account names](https://www.mpsontwitter.co.uk/list).

## The affects of data augmentation
### Restaurant and using the same targets in the training dataset
To get a list of target words perform the following:
``` bash
python target_word_list.py data/splits/Restaurant\ Train target_words/restaurant/restaurant_train.json
```
This will then create a list of unique target words that have come from the Restaurant training dataset and save them as a list within `target_words/restaurant/restaurant_train.json`. This can then be used to expand the current training dataset based on switching targets that are similar within the same sentence.

#### Expanding through language model
The following command with create a new json type file where each new line is a json dictionary that corresponds to one sample in the original TDSA training dataset but with 3 new fields; `alternative_targets`, `alternative_perplexity`, and `original_perplexity`:
``` bash
python augment_transformer.py data/splits/Restaurant\ Train ../yelp_language_model_save_large/model.tar.gz ./target_words/restaurant/restaurant_train.json ./original_augmentation_datasets/restaurant/yelp_lm.json 'spacy' --cuda --batch_size 15
```

#### Expanding by using the Domain Specific Embedding
``` bash
python augment_embedding.py data/splits/Restaurant\ Train ./embeddings/yelp/lower\ case\ phrase\ stanford\ 300D ./target_words/restaurant/restaurant_train.json ./original_augmentation_datasets/restaurant/embedding.json 'spacy' --lower
```
#### Plotting the domain specific embeddings similarity scores
Here we want to know the distrbution of the similarity scores so that we can create a threshold value. This threshold value is a lot easier for the language model as we can use the perplexity score of the original sentence and only choose targets that are equal or lower to that original perplexity score.

The similarity scores that we will use to create this distribution will come from the expanded dataset above. Even though this will cause a bais towards targets the occur more frequently this bias comes from the training data so we are going to keep that bias.
``` bash
python embedding_similarity_dist.py original_augmentation_datasets/restaurant/embedding.json ./images/embedding_similarity_dist/restaurant.png 10.0

python embedding_similarity_dist.py original_augmentation_datasets/restaurant/embedding.json ./images/embedding_similarity_dist/restaurant.png 5.0
```
This will show that the simiarity value of 0.36 (0.418) will cover 10% (5%) of the simialrity values within the augmented dataset. The plot returned from this command shows that the data is not normally distributed and this is confirmed by the `D’Agostino and Pearson’s` normality test. 

#### Creating new Training datasets
Here we show how we create **K** best alternative target datasets and **K Threshold** alternative datasets:
##### K
This is where we choose the **K** most similar targets based on either the language model or the embedding. Below is the command to run to create both of these datasets respectively:
``` bash
python create_datasets.py original_augmentation_datasets/restaurant/yelp_lm.json augmented_data/restaurant/no_additional_targets/lm_10_no_threshold.json 10 --lm
python create_datasets.py original_augmentation_datasets/restaurant/embedding.json augmented_data/restaurant/no_additional_targets/embedding_10_no_threshold.json 10 --embedding
```
Where in both cases we can see that **K** is 10. We repeat this same process for `[2,3,5]` values of **K**. For this we can run the following script:

``` bash
./create_datasets.sh /home/andrew/Envs/example_augmentation/bin/python restaurant
```

##### K Threshold
This is the same as [above](#k) except that we restrict the **K** most similar to only those **K** that pass some sort of threshold, in the case of the language model this is that the **K** targets when within the sentence the perplexity of the sentence is lower or equal to the same sentence but with the original target. In the embedding case it's not context/sentence specific rather we have to define up front a specific similarity score that the **K** targets have to be greater or equal to the similarity of the original target. To inform us on the similarity threshold to use we look at the similarity plot produced in the [above section](#plotting-the-domain-specific-embeddings-similarity-scores) and from this we have decided 0.418 as it will only allow the top 5% of the most similar targets through and hopefully increase precision when **K** is large. The command top produce the threshold dataset is shown below for **K** equal to 10:
``` bash
python create_datasets.py original_augmentation_datasets/restaurant/yelp_lm.json augmented_data/restaurant/no_additional_targets/lm_10.json 10 --lm --threshold 1
python create_datasets.py original_augmentation_datasets/restaurant/embedding.json augmented_data/restaurant/no_additional_targets/embedding_10.json 10 --embedding --threshold 0.418
``` 
We repeat this same process for `[2,3,5]` values of **K**, without changing the threshold limit for the embedding which is **0.418**. For this we can run the following script:

``` bash
./create_datasets.sh /home/andrew/Envs/example_augmentation/bin/python restaurant 0.418
```

#### The affects this has on modelling
First to ensure that the learning rates that we have selected in the model configurations are suitable we can run the following to plot learning rate against loss for the first 100 batches in the training data: (Currently one problem with this method is that when we do it for several modls at the same time it plots over each other)
``` bash
python find_lr_models.py ./data/splits/Restaurant\ Train results/learning_rates/ ./model_configs/ Restaurant /tmp/find_lr.log
```

``` bash
./restaurant_run_script.sh /home/andrew/Envs/example_augmentation/bin/python ./model_configs/standard
```
Here we show the affects that data augmentation has on the sentiment models. The models that we shall use are the following:
1. IAN
2. TDSLTM

Plotting the results, we can use the following command to plot the results for Validation and Test sets with both Macro F1 and Accuracy metrics:
``` bash
python vis_results.py 5 results/augmentation/no_additional_targets/ data/splits/Restaurant\ Test 'Macro F1' ./images/results/restaurant/augmentation/no_additional_targets_macro_f1_test.png Restaurant
python vis_results.py 5 results/augmentation/no_additional_targets/ data/splits/Restaurant\ Val 'Macro F1' ./images/results/restaurant/augmentation/no_additional_targets_macro_f1_val.png Restaurant --val
python vis_results.py 5 results/augmentation/no_additional_targets/ data/splits/Restaurant\ Test 'Accuracy' ./images/results/restaurant/augmentation/no_additional_targets_accuracy_test.png Restaurant
python vis_results.py 5 results/augmentation/no_additional_targets/ data/splits/Restaurant\ Val 'Accuracy' ./images/results/restaurant/augmentation/no_additional_targets_accuracy_val.png Restaurant --val
```



If we open `./augmentation_sentence_examples/restaurant/embedding.tsv` we can see the sentence on line 24 is a problem with regards to its suggested target replacements:

sentence: `It's also attached to Angel's Share, which is a cool, more romantic [bar]...`
related targets: `bars(0.611)`, `pub(0.5102)`, `bartender(0.4999)`, `bartenders(0.4885)`, `counter(0.4596)`

"dining experience", "date spot", "all you can eat deal", "icing on the cake", "place", "spot", "setting"

As we can see the first is wrong as it is non-singular, the second and fifth are plausible but the third and fourth are completely wrong but are related by topic.


### Laptop and using the same targets in the training dataset (Same as above from the restaurant only showing the commands here for reproducibility reasons)
To get a list of target words perform the following:
``` bash
python target_word_list.py data/splits/Laptop\ Train target_words/laptop/laptop_train.json
```
This will then create a list of unique target words that have come from the Laptop training dataset and save them as a list within `target_words/laptop/laptop_train.json`. This can then be used to expand the current training dataset based on switching targets that are similar within the same sentence.

#### Expanding through language model
The following command with create a new json type file where each new line is a json dictionary that corresponds to one sample in the original TDSA training dataset but with 3 new fields; `alternative_targets`, `alternative_perplexity`, and `original_perplexity`:
``` bash
python augment_transformer.py data/splits/Laptop\ Train ../amazon_language_model_save_large/model.tar.gz ./target_words/laptop/laptop_train.json ./original_augmentation_datasets/laptop/amazon_lm.json 'spacy' --cuda --batch_size 15
```

#### Expanding by using the Domain Specific Embedding
``` bash
python augment_embedding.py data/splits/Laptop\ Train ./embeddings/amazon/lower\ case\ phrase\ stanford\ 300D ./target_words/laptop/laptop_train.json ./original_augmentation_datasets/laptop/embedding.json 'spacy' --lower
```
#### Plotting the domain specific embeddings similarity scores
Here we want to know the distrbution of the similarity scores so that we can create a threshold value. This threshold value is a lot easier for the language model as we can use the perplexity score of the original sentence and only choose targets that are equal or lower to that original perplexity score.

The similarity scores that we will use to create this distribution will come from the expanded dataset above. Even though this will cause a bais towards targets the occur more frequently this bias comes from the training data so we are going to keep that bias.
``` bash
python embedding_similarity_dist.py original_augmentation_datasets/laptop/embedding.json ./images/embedding_similarity_dist/laptop.png 10.0

python embedding_similarity_dist.py original_augmentation_datasets/laptop/embedding.json ./images/embedding_similarity_dist/laptop.png 5.0
```
This will show that the simiarity value of 0.336 (0.404) will cover 10% (5%) of the simialrity values within the augmented dataset. The plot returned from this command shows that the data is not normally distributed and this is confirmed by the `D’Agostino and Pearson’s` normality test. 

#### Creating new Training datasets
Here we show how we create **K** best alternative target datasets and **K Threshold** alternative datasets:
##### K
This is where we choose the **K** most similar targets based on either the language model or the embedding. Below is the command to run to create both of these datasets respectively:
``` bash
python create_datasets.py original_augmentation_datasets/laptop/amazon_lm.json augmented_data/laptop/no_additional_targets/lm_10_no_threshold.json 10 --lm
python create_datasets.py original_augmentation_datasets/laptop/embedding.json augmented_data/laptop/no_additional_targets/embedding_10_no_threshold.json 10 --embedding
```
Where in both cases we can see that **K** is 10. We repeat this same process for `[2,3,5]` values of **K**. For this we can run the following script:

``` bash
./create_datasets.sh /home/andrew/Envs/example_augmentation/bin/python laptop
```

##### K Threshold
This is the same as [above](#k) except that we restrict the **K** most similar to only those **K** that pass some sort of threshold, in the case of the language model this is that the **K** targets when within the sentence the perplexity of the sentence is lower or equal to the same sentence but with the original target. In the embedding case it's not context/sentence specific rather we have to define up front a specific similarity score that the **K** targets have to be greater or equal to the similarity of the original target. To inform us on the similarity threshold to use we look at the similarity plot produced in the [above section](#plotting-the-domain-specific-embeddings-similarity-scores) and from this we have decided 0.404 as it will only allow the top 5% of the most similar targets through and hopefully increase precision when **K** is large. The command top produce the threshold dataset is shown below for **K** equal to 10:
``` bash
python create_datasets.py original_augmentation_datasets/laptop/amazon_lm.json augmented_data/laptop/no_additional_targets/lm_10.json 10 --lm --threshold 1
python create_datasets.py original_augmentation_datasets/laptop/embedding.json augmented_data/laptop/no_additional_targets/embedding_10.json 10 --embedding --threshold 0.404
``` 
We repeat this same process for `[2,3,5]` values of **K**, without changing the threshold limit for the embedding which is **0.404**. For this we can run the following script:

``` bash
./create_datasets.sh /home/andrew/Envs/example_augmentation/bin/python laptop 0.404
```

#### The affects this has on modelling
First to ensure that the learning rates that we have selected in the model configurations are suitable we can run the following to plot learning rate against loss for the first 100 batches in the training data: (Currently one problem with this method is that when we do it for several modls at the same time it plots over each other)
``` bash
python find_lr_models.py ./data/splits/Laptop\ Train results/learning_rates/ ./model_configs/ Laptop /tmp/find_laptop_lr.log
```

``` bash
./laptop_run_script.sh /home/andrew/Envs/example_augmentation/bin/python ./model_configs/standard
```
Here we show the affects that data augmentation has on the sentiment models. The models that we shall use are the following:
1. IAN
2. TDSLTM

Plotting the results, we can use the following command to plot the results for Validation and Test sets with both Macro F1 and Accuracy metrics:
``` bash
python vis_results.py 5 results/augmentation/no_additional_targets/ data/splits/Laptop\ Test 'Macro F1' ./images/results/laptop/augmentation/no_additional_targets_macro_f1_test.png Laptop
python vis_results.py 5 results/augmentation/no_additional_targets/ data/splits/Laptop\ Val 'Macro F1' ./images/results/laptop/augmentation/no_additional_targets_macro_f1_val.png Laptop --val
python vis_results.py 5 results/augmentation/no_additional_targets/ data/splits/Laptop\ Test 'Accuracy' ./images/results/laptop/augmentation/no_additional_targets_accuracy_test.png Laptop
python vis_results.py 5 results/augmentation/no_additional_targets/ data/splits/Laptop\ Val 'Accuracy' ./images/results/laptop/augmentation/no_additional_targets_accuracy_val.png Laptop --val
```



If we open `./augmentation_sentence_examples/restaurant/embedding.tsv` we can see the sentence on line 24 is a problem with regards to its suggested target replacements:

sentence: `It's also attached to Angel's Share, which is a cool, more romantic [bar]...`
related targets: `bars(0.611)`, `pub(0.5102)`, `bartender(0.4999)`, `bartenders(0.4885)`, `counter(0.4596)`

"dining experience", "date spot", "all you can eat deal", "icing on the cake", "place", "spot", "setting"

As we can see the first is wrong as it is non-singular, the second and fifth are plausible but the third and fourth are completely wrong but are related by topic.

## Anaylsing the results of K
Here we want to know if **K** is significant or not, furthermore we will expore this in two ways:

1. Is the Best K for each model and augmentation technique significantly better than the worse K?
2. Is the Best K for each model and augmentation technique sigificantly better than the next best K?
3. Is there a trend of best K's and significantly worse K's? -- This is shown through the number of times K is best for each metric and data split and the number of times a K is significantly worse than the best K.
3. Given all the significantly best and worse K pairs is there an overall best and worse K from all of the model and augmentation pairs?

We are going to break down the code commands to generate the scores for these based on Metric and data split (Validation or Test). For each run it calculates the significants based on one-tailed paired bootstrap test with 10,000 bootstrap samples. As each of the models and augmentation techniques have been run 5 times to take into account the random seed problem we will take the median best model for each to compare significant values.

### Validation
#### Accuracy

We will break this down for both the validation and test sets. For the validation sets for both Macro F1 and Accuracy scores for the laptop dataset:
``` bash
./is_k_significant.sh ~/Envs/example_augmentation/bin/python ./results/augmentation/no_additional_targets/ ./data/splits/Laptop\ Val 'Accuracy' Laptop 10000 true true
./is_k_significant.sh ~/Envs/example_augmentation/bin/python ./results/augmentation/no_additional_targets/ ./data/splits/Restaurant\ Val 'Accuracy' Restaurant 10000 true true
```
The results for this can be found in the following [pdf](./latex/k_results/validation_accuracy.pdf) and [latex](./latex/k_results/validation_accuracy.latex) file (pdf is a rendering of the latex). As well as the number of times K was best and worse [pdf](./latex/k_results/validation_accuracy_best_worse.pdf) and [latex](./latex/k_results/validation_accuracy_best_worse.latex) file.

#### Macro F1
``` bash
./is_k_significant.sh ~/Envs/example_augmentation/bin/python ./results/augmentation/no_additional_targets/ ./data/splits/Laptop\ Val 'Macro F1' Laptop 10000 true true
./is_k_significant.sh ~/Envs/example_augmentation/bin/python ./results/augmentation/no_additional_targets/ ./data/splits/Restaurant\ Val 'Macro F1' Restaurant 10000 true true
```

The results for this can be found in the following [pdf](./latex/k_results/validation_macro_f1.pdf) and [latex](./latex/k_results/validation_macro_f1.latex) file.

### Test
#### Accuracy
``` bash
./is_k_significant.sh ~/Envs/example_augmentation/bin/python ./results/augmentation/no_additional_targets/ ./data/splits/Laptop\ Test 'Accuracy' Laptop 10000 false true
./is_k_significant.sh ~/Envs/example_augmentation/bin/python ./results/augmentation/no_additional_targets/ ./data/splits/Restaurant\ Test 'Accuracy' Restaurant 10000 false true
```
#### Macro F1
``` bash
./is_k_significant.sh ~/Envs/example_augmentation/bin/python ./results/augmentation/no_additional_targets/ ./data/splits/Laptop\ Test 'Macro F1' Laptop 10000 false true
./is_k_significant.sh ~/Envs/example_augmentation/bin/python ./results/augmentation/no_additional_targets/ ./data/splits/Restaurant\ Test 'Macro F1' Restaurant 10000 false true
```

## Converting the word vectors from binary file to text file
``` bash
python from_vector_to_txt.py ./embeddings/yelp/lower\ case\ phrase\ stanford\ 300D ./embeddings/yelp/ds_embedding.txt
python from_vector_to_txt.py ./embeddings/amazon/lower\ case\ phrase\ stanford\ 300D ./embeddings/amazon/ds_embedding.txt
```

## Are language models embedding better than embeddings and is domain speicifc required?
``` bash
./lm_embedding_run_script.sh /home/andrew/Envs/example_augmentation/bin/python
```
and to visulise the results:
``` bash
python vis_domain_results.py 5 ./results/ data/splits/Laptop\ Test 'Macro F1' images/results/laptop/domain_specific/macro_f1_test_val.png Laptop --val_fp data/splits/Laptop\ Val
python vis_domain_results.py 5 ./results/ data/splits/Laptop\ Test 'Accuracy' images/results/laptop/domain_specific/accuracy_test_val.png Laptop --val_fp data/splits/Laptop\ Val
python vis_domain_results.py 5 ./results/ data/splits/Restaurant\ Test 'Macro F1' images/results/restaurant/domain_specific/macro_f1_test_val.png Restaurant --val_fp data/splits/Restaurant\ Val
python vis_domain_results.py 5 ./results/ data/splits/Restaurant\ Test 'Accuracy' images/results/restaurant/domain_specific/accuracy_test_val.png Restaurant --val_fp data/splits/Restaurant\ Val
```

## What happens when augmentation meets domain specific LM?
ATAE Laptop with DS LM and Glove:
``` bash
./laptop_run_script.sh /home/andrew/Envs/example_augmentation/bin/python ./model_configs/Laptop_ds_lm_embedding 'atae' 'atae_ds_lm_embedding'
```

## Extra baselines
python run_models.py 5 ./data/splits/ ./results/baseline ./model_configs/standard Laptop ./log_dir/Laptop_baseline_extra.log --model_names "lstm" "lstm_random" "ds_elmo_t_embedding_tune_laptop" "ds_elmo_t_embedding_laptop" "ds_elmo_t_laptop" "elmo_t" --model_name_save_names "lstm" "lstm_random" "ds_elmo_t_embedding_tune" "ds_elmo_t_embedding" "ds_elmo_t" "elmo_t"
python run_models.py 5 ./data/splits/ ./results/baseline ./model_configs/standard Restaurant ./log_dir/Restaurant_baseline_extra.log --model_names "lstm" "lstm_random" "ds_elmo_t_embedding_tune_restaurant" "ds_elmo_t_embedding_restaurant" "ds_elmo_t_restaurant" "elmo_t" --model_name_save_names "lstm" "lstm_random" "ds_elmo_t_embedding_tune" "ds_elmo_t_embedding" "ds_elmo_t" "elmo_t"


python run_models.py 1 ./data/splits/ ./results/baseline ./model_configs/standard Laptop ./log_dir/Laptop_baseline_extra_fine.log --model_names "ds_elmo_t_fine_tune_laptop" --model_name_save_names "ds_elmo_t_fine_tune" - Acc val, test 74.46, 77, macro f1 val, test 66.85, 71 epoch 30

## References

1. [SemEval-2014 Task 4: Aspect Based Sentiment Analysis](https://aclanthology.info/papers/S14-2004/s14-2004)
2. [TDParse: Multi-target-specific sentiment recognition on Twitter](https://aclanthology.info/papers/E17-1046/e17-1046)

## Number of instances
Laptop Train - 1851 - 58 batches with batch size 32
Restaurant Train - 2882 - 91 batches with batch size 32


## See the amount of the unlabelled data in amazon, yelp, and elections that have bad unicode
To do this we are going to use the [ftfy package](https://pypi.org/project/ftfy/). First we are going to see the scale of the problem using the following script:
``` bash
python text_encoding_issues_ftfy.py ../MP-Tweets/filtered_split_train.txt
python text_encoding_issues_ftfy.py ../amazon/filtered_split_train.txt
python text_encoding_issues_ftfy.py ../yelp/splits/filtered_split_train.txt
```
However the problem is more difficult than I relaise as this needs to be fixed before the tokenization happens as shown below:
```python
ftfy_text = 'All mounts have different tv \'s on them , One is a 50&#34 ; , 40&#34 ; and a 32". Product comes with multiple screw for different tvs .\n'
text_before_hand = "All mounts have different tv 's on them , One is a 50&#34 ; , 40&#34 ; and a 32&#34;. Product comes with multiple screw for different tvs .\n"
```
As we can see it should have 40 and 50 as 40" and 50" like it corrected the 32. However this has not happened due to the tokenization.

## Training a Target Extraction method:
We want to find new targets within large samples of text so that we can then uses these to help augmentation. To do so first we must train our Target Extraction models. We will use a standard LSTM based approach and use the domain specific ELMo Transformer models to help with the word representations. We shall do this for each of the datasets.

### Amazon (SemEval 2014 Laptop domain)
``` bash
python target_extraction_train_predict.py semeval_2014 --train_fp ../../Music/original_target_datasets/semeval_2014/SemEval\'14-ABSA-TrainData_v2\ \&\ AnnotationGuidelines/Laptop_Train_v2.xml --test_fp ../../Music/original_target_datasets/semeval_2014/ABSA_Gold_TestData/Laptops_Test_Gold.xml --number_to_predict_on 1000000 --batch_size 256 target_extraction_configs/amazon.jsonnet ./target_extract_models/amazon ../amazon/filtered_split_train.txt /tmp/amazon_predicted_targets.txt
```
This should produce a Test F1 score of around: 0.85 (0.85423197492163) which is around the state-of-the-art performance ([0.8426](https://www.aclweb.org/anthology/N19-1242)), this also takes around 82 minutes to make predictions for all 1,000,000 sentences.

### Yelp (SemEval 2014 Restaurant domain)
``` bash
python target_extraction_train_predict.py --train_fp ../../Music/original_target_datasets/semeval_2014/SemEval\'14-ABSA-TrainData_v2\ \&\ AnnotationGuidelines/Restaurants_Train_v2.xml --test_fp ../../Music/original_target_datasets/semeval_2014/ABSA_Gold_TestData/Restaurants_Test_Gold.xml --number_to_predict_on 1000000 --batch_size 256 semeval_2014 ./target_extraction_configs/yelp.jsonnet ./target_extract_models/yelp ../yelp/splits/filtered_split_train.txt /tmp/yelp_predicted_targets.txt
```
This should produce a Test F1 score of around 0.88 (0.882843352347521) which beats the state-of-the-art on this dataset ([85.61](https://www.ijcai.org/proceedings/2018/0583.pdf)), this also takes around 69 minutes to make predictions for all 1,000,000 sentences.

### MP Tweets (Twitter Election dataset)
``` bash
python target_extraction_train_predict.py --number_to_predict_on 1000000 --batch_size 256 election_twitter ./target_extraction_configs/mp.jsonnet ./target_extract_models/mp ../MP-Tweets/filtered_split_train.txt /tmp/mp_predicted_targets.txt
```
This should produce a Test F1 score of around 0.8778 (0.8778369844089204) (no baseline paper to compare to), this also takes around 104 minutes to make predictions fro all 1,000,000 sentences.

# Extract the predicted targets
The data from the predicted targets can be found at the following Path `../predicted_targets_train.txt` which used a state of the art Target Extraction method. We want to first find all of the targets and then find the related confidence scores.

