#!/bin/bash
python_exec=$1
result_dir=$2
data_dir=$3
metric=$4
domain=$5
bootstrap_samples=$6
is_val=$7
verbose=$8

for p in 0.05 0.1
do
    for worse_second in 'worse' 'second_best'
    do
        if [ $worse_second = 'second_best' ]
        then
            if [ $is_val = 'true' ]
            then
                if [ $verbose = 'true' ]
                then
                    $python_exec is_k_significant.py 5 "$result_dir" "$data_dir" "$metric" "$domain" $p --bootstrap_samples $bootstrap_samples --val --second_best --verbose
                else
                    $python_exec is_k_significant.py 5 "$result_dir" "$data_dir" "$metric" "$domain" $p --bootstrap_samples $bootstrap_samples --val --second_best
                fi
            else
                if [ $verbose = 'true' ]
                then
                    $python_exec is_k_significant.py 5 "$result_dir" "$data_dir" "$metric" "$domain" $p --bootstrap_samples $bootstrap_samples --second_best --verbose
                else
                    $python_exec is_k_significant.py 5 "$result_dir" "$data_dir" "$metric" "$domain" $p --bootstrap_samples $bootstrap_samples --second_best
                fi
            fi
        else
            if [ $is_val = 'true' ]
            then
                if [ $verbose = 'true' ]
                then
                    $python_exec is_k_significant.py 5 "$result_dir" "$data_dir" "$metric" "$domain" $p --bootstrap_samples $bootstrap_samples --val --verbose
                else
                    $python_exec is_k_significant.py 5 "$result_dir" "$data_dir" "$metric" "$domain" $p --bootstrap_samples $bootstrap_samples --val
                fi
            else
                if [ $verbose = 'true' ]
                then
                    $python_exec is_k_significant.py 5 "$result_dir" "$data_dir" "$metric" "$domain" $p --bootstrap_samples $bootstrap_samples --verbose
                else
                    $python_exec is_k_significant.py 5 "$result_dir" "$data_dir" "$metric" "$domain" $p --bootstrap_samples $bootstrap_samples
                fi
            fi
        fi
    done
done