#!/bin/bash
#
language=$1

for batchsize in  128  32 
do
    for hidden_dim in 2048  1024 
    do
        for layer_num in 3  2 1   
        do
            for weight_dropout_in in 0.05 0.3 0.4
            do
                for weight_dropout_hidden in 0.05  0.3 0.4
                do
                    for char_dropout_prob in 0.05 0.2
                    do
                        for learning_rate in 0.05 0.5 1
                        do
			    echo python3 /private/home/mbaroni/acqdiv/char-lms/lm-acqdiv.py --language $language --batchSize $batchsize --char_embedding_size 100 --hidden_dim $hidden_dim --layer_num $layer_num --weight_dropout_in $weight_dropout_in --weight_dropout_hidden $weight_dropout_hidden --char_dropout_prob $char_dropout_prob --char_noise_prob 0.0 --learning_rate $learning_rate --sequence_length 50
                        done
                    done
                done
            done
        done
    done
done




                   
