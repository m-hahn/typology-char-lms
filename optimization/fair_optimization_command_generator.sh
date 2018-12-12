#!/bin/bash
#
language=$1

for batchsize in 32 
do
    for hidden_dim in 1024 512
    do
	for char_embedding_size in 50 100
	do
            for layer_num in 2 1   
            do
		for weight_dropout_in in 0.05 0.3 0.5
		do
                    for weight_dropout_hidden in 0.05 0.3 0.5
                    do
			for char_dropout_prob in 0.05
			do
                            for learning_rate in 0.05 1 10
                            do
				lossfilename="/private/home/mbaroni/acqdiv/char-lms/optimization/"$language"_hp_final/loss_batchsize_"$batchsize"_hidden_dim_"$hidden_dim"_char_embedding_size_"$char_embedding_size"_layer_num_"$layer_num"_weight_dropout_in_"$weight_dropout_in"_weight_dropout_hidden_"$weight_dropout_hidden"_char_dropout_prob_"$char_dropout_prob"_learning_rate_"$learning_rate".txt"

				echo python3 /private/home/mbaroni/acqdiv/char-lms/acqdiv_split/lm-acqdiv-split.py --language $language --batchSize $batchsize --char_embedding_size $char_embedding_size --hidden_dim $hidden_dim --layer_num $layer_num --weight_dropout_in $weight_dropout_in --weight_dropout_hidden $weight_dropout_hidden --char_dropout_prob $char_dropout_prob --char_noise_prob 0.0 --learning_rate $learning_rate --sequence_length 50 --out-loss-filename $lossfilename
                            done
			done
                    done
                done
            done
        done
    done
done




                   
