#!/bin/bash

echo "bias used is $1"
echo "using GPU $2"

export BERT_DIR=/home/bert/uncased_L-12_H-768_A-12

CUDA_VISIBLE_DEVICES=$2 python run_classifier.py \
	  --task_name=QQP\
	    --do_train=true \
	      --do_eval=true \
	       --do_predict=true \
	         --data_dir=/home/data/QQP \
		   --vocab_file=$BERT_DIR/vocab.txt \
		     --bert_config_file=$BERT_DIR/bert_config.json \
		       --init_checkpoint=$BERT_DIR/bert_model.ckpt \
		         --max_seq_length=128\
			   --train_batch_size=32\
			     --learning_rate=3e-5 \
			       --num_train_epochs=3.0 \
			         --output_dir=/home/bert/qqp_models/$1 \
				  --which_bias=$1 \
				  --debiasmode=poe

