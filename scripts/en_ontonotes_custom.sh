#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



config_path=mrc-for-flat-nested-ner/configs/eng_base_case_bert.json
base_path=mrc-for-flat-nested-ner
bert_model='bert-base-cased'
task_name=ner
max_seq_length=150
num_train_epochs=4
warmup_proportion=-1
seed=3306
data_dir=/media/yaroslav/DATA/datasets/nlp/ner/ontonotes_v3_mrc
data_sign=en_onto
checkpoint=28000
gradient_accumulation_steps=4
learning_rate=6e-6
train_batch_size=2
dev_batch_size=2
test_batch_size=2
smooth=2.0
export_model=/media/yaroslav/DATA/models/mrc-ontonotes-test.th
output_dir=${export_model}



CUDA_VISIBLE_DEVICES=0 python3 ${base_path}/run/run_query_ner.py \
--config_path ${config_path} \
--data_dir ${data_dir} \
--bert_model ${bert_model} \
--max_seq_length ${max_seq_length} \
--train_batch_size ${train_batch_size} \
--dev_batch_size ${dev_batch_size} \
--test_batch_size ${test_batch_size} \
--checkpoint ${checkpoint} \
--learning_rate ${learning_rate} \
--num_train_epochs ${num_train_epochs} \
--warmup_proportion ${warmup_proportion} \
--export_model ${export_model} \
--output_dir ${output_dir} \
--data_sign ${data_sign} \
--gradient_accumulation_steps ${gradient_accumulation_steps} 
