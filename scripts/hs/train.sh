#!/bin/bash
set -e

seed=0
# mined_num=$1
mod='hard'
freq=3
vocab="data/hs/vocab_${mod}.freq${freq}.bin"
train_file="data/hs/train_${mod}.bin"
dev_file="data/hs/dev_${mod}.bin"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
lr=0.001
lr_decay=0.5
batch_size=10
max_epoch=10
beam_size=15
lstm='lstm'  # lstm
lr_decay_after_epoch=15
model_name=${mod}_hs.${lstm}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dr${dropout}.lr${lr}.lr_de${lr_decay}.lr_da${lr_decay_after_epoch}.beam${beam_size}.$(basename ${vocab}).$(basename ${train_file}).glorot.par_state.seed${seed}

# commandline="-batch_size 10 -max_epoch 200 -valid_per_batch 280 -save_per_batch 280 -decode_max_time_step 350 -optimizer adadelta -rule_embed_dim 128 -node_embed_dim 64 -valid_metric bleu"
# echo "**** Writing results to logs/conala/${model_name}.log ****"
# mkdir -p logs/conala
# echo commit hash: `git rev-parse HEAD` > logs/conala/${model_name}.log

# python3.7 tmpTest.py \
#     --mod ${mod}

# CUDA_VISIBLE_DEVICES=2 python3.7 -u exp.py \
    # --cuda \
python -u exp.py \
    --seed ${seed} \
    --mod ${mod} \
    --relation \
    --mode train \
    --batch_size ${batch_size} \
    --asdl_file asdl/lang/py3/py3_asdl.simplified.txt \
    --transition_system python3 \
    --evaluator hs_evaluator \
    --train_file ${train_file} \
    --dev_file ${dev_file} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 50 \
    --glorot_init \
    --lr ${lr} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --max_epoch ${max_epoch} \
    --beam_size ${beam_size} \
    --log_every 50 \
    --decode_max_time_step 350 \
--save_to saved_models/hs/${model_name} 2>&1 | tee logs/hs/${model_name}.log

# . scripts/hs/test.sh saved_models/hs/${model_name}.bin 2>&1 | tee -a logs/hs/${model_name}.log

