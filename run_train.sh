#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)

. ./path.sh || exit 1

stage=-1
stop_stage=-1

data=data
data_type="shard"  # shard/raw

config=conf/ens_kd/ens_wavlm_campplus.yaml
exp_dir=exp/EnsDistill_WavLM_CAMPPlus-G2-TSTP-emb512-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150
model_path=$exp_dir/models/enkd_campplus.pt

gpus="[0,1,2]"

trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
score_norm_method="asnorm"  # asnorm/snorm
top_n=300

. tools/parse_options.sh || exit 1

echo "Start training ..."
num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/bin/train.py --config $config \
    --exp_dir ${exp_dir} \
    --gpus $gpus \
    --num_avg ${num_avg} \
    --data_type "${data_type}" \
    --train_data ${data}/vox2_dev/${data_type}.list \
    --train_label ${data}/vox2_dev/utt2spk \
    --reverb_data ${data}/rirs/lmdb \
    --noise_data ${data}/musan/lmdb \
    ${checkpoint:+--checkpoint $checkpoint}

echo "Do model average ..."
avg_model=$exp_dir/models/avg_model.pt
python wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}
