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

gpus="[0]"

trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
score_norm_method="asnorm"  # asnorm/snorm
top_n=300

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract embeddings ..."
  local/extract_vox.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 4 --gpus $gpus --data_type $data_type --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Score ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Score norm ..."
  local/score_norm.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set vox2_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi