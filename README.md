# Emphasized Non-Target Speaker Knowledge in Knowledge Distillation for Speaker Verification

This Repository contains the code and pretrained models for the following ICASSP 2024 paper:

* **Title** : Emphasized Non-Target Speaker Knowledge in Knowledge Distillation for Speaker Verification
* **Autor** : Duc-Tuan Truong, Ruijie Tao, Jia Qi Yip, Kong Aik Lee, Eng Siong Chng

# Prerequisites

## Environment Setting
Follow the below commands to install the required packages for preparing the dataset, training and testing the model.

``` sh
conda create -n wespeaker python=3.9
conda activate wespeaker
conda install pytorch=1.12.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Datasets

We used VoxCeleb dataset for training and test. For noise augmentation, we used the MUSAN and RIRS corpus. To download and preprocesing data, please run the following snippet  

``` sh
bash prepare_data.sh --stage 1 --stop_stage 2
```

## Pretrained Model
The pretrained teacher model WavLM-Large can be found at [microsoft/UniSpeech](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification) and download at [link](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view?usp=sharing). The WavLM-Large checkpoint should be put at `pretrained_model/`

We have uploaded pretrained models of our experiments. You can download pretrained models from [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/truongdu001_e_ntu_edu_sg/EpjkpezMPC9Fmrng3w73iVcB7aFORZoiNWg458Y3RlOGOA?e=FhYn6Q) and put in the corresponding directory in `exp/` folder. 

## Training
To perform model training, run:
```
bash run_train.sh --config=/path/to/exp_config.yaml --exp_dir=/path/to/exp_dir
```

## Testing
To run evaluation on VoxCeleb evaluation sets: Vox-O, Vox-H, and Vox-E. 
```
bash run_test.sh --stage 1 --stop_stage 3 --config=/path/to/exp_config.yaml --exp_dir=/path/to/exp_dir --model_path=/path/to/pretrained_model.pt
```

## Citation

If you find our repository valuable for your work, please consider citing our paper:
```
@INPROCEEDINGS{enskd,
  author={Truong, Duc-Tuan and Tao, Ruijie and Yip, Jia Qi and Aik Lee, Kong and Chng, Eng Siong},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Emphasized Non-Target Speaker Knowledge in Knowledge Distillation for Automatic Speaker Verification}, 
  year={2024},
  volume={},
  number={},
  pages={10336-10340},
  doi={10.1109/ICASSP48485.2024.10447160}}
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

### Acknowledge

Our work is built upon the [wenet-e2e/wespeaker](https://github.com/wenet-e2e/wespeaker) toolkit. We also follow some parts of the following codebases:

[microsoft/UniSpeech](https://github.com/microsoft/UniSpeech) (for WavLM model architechture).

[megvii-research/mdistiller](https://github.com/megvii-research/mdistiller) (for Decouple Knowledge Distillation).

Thanks for these authors for sharing their work!
