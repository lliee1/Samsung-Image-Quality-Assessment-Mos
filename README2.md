# 2023 Samsung AI Challenge : Image Quality Assessment - Mos (Train, Inference Explain)
~~~md
# 1. train, test image를 다음 경로에 넣어주세요
train_image_path : dacon/data/train
test_image_path : dacon/data/test
~~~

~~~md
# Model Train

# 경로 이동 (중요)
cd dacon/lightning-hydra-template/src

# train 명령어
python train.py model=maniqa_{model_size}_model data=maniqa_{model_size}_data trainer.devices={num_your_device} trainer=ddp

{model_size} = 384,448,640

예시) 
[384 model, gpu 4개, ddp 사용]
>> python train.py model=maniqa_384_model data=maniqa_384_data trainer.devices=4 trainer=ddp

[448 model, gpu 4개, ddp 사용]
>> python train.py model=maniqa_448_model data=maniqa_448_data trainer.devices=4 trainer=ddp

[640 model, gpu 4개, ddp 사용]
>> python train.py model=maniqa_640_model data=maniqa_640_data trainer.devices=4 trainer=ddp




~~~

~~~md
# 각 모델에는 5fold를 실험할 수 있는 환경이 구성되어 있습니다.
default fold는 fold0. (fold0,1,2,3,4으로 실험 가능)


# 다른 fold의 dataset으로 train하고 싶다면, 하단 경로의 yaml파일에 들어가 csv 이름을 변경해야합니다.
'dacon/lightning-hydra-template/configs/data/maniqa_{model_size}_data.yaml' 
{model_size} = 384, 448, 640


Ex) fold0
train_csv_file: ${paths.data_dir}/train_only_mos/train_df_fold0.csv
valid_csv_file: ${paths.data_dir}/train_only_mos/val_df_fold0.csv 


Ex) fold1
train_csv_file: ${paths.data_dir}/train_only_mos/train_df_fold1.csv
valid_csv_file: ${paths.data_dir}/train_only_mos/val_df_fold1.csv 

Ex) fold2
train_csv_file: ${paths.data_dir}/train_only_mos/train_df_fold2.csv
valid_csv_file: ${paths.data_dir}/train_only_mos/val_df_fold2.csv 

Ex) fold3
train_csv_file: ${paths.data_dir}/train_only_mos/train_df_fold3.csv
valid_csv_file: ${paths.data_dir}/train_only_mos/val_df_fold3.csv 

Ex) fold4
train_csv_file: ${paths.data_dir}/train_only_mos/train_df_fold4.csv
valid_csv_file: ${paths.data_dir}/train_only_mos/val_df_fold4.csv 



# 모델은 총 384, 448, 640 model 3개, fold는 0,1,2,3,4로, 모든 train을 마치게 되면, 15개의 weight.ckpt를 얻을 수 있습니다.
# weight는 dacon/lightning-hydra-template/logs/train/runs/{돌린 날짜와 시간}/checkpoints 에 저장됩니다.
~~~

---

~~~md
# Inference에 사용한 weight는 함께 첨부드린 weight.zip파일에 있으며
# zip파일에 있는 15개의 weight를 dacon/lightning-hydra-template/weight 경로에 넣어주시면 하단의 명령어를 이용해 Inference 가능합니다.
~~~
~~~md
# Model Inference

# 경로 이동 (중요)
cd dacon/lightning-hydra-template/src

# Inference 명령어
python eval.py ckpt_path=../weight/{your_weight_file} model=maniqa_{model_size}_model data=maniqa_{model_size}_data model.name={name_of_your_inference_csv_file}


예시)
[384 model의 384_fold0.ckpt 사용]
python eval.py ckpt_path=../weight/384_fold0.ckpt model=maniqa_384_model data=maniqa_384_data model.name=384_fold0


[448 model의 448_fold1.ckpt 사용]
python eval.py ckpt_path=../weight/448_fold1.ckpt model=maniqa_448_model data=maniqa_448_data model.name=448_fold1


[640 model의 640_fold2.ckpt 사용]
python eval.py ckpt_path=../weight/640_fold2.ckpt model=maniqa_448_model data=maniqa_448_data model.name=640_fold2


# 위 예시를 활용해서 각 모델(384, 448, 640)에 fold0,1,2,3,4 weight를 사용하여 Inference를 진행하면,
# 15개의 csv파일을 얻을 수 있습니다.
# csv파일은 dacon/result_csv에 저장되며, 최종 final.csv는 dacon/result_csv/15fold.py를 실행시키면 최종 mos 결과를 얻을 수 있습니다.

~~~

---

### Score Ranking
|Type|score|Rank|
| :---: | :---: | :---: |
| Public | 0.7874 | 6 |
| Private | 0.57225 | 5 |
---


### [report](Dacon_challenge_math_giveup.pdf)