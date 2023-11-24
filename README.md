# 2023 Samsung AI Challenge : Image Quality Assessment - Mos

- 카메라 영상 화질 정량 평가 및 자연어 정성 평가를 동시 생성하는 알고리즘 개발

- 종합 등수 : 6th
- coworker : [화질 정성 평가 code](https://github.com/GeonHyeock/Samsung-Image-Quality-Assessment-Captioning)

## Member🔥
| [박주용](https://github.com/lliee1)| [허건혁](https://github.com/GeonHyeock) |
| :-: | :-: |
| <img src="https://avatars.githubusercontent.com/lliee1" width="100"> | <img src="https://avatars.githubusercontent.com/GeonHyeock" width="100"> |
***


## Index
* [Competition imformation](#competition-imformation)
* [Code reproduction](#code-reproduction)
***

### Competition imformation

- 주관 : 삼성전자 SAIT
- 운영 : 데이콘
- 대회 : [link](https://dacon.io/competitions/official/236134/overview/description)

대회 기간 : 2023-08-21-11:00 ~ 2023-10-02-10:00

Input : 사진 \
Output : 화질 평가 점수(float)

평가 산식 : $(PLCC + SLCC)\over2$

---
### Set Environment
1. clone repo

2. build image
~~~md
docker build -t my_image:1.0.0 .

(you can change name and tag)
~~~

3. docker run
~~~md
docker run -it --name {name_you_want} --gpus all --ipc=host my_image:1.0.0 /bin/bash
~~~




---
### Code reproduction

※ [raw_data](https://dacon.io/competitions/official/236134/data)를 [data folder](data)에 저장

~~~md
# Data preprocessing
train_image_path : /data/train
test_image_path : /data/test
~~~

~~~md
# Model Train
cd /lightning-hydra-template/src
python train.py model=maniqa_{model_size}_model data=maniqa_{model_size}_data trainer.devices={num_your_device} trainer=ddp

{model_size} = 384,448,640

~~~

~~~md
# Another fold for training
default fold is fold0. (There are fold0,1,2,3,4)

if you want another fold for train,
you should change train,val_csv_file directory at 

'/lightning-hydra-template/configs/data/maniqa_{model_size}_data.yaml' 
{model_size} = 384, 448, 640

Ex) fold1
train_csv_file: ${paths.data_dir}/train_only_mos/train_df_fold1.csv
valid_csv_file: ${paths.data_dir}/train_only_mos/val_df_fold1.csv 

~~~

---
### Insert your weight.ckpt into /lightning-hydra-template/weight

~~~md
# Model Inference
cd /lightning-hydra-template/src
python eval.py ckpt_path=../weight/{your_weight_file} model=maniqa_{model_size}_model data=maniqa_{model_size}_data model.name={name_of_your_inference_csv_file}
~~~

---

### Score Ranking
|Type|score|Rank|
| :---: | :---: | :---: |
| Public | 0.7874 | 6 |
| Private | 0.57225 | 5 |
---


### [report](Dacon_challenge_math_giveup_final.pdf)


### [코드 재현을 위한 상세 설명](data/README.md)