# 2023 Samsung AI Challenge : Image Quality Assessment - Mos

- 카메라 영상 화질 정량 평가 및 자연어 정성 평가를 동시 생성하는 알고리즘 개발

- 종합 등수 : 6th
- [화질 정성 평가 code]()

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

평가 산식 : (PLCC + SLCC) / 2

---

### Code reproduction

1. [raw_data](https://dacon.io/competitions/official/236134/data)를 [data folder](data)에 저장 

~~~md
# Data preprocessing
train_image_path : dacon/data/train
test_image_path : dacon/data/test
~~~

~~~md
# Model Train
cd dacon/lightning-hydra-template/src
python train.py model=maniqa_{model_size}_model data=maniqa_{model_size}_data trainer.devices={num_your_device} trainer=ddp

model_size = (384,448,640)
~~~


you must insert your weight.ckpt into dacon/lightning-hydra-template/weight
~~~md
# Model Inference
cd dacon/lightning-hydra-template/src
python eval.py ckpt_path=../weight/{your_weight_file} model=maniqa_{model_size}_model data=maniqa_{model_size}_data model.name={name_of_your_inference_csv_file}
~~~

---

### Score Ranking
|Type|score|Rank|
| :---: | :---: | :---: |
| Public | 0.7874 | 6 |
| Private | 0.57225 | 5 |
---


### [report](Dacon_challenge.pdf)