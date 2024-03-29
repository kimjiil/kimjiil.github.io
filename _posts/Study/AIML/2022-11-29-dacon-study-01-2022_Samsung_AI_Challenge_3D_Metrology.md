---
title: "Dacon Study (1) - 2022 Samsung AI Challenge 3D Metrology "
tags:
  - Pytorch
  - Deep Learning
  - Dacon
categories:
  - AI/ML Study
date: 2022-11-29
toc: true
toc_sticky: true
toc_icon: "sticky-note"
use_math: true
last_modified_at: 2023-01-05T13:14:34
---

<hr/>

실험 결과 링크    
[[wandb link]](https://wandb.ai/kimjiil2013/221227_Samsung_sem?workspace=user-kimjiil2013){:target="_blank"}

코드 및 자료 링크    
[[github link]](https://github.com/kimjiil/AIML_Competition/tree/main/2022-Samsung-AI-Challenge-3D-Metrology){:target="_blank"}

<hr/>

### 어떤 대회인지

[[대회 링크]](https://dacon.io/competitions/official/235954/overview/description){:target="_blank"}

#### 배경

> 최근 반도체 구조의 폭, 물성 등 정량적으로 Monitoring하는 계측 분야가 반도체 구조가 미세화, 복잡화되면서 더욱 중요해지고 있으며,
> 이 분야에 AI 알고리즘을 개발하고자 하는 시도가 반도체 제조사에서 다양하게 이루어지고 있습니다.
>
> 대표적인 반도체 계측 방식은 상부에서 촬영한 (Top-down) SEM (주사 전자 현미경, Scanning Electron Microscope) 영상을 활용하는 것으로, 
> 구조별 2차원 정보인 폭/두께 계측으로 한정되어 사용되어 있으며, 현재 깊이를 계측하기 위해서 OCD (Optical Critical Dimension), 
> TEM (Transmission Electron Microscope) 영상 등을 활용하고 있습니다.



#### 설명

- Top-down으로 취득한 SEM 영상으로부터 깊이 (Depth, 깊을수록 작은 값)를 예측

<p align="center">
<img src="/assets/images/2022-11-29-dacon-study-01-2022_Samsung_AI_Challenge_3D_Metrology/description_image.webp"
height="80%" width="80%">
</p>

<hr/>

### Dataset

[2022 Samsung AI Challenge (3D Metrology) Dataset Info.]
    
- Train Dataset (학습용 데이터셋, 학습 가능) - 총 60664개
  - SEM [폴더] : 실제 SEM 영상을 Hole 단위로 분할한 영상 (8bit Gray 영상)
  - average_depth.csv : 전체 SEM 영상과 대응되는 평균 Depth

<p align="center">
<img src="/assets/images/2022-11-29-dacon-study-01-2022_Samsung_AI_Challenge_3D_Metrology/train_sem_image.webp"
height="80%" width="80%">
<figcaption align="center"> Train SEM Image </figcaption>
</p>

<p align="center">
<img src="/assets/images/2022-11-29-dacon-study-01-2022_Samsung_AI_Challenge_3D_Metrology/dataset_fig_01.webp"
height="60%" width="60%">
<figcaption align="center"> Train SEM Image의 평균 pixel 값, 대응되는 평균 Depth의 분포 </figcaption>
</p>


- Simulation Dataset (학습용 데이터셋, 학습 가능) - 총 259956개
  - SEM [폴더] : Simulator을 통해 생성한 Hole 단위 SEM 영상 (실제 SEM 영상과 유사하나, 대응 관계는 없음)
  - Depth [폴더] : Simulator을 통해 얻은 SEM 영상과 Pixel별로 대응되는 Depth Map
  - Depth 이미지 1개당 2개의 Simulator Hole 단위 SEM 영상이 Pair하게 매칭됩니다. (Name_itr0, Name_itr1)

<p align="center">
<img src="/assets/images/2022-11-29-dacon-study-01-2022_Samsung_AI_Challenge_3D_Metrology/sim_sem_image.webp"
height="80%" width="80%">
<figcaption align="center"> Simulation SEM Image </figcaption>
</p>

<p align="center">
<img src="/assets/images/2022-11-29-dacon-study-01-2022_Samsung_AI_Challenge_3D_Metrology/sim_depth_image.webp"
height="80%" width="80%">
<figcaption align="center"> Simulation Depth Image </figcaption>
</p>

<p align="center">
<img src="/assets/images/2022-11-29-dacon-study-01-2022_Samsung_AI_Challenge_3D_Metrology/dataset_fig_02.webp"
height="60%" width="60%">
<figcaption align="center"> Simulation SEM Image와 Depth Image의 평균 pixel값의 분포 </figcaption>
</p>


- Test Dataset (평가를 위한 테스트 데이터셋, 학습 불가능) - 총 25988개 
  - SEM [폴더] : 실제 SEM 영상을 Hole 단위로 분할한 영상 (8bit Gray 영상)

- sample_submission.zip (제출 양식) - 총 25988개
  - 실제 Hole 단위 SEM 영상으로부터 추론한 Depth Map (webp 파일)

<hr/>

### Method

#### 1. Basic AE Model(Tutorial)
대회에서 주어진 Basic Code는 AutoEncoder 모델로 공유된 코드로 학습 진행

<script src="https://gist.github.com/kimjiil/38bfd83aeeb345148a23a8530ed1cc1e.js"></script>

#### 2. Cycle GAN Model

대회에서 주어진 데이터셋은 Simulator로 생성한 SEM 이미지에 대응 되는 depth map과 실제 데이터인 Train 이미지에 대응되는 avg depth값이다.
simulation sem image가 실제 sem image와 같은 이미지면 단순히 simulation sem image를 입력으로 받고 depth map을 결과로
하는 생성모델을 만들어 학습하면 될거라고 생각했다.


<p align="center">
<img src="/assets/images/2022-11-29-dacon-study-01-2022_Samsung_AI_Challenge_3D_Metrology/dataset_fig_03.webp"
height="60%" width="60%">
<figcaption align="center"> Simulation SEM Image와 Train(Real) SEM Image의 분포 차이 </figcaption>
</p>


simulation sem image와 train sem image의 분포를 단순히 이미지의 픽셀 평균값을 이용해서 표현해보니 위와 같다. 
분포의 경향은 비슷하지만 값의 차이가 커서 다른 이미지로 간주하고 생성모델을 이용해 
1차적으로 train sem image로 부터 simulation sem image를 생성하고 다시 simulation sem image로 부터 depth map을 생성하는
구조를 생각했다.

simulation sem image와 대응되는 depth map 이미지는 있지만 공지에서 "simulation sem image는 실제 SEM 영상과, 대응 관계는 없음"이라고 했으므로
대응 되는 데이터쌍이 없다.

<p align="center">
<img src="/assets/images/2022-11-29-dacon-study-01-2022_Samsung_AI_Challenge_3D_Metrology/cyclegan_fig_01.webp"
height="60%" width="60%">
<figcaption align="center"> Cycle Gan 구조 </figcaption>
</p>

simulation과 train에서 대응되는 데이터쌍이 없기 때문에 unpair dataset에서 domain shift학습이 가능한 cycle gan을 생성모델로 선택했다.

먼저 데이터 쌍이 없는 simulation/train에 대해서는 다음과 같이 generator를 학습시켰다.

<p align="center">
<img src="/assets/images/2022-11-29-dacon-study-01-2022_Samsung_AI_Challenge_3D_Metrology/training_gen_simtotrain.webp"
height="60%" width="60%">
<figcaption align="center"> simulation/train generator training </figcaption>
</p>

simulation sem/depth 데이터셋도 마찬가지로 기본적인 cycle gan loss로 학습한 결과 다음과 같이 수렴했다.

<p align="center">
<img src="/assets/images/2022-11-29-dacon-study-01-2022_Samsung_AI_Challenge_3D_Metrology/training_result_01.webp"
height="60%" width="60%">
<figcaption align="center"> simulation sem/depth generator training result </figcaption>
</p>

simulation sem 이미지의 분포와 실제 각 case(110, 120, 130, 140)에 대해서 대응되는 실제 depth map의 분포는 왼쪽과 같이 일정한 간격을 두는
분포를 가진다. 하지만 실제 학습 결과 모든 case가 전체 분포를 합친 결과로 수렴했다.

<p align="center">
<img src="/assets/images/2022-11-29-dacon-study-01-2022_Samsung_AI_Challenge_3D_Metrology/training_gen_semtodepth.webp"
height="60%" width="60%">
<figcaption align="center"> simulation sem/depth generator training </figcaption>
</p>

simulation sem과 depth의 대응되는 쌍에 대해 다음과 같은 Guided Loss L1인 주황색 실선을 추가하여 밝기 값에 대해서 더 잘 학습되도록 했다.


<p align="center">
<img src="/assets/images/2022-11-29-dacon-study-01-2022_Samsung_AI_Challenge_3D_Metrology/model structure.webp"
height="60%" width="60%">
<figcaption align="center"> case별 학습 및 모델 개요 </figcaption>
</p>

한개의 generator로만 학습할 경우 위와 같이 4개의 case를 합친 분포로 학습되는 경향이 보여 이를 4개의 case로 나눠 각각의 case에 대해서 
학습되도록 구성했다. 나눠진 case에 대해서 따로 cnn classifier를 추가하여 case 1~4에 대해서 분류하도록 학습했다. 

전체적인 과정은 classifier에서 case를 분류하면 그에 대응되는 generator를 가져와 이를 통해 depth를 추론한다

### Jupyter Notebook

<script src="https://gist.github.com/kimjiil/3a74dd262a411dae5d95463e10d1e5c9.js"></script>


submission dataset을 제출한 결과 아래와 같이 대략적인 성능이 나왔다.

<p align="center">
<img src="/assets/images/2022-11-29-dacon-study-01-2022_Samsung_AI_Challenge_3D_Metrology/cyclegan_result.webp"
height="100%" width="100%">
</p>

<p align="center">
<img src="/assets/images/2022-11-29-dacon-study-01-2022_Samsung_AI_Challenge_3D_Metrology/leader_board_chart.webp"
height="100%" width="100%">
<figcaption align="center"> 성능 리더보드 </figcaption>
</p>