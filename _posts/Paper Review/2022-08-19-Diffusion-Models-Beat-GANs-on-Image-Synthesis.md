---
title: "[논문 리뷰]Diffusion Models Beat GANs on Image Synthesis(작성중)"
tags:
  - Diffusion Model
  - GAN
  - Image Generation
categories:
  - Image Generation Paper 
date: 2022-08-19
toc: true
toc_sticky: true
toc_icon: "sticky-note"
use_math: true
---

### [1] Intro
- 지난 몇년 동안 generative model은 거의 사람과 유사한 능력(language, Image, Speech, music 분야에서) 얻음
  - 이러한 모델들은 텍스트 입력으로 부터 이미지를 생성하거나 유용한 feature representation을 학습과 같은 여러 분야에서 여러 방법으로 사용된다.
  - 아직 개선점이 많지만 이런 모델들이 실제 이미지나 sound를 지금도 생성하고 있고 더나은 generative 모델은 그래픽 디자인, 게임과 같은 셀수 없는 더 넓은 분야에 영향을 미칠 것이다.
- 지금에도 GANs 모델은 sample의 quality를 측정하는 방법인 FID[23], Inception Score(IS)[54], Precision[32]으로 평가되는 대부분의 image generation task에서 SOTA를 유지하고 있다.
  - 이미지 생성 Quality를 평가하는 FID, IS, Precision 방법은 이미지 생성에서 Diversity(다양성)을 잘 평가하지 못함.
  - 그래서 GAN모델은 SOTA likelihood-based models[51, 43, 42]보다 Diversity(다양성)면에서 좋지 않은 모습을 보여준다.
  - 게다가 GANs 모델은 hyperparameter와 regularizer를 신중하게 선택하지 않으면 수렴하지 않기 떄문에 모델을 학습하는데 난이도가 높다.
- GAN 모델은 SOTA를 유지하고 있는 반면 새로운 도메인에 적용하고 확장시키는데 어려움을 가지고 있다.
  - 결과적으로 많은 연구들이 likelihood-based models[51, 25, 42, 9]을 통해 GAN과 비슷한 수준의 sample quality를 갖도록 진행 되었다.
  - likelihood-based model들은 GAN모델 보다 학습이 쉽고 확장하기도 편하고 더욱 많은 다양성을 갖지만 여전히 GAN보다 visual sample quality가 떨어졌다.
  - 게다가 VAE를 제외하고는 likelihood-based model가 이미지를 생성하는 시간은 GAN보다 느리다.
  - 