---
title: "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
tags:
  - Diffusion Model
  - Generative Model
categories:
  - Deep Learning Paper

date: 2023-01-17
toc: true
toc_sticky: true
toc_icon: "sticky-note"
use_math: true
last_modified_at: 2023-01-20T11:30:40
---


## Abstract

- 머신러닝에서 가장 중요한 문제는 learning, sampling, inference, evalution에서 계산하기 쉬운 유연하고 친숙한 확률분포를 사용해서 
복잡한 데이터셋을 모델링하는 것과 관련이 깊다.

- 다루기쉬우면서 동시에 유연한 접근법을 개발했다. 가장 중요한 아이디어는 non-equilibrium statistical physical로 부터 영감을 받았다. 
이 아이디어는 반복적인 forward diffusion process를 통해 전체적인 데이터 분포 구조를 천천히 파괴한다. 
- 그런 이후 매우 유연하고 다루기 쉬운 데이터의 generative model을 생성하기 위해 데이터에 저장된 구조인 reverse diffusion process를 학습한다.

- 이 접근 방식을 통해 수천 개의 계층 또는 시간 단계가 있는 심층 생성 모델에서 확률을 빠르게 학습, 
샘플링 및 평가할 수 있을 뿐만 아니라 학습된 모델에서 조건부 및 사후 확률을 계산할 수 있습니다.

## Introduction

- 확률 모델은 tractability, flexibilty의 2개의 상반된 목표로 부터 고통받았다. tractable한 모델은 데이터를 쉽게 학습하지만 이러한 모델은 
풍부한 데이터셋에서 전체 구조를 적절하게 묘사하는 것은 불가능하다.

- 반면에 flexible한 모델은 arbitrary한 data에 잘 학습된다. 예를들어 모델을 어떤 non-negative한 
flexible distribution $p(x)= \frac{\phi(x)}{Z}$로 부터 추출된 함수 $\phi (x)$로도 정의가 가능하다. 여기서 $Z$는 normalization constant이다.

  - 하지만 이러한 normalization constant는 일반적으로 매우 다루기 어렵다(intractable).
  - 이런 flexible model로 evaluating, trainging, drawing sample을 하는건 매우 많은 비용을 요구하는 Monte Carlo process를 필요로 한다.

- 분석적인 근사법의 다양성은 이런 제거하지 못하는 trade-off를 개선하기 위해 존재하는데 예를들어 mean field theory and its expansions (T. 1982; Tanaka, 1998),
variational Bayes(Jordan et al., 1999), constrasive divergence(Welling & Hinton, 2002; Hinton, 2002), minimum probability flow(Sohl-Dickstein et al. 2011b;a),
minimum KL-constraction (Lyu, 2011), proper scoring rules (Gneiting & Raftery, 2007; Parry et al., 2012), score matching( Hyvarinen, 2005)
psedolikelihood(Besag, 1975), loopy belief propagation(Murphy et al. 1999) 등 많고 Non-parametric methods (Gershman & Blei, 2012) 역시 매우 효과적이다.

### 1.1 Diffusion probabilistic models

- 다음을 따르는 probabilistic model를 정의하는 뛰어난 방법을 제안한다.
    1. extreme flexibility in model structure
    2. exact sampling 
    3. posterior를 계산하기 위해 다른 확률 분포 끼리 쉬운 곱셈
    4. the model log likelihood와 the probability of individual states를 평가하기 매우 쉬움

- 이 방법은 Markov chain을 사용해서 점진적으로 어떤 분포로부터 다른 분포로 변환하는 것이다. 이 방법은 non-equilibrium statistical physics(Harzynski, 1997)와 
sequential Monte Carlo (Neal, 2001)에서 사용되었다.

- diffusion process를 사용해서 잘 알려진 단순한 분포에서 (예를들어 Gaussian) target data 분포로 변환하는 generative Markov chain을 만든다.

- 다르게 정의된 모델을 대략적으로 평가하기 위해 이 Markov chain을 사용하는 대신 확률 모델을 Markov chain의 끝점으로 명시적으로 정의한다.

- diffusion chain에서 각 단계는 analytically evaluable probability을 가지고 있기 때문에 full-chain 역시 analytically evaluable이다.

---

- 이 framework에서 학습하는 것은 diffusion process에서 작은 방해(perturbations) Noise을 추정하는 것과 관련있다. 
- small perturbation을 추정하는 것은 single, non-analytically-normalizable의 가능성을 내포한 함수의 전체 확률을 추정하는 것보다 더 다루기 쉽다.
- 게다가 diffusion process은 any smooth target distribution을 목적으로 하기 때문에 어떠한 arbitrary한 형태의 data distribution도 정확히 담아낼수 있다.

---

- 이런 diffusion probabilistic model의 유용함을 보이기 위해 2차원의 스위스 롤 형태, binary sequence, MNIST를 위한 log likelihood 모델을 학습해보았다.


### 1.2 Relationship to other work

- The wake-sleep 알고리즘(Hinton, 1995; Dayan et al., 1995)은 각각 서로 대응되는 추론 확률 모델과 생성 확률 모델을 학습하는 방법을 제시했다.
  - 이 접근법은 간간히 작은 연구들은(Sminchisescu et al. 2006; Kavukcuoglu et al., 2010) 있었지만 거의 20년동안 크게 연구되고 있지 않은채 남아 있었다.
  - 최근에 이 방법을 개발하는데 많은 연구가 진행되고 있다.
  - (Kingma &Welling, 2013; Gregor et al., 2013; Rezende et al., 2014; Ozair & Bengio, 2014) variational learning 및 inference 알고리즘은 
잠재 변수(latent variable)에 대한 유연한 생성 모델 및 사후 분포를 서로에 대해 직접 훈련할 수 있도록 개발되었습니다.

---

- 이러한 논문들의 variational bound은 우리의 training objective에서 사용된 것과 유사하거나 더 이전의 연구된 것(Sminchisescu et al., 2006)과 유사하다.
- 하지만 우리의 motivation과 model 형태 둘다 다르고 현재의 연구들은 다음의 이러한 technique와 관련된 차이점과 이점을 채택하고 있다.

  1. 우리는 변형 베이지안 방법이 아닌 물리학, quasi-static process 및 annealed importance sampling의 아이디어를 사용하여 프레임워크를 개발합니다.
  2. 서로 다른 확률 분포와 학습된 분포의 곱셈이 얼마나 쉽게 수행되는지 보인다.(사후 확률을 계산하기 위해 조건부 확률과의 곱셈)
  3. inference와 generative model간의 objective에서의 불균형 때문에 varaiational inference method에서 특히 infernece model을 학습하는 것이 어렴다는 것을 설명한다.
  4. 몇개 안되는 layer를 다루기 보다 수천개의 layer를 가진 모델을 학습한다.
  5. 각 layer에서 entropy production의 하한 상한 경계선을 준다.


- 확률 모델을 학습하기 위한 관련된 기술들이 많이 있다. 이하에서 요약 ~~~~~~~~~~~

Related ideas from physics include the Jarzynski equality
(Jarzynski, 1997), known in machine learning as An

























