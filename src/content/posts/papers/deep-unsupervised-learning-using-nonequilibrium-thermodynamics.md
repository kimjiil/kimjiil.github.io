---
title: "[Diffusion 계보 ①] Deep Unsupervised Learning using Nonequilibrium Thermodynamics (2015)"
date: 2023-01-17
updated: 2026-08-18
category: deep-learning-paper
tags:
  - "Diffusion Model"
  - "Generative Model"
---

<span style="font-size:17pt">
<b>Deep Unsupervised Learning using Nonequilibrium Thermodynamics</b>
</span>

<a href="https://arxiv.org/abs/1503.03585" target="_blank"><b>[PDF]</b></a>
, <b><span style="color: #F2AA4C">Generative Model</span></b>, Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli (Stanford, ICML 2015)

### <span style="color: #ffd33d">Summary</span>

**Diffusion model의 원조 논문**이다. 비평형 통계물리학(non-equilibrium statistical physics)의 아이디어를 가져와서,
데이터 분포를 Markov chain으로 **천천히 파괴(forward diffusion)** 한 다음 그 **역과정(reverse diffusion)을 신경망으로 학습**하면
임의의 복잡한 분포도 tractable 하게 모델링할 수 있다는 프레임워크를 처음 제시했다.

- forward process를 아주 작은 step으로 쪼개면, 각 step의 reverse도 forward와 **같은 함수 형태**(Gaussian이면 Gaussian)를 가진다는 것이
핵심 관찰이다. 그래서 reverse의 (평균, 공분산)만 신경망으로 추정하면 된다.
- 학습은 log likelihood의 lower bound $K$를 최대화하는 것으로 하고, 이 bound가 forward posterior와 reverse 사이의
KL divergence 합으로 정리된다는 것을 유도한다. (이 유도가 5년 뒤 DDPM의 $L_{vlb}$ 그대로다)
- 학습된 분포에 다른 분포 $r(x)$를 **곱하는 연산**(posterior 계산)이 쉽다는 것도 보이는데, 이것이 나중에
classifier guidance의 수학적 원형이 된다.

2015년 당시에는 MNIST/CIFAR-10 수준의 실험으로 끝났고 주목을 못 받다가, 2020년 DDPM[2]이 이 프레임워크를 다듬어서
고품질 이미지 생성에 성공하면서 다시 발굴된 논문이다. 계보를 따라가려면 이 논문 →
[DDPM 계열 정리 + Diffusion Models Beat GANs 리뷰](/posts/diffusion-models-beat-gans-on-image-synthesis/) 순서로 읽는 것을 추천.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[1] Abstract & Introduction</span>

#### <span style="color: #4682B4">Abstract</span>

- 머신러닝에서 가장 중요한 문제는 learning, sampling, inference, evalution에서 계산하기 쉬운 유연하고 친숙한 확률분포를 사용해서
복잡한 데이터셋을 모델링하는 것과 관련이 깊다.

- 다루기쉬우면서 동시에 유연한 접근법을 개발했다. 가장 중요한 아이디어는 non-equilibrium statistical physical로 부터 영감을 받았다.
이 아이디어는 반복적인 forward diffusion process를 통해 전체적인 데이터 분포 구조를 천천히 파괴한다.
- 그런 이후 매우 유연하고 다루기 쉬운 데이터의 generative model을 생성하기 위해 데이터에 저장된 구조인 reverse diffusion process를 학습한다.

- 이 접근 방식을 통해 수천 개의 계층 또는 시간 단계가 있는 심층 생성 모델에서 확률을 빠르게 학습,
샘플링 및 평가할 수 있을 뿐만 아니라 학습된 모델에서 조건부 및 사후 확률을 계산할 수 있습니다.

#### <span style="color: #4682B4">Introduction</span>

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

#### <span style="color: #4682B4">1.1 Diffusion probabilistic models</span>

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

#### <span style="color: #4682B4">1.2 Relationship to other work</span>

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

- 물리학 쪽 관련 아이디어로는 **Jarzynski equality**(Jarzynski, 1997)가 있다. 머신러닝에서는
**Annealed Importance Sampling(AIS)**(Neal, 2001)로 알려져 있는데, 중간 분포들의 sequence를 거쳐가며 importance weight를
곱해나가면 두 분포 사이의 ratio(물리에서는 자유에너지 차이)를 계산할 수 있다는 내용이다.
아래 2.3에서 model probability를 계산하는 트릭이 정확히 이 구조다.
- 그 외에도 Langevin dynamics(Neal, 2011), score matching과 denoising autoencoder의 관계(Vincent, 2011),
Kolmogorov forward/backward equation 등과 연결된다. 특히 **denoising autoencoder를 무한히 쌓은 것**으로 해석할 수 있다는 점이
이후 연구(DDPM의 $\epsilon$ 예측)로 이어지는 포인트다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[2] Algorithm</span>

표기법: 논문은 timestep을 위첨자로 쓴다. $x^{(0)}$가 원본 데이터, $x^{(T)}$가 완전히 파괴된 노이즈다.
(DDPM 표기로는 $x_0, x_T$에 대응)

#### <span style="color: #4682B4">2.1 Forward Trajectory</span>

- 데이터 분포 $q(x^{(0)})$를 Markov diffusion kernel $T_{\pi}(y|y';\beta)$를 반복 적용해서
**analytically tractable한 분포** $\pi(y)$로 점진적으로 변환한다.

$$
    q(x^{(t)}|x^{(t-1)}) = T_{\pi}\big(x^{(t)}|x^{(t-1)};\,\beta_t\big)
    ,\qquad
    q(x^{(0 \cdots T)}) = q(x^{(0)})\prod_{t=1}^{T}{q(x^{(t)}|x^{(t-1)})}
$$

- $\beta_t$는 step $t$에서의 diffusion rate이다. 논문은 Gaussian과 Binomial 두 종류의 kernel로 실험한다.

| | Gaussian (연속 데이터) | Binomial (이진 데이터) |
|---|---|---|
| $\pi(x^{(T)})$ | $\mathcal{N}(x^{(T)}; 0, \mathbf{I})$ | $\mathcal{B}(x^{(T)}; 0.5)$ |
| $q(x^{(t)}\|x^{(t-1)})$ | $\mathcal{N}\big(x^{(t)}; \sqrt{1-\beta_t}\,x^{(t-1)},\, \beta_t\mathbf{I}\big)$ | $\mathcal{B}\big(x^{(t)}; x^{(t-1)}(1-\beta_t) + 0.5\beta_t\big)$ |

- Gaussian kernel이 바로 DDPM의 forward 그 식이다. $\sqrt{1-\beta_t}$로 수축시키면서 $\beta_t$만큼 노이즈를 넣으면
분산이 1로 유지되면서 $T \rightarrow \infty$에서 $\mathcal{N}(0,\mathbf{I})$로 수렴한다.

#### <span style="color: #4682B4">2.2 Reverse Trajectory</span>

- 생성 모델은 같은 궤적을 **역방향으로 되짚는** Markov chain으로 정의된다. 시작점은 tractable한 $\pi$다.

$$
    p(x^{(T)}) = \pi(x^{(T)})
    ,\qquad
    p(x^{(0 \cdots T)}) = p(x^{(T)})\prod_{t=1}^{T}{p(x^{(t-1)}|x^{(t)})}
$$

- 이 논문의 **핵심 관찰**: continuous diffusion에서 $\beta$가 충분히 작으면(step을 잘게 쪼개면),
**reverse kernel도 forward kernel과 동일한 함수 형태**를 가진다. (Feller, 1949)
  - forward가 Gaussian이면 reverse도 Gaussian, forward가 binomial이면 reverse도 binomial.
  - step이 클수록 reverse는 복잡한 multimodal 분포가 되지만, 잘게 쪼개면 단순한 unimodal로 근사 가능하다는 것.
- 그래서 학습해야 할 것은 각 step의 **Gaussian의 평균과 공분산** (binomial이면 bit flip 확률) 뿐이다.

$$
    p(x^{(t-1)}|x^{(t)}) = \mathcal{N}\big(x^{(t-1)};\; f_{\mu}(x^{(t)}, t),\; f_{\Sigma}(x^{(t)}, t)\big)
$$

- $f_{\mu}, f_{\Sigma}$를 신경망(아래 [3]에서 구조 설명)으로 추정한다. 이 두 함수를 정의하는 것이 알고리즘의 전부고,
계산 비용도 이 함수들의 비용 × $T$가 전부다.

#### <span style="color: #4682B4">2.3 Model Probability</span>

- 생성 모델이 데이터에 주는 확률은 중간 궤적을 전부 적분(marginalize)해야 해서 그대로는 intractable 하다.

$$
    p(x^{(0)}) = \int{dx^{(1 \cdots T)}\; p(x^{(0 \cdots T)})}
$$

- 여기서 annealed importance sampling / Jarzynski equality 트릭을 쓴다. forward 궤적 $q(x^{(1 \cdots T)}|x^{(0)})$를
곱하고 나누면, forward에서 샘플링한 궤적 하나로 평가 가능한 기대값 형태가 된다.

$$
    \begin{split}
    p(x^{(0)}) &= \int{dx^{(1 \cdots T)}\; q(x^{(1 \cdots T)}|x^{(0)})\,
        \frac{p(x^{(0 \cdots T)})}{q(x^{(1 \cdots T)}|x^{(0)})}}
    \\ &= \int{dx^{(1 \cdots T)}\; q(x^{(1 \cdots T)}|x^{(0)})\; \cdot\;
        p(x^{(T)})\prod_{t=1}^{T}{\frac{p(x^{(t-1)}|x^{(t)})}{q(x^{(t)}|x^{(t-1)})}}}
    \end{split}
$$

- 즉 forward로 노이즈를 입혀보면서 각 step에서 reverse/forward 확률 비율만 곱해나가면 된다.
$\beta$가 작으면 forward와 reverse가 거의 같은 분포라 이 비율이 1에 가깝고, **single sample로도 평가가 잘 된다**
(quasi-static process에 대응).

#### <span style="color: #4682B4">2.4 Training — Lower Bound $K$ 유도</span>

- 학습은 log likelihood를 최대화하는 것인데, 위 식에 Jensen's inequality를 적용하면 lower bound $K$가 나온다.

$$
    L = \int{dx^{(0)}\, q(x^{(0)})\log{p(x^{(0)})}} \;\ge\; K
$$

- 그리고 이 $K$는 (Sohl-Dickstein 논문 Appendix B에서) **KL divergence와 entropy들의 합**으로 정리된다.

$$
    K = -\sum_{t=2}^{T}{\int{dx^{(0)}dx^{(t)}\; q(x^{(0)}, x^{(t)})\;
        D_{KL}\big( q(x^{(t-1)}|x^{(t)}, x^{(0)}) \,\|\, p(x^{(t-1)}|x^{(t)}) \big)}}
    \\ + H_q(X^{(T)}|X^{(0)}) - H_q(X^{(1)}|X^{(0)}) - H_p(X^{(T)})
$$

<details>
<summary> <span style="color: #ffd33d">lower bound K 전체 유도 (Jensen → Bayes 뒤집기 → entropy 정리) 펼치기/접기</span> </summary>

- **Step 1 — Jensen's inequality.** 2.3의 식을 로그 안에 넣고 기대값을 밖으로 꺼낸다.

$$
    \begin{split}
    L &= \int{dx^{(0)}\,q(x^{(0)})\,\log\left[
        \int{dx^{(1 \cdots T)}\, q(x^{(1 \cdots T)}|x^{(0)})\,
        p(x^{(T)})\prod_{t=1}^{T}{\frac{p(x^{(t-1)}|x^{(t)})}{q(x^{(t)}|x^{(t-1)})}}} \right]}
    \\ &\ge \int{dx^{(0 \cdots T)}\; q(x^{(0 \cdots T)})\,
        \log\left[ p(x^{(T)})\prod_{t=1}^{T}{\frac{p(x^{(t-1)}|x^{(t)})}{q(x^{(t)}|x^{(t-1)})}} \right]} =: K
    \end{split}
$$

- **Step 2 — $t=1$ 항 분리와 Bayes 뒤집기.** 곱을 풀고, $t \ge 2$의 forward 항을 posterior로 뒤집는다.
(Markov 성질 때문에 $x^{(0)}$를 조건에 추가해도 같다)

$$
    q(x^{(t)}|x^{(t-1)}) = q(x^{(t)}|x^{(t-1)}, x^{(0)})
    = \frac{q(x^{(t-1)}|x^{(t)}, x^{(0)})\; q(x^{(t)}|x^{(0)})}{q(x^{(t-1)}|x^{(0)})}
$$

- 대입하면 $\frac{q(x^{(t)}|x^{(0)})}{q(x^{(t-1)}|x^{(0)})}$ 부분이 telescoping으로 연쇄 약분되어
$\frac{q(x^{(T)}|x^{(0)})}{q(x^{(1)}|x^{(0)})}$만 남는다.

$$
    K = \mathbb{E}_q\left[
        \log{p(x^{(T)})}
        + \sum_{t=2}^{T}{\log{\frac{p(x^{(t-1)}|x^{(t)})}{q(x^{(t-1)}|x^{(t)},x^{(0)})}}}
        + \log{\frac{p(x^{(0)}|x^{(1)})\,q(x^{(1)}|x^{(0)})}{q(x^{(T)}|x^{(0)})}}
        - \log{q(x^{(1)}|x^{(0)})}
    \right]
$$

- **Step 3 — KL과 entropy로 정리.** 기대값 안의 각 항을 모으면
  - $\sum_{t \ge 2}$ 항 → $-\sum D_{KL}(q(x^{(t-1)}|x^{(t)},x^{(0)}) \| p(x^{(t-1)}|x^{(t)}))$
  - $-\mathbb{E}[\log q(x^{(T)}|x^{(0)})]$ → $H_q(X^{(T)}|X^{(0)})$
  - $-\mathbb{E}[\log q(x^{(1)}|x^{(0)})]$ → $-H_q(X^{(1)}|X^{(0)})$ (부호 주의)
  - $\mathbb{E}[\log p(x^{(T)})]$ → $-H_p(X^{(T)})$: reverse 시작 분포가 $\pi$로 고정이라 학습과 무관한 상수
  - ($p(x^{(0)}|x^{(1)})$ 항은 논문에서 edge effect로 처리, DDPM의 $L_0$에 대응) $\qquad\blacksquare$

- 결국 **학습으로 줄일 수 있는 것은 KL 항들 뿐**이고, 이는 "각 step에서 reverse $p$가 forward posterior $q$를
얼마나 잘 흉내내는가"이다. forward posterior $q(x^{(t-1)}|x^{(t)},x^{(0)})$는 Gaussian으로 정확히 계산되므로
(유도는 [Diffusion Models Beat GANs 리뷰](/posts/diffusion-models-beat-gans-on-image-synthesis/)의 2.2 접기 참고)
**두 Gaussian 사이의 KL → closed form**으로 학습이 된다. 이 구조가 DDPM의 $L_{vlb}$와 완전히 동일하다.

</details>

<hr/> <!-- 수평선 -->

- forward와 reverse가 정확히 일치하는 quasi-static 극한에서 등호가 성립한다($L = K$).
즉 $\beta$를 잘게 쪼갤수록 bound가 tight 해진다.

#### <span style="color: #4682B4">2.5 Diffusion Rate $\beta_t$ 설정</span>

- forward의 $\beta_t$ 스케줄은 성능에 중요하다. (AIS나 열역학에서도 중간 분포 스케줄이 결과 품질을 좌우하는 것과 동일)
- **Gaussian**: $\beta_{2 \cdots T}$를 $K$에 대한 **gradient ascent로 직접 학습**한다.
  - 단 $\beta_1$은 overfitting 방지를 위해 작은 상수로 고정하고,
  - $\beta_t$에 대한 미분이 잘 흐르도록 sampling 시 frozen noise를 사용한다 (VAE의 reparameterization trick과 같은 아이디어).
- **Binomial**: 매 step 원본 신호의 $1/T$씩을 지우는 schedule $\beta_t = (T-t+1)^{-1}$을 사용한다.
- 참고로 DDPM은 이걸 학습하지 않고 linear schedule 상수로 고정했고, IDDPM에서 cosine schedule로 개선한다.

#### <span style="color: #4682B4">2.6 학습된 분포에 다른 분포 곱하기 (→ classifier guidance의 원형)</span>

- inpainting이나 denoising을 하려면 모델 분포 $p(x^{(0)})$에 다른 분포 $r(x^{(0)})$(관측된 픽셀 고정, 노이즈 모델 등)를
곱한 **새로운 분포** $\tilde{p}(x^{(0)}) \propto p(x^{(0)})\,r(x^{(0)})$에서 샘플링해야 한다.
- 보통의 생성 모델에서 이건 매우 어려운 일인데, diffusion은 **각 step에 $r$을 끼워넣는 것**으로 해결된다.
각 timestep의 분포를 $\tilde{p}(x^{(t)}) \propto p(x^{(t)})\,r(x^{(t)})$로 정의하면 reverse kernel만 수정하면 된다.

$$
    \tilde{p}(x^{(t-1)}|x^{(t)}) \propto p(x^{(t-1)}|x^{(t)})\; r(x^{(t-1)})
$$

<details>
<summary> <span style="color: #ffd33d">r(x)가 완만할 때 perturbed Gaussian이 되는 증명 펼치기/접기</span> </summary>

- $r(x^{(t-1)})$이 reverse kernel의 분산 대비 충분히 완만(smooth)하면, Gaussian 평균 근처에서 1차 Taylor 전개가 가능하다.

$$
    \log{r(x^{(t-1)})} \approx \log{r(\mu)} + (x^{(t-1)}-\mu)^{\top}
        \underbrace{\nabla_{x}\log{r(x)}\big|_{x=\mu}}_{=:\,g}
$$

- reverse kernel $\mathcal{N}(\mu, \Sigma)$의 log density와 더하고 완전제곱식으로 정리하면
(전개 과정은 [Diffusion Models Beat GANs 리뷰](/posts/diffusion-models-beat-gans-on-image-synthesis/)의
Perturbed Gaussian 증명과 완전히 동일)

$$
    \tilde{p}(x^{(t-1)}|x^{(t)}) \approx \mathcal{N}\big(x^{(t-1)};\; \mu + \Sigma\,g,\; \Sigma\big)
$$

- 즉 **평균만 $\Sigma \nabla\log r$ 방향으로 shift** 하면 된다. $\blacksquare$
- 2021년 ADM의 classifier guidance는 $r(x^{(t)}) = p_{\phi}(y|x^{(t)})$로 둔 정확히 이 정리의 응용이다.
6년 전에 수학은 이미 다 준비되어 있었던 셈.

</details>

<hr/> <!-- 수평선 -->

- $r(x^{(t)})$가 완만하지 않은 경우(예: inpainting에서 관측 픽셀을 delta로 고정)에도, 관측 픽셀을 매 step 강제로 세팅하는
방식으로 정확히 처리 가능하다. 논문의 inpainting 실험이 이 방식이다.

#### <span style="color: #4682B4">2.7 Entropy 상한/하한</span>

- forward process가 알려져 있으므로 각 step의 조건부 entropy에 대한 analytic한 상한/하한을 줄 수 있고,
이걸로 log likelihood 자체의 상한/하한도 계산할 수 있다.

$$
    H_q(X^{(t)}|X^{(t-1)}) + H_q(X^{(t-1)}|X^{(0)}) - H_q(X^{(t)}|X^{(0)})
    \;\le\; H_q(X^{(t-1)}|X^{(t)})
    \;\le\; H_q(X^{(t)}|X^{(t-1)})
$$

- 상한/하한 모두 $q$의 알려진 Gaussian들로만 구성되어 있어서 계산 가능하다. (유도는 논문 Appendix A)

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[3] Model Architecture — 네트워크 뜯어보기</span>

reverse kernel의 $f_{\mu}(x^{(t)}, t)$, $f_{\Sigma}(x^{(t)}, t)$를 어떻게 만들었는지. (논문 Appendix D)
요즘 UNet 기반 diffusion과 비교하면 소박하지만 구조적 아이디어는 이미 다 들어있다.

- **Multi-scale convolutional network**: 이미지 실험에서는 conv layer들을 여러 scale에서 병렬로 돌린다.
  - 각 scale branch는 `mean pooling으로 downsample → conv → upsample(복제)` 구조로, 넓은 receptive field를 싸게 확보한다.
  - 지금 UNet의 down/up path가 하는 역할을 단순화한 형태라고 보면 된다.
- **Readout head 분리**: 공통 feature를 뽑은 뒤 $\mu$용, $\Sigma$용 head를 따로 둔다.
  - $\Sigma$ 출력은 sigmoid를 거쳐 $\beta_t$ 근처에서 안정적으로 파라미터화한다. (분산이 음수가 되거나 폭주하는 것 방지)
- **Timestep 의존성**: $t$마다 네트워크를 따로 두는 게 아니라, **시간축 basis function(bump function)들의
선형결합**으로 readout weight가 $t$에 따라 부드럽게 변하게 한다.
  - $w(t) = \sum_{j}{g_j(t)\,w_j}$ 형태 ($g_j$는 soft한 시간 bump).
  - 지금의 sinusoidal timestep embedding + MLP 주입과 같은 문제("하나의 네트워크로 모든 $t$를 처리")를 푸는 2015년식 해법이다.
- 이미지가 아닌 저차원 실험(swiss roll 등)에서는 그냥 MLP(radial basis 포함)를 쓴다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[4] Experiments</span>

- 다양한 종류의 데이터로 프레임워크의 범용성을 보인다. 전부 같은 알고리즘으로 학습된다.
  - **Swiss roll (2D toy)**: forward가 나선 구조를 서서히 지워 Gaussian이 되고, 학습된 reverse가
Gaussian에서 나선을 복원하는 과정을 시각화. reverse의 drift(평균 이동)가 데이터 manifold를 향하는 것을 보여준다.
  - **Binary heartbeat sequence**: 주기 5의 이진 시퀀스. binomial diffusion으로 학습해서 이산 데이터에도 동작함을 확인.
  - **자연 이미지**: MNIST, CIFAR-10, dead leaves 모델, 나무껍질(bark) 텍스처.
- **MNIST 정량 비교** (Parzen window 기반 log likelihood, 당시 표준 프로토콜):

| Model | Log Likelihood |
|---|---|
| Stacked CAE | 121 ± 1.6 |
| DBN | 138 ± 2 |
| Deep GSN | 214 ± 1.1 |
| **Diffusion (이 논문)** | **220 ± 1.9** |
| Adversarial net (GAN, 같은 해) | 225 ± 2 |

- 당시 기준으로 GSN을 넘고 갓 나온 GAN과 비등한 수준이었다.
- **Inpainting / Denoising**: 2.6의 분포 곱하기를 이용해 bark 이미지의 가운데 100×100을 지우고 복원하거나,
Gaussian 노이즈가 낀 dead leaves 이미지를 denoising 한다. 별도 재학습 없이 학습된 모델 그대로 수행된다는 게 포인트.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[5] 의의와 한계 & 개인적인 생각</span>

- **이 논문이 만든 것**: forward/reverse Markov chain 프레임워크, posterior 기반 KL 학습 objective($L_{vlb}$의 원형),
분포 곱하기(guidance의 원형), 이산/연속 diffusion 둘 다. 사실상 diffusion model의 뼈대 전부다.
- **DDPM(2020)이 바꾼 것과 비교**하면 이 논문의 위치가 명확해진다.

| | 이 논문 (2015) | DDPM (2020) |
|---|---|---|
| 예측 대상 | 평균 $f_{\mu}$, 공분산 $f_{\Sigma}$ 직접 예측 | 노이즈 $\epsilon$ 예측 (재파라미터화) |
| 공분산 | 학습 | $\beta_t\mathbf{I}$ 상수 고정 |
| $\beta_t$ | gradient ascent로 학습 | linear schedule 고정 |
| Loss | $K$ (전체 variational bound) | $L_{simple}$ (weight 버린 MSE) |
| 네트워크 | multi-scale conv + bump 시간 basis | UNet + sinusoidal embedding |
| 결과 | MNIST 수준 | 고해상도 자연 이미지 |

- 프레임워크는 2015년에 완성됐는데 결과가 터진 건 5년 뒤라는 게 흥미롭다. 바뀐 건 수학이 아니라
**파라미터화($\epsilon$ 예측)와 objective 단순화, 그리고 네트워크/컴퓨팅**이었다.
- 2.6의 "분포 곱하기"가 classifier guidance([ADM 리뷰](/posts/diffusion-models-beat-gans-on-image-synthesis/) 4장)로,
$K$의 KL 분해가 DDPM의 $L_{vlb}$로 그대로 이어지므로, diffusion 수식의 출처가 궁금할 때 돌아와서 읽기 좋은 논문이다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">Reference</span>

- [1] J. Sohl-Dickstein et al., "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (ICML 2015)
- [2] J. Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- [3] A. Nichol & P. Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (ICML 2021)
- [4] P. Dhariwal & A. Nichol, "Diffusion Models Beat GANs on Image Synthesis" (NeurIPS 2021)
- [5] R. Neal, "Annealed Importance Sampling" (2001) / C. Jarzynski, "Equilibrium free-energy differences from nonequilibrium measurements" (1997)
