---
title: "[논문 리뷰]Diffusion Models Beat GANs on Image Synthesis"
date: 2022-08-19
updated: 2026-08-18
category: image-generation-paper
tags:
  - "Diffusion Model"
  - "GAN"
  - "Image Generation"
---

<span style="font-size:17pt">
<b>Diffusion Models Beat GANs on Image Synthesis</b>
</span>

<a href="https://arxiv.org/abs/2105.05233" target="_blank"><b>[PDF]</b></a>
, <b><span style="color: #F2AA4C">Image Generation</span></b>, Prafulla Dhariwal, Alex Nichol (OpenAI, NeurIPS 2021)

### <span style="color: #ffd33d">Summary</span>

이 논문을 간단하게 요약하면, diffusion model이 image generation에서 GAN을 이기지 못했던 이유를
**(1) 모델 아키텍처가 GAN 계열만큼 정제되지 않았다**, **(2) GAN은 fidelity와 diversity를 trade-off 할 수 있는 장치(truncation trick 등)가
있는데 diffusion은 없었다** 이 두가지로 진단하고 각각을 해결한 논문이다.

첫번째 문제는 UNet 구조를 하나하나 ablation 하면서(채널 수 vs depth, attention head 수, multi-resolution attention,
BigGAN 스타일의 residual up/down sampling block, Adaptive Group Normalization) 최적의 구조인 **ADM**(Ablated Diffusion Model)을 찾는 것으로 해결한다.

두번째 문제는 **classifier guidance**로 해결한다. noisy한 이미지 $x_t$에서 class $y$를 예측하도록 학습된 classifier $p_{\phi}(y|x_t)$의
gradient $\nabla_{x_t} \log{p_{\phi}(y|x_t)}$를 sampling 과정의 평균에 더해줘서 sample이 해당 class 방향으로 생성되도록 유도한다.
이때 gradient에 scale $s$를 곱해서 $s$가 커질수록 fidelity ↑ / diversity ↓ 로 trade-off가 가능해지는데 이게 GAN의 truncation trick과
같은 역할을 하게 된다.

결과적으로 ImageNet 128×128에서 FID 2.97, 256×256에서 FID 4.59, 512×512에서 FID 7.72를 달성해서 BigGAN-deep을 이겼고,
DDIM sampler를 쓰면 **25 step만으로도** BigGAN-deep보다 좋은 FID를 유지한다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[1] Intro</span>

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
- Diffusion model은 likelihood-based model의 한 종류로 최근에 CIFAR-10[27]에서 SOTA를 달성했고 distribution coverage(다양성),
정적인 학습 objective, 쉬운 확장성 같은 좋은 성질들을 가지고 있다.
  - 하지만 LSUN, ImageNet 같은 어려운 데이터셋에서는 여전히 GAN에게 밀리고 있었다.
  - Nichol & Dhariwal[43]은 diffusion model이 학습/샘플링 계산량이 늘어날수록 성능이 좋아진다는 것을 발견했지만
그래도 BigGAN-deep[5]과의 격차는 남아있었다.
- 논문은 diffusion model과 GAN의 격차가 나는 이유를 최소 2가지라고 가정한다.
  1. 최근 GAN 계열 연구들이 사용하는 모델 아키텍처는 매우 많이 탐색되고 정제되어 있다. (diffusion은 아직임)
  2. GAN은 diversity를 포기하고 fidelity를 얻는 trade-off가 가능해서, 전체 distribution을 커버하지 않는 대신 high quality sample을 만든다.
- 그래서 논문은 먼저 모델 아키텍처를 개선하고, 그 다음 fidelity ↔ diversity trade-off 장치를 설계해서
이 두가지 이점을 diffusion model에 가져온다. 결과적으로 GAN을 넘어서는 SOTA를 달성한다.
  - ImageNet 128×128 FID 2.97, 256×256 FID 4.59, 512×512 FID 7.72
  - DDIM[57]으로 25 step만 샘플링해도 BigGAN-deep과 비슷하거나 더 좋은 FID를 유지
  - upsampling diffusion model과 classifier guidance를 결합하면 256×256 FID 3.94, 512×512 FID 3.85까지 개선

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[2] Background: Diffusion Model 정리</span>

논문을 읽기 전에 DDPM[25] 계열 수식을 한번 정리하고 간다. 이 부분은 논문의 Section 2 + Appendix B 내용이고,
[Deep Unsupervised Learning using Nonequilibrium Thermodynamics](/posts/deep-unsupervised-learning-using-nonequilibrium-thermodynamics/) 리뷰에서
다뤘던 부분의 연장선이다.

#### <span style="color: #4682B4">2.1 Forward Process</span>

- Diffusion model은 데이터 $x_0 \sim q(x_0)$에 노이즈를 점진적으로 추가하는 **forward process** $q$를 정의한다.
- 각 step은 variance schedule $\beta_1, ..., \beta_T$를 따르는 Gaussian Markov chain이다.

$$
    q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\,x_{t-1},\; \beta_t \mathbf{I})
$$

- 이 process의 좋은 성질은 임의의 timestep $t$의 $x_t$를 $x_0$에서 **한번에** 샘플링 할 수 있다는 것이다.
$\alpha_t := 1-\beta_t$, $\bar{\alpha}_t := \prod_{s=1}^{t}{\alpha_s}$로 정의하면

$$
    q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\,x_0,\; (1-\bar{\alpha}_t)\mathbf{I})
    \quad \Leftrightarrow \quad
    x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

<details>
<summary> <span style="color: #ffd33d"><code>q(x_t | x_0)</code> closed form 증명 펼치기/접기</span> </summary>

- reparameterization trick으로 한 step을 풀어쓰면 $x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_{t-1}$ 이다.
- $x_{t-1}$을 다시 한 step 풀어서 대입한다.

$$
    \begin{split}
    x_t &= \sqrt{\alpha_t}\left( \sqrt{\alpha_{t-1}}\,x_{t-2} + \sqrt{1-\alpha_{t-1}}\,\epsilon_{t-2} \right) + \sqrt{1-\alpha_t}\,\epsilon_{t-1}
    \\  &= \sqrt{\alpha_t \alpha_{t-1}}\,x_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-2} + \sqrt{1-\alpha_t}\,\epsilon_{t-1}
    \end{split}
$$

- 여기서 $\epsilon_{t-2}, \epsilon_{t-1}$은 서로 독립인 standard Gaussian이다.
독립인 두 Gaussian의 합 $\mathcal{N}(0, \sigma_1^2\mathbf{I}) + \mathcal{N}(0, \sigma_2^2\mathbf{I}) = \mathcal{N}(0, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$
성질을 이용하면 두 노이즈 항이 하나로 합쳐진다.

$$
    \begin{split}
    \sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-2} + \sqrt{1-\alpha_t}\,\epsilon_{t-1}
    &\sim \mathcal{N}\big(0,\; [\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t)]\mathbf{I}\big) \\
    &= \mathcal{N}\big(0,\; (1-\alpha_t\alpha_{t-1})\mathbf{I}\big)
    \end{split}
$$

- 따라서 $x_t = \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\,\bar{\epsilon}$ 형태가 되고,
같은 과정을 $x_0$까지 반복(귀납)하면

$$
    x_t = \sqrt{\textstyle\prod_{s=1}^{t}\alpha_s}\;x_0 + \sqrt{1-\textstyle\prod_{s=1}^{t}\alpha_s}\;\epsilon
        = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon \qquad \blacksquare
$$

</details>

<hr/> <!-- 수평선 -->

- $T$가 충분히 크고 schedule이 적절하면 $\bar{\alpha}_T \rightarrow 0$이 되어 $x_T$는 거의 pure Gaussian noise $\mathcal{N}(0,\mathbf{I})$가 된다.

#### <span style="color: #4682B4">2.2 Reverse Process와 Posterior</span>

- 생성은 forward의 반대 방향, 즉 $q(x_{t-1}|x_t)$를 알면 $x_T \sim \mathcal{N}(0,\mathbf{I})$에서 시작해서 노이즈를 점점 제거하며 $x_0$를 만들 수 있다.
- 하지만 $q(x_{t-1}|x_t)$는 전체 데이터 분포에 의존하기 때문에 intractable 하다. 그래서 이걸 신경망으로 근사한다.
($\beta_t$가 충분히 작으면 reverse도 Gaussian으로 잘 근사된다는 것이 알려져 있다[56])

$$
    p_{\theta}(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\; \mu_{\theta}(x_t, t),\; \Sigma_{\theta}(x_t, t))
$$

- 재밌는 점은 $x_0$를 조건으로 주면 posterior $q(x_{t-1}|x_t, x_0)$는 **정확한 Gaussian으로 계산이 가능**하다는 것이다.
이게 학습 target이 된다.

$$
    q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1};\; \tilde{\mu}_t(x_t, x_0),\; \tilde{\beta}_t\mathbf{I})
$$

$$
    \tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,x_0
    + \frac{\sqrt{\alpha_t}\,(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,x_t
    ,\qquad
    \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t
$$

<details>
<summary> <span style="color: #ffd33d">posterior <code>q(x_(t-1) | x_t, x_0)</code> 유도 (완전제곱식 정리) 펼치기/접기</span> </summary>

- 베이즈 정리로 posterior를 forward process의 항들로 바꾼다. 세 항 모두 위에서 구한 Gaussian이다.

$$
    q(x_{t-1}|x_t, x_0) = q(x_t|x_{t-1}, x_0)\,\frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}
    = q(x_t|x_{t-1})\,\frac{q(x_{t-1}|x_0)}{q(x_t|x_0)} \quad (\because \text{Markov property})
$$

- exponent만 모아서 정리한다. ($x_{t-1}$에 대한 함수로 보고, $x_{t-1}$과 무관한 항은 상수 $C$로 몰아넣음)

$$
    \begin{split}
    \log q(x_{t-1}|x_t,x_0) &= -\frac{1}{2}\left[
        \frac{(x_t - \sqrt{\alpha_t}\,x_{t-1})^2}{\beta_t}
        + \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\,x_0)^2}{1-\bar{\alpha}_{t-1}}
        - \frac{(x_t - \sqrt{\bar{\alpha}_t}\,x_0)^2}{1-\bar{\alpha}_t}
    \right]
    \\ &= -\frac{1}{2}\left[
        \left( \frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}} \right) x_{t-1}^2
        - 2\left( \frac{\sqrt{\alpha_t}}{\beta_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0 \right) x_{t-1}
    \right] + C
    \end{split}
$$

- Gaussian의 exponent는 $-\frac{1}{2\sigma^2}(x-\mu)^2 = -\frac{1}{2}\left[\frac{1}{\sigma^2}x^2 - \frac{2\mu}{\sigma^2}x\right] + C$ 형태이므로
$x_{t-1}^2$의 계수에서 분산이, $x_{t-1}$의 계수에서 평균이 나온다.

- 먼저 분산부터 계산하면

$$
    \begin{split}
    \frac{1}{\tilde{\beta}_t} &= \frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}
    = \frac{\alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t}{\beta_t(1-\bar{\alpha}_{t-1})}
    = \frac{\alpha_t - \bar{\alpha}_t + 1 - \alpha_t}{\beta_t(1-\bar{\alpha}_{t-1})}
    = \frac{1-\bar{\alpha}_t}{\beta_t(1-\bar{\alpha}_{t-1})}
    \\ \therefore \tilde{\beta}_t &= \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t
    \end{split}
$$

- 평균은 ($\mu = \sigma^2 \times$ 일차항 계수의 절반)

$$
    \begin{split}
    \tilde{\mu}_t &= \left( \frac{\sqrt{\alpha_t}}{\beta_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0 \right)
        \cdot \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t
    \\ &= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,x_t
        + \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,x_0 \qquad \blacksquare
    \end{split}
$$

</details>

<hr/> <!-- 수평선 -->

#### <span style="color: #4682B4">2.3 학습 Objective (ELBO)</span>

- VAE와 동일하게 log likelihood의 variational lower bound(ELBO)를 최적화한다.

$$
    \mathbb{E}\left[-\log p_{\theta}(x_0)\right]
    \le \mathbb{E}_q\left[ -\log\frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_0)} \right] =: L_{vlb}
$$

- 이 bound는 다음처럼 timestep 별 KL divergence의 합으로 분해된다.

$$
    L_{vlb} = \underbrace{D_{KL}(q(x_T|x_0)\,\|\,p(x_T))}_{L_T}
    + \sum_{t=2}^{T}{\underbrace{D_{KL}(q(x_{t-1}|x_t,x_0)\,\|\,p_{\theta}(x_{t-1}|x_t))}_{L_{t-1}}}
    \underbrace{-\log p_{\theta}(x_0|x_1)}_{L_0}
$$

<details>
<summary> <span style="color: #ffd33d">ELBO 분해 유도 펼치기/접기</span> </summary>

- Jensen's inequality로 lower bound를 만들고 Markov 성질로 곱을 풀어쓴다.

$$
    \begin{split}
    -\log p_{\theta}(x_0) &= -\log \int p_{\theta}(x_{0:T})\,dx_{1:T}
    = -\log \int q(x_{1:T}|x_0)\,\frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_0)}\,dx_{1:T}
    \\ &\le -\mathbb{E}_{q(x_{1:T}|x_0)}\left[ \log\frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_0)} \right]
    \quad (\because \text{Jensen's inequality})
    \\ &= \mathbb{E}_q\left[ -\log p(x_T) - \sum_{t=1}^{T}{\log\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})}} \right]
    \end{split}
$$

- 여기서 핵심 트릭은 $q(x_t|x_{t-1})$을 베이즈 정리로 뒤집으면서 $x_0$를 조건에 추가하는 것이다. ($t \ge 2$)

$$
    q(x_t|x_{t-1}) = q(x_t|x_{t-1}, x_0) = \frac{q(x_{t-1}|x_t, x_0)\,q(x_t|x_0)}{q(x_{t-1}|x_0)}
$$

- 대입해서 정리하면 telescoping으로 $q(x_t|x_0)$ 항들이 연쇄적으로 약분된다.

$$
    \begin{split}
    L_{vlb} &= \mathbb{E}_q\left[ -\log p(x_T)
        - \sum_{t=2}^{T}{\log\left( \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)} \cdot \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)} \right)}
        - \log\frac{p_{\theta}(x_0|x_1)}{q(x_1|x_0)} \right]
    \\ &= \mathbb{E}_q\left[ -\log\frac{p(x_T)}{q(x_T|x_0)}
        - \sum_{t=2}^{T}{\log\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}}
        - \log p_{\theta}(x_0|x_1) \right]
    \\ &= D_{KL}(q(x_T|x_0)\,\|\,p(x_T))
        + \sum_{t=2}^{T}{D_{KL}(q(x_{t-1}|x_t,x_0)\,\|\,p_{\theta}(x_{t-1}|x_t))}
        - \mathbb{E}_q\left[\log p_{\theta}(x_0|x_1)\right] \qquad \blacksquare
    \end{split}
$$

- $L_T$는 학습 파라미터가 없어서 상수이고(단, $\bar{\alpha}_T \approx 0$이어야 작음), $L_{t-1}$은 **두 Gaussian 사이의 KL**이라 closed form으로 계산된다.

$$
    D_{KL}(\mathcal{N}(\mu_1,\Sigma_1)\,\|\,\mathcal{N}(\mu_2,\Sigma_2))
    = \frac{1}{2}\left[ \log\frac{|\Sigma_2|}{|\Sigma_1|} - d + tr(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^{\top}\Sigma_2^{-1}(\mu_2-\mu_1) \right]
$$

</details>

<hr/> <!-- 수평선 -->

#### <span style="color: #4682B4">2.4 $\epsilon$-parameterization과 $L_{simple}$</span>

- DDPM[25]의 핵심 발견은 $\mu_{\theta}$를 직접 예측하는 것보다 **노이즈 $\epsilon$을 예측**하는 것이 더 잘 된다는 것이다.
- $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ 에서 $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon)$을
posterior 평균 $\tilde{\mu}_t$ 식에 대입하면

$$
    \begin{split}
    \tilde{\mu}_t &= \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\cdot\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon)
        + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t
    \\ &= \frac{1}{\sqrt{\alpha_t}}\left( \frac{\beta_t}{1-\bar{\alpha}_t}
        + \frac{\alpha_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \right)x_t
        - \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}\,\epsilon
    \\ &= \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon \right)
    \end{split}
$$

- 그래서 네트워크 $\epsilon_{\theta}(x_t, t)$가 $\epsilon$을 예측하게 하고 평균은 위 식으로 계산한다.
이때 $L_{t-1}$의 KL을 전개하면 weighting term이 붙은 $\epsilon$의 MSE가 되는데, DDPM은 이 weight를 버린
단순화된 loss가 실제로 sample quality에 더 좋다는 것을 보였다.

$$
    L_{simple} = \mathbb{E}_{t, x_0, \epsilon}\left[ \| \epsilon - \epsilon_{\theta}(x_t, t) \|^2 \right]
$$

#### <span style="color: #4682B4">2.5 Improved DDPM (learned $\Sigma$, hybrid objective)</span>

- DDPM은 $\Sigma_{\theta}$를 학습하지 않고 $\sigma_t^2 = \beta_t$ 또는 $\tilde{\beta}_t$로 고정했다. (둘의 성능 차이는 거의 없었음)
- Nichol & Dhariwal[43]은 sampling step을 줄일 때는 $\Sigma_{\theta}$ 학습이 중요하다는 것을 보였고, 두 극단값 사이를
interpolation 하는 형태로 파라미터화 했다. (네트워크가 차원별로 $v$를 출력)

$$
    \Sigma_{\theta}(x_t, t) = \exp\big( v\log{\beta_t} + (1-v)\log{\tilde{\beta}_t} \big)
$$

- $L_{simple}$은 $\Sigma$에 대한 gradient가 없으므로 hybrid objective를 사용한다.
($L_{vlb}$ 항에서 $\mu_{\theta}$에는 stop-gradient를 걸어서 $\Sigma$ 학습에만 쓰이게 함, $\lambda = 0.001$)

$$
    L_{hybrid} = L_{simple} + \lambda L_{vlb}
$$

- 이 논문(ADM)도 이 세팅($L_{hybrid}$ + learned $\Sigma$)을 그대로 가져와서 **25~250 step 샘플링**이 가능하게 한다.

#### <span style="color: #4682B4">2.6 DDIM</span>

- Song et al.[57]은 같은 marginal $q(x_t|x_0)$를 유지하면서 non-Markovian forward를 정의해서
**deterministic한 sampling**($\sigma = 0$)이 가능한 DDIM을 제안했다.

$$
    x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}
    \underbrace{\left( \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_{\theta}(x_t)}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{predicted } x_0}
    + \sqrt{1-\bar{\alpha}_{t-1}}\cdot\epsilon_{\theta}(x_t)
$$

- 이 논문에서는 sampling step이 50 미만일 때는 DDIM이 더 유리하다는 것을 확인하고 25 step 실험에 사용한다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[3] Architecture Improvements — UNet 뜯어보기</span>

이 논문의 첫번째 기여. Ho et al.[25]의 UNet을 베이스라인으로 두고 구조를 하나씩 바꿔가며 ablation 한다.
먼저 베이스라인 UNet 구조부터 정리하면:

- **전체 구조**: downsampling path → middle block → upsampling path의 UNet 구조에, 같은 해상도의 down/up block끼리
skip connection으로 연결된다.
- **Residual Block**: 각 해상도마다 residual block이 반복된다. 하나의 residual block은
`GroupNorm → SiLU → 3×3 Conv → (timestep embedding 주입) → GroupNorm → SiLU → Dropout → 3×3 Conv` + skip connection 구조.
- **Timestep embedding**: $t$를 Transformer의 sinusoidal positional embedding으로 인코딩한 후 2-layer MLP를 통과시키고,
각 residual block마다 projection 해서 더해준다. (class-conditional 모델이면 class embedding도 여기에 더함)
- **Attention Block**: 16×16 해상도에 self-attention block 하나를 사용. `GroupNorm → 1×1 conv로 QKV 생성 → attention → 1×1 conv` 형태이고
이것도 residual로 더해진다.
- **Down/Up sampling**: downsampling은 stride-2 conv(또는 average pool), upsampling은 nearest 2배 + conv.

여기서 논문이 실험한 변경점들:

1. **모델 크기를 유지하면서 depth vs width 조절**
   - depth를 늘리면(residual block 수 증가) 성능은 좋아지지만 같은 성능의 width 증가 대비 학습 시간(wall-clock)이 더 걸린다.
   - 그래서 최종 모델은 depth(block 2개)를 유지하고 width를 키우는 쪽을 선택한다.
2. **Attention head 수 증가**
   - head 수를 늘리거나 head당 채널 수를 줄이면 FID가 개선된다.
   - 최종적으로 **head당 64 channels**로 고정하는 것이 가장 좋았다.
3. **Multi-resolution attention**
   - attention을 16×16 하나가 아니라 **32×32, 16×16, 8×8** 해상도 모두에 사용한다.
4. **BigGAN residual block**
   - up/downsampling을 별도 layer로 하지 않고 BigGAN[5] 스타일로 **residual block 내부에서** up/downsampling을 수행한다.
   - 이 변경이 ablation 중에서 FID 개선폭이 가장 컸다.
5. **Residual connection rescaling ($\frac{1}{\sqrt{2}}$)**
   - StyleGAN 계열에서 쓰던 트릭인데, diffusion에서는 **성능 개선이 없어서 채택하지 않는다.**

#### <span style="color: #4682B4">3.1 Adaptive Group Normalization (AdaGN)</span>

- 추가로 timestep/class embedding을 주입하는 방식을 바꾼다. 기존 DDPM은 embedding을 feature에 단순히 더했는데(AdaIN처럼
스타일을 주입하는 StyleGAN에서 영감), 논문은 GroupNorm 이후에 **channel-wise scale & shift**로 주입한다.

$$
    AdaGN(h, y) = y_s \cdot GroupNorm(h) + y_b
$$

- 여기서 $h$는 residual block의 첫 conv를 통과한 feature이고, $y = [y_s, y_b]$는 timestep embedding과 class embedding을
linear projection한 값이다.
- 단순 addition 대비 FID가 유의미하게 개선되어 모든 residual block에 적용한다.

<details>
<summary> <span style="color: #4682B4">비슷한 conditioning 기법들 비교 펼치기/접기</span> </summary>

- 이 계열 기법들은 전부 "조건 정보를 normalization의 affine 파라미터로 주입한다"는 같은 아이디어의 변형이다.

| 기법 | 수식 형태 | 사용처 |
|---|---|---|
| Conditional BatchNorm | $\gamma(y)\cdot BN(h) + \beta(y)$ | cGAN, BigGAN |
| AdaIN | $\sigma(y)\cdot \frac{h-\mu(h)}{\sigma(h)} + \mu(y)$ | StyleGAN |
| FiLM | $\gamma(y)\cdot h + \beta(y)$ (norm 없이) | VQA 등 |
| **AdaGN** | $y_s\cdot GN(h) + y_b$ | **ADM (이 논문)** |

- BatchNorm은 batch 통계에 의존해서 batch 크기에 민감한데, diffusion 학습은 큰 이미지 + 큰 모델 때문에 GPU당 batch가 작아서
GroupNorm 기반이 안정적이다.

</details>

<hr/> <!-- 수평선 -->

#### <span style="color: #4682B4">3.2 최종 아키텍처 (ADM)</span>

- 위 ablation을 종합한 최종 구성:
  - variable width (해상도별 채널 multiplier)
  - resolution당 **2 residual blocks**
  - **32/16/8 해상도에 multi-head attention, head당 64 channels**
  - **BigGAN 스타일 residual up/downsampling**
  - **AdaGN**으로 timestep/class embedding 주입
- 이 구조를 논문에서 **ADM**(Ablated Diffusion Model)이라고 부르고, classifier guidance가 붙으면 **ADM-G**,
upsampling stack이 붙으면 **ADM-U**라고 부른다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[4] Classifier Guidance</span>

이 논문의 두번째이자 가장 중요한 기여. GAN의 conditional 생성은 class 정보를 매우 적극적으로 쓰는데(class-conditional BatchNorm,
Discriminator에 auxiliary classifier[41] 등), diffusion은 AdaGN에 class embedding을 넣는 정도라 class 정보 활용이 약하다.
그래서 **별도의 classifier의 gradient로 sampling 과정 자체를 조종한다**.

- 핵심 아이디어: noisy 이미지 $x_t$에서 class를 맞추는 classifier $p_{\phi}(y|x_t, t)$를 학습하고,
그 gradient $\nabla_{x_t}\log{p_{\phi}(y|x_t, t)}$를 이용해 sampling 평균을 class $y$ 방향으로 이동시킨다.
- classifier는 **noise가 낀 $x_t$에 대해 학습**되어야 한다. (일반 classifier는 noisy 입력에서 gradient가 무의미함)
- classifier 구조는 ADM의 downsampling path + 8×8에서 attention pooling 하는 형태를 사용한다.

#### <span style="color: #4682B4">4.1 Conditional Reverse Process 유도</span>

- label $y$를 조건으로 하는 reverse process는 다음과 같이 unconditional reverse × classifier로 분해된다.
($Z$는 normalizing constant)

$$
    p_{\theta,\phi}(x_t|x_{t+1}, y) = Z\, p_{\theta}(x_t|x_{t+1})\, p_{\phi}(y|x_t)
$$

<details>
<summary> <span style="color: #ffd33d">위 분해가 성립하는 이유 (논문 Eq. 2~3) 펼치기/접기</span> </summary>

- forward process에서 $y$가 주어져도 noising은 동일하므로 $\hat{q}(x_{t+1}|x_t, y) = q(x_{t+1}|x_t)$로 정의하면,
label이 있는 forward process $\hat{q}$의 reverse는

$$
    \begin{split}
    \hat{q}(x_t|x_{t+1}, y) &= \frac{\hat{q}(x_t, x_{t+1}, y)}{\hat{q}(x_{t+1}, y)}
    = \frac{\hat{q}(x_t, x_{t+1}, y)}{\hat{q}(y|x_{t+1})\,\hat{q}(x_{t+1})}
    \\ &= \frac{\hat{q}(x_t|x_{t+1})\,\hat{q}(y|x_t, x_{t+1})\,\hat{q}(x_{t+1})}{\hat{q}(y|x_{t+1})\,\hat{q}(x_{t+1})}
    \\ &= \frac{\hat{q}(x_t|x_{t+1})\,\hat{q}(y|x_t)}{\hat{q}(y|x_{t+1})}
    \quad (\because y \perp x_{t+1} \,|\, x_t \text{, noising은 label과 무관})
    \end{split}
$$

- 분모 $\hat{q}(y|x_{t+1})$은 $x_t$와 무관하므로 상수 $Z$로 취급 가능하다. 즉 unconditional reverse에
classifier 항을 곱한 것이 conditional reverse가 된다. $\hat{q}(x_t|x_{t+1}) = q(x_t|x_{t+1})$을 $p_{\theta}$로,
$\hat{q}(y|x_t)$를 $p_{\phi}$로 근사하면 위 식이 된다. $\blacksquare$

</details>

<hr/> <!-- 수평선 -->

- 문제는 이 분포에서 직접 샘플링하는게 일반적으로 intractable 하다는 것. 논문은 이걸 **perturbed Gaussian**으로 근사한다.

#### <span style="color: #4682B4">4.2 Perturbed Gaussian 근사 (핵심 증명)</span>

결론부터 쓰면, $p_{\theta}(x_t|x_{t+1}) = \mathcal{N}(\mu, \Sigma)$일 때

$$
    p_{\theta,\phi}(x_t|x_{t+1}, y) \approx \mathcal{N}(\mu + \Sigma g,\; \Sigma), \qquad
    g = \nabla_{x_t}\log{p_{\phi}(y|x_t)}\big|_{x_t=\mu}
$$

즉 **평균을 $\Sigma g$만큼 classifier gradient 방향으로 shift** 시키기만 하면 된다.

<details>
<summary> <span style="color: #ffd33d">Perturbed Gaussian 증명 (논문 Eq. 3~10, Taylor 전개 + 완전제곱) 펼치기/접기</span> </summary>

- unconditional reverse의 log를 쓰면 ($C$는 $x_t$와 무관한 상수)

$$
    \log{p_{\theta}(x_t|x_{t+1})} = -\frac{1}{2}(x_t-\mu)^{\top}\Sigma^{-1}(x_t-\mu) + C
$$

- diffusion 후반부($t$가 작을 때)로 갈수록 $\|\Sigma\| \rightarrow 0$, 즉 분포가 $\mu$ 주변에 뾰족하게 몰리므로
$\log{p_{\phi}(y|x_t)}$를 $x_t = \mu$ 근처에서 **1차 Taylor 전개**로 근사할 수 있다.

$$
    \begin{split}
    \log{p_{\phi}(y|x_t)} &\approx \log{p_{\phi}(y|x_t)}\big|_{x_t=\mu}
        + (x_t-\mu)^{\top} \underbrace{\nabla_{x_t}\log{p_{\phi}(y|x_t)}\big|_{x_t=\mu}}_{=:\,g}
    \\ &= (x_t-\mu)^{\top}g + C_1
    \end{split}
$$

- 두 log를 더하면

$$
    \begin{split}
    \log\big( p_{\theta}(x_t|x_{t+1})\,p_{\phi}(y|x_t) \big)
    &\approx -\frac{1}{2}(x_t-\mu)^{\top}\Sigma^{-1}(x_t-\mu) + (x_t-\mu)^{\top}g + C_2
    \end{split}
$$

- $u := x_t - \mu$로 치환하고 완전제곱식으로 정리한다.

$$
    \begin{split}
    -\frac{1}{2}u^{\top}\Sigma^{-1}u + u^{\top}g
    &= -\frac{1}{2}\left[ u^{\top}\Sigma^{-1}u - 2u^{\top}g \right]
    \\ &= -\frac{1}{2}\left[ u^{\top}\Sigma^{-1}u - u^{\top}\Sigma^{-1}(\Sigma g) - (\Sigma g)^{\top}\Sigma^{-1}u \right]
    \quad (\because \Sigma^{-1}\Sigma = \mathbf{I},\; \Sigma^{\top} = \Sigma)
    \\ &= -\frac{1}{2}\left[ (u - \Sigma g)^{\top}\Sigma^{-1}(u - \Sigma g) - g^{\top}\Sigma g \right]
    \\ &= -\frac{1}{2}(u - \Sigma g)^{\top}\Sigma^{-1}(u - \Sigma g) + \underbrace{\frac{1}{2}g^{\top}\Sigma g}_{\text{const}}
    \end{split}
$$

- 다시 $u = x_t - \mu$를 되돌리면

$$
    \log\big( p_{\theta}(x_t|x_{t+1})\,p_{\phi}(y|x_t) \big)
    \approx -\frac{1}{2}(x_t - \mu - \Sigma g)^{\top}\Sigma^{-1}(x_t - \mu - \Sigma g) + C_3
$$

- 이건 정확히 평균이 $\mu + \Sigma g$이고 공분산이 $\Sigma$인 Gaussian의 log density 형태이다.
상수항은 normalizing constant $Z$에 흡수된다. $\blacksquare$

$$
    \therefore \quad p_{\theta,\phi}(x_t|x_{t+1}, y) \approx \mathcal{N}(\mu + \Sigma g,\; \Sigma)
$$

</details>

<hr/> <!-- 수평선 -->

- 이걸 그대로 sampling loop에 넣으면 **Algorithm 1** (classifier guided DDPM sampling)이 된다.

```text
Algorithm 1: Classifier guided diffusion sampling (DDPM)
Input: class label y, gradient scale s
x_T ← sample from N(0, I)
for t = T, ..., 1 do
    μ, Σ ← μ_θ(x_t), Σ_θ(x_t)
    x_{t-1} ← sample from N(μ + s·Σ·∇_{x_t} log p_φ(y|x_t), Σ)
end for
return x_0
```

#### <span style="color: #4682B4">4.3 DDIM을 위한 Score 기반 유도</span>

- DDIM은 deterministic이라 위처럼 "Gaussian 평균 shift"로 해석할 수 없다. 대신 **score function** 관점으로 유도한다.
- $\epsilon_{\theta}$와 score의 관계: $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})$에서

$$
    \nabla_{x_t}\log{p_{\theta}(x_t)} = -\frac{1}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_{\theta}(x_t)
$$

<details>
<summary> <span style="color: #ffd33d">ε ↔ score 관계 유도 펼치기/접기</span> </summary>

- Gaussian $q(x_t|x_0)$의 log density를 $x_t$로 미분하면

$$
    \nabla_{x_t}\log{q(x_t|x_0)}
    = -\frac{x_t - \sqrt{\bar{\alpha}_t}\,x_0}{1-\bar{\alpha}_t}
    = -\frac{\sqrt{1-\bar{\alpha}_t}\,\epsilon}{1-\bar{\alpha}_t}
    = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}
$$

- (마지막에 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$을 대입)
- denoising score matching 관점에서 $\epsilon_{\theta}$가 $\epsilon$의 conditional expectation을 근사하므로
marginal score도 $-\epsilon_{\theta}(x_t)/\sqrt{1-\bar{\alpha}_t}$로 근사된다. $\blacksquare$

</details>

<hr/> <!-- 수평선 -->

- conditional 분포 $p(x_t)\,p(y|x_t)$의 score는 두 score의 합이다.

$$
    \nabla_{x_t}\log{\big( p_{\theta}(x_t)\,p_{\phi}(y|x_t) \big)}
    = -\frac{1}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_{\theta}(x_t) + \nabla_{x_t}\log{p_{\phi}(y|x_t)}
$$

- 이걸 다시 $\epsilon$ 형태로 되돌리면 **guidance가 반영된 새로운 노이즈 예측** $\hat{\epsilon}$이 정의된다.

$$
    \hat{\epsilon}(x_t) := \epsilon_{\theta}(x_t) - \sqrt{1-\bar{\alpha}_t}\;\nabla_{x_t}\log{p_{\phi}(y|x_t)}
$$

- DDIM sampling 식의 $\epsilon_{\theta}$ 자리에 $\hat{\epsilon}$을 그대로 넣으면 **Algorithm 2** (classifier guided DDIM)가 된다.

```text
Algorithm 2: Classifier guided DDIM sampling
Input: class label y, gradient scale s
x_T ← sample from N(0, I)
for t = T, ..., 1 do
    ε̂ ← ε_θ(x_t) − √(1−ᾱ_t) · s · ∇_{x_t} log p_φ(y|x_t)
    x_{t-1} ← √ᾱ_{t-1} · (x_t − √(1−ᾱ_t)·ε̂)/√ᾱ_t + √(1−ᾱ_{t-1}) · ε̂
end for
return x_0
```

#### <span style="color: #4682B4">4.4 Gradient Scale $s$ — GAN의 truncation trick에 대응</span>

- 실험해보니 scale 1.0으로는 원하는 class가 잘 안나오는 경우가 있어서(sample의 절반 정도만 의도한 class로 분류됨),
gradient에 $s > 1$을 곱하니 class 일치율이 거의 100%로 올라가고 fidelity도 좋아졌다.
- $s$를 곱하는 것의 의미를 수식으로 보면

$$
    s\cdot\nabla_{x_t}\log{p_{\phi}(y|x_t)} = \nabla_{x_t}\log{\frac{1}{Z}p_{\phi}(y|x_t)^s}
$$

- 즉 $p(y|x)^s$ 라는 **재정규화된 더 뾰족한 분포**를 쓰는 것과 같다. $s > 1$이면 분포가 classifier가 확신하는
mode 쪽으로 집중되고, 그 결과 **diversity를 희생하고 fidelity를 얻는다.**
- 이게 정확히 GAN의 truncation trick[5]이 하던 역할이고, 논문의 가설 2번(GAN 우위의 원인)을 diffusion에서 재현한 것이다.
- 실제로 $s$를 조절하면 precision(fidelity)과 recall(diversity)이 반대로 움직이는 것이 실험으로 확인된다.
  - $s$ ↑ → precision ↑, IS ↑ / recall ↓
  - FID는 fidelity와 diversity 둘 다 반영하므로 중간 어딘가에서 최적값이 나옴
- 재밌는 점: **conditional diffusion model($p(y)$를 AdaGN으로 주입) + guidance를 같이 쓰는게 최고 성능**이다.
unconditional model + guidance 만으로도 conditional model에 가까운 FID가 나온다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[5] Results</span>

#### <span style="color: #4682B4">5.1 SOTA 비교</span>

- LSUN (unconditional, 256×256): StyleGAN 계열과 비교

| Dataset | 이전 SOTA (GAN) | ADM (dropout) |
|---|---|---|
| LSUN Bedroom | StyleGAN 2.35 | **1.90** |
| LSUN Horse | StyleGAN2 3.84 | **2.57** |
| LSUN Cat | StyleGAN2 7.25 | **5.57** |

- ImageNet (class-conditional): BigGAN-deep과 비교 (FID, 낮을수록 좋음)

| 해상도 | BigGAN-deep | ADM | ADM-G (guidance) | ADM-G, ADM-U |
|---|---|---|---|---|
| 128×128 | 6.02 | 5.91 | **2.97** | - |
| 256×256 | 6.95 | 10.94 | 4.59 | **3.94** |
| 512×512 | 8.43 | 23.24 | 7.72 | **3.85** |

- guidance 없이도(ADM) 128×128에서 이미 BigGAN-deep을 이기고, guidance를 붙이면(ADM-G) 모든 해상도에서 이긴다.
- precision/recall로 보면 BigGAN-deep은 precision(fidelity)이 높은 대신 recall(diversity)이 낮고,
ADM-G는 **비슷한 precision에서 더 높은 recall**을 유지한다. 즉 분포 커버리지가 더 좋다.

#### <span style="color: #4682B4">5.2 Guidance vs Upsampling</span>

- 저해상도 모델 + upsampling diffusion model(ADM-U) 조합도 FID를 크게 개선하는데, guidance와는 **개선하는 축이 다르다.**
  - upsampling: precision은 유지하면서 recall을 개선 (base 모델의 diversity를 유지한 채 detail만 보강)
  - guidance: recall을 희생하고 precision을 개선
- 그래서 두 개를 **같이 쓰면**(guidance로 base를 뾰족하게 + upsampler로 해상도 복원) 최고 성능이 나온다.
(256×256 FID 3.94, 512×512 FID 3.85)

#### <span style="color: #4682B4">5.3 Sampling 속도</span>

- 250 step DDPM sampling이 기본이지만, learned $\Sigma$ 덕분에 step을 줄여도 성능 하락이 적고,
**DDIM 25 step만으로도 BigGAN-deep보다 나은 FID**를 얻는다.
- 그래도 GAN의 single forward pass 대비하면 여전히 느리다. (이 논문의 한계로 명시함)

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[6] Limitations & 개인적인 생각</span>

- 논문이 직접 언급하는 한계:
  1. 여전히 sampling이 GAN보다 느리다. (multiple denoising step)
  2. classifier guidance는 **label이 있는 데이터셋에서만** 가능하다. unlabeled 데이터로 확장하려면
     clustering 기반 synthetic label이나 discriminative model로 sample quality를 유도하는 방향을 future work로 제시한다.
- classifier의 gradient를 생성 과정에 주입한다는 아이디어는 adversarial example과 수학적으로 완전히 같은 연산인게 재밌다.
(classifier가 원하는 class로 분류하도록 입력을 gradient 방향으로 조금씩 미는 것)
차이점은 diffusion은 매 step 노이즈 제거와 같이 진행되기 때문에 자연스러운 이미지 manifold 위에서 움직인다는 점.
- 이 논문의 classifier guidance는 이후 **classifier-free guidance**(Ho & Salimans)로 발전해서 classifier 없이
conditional/unconditional 모델의 출력 차이로 같은 효과를 내게 되고, 그게 지금의 Stable Diffusion, DALL·E 계열까지 이어진다.
그 흐름의 출발점이 된 논문이라 diffusion 계보에서 반드시 읽어야 하는 논문이라고 생각한다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">Reference</span>

- [5] A. Brock et al., "Large Scale GAN Training for High Fidelity Natural Image Synthesis" (BigGAN)
- [25] J. Ho et al., "Denoising Diffusion Probabilistic Models" (DDPM)
- [43] A. Nichol & P. Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (IDDPM)
- [56] J. Sohl-Dickstein et al., "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
- [57] J. Song et al., "Denoising Diffusion Implicit Models" (DDIM)
