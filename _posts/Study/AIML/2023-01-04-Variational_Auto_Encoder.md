---
title: "Variational AutoEncoder(VAE)"
tags:
  - Pytorch
categories:
  - AI/ML Study
date: 2023-01-04
toc: true
toc_sticky: true
toc_icon: "sticky-note"
use_math: true
last_modified_at: 2023-01-12T18:17:35
---

# Variational AutoEncoder

---

## 문제 정의

데이터는 $\mathcal{X} = \lbrace x\_{1},\, x\_{2} ,\, \ldots,\, x\_{n} \rbrace$ 일때

VAE는 Data $\mathcal{X}$를 잘 표현하는 z의 분포를 찾아서 이 분포로 부터 샘플링한 z값으로 새로운(unseen) 데이터 $x$를 생성하는 것이 목표

x에 대한 z의 분포인 posterior $p(z\|x)$를 찾아야된다.

---

## Intractable value

### Evidence

Maximum Likelihood Estimation(MLE)이나 Maximum A Posterior(MAP)와는 다르게 posterior인 $p(z\|x)$의 최대값만 찾으면 되는게 아니라
전체 분포를 알아야된다.

$$
    p(z\|x) = \frac{p(x|z) p(z)}{ \int p(x|z) p(z) dz}
$$

posterior의 베이지안 방정식을 살펴보면 실제 posterior의 분포를 알기위해서 likelihood와 prior 그리고 분모인 evidence 부분을 모두 계산해야 된다.

likelihood는 데이터 관찰을 통해 계산이 가능하고 $p(z)$는 VAE에서 $\mathcal{N}(0, 1^2)$으로 정의되어 쉽게 계산가능하다.

하지만 분모인 evidence는 z에 대한 적분으로 전체 z에 대해 모두 계산이 되기 때문에 거의 계산이 불가능하다.

### Output

위에서 운이 좋게 evidence 부분을 계산해 값을 구했다고 해도 마찬가지로 output에서도 똑같은 문제가 발생한다.

단순히 output을 구하면 MLE, MAP와 다를게 없기 때문에 마찬가지로 z에 따른 output에 대한 기대값을 구해야된다.


$$
    \mathbb{E}_{z \sim p(z|x)} \left[ y(z) \right] = \int y(z) p(z|x) dz
$$

여기서, y(z)는 output에 관한 함수임 

적분식에 $p(z\|x)$이 포함되어 마찬가지로 전체 z에 대해 적분해야 되는데 계산이 거의 불가능하다.

---

## Variable Inference

위와 같이 posterior $p(z\|x)$를 계산하는데 어려움이 많으니깐 대신 우리가 알고 있고 
parameter $\theta$도 적은 함수인 $q(z\|\theta)$에 근사시켜 대신 구하는 방법인 변분 추론(Variable Inference)을 사용한다.

두 분포가 얼마나 닮았는지는 분포의 유사도를 측정하는 KL-Divergence를 사용한다.

$$
    D_{KL} \left( q(z|\theta) || p(z|x) \right) = \int q(z|\theta) \log \frac{q(z|\theta}{p(z|x} \, dz
$$

여기서 $\theta$는 encoder 함수인 $f_{\mu}, \, f_{\sigma}$를 통해 계산된 값을 나타냄 $\theta_i = (f_{\mu}(x_i), \, f_{\sigma}(x_i)))$

## Objective

x에 대한 적절한 z의 분포를 찾았는지는 전체 z에 대한 likelihood를 계산해보면 된다. 이 값이 크면 클수록 x의 적절한 z분포일 확률이 높다는 뜻이다

$$
    \sum^{all}_{z \in \mathbb{Z}} likelihood \times prior = \int p(x|z) p(z) \, dz 
$$

우변은 marginal likelihood인 $p(x)$와 같다.

결국 marginal likelihood인 $p(x)$을 최대화하면 된다. 하지만 위에서 언급한 것처럼 intractable value인 $p(x)$는 계산이 거의 불가능하다.

그래서 대신 marginal likelihood을 계산 가능한 값으로 이루어진 Lower Bound를 찾아 Lower Bound를 최대화하는 전략인 
Evidence Lower Bound(ELBO)를 사용해서 간접적으로 최대화한다.

### Evidence Lower Bound(ELBO)

ELBO를 유도하기 위해 KL-Divergence의 항상 0보다 크거나 같은 특성을 사용한다.

$$
    D_{KL} \left( q(z|\theta) || p(z|x) \right) \ge 0
$$

위의 식을 전개해보면 다음과 같다.

$$
    \begin{split}
    D_{KL} (q(z|\theta) || p(z|x)) &= \int q(z|\theta) \log \frac {q(z|\theta) p(x)}{ p(x|z) p(z) } dz \\
            &= \int q(z|\theta) \log p(x) dz - \int q(z|\theta) \log p(x|z) dz + \int q(z|\theta) log \frac{q(z|\theta)}{p(z)} dz \\
            &=  \log p(x) - \mathbb{E}_{z \sim q(z|\theta)} \big[ log p(x|z) \big] + D_{KL} \big(q(z|\theta) || p(z) \big)
    \end{split}
$$

$$
    \log p(x) - \mathbb{E}_{z \sim q(z|\theta)} \big[ log p(x|z) \big] + D_{KL} \big(q(z|x) || p(z) \big) \ge 0 \\
    \log p(x) \ge \mathbb{E}_{z \sim q(z|\theta)} \big[ log p(x|z) \big] - D_{KL} \big(q(z|\theta) || p(z) \big) \\
    
    ELBO(Evidence \; Lower \; Bound) = \mathbb{E}_{z \sim q(z|\theta)} \big[ log p(x|z) \big] - D_{KL} \big(q(z|\theta) || p(z) \big)
$$

---


# 초안

$p(x\|z)$ 가우시안 분포인 z에서 원래 데이터인 x로 가는 매핑을 통해 새로운 데이터를 생성하는 분포를 알아야함

데이터를 표현하는 likelihood의 합 $p(x) = \int p(x\|z) p(z) dz$ 이 최대가 되야함 marginal likelihood

사후확률 $p(z\|x)$을 잘 알고있는 $q(z\|x)$에 변분 추론을 통해 근사하려함

근사식

$$
    D_{KL} (q(z|x) || p(z|x)) = \int q(z|x) \log \frac{q(z|x)}{p(z|x)} dz 
$$

p(z)는 가우시안 

$$
    p(z|x) = \frac{p(x|z) p(z)}{ p(x) } \\
    
    \begin{split}
    D_{KL} (q(z|x) || p(z|x)) &= \int q(z|x) \log \frac {q(z|x) p(x)}{ p(x|z) p(z) } dz \\
            &= \int q(z|x) \log p(x) dz - \int q(z|x) \log p(x|z) dz + \int q(z|x) log \frac{q(z|x)}{p(z)} dz \\
            &=  \log p(x) - \mathbb{E}_{z \sim q(z|x)} \big[ log p(x|z) \big] + D_{KL} \big(q(z|x) || p(z) \big)
    \end{split}
$$


likelihood인 $p(x|z)$가 매핑이 잘되면 likelihood와 prior의 적분합인 데이터 $p(x)$를 잘표현하게되고
marginal likelihood인 $p(x)$의 값이 최대값을 가지면 됨


log는 단조함수이므로 $\log p(x)$가 최대면 $p(x)$도 최대

$\log p(x)$를 직접적으로 계산하기는 어려움으로 대신 lower bound가 있다면 lower bound를 커지게 하면 자동적으로 $\log p(x)$도 커짐

여기서 $D\_{KL} \big( q(z\|x) \|\| p(z\|x) \big)$는 kl divergence의 특성상 항상 0보다 큼

$$
    D_{KL} \big( q(z|x) || p(z|x) \big) \ge 0 \\
    \log p(x) - \mathbb{E}_{z \sim q(z|x)} \big[ log p(x|z) \big] + D_{KL} \big(q(z|x) || p(z) \big) \ge 0 \\
    \log p(x) \ge \mathbb{E}_{z \sim q(z|x)} \big[ log p(x|z) \big] - D_{KL} \big(q(z|x) || p(z) \big) \\
    
    ELBO(Evidence \; Lower \; Bound) = \mathbb{E}_{z \sim q(z|x)} \big[ log p(x|z) \big] - D_{KL} \big(q(z|x) || p(z) \big)
$$


$$
    marginal \; log-likelihood \quad \log p(x) \\
    \begin{split}
        \log p(x) &= D_{KL} \big( q(z|x) || p(z|x) \big) + \mathbb{E}_{z \sim q(z|x)} \big[ \log p(x|z) \big] - D_{KL} \big( q(z|x) || p(z) \big) \\
            &= D_{KL} \big( q(z|x) || p(z|x) \big) + ELBO
    \end{split}
$$


여기서 ELBO 값이 커지면 Variable Inference 부분인 $D\_{KL} \big( q(z\|x) \|\| p(z\|x) \big)$의 값이 작아짐.

Loss값은 ELBO값을 크게 만들면 됨

ELBO 값중 $\mathbb{E}\_{z \sim q(z\|x)} \big\[ \log p(x\|z) \big\]$을 실제 계산값으로 만들어야됨

$$
    p(x|z) = \frac{1}{\sigma \sqrt{2 \pi}} \exp \bigg(-\frac{(x-\hat{x}_{mu})^2}{ 2 \sigma^2 } \bigg) \\
    
    \begin{split}
    \log p(x|z) &= -log(\sigma \sqrt{2 \pi}) - \frac{(x-\hat{x}_{mu})^2}{ 2 \sigma^2 } \\
    \end{split}
$$

몬테카를로 근사에 의해 다음값으로 근사가능

$$  
    \begin{split}
    \mathbb{E}_{z \sim q(z|x)} \big[ \log p(x|z) \big] &\approx \frac{1}{N} \sum\log p(x|z) dz \\
            &= - \frac{1}{N} \sum \bigg( \frac{(x-\hat{x})^2}{ 2 \sigma^2 } + log(\sigma \sqrt{2 \pi}) \bigg)
    \end{split}
$$


$z$ 는 encoder에 의해 $input \; \hat{x}$를 latent space에 embedding한 값     

$x$ 는 decoder에서 나오는 예측 이미지(normalize된 상태) sigma 는 1

$$
    \mathbb{E}_{z \sim q(z|x)} \big[ \log p(x|z) \big] \approx - \frac{1}{N} \sum \bigg( \frac{(x-\hat{x})^2}{ 2 } \bigg)
$$

$D\_{KL} \big( q(z\|x) \|\| p(z) \big)$ 값 계산

$p(z)$ 는 $\sigma = 1, \mu =0$인 gaussian distribution 


$$
    \log q(z|x) = - log \sigma_1 - \frac{1}{2} log 2 \pi - \frac{(z - \mu_1)^2}{2 \sigma_1^2} \\
    \log p(z) = - log \sigma_2 - \frac{1}{2} log 2 \pi - \frac{(z - \mu_2)^2}{2 \sigma_2^2}
$$

$$
    \begin{split}
    D_{KL} \big( q(z|x) || p(z) \big) &= \int q(z|x) \log \frac{q(z|x)}{p(z)} dz \\
        &= \int q(z|x) \bigg[ log \frac{\sigma_2}{\sigma_1} - \frac{(z - \mu_1)^2}{2 \sigma_1^2} + \frac{(z - \mu_2)^2}{2 \sigma_2^2} \bigg] dz
    \end{split}
$$

$$
\begin{split}
\left( 1 \right)  \; \int q(z|x) \bigg[ log \frac{\sigma_2}{\sigma_1} \bigg] dz &= log \frac{\sigma_2}{\sigma_1} \int q(z|x) dz \\
            &= log \frac{\sigma_2}{\sigma_1} \cdot 1
\end{split}
$$

$$
\begin{split}
\left( 2 \right) \; \int q(z|x) \cdot \frac{(z - \mu_1)^2}{2 \sigma_1^2} dz &= \int \frac{1}{\sigma_1 \sqrt{ 2 \pi}} 
            \exp \left( \frac{-(z - \mu_1)^2}{ 2 \sigma_1^2} \right) \cdot \frac{(z - \mu_1)^2}{2 \sigma_1^2}dz , \quad t = \frac{z - \mu_1}{ \sigma_1 \sqrt{2} } \\
    &= \frac{1}{\sigma_1 \sqrt{ 2 \pi }} \int t^2 \exp ( -t^2 ) \sigma_1 \sqrt{2} dt  \;, \quad dz = \sigma_1 \sqrt{2} dt \\
    &= \frac{1}{\sqrt{\pi}} \int t^2 \exp(-t^2) dt \\ 
    &= \frac{1}{\sqrt{\pi}} \frac{\sqrt{\pi}}{2} = \frac{1}{2}
\end{split}
$$


$$  
    \begin{split}
        \int^{\infty}_{0} x^2 e^{-x^2} dx &= [x \cdot -\frac{1}{2} e^{-x^2}]^{\infty}_{0} + \frac{1}{2} \int^{\infty}_{0} e^{-x^2} dx \\
            &= 0 + \frac{\sqrt{\pi}}{4} \\
        
    \end{split}
$$

가우스 적분

$$
    \int^{\infty}_{-\infty} e^{-x^2} dx = \sqrt{\pi} \\
    \int^{\infty}_{0} x^n e^{-x^m} dx = \frac {1}{m} \Gamma (\frac{n+1}{m})
$$

감마 함수

$$
    \Gamma(s) = \int^{\infty}_{0} e^{-t}t^{s} \frac{dt}{t}
$$

$$
\begin{split}
\left( 3 \right) \; \int q(z|x) \cdot \frac{(z-\mu_2)^2}{2 \sigma^2_{2}} dz &= \int \frac{1}{\sigma_1 \sqrt{ 2 \pi}}  \exp \left( \frac{-(z - \mu_1)^2}{ 2 \sigma_1^2} \right)
    \cdot \frac{(z-\mu_2)^2}{2 \sigma^2_{2}} dz \\
        &= \frac{1}{\sigma_2^2 \sigma_1 \sqrt{ 2 \pi}} \int \exp \left( \frac{-(z - \mu_1)^2}{ 2 \sigma_1^2} \right) \cdot [ z^2 - 2 \mu_2 z +  \mu_2^2 ] dz \\
\end{split}
$$

$$ 
\begin{split}
\left( 3 \text{-} 1 \right) \; \int \exp \left( \frac{-(z - \mu_1)^2}{ 2 \sigma_1^2} \right) \cdot z^2 dz
    &= \int e^{-t^2} (\sigma_1 \sqrt{2} t + \mu_1)^2 \sigma_1 \sqrt{2} \, dt  \;,\quad t= \frac{z - \mu_{1}}{\sigma_{1} \sqrt{2}} \\
    &= \sigma_1 \sqrt{2} \int 2\sigma^2_1 t^2 e^{-t^2} + 2\sqrt{2} \sigma_1 \mu_1 t e^{-t^2} + \mu_1^2 e^{-t^2} \, dt \\
    &= 2 \sqrt{2} \sigma_1^3 \int t^2 e^{-t^2} \, dt + 4 \sigma_{1}^2 \mu_1 \int t e^{-t^2} \, dt + \sigma_1 \mu_1^2 \sqrt{2} \int e^{-t^2} \, dt \\
    &= 2 \sqrt{2} \sigma_1^3 \cdot \frac{\sqrt{\pi}}{2} + 4 \sigma_{1}^2 \mu_1 \cdot 0 + \sigma_1 \mu_1^2 \sqrt{2} \cdot \sqrt{\pi} \\
    &= \sigma^3_1 \sqrt{2 \pi} + \sigma_1 \mu_1^2 \sqrt{2 \pi}
\end{split}
$$

$$
\begin{split}
\left( 3\text{-}2 \right) \; \int \exp \left( \frac{-(z - \mu_1)^2}{ 2 \sigma_1^2} \right) \cdot (2 \mu_2 z) dz 
    &= 2 \mu_2 \int e^{-t^2} (\sigma_1 \sqrt{2} t + \mu_1) \sigma_1 \sqrt{2} \, dt \; , \quad t= \frac{z - \mu_{1}}{\sigma_{1} \sqrt{2}} \\
    &= 4 \sigma_{1}^{2} \mu_2 \int t e^{-t^2} \, dt + 2 \sqrt{2} \sigma_1 \mu_1 \mu_2 \int e^{-t^2} \, dt \\
    &= 4 \sigma_{1}^{2} \mu_2 \cdot 0 + 2 \sqrt{2} \sigma_1 \mu_1 \mu_2 \cdot \sqrt{\pi} \\
    &= 2 \sqrt{2 \pi} \sigma_1 \mu_1 \mu_2
\end{split}
$$

$$
\begin{split}
\left (3 \text{-} 3 \right) \; \int \exp \left( \frac{-(z - \mu_1)^2}{ 2 \sigma_1^2} \right) \cdot \mu^2_2 \, dz 
    &= \mu^2_2 \int e^{-t^2} \sigma_1 \sqrt{2} \, dt  \; , \quad t= \frac{z - \mu_{1}}{\sigma_{1} \sqrt{2}} \\
    &= \mu^2_2 \sigma_1 \sqrt{2} \cdot \sqrt{\pi} =  \mu^2_2 \sigma_1 \sqrt{2 \pi}
\end{split}
$$

$$  
\begin{split}
    \left( 3 \right) &= \frac{1}{\sigma_1 \sigma_2^2 \sqrt{2 \pi}} \big[ \left( 3\text{-} 1\right) - \left( 3\text{-} 2\right) + \left( 3\text{-} 3\right) \big] \\
     &= \frac{1}{\sigma_1 \sigma_2^2 \sqrt{2 \pi}} \big[ 
            \sigma^3_1 \sqrt{2 \pi} + \sigma_1 \mu_1^2 \sqrt{2 \pi} -
            2 \sqrt{2 \pi} \sigma_1 \mu_1 \mu_2 +
            \mu^2_2 \sigma_1 \sqrt{2 \pi}
        \big] \\
    &= \frac{\sigma_1^2 + \mu_1^2 - 2\mu_1 \mu_2 + \mu^2_2}{\sigma_2^2} \\
    &= \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{\sigma_2^2}
    
\end{split}
$$

$$
\begin{split}
    D_KL(q(z|x) || p(z)) &= (1) - (2) + (3) \\
        &= \log \frac{\sigma_2}{\sigma_1} - \frac{1}{2} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{\sigma_2^2}
\end{split}
$$

여기서 $p(z)$ $\sigma\_2=1, \; \mu\_2=0$인 gaussian 분포 함수임

$$
    D_KL(q(z|x) || p(z)) = - \log \sigma_1 - \frac{1}{2} + \sigma_1^2 + \mu_1^2 
$$

그래서 실제 Loss는 ELBO를 최대화하는 것이므로 -ELBO를 최소화하는 것과 같다 식은 다음과 같다.

$$
\begin{split}
    Loss &= -ELBO \\
        &= - \mathbb{E}_{z \sim q(z|x)} \big[ \log p(x|z) \big] + D_{KL} \big( q(z|x) || p(z) \big) \\ 
        &= \frac{1}{N} \sum \bigg( \frac{(x-\hat{x})^2}{ 2 } \bigg) + \sigma_1^2 + \mu_1^2 - \log \sigma_1 - \frac{1}{2} \\
\end{split}
$$

사실상 ELBO는 recontruction error와 $q(z\|x)$를 gaussian 분포인 $p(z)=\mathcal{N}(0, 1^2)$에 의해 regularization되는 항이 합해짐

## Reference

[https://hyeongminlee.github.io/post/bnn002_mle_map/](https://hyeongminlee.github.io/post/bnn002_mle_map/)