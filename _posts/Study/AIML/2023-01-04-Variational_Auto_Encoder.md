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
last_modified_at: 2023-01-16T15:55:56
---

# Variational AutoEncoder

## 문제 정의

데이터는 $\mathcal{X} = \lbrace x\_{1},\, x\_{2} ,\, \ldots,\, x\_{n} \rbrace$ 일때

VAE는 Data $\mathcal{X}$를 잘 표현하는 z의 분포를 찾아서 이 분포로 부터 샘플링한 z값으로 새로운(unseen) 데이터 $x$를 생성하는 것이 목표

가지고 있는 데이터로 x에 대한 z의 분포인 posterior $p(z\|x)$를 찾아야된다.

이때 posterior $p(z\|x)$는 계산하기 어려워 대신 $q(z\|\theta)$로 근사하는 변분추론을 사용하고 $\theta=(f\_\mu(x), f\_\sigma(x)))$는 encoder 함수 $f$에
의해 계산된 값으로 구성된다.

decoder는 $x=y(z\|\theta)$ 함수를 사용하고 예측된 값의 확률 분포는 $p(x\|z)$이다.

## Intractable value

### Evidence

Maximum Likelihood Estimation(MLE)이나 Maximum A Posterior(MAP)와는 다르게 posterior인 $p(z\|x)$의 최대값만 찾으면 되는게 아니라
전체 분포를 알아야된다.

$$
    p(z|x) = \frac{p(x|z) p(z)}{ \int p(x|z) p(z) dz}
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

여기서, y(z)는 output에 관한 함수이고 딥러닝에서 neural network를 나타낸다. 

적분식에 $p(z\|x)$이 포함되어 마찬가지로 전체 z에 대해 적분해야 되는데 계산이 거의 불가능하다.


## Variable Inference


위와 같이 posterior $p(z\|x)$를 계산하는데 어려움이 많으니깐 대신 우리가 알고 있고 
parameter $\theta$도 적은 함수인 $q(z\|\theta)$에 근사시켜 대신 구하는 방법인 변분 추론(Variable Inference)을 사용한다.

두 분포가 얼마나 닮았는지는 분포의 유사도를 측정하는 KL-Divergence를 사용한다.

$$
    D_{KL} \left( q(z|\theta) \, || \, p(z|x) \right) = \int q(z|\theta) \log \frac{q(z|\theta)}{p(z|x)} \, dz
$$

여기서 $\theta$는 encoder 함수인 $f_{\mu}, \, f_{\sigma}$를 통해 계산된 값을 나타냄 $\theta_i = (f_{\mu}(x_i), \, f_{\sigma}(x_i))$

## Objective

x에 대한 적절한 z의 분포를 찾았는지는 전체 z에 대한 likelihood를 계산해보면 된다. 이 값이 크면 클수록 x의 적절한 z분포일 확률이 높다는 뜻이다

$$
    \sum^{all}_{z \in \mathbb{Z}} likelihood \times prior = \int p(x|z) \, p(z) \, dz 
$$

우변은 marginal likelihood인 $p(x)$와 같다.

결국 marginal likelihood인 $p(x)$을 최대화하면 된다. 하지만 위에서 언급한 것처럼 intractable value인 $p(x)$는 계산이 거의 불가능하다.

그래서 대신 marginal likelihood을 계산 가능한 값으로 이루어진 Lower Bound를 찾아 Lower Bound를 최대화하는 전략인 
Evidence Lower Bound(ELBO)를 사용해서 간접적으로 최대화한다.

### Evidence Lower Bound(ELBO) - 유도

---

ELBO를 유도하기 위해 변분 추론을 위한 위의 식을 전개해보면 다음과 같다.

$$
    \begin{split}
    D_{KL} (q(z|\theta) \, || \, p(z|x)) &= \int q(z|\theta) \log \frac {q(z|\theta) p(x)}{ p(x|z) p(z) } dz \\
            &= \int q(z|\theta) \log p(x) dz - \int q(z|\theta) \log p(x|z) dz + \int q(z|\theta) log \frac{q(z|\theta)}{p(z)} dz \\
            &=  \log p(x) - \mathbb{E}_{z \sim q(z|\theta)} \big[ log p(x|z) \big] + D_{KL} \big(q(z|\theta) \, || \, p(z) \big)
    \end{split}
$$

evidence의 Lower Bound를 찾기 위해 KL-Divergence의 항상 0보다 크거나 같은 특성을 사용한다.

$$
    D_{KL} \left( q(z|\theta) \, || \, p(z|x) \right) \ge 0
$$  

위에서 전개된 우변을 evidence인 $\log p(x)$에 관한 식으로 정리하면 다음과 같이 ELBO가 나오게 된다.

$$
    \log p(x) - \mathbb{E}_{z \sim q(z|\theta)} \big[ \log p(x|z) \big] + D_{KL} \big(q(z|\theta) \, || \, p(z) \big) \ge 0 \\
    \log p(x) \ge \mathbb{E}_{z \sim q(z|\theta)} \big[ \log p(x|z) \big] - D_{KL} \big(q(z|\theta) \, || \, p(z) \big) \\
$$

여기서 우변은 계산가능하므로 Evidence의 Lower Bound으로 사용할수 있다.

$$
    ELBO(Evidence \; Lower \; Bound) = \mathbb{E}_{z \sim q(z|\theta)} \big[ \log p(x|z) \big] - D_{KL} \big(q(z|\theta) \, || \, p(z) \big)
$$

이제 evidence인 marginal log-likelihood $\log p(x)$의 lower bound를 찾았으니 이 값을 최대화 시키면 된다. 
이 값을 딥러닝에서 학습시키기 위해 실제 계산이 가능한 값들로 바꿔야한다.

### Evidence Lower Bound(ELBO) - 변환 

ELBO는 reconstruction error인 $\mathbb{E}\_{z \sim q(z\|\theta)} [ \log p(x\|z) ]$와 $p(z)$에 대한 regularization인 
$D\_{KL} (q(z\|\theta) \, \|\| \, p(z) )$ 2개의 항으로 구성되어 있다.

#### (1) Reconstruction Error

$\mathbb{E}\_{z \sim q(z\|\theta)} [ \log p(x\|z) ]$가 reconstruction error인 이유는 $z$를 encoder인 $q(z\|\theta)$으로 부터 
샘플링하고 다시 $z$를 decoder인 $p(x\|z)$를 통해 x로 복원하기 때문이다.

먼저 첫번째 항인 Reconstruction Error는 정확한 값을 구하기 위해선 전체 z에 대한 기대값을 적분해야 한다.

실제로 모든 z에 대한 계산은 거의 불가능 하므로 Monte Carlo Approximation으로 다음과 같이 근사가 가능하다.

$$
    E_{x \sim p(x)}[f(x)] = \int f(x) \, p(x) \, dx \approx \frac{1}{N} \sum^{N}_{i=1}f(x_i)
$$

어떤 분포로 부터 $N$개의 Sample을 추출해서 $f(x)$의 기대값을 근사할 수 있다. 

$$
    \mathbb{E}_{z \sim q(z|\theta)} \big[ \log p(x|z) \big] = \int \log p(x|z) \, q(z|\theta) \, dz \approx \frac{1}{N} \sum^{N}_{i=1} \log p(x|z)
$$

$p(x\|z)$는 $q(z\|\theta)$로 부터 sampling한 $z$를 Decoder 함수인 $y(z\|\theta)$를 통해 예측한 x의 확률 밀도 함수이다.

식으로 다음과 같이 표현할 수 있다.

$$
    p(x|z) = \frac{1}{\sigma \sqrt{2 \pi}} \exp (- \frac{(x - y(z|\theta))^2}{2 \sigma^2})
$$


<blockquote>
예를 들어, 키를 보고 몸무게를 예측하는 모델에서 어떤 모델을 학습으로 구한 parameter $w$를 통해 실제 몸무게($t$)를 예측한다고 할때

$$ t = y(x|w) $$
</blockquote>
>
> 이때 실제 몸무게($t$)와 내가 예측한 몸무게($y$)가 정확하게 일치한다고 말하기 어렵다. 실제 데이터의 특성상 170cm인 사람중에는
몸무게가 69kg도 있고 70kg일 수도 있다.        
> 
> 그래서 실제 몸무게(t)는 예측한 몸무게(y)를 평균으로 하고 특정한 값 $\sigma$를 표준편차로 하는 Gaussian Distribution을 따른다고 말하는게
더 정확한 표현이다. ($\sigma$는 내가 예측한 값에 대한 신뢰도를 나타낸다.)


$$
    \log p(x|z) = -log(\sigma \sqrt{2 \pi}) - \frac{(x - y(z|\theta))^2}{ 2 \sigma^2 }
$$

위 식을 아까 근사한 기대값에 대입해보면

$$
    \frac{1}{N} \sum^{N}_{i=1} \log p(x|z) = \frac{1}{N} \sum^{N}_{i=1} \bigg( - \frac{(x - y(z|\theta))^2}{ 2 \sigma^2 } -log(\sigma \sqrt{2 \pi}) \bigg)
$$

이를 최대화해야할 값인 $\mathbb{E}\_{z \sim q(z\|\theta)} [ \log p(x\|z) ]$으로 정리하면 다음과 같이 정리가 된다.

$$
    \begin{split}
        \mathbb{E}_{z \sim q(z|\theta)} \big[ \log p(x|z) \big]  &\approx \frac{1}{N} \sum^{N}_{i=1} \log p(x|z) \\
                &= \frac{1}{N} \sum^{N}_{i=1} \bigg( - \frac{(x - y(z|\theta))^2}{ 2 \sigma^2 } -log(\sigma \sqrt{2 \pi}) \bigg) \\
                &\approx - \frac{1}{N} \sum^{N}_{i=1} (x - y(z|\theta))^2
    \end{split}
$$

고정된 값인 $\sigma$와 $\log \sigma\sqrt{2 \pi}$은 maximize에 영향이 없으므로 간단히 정리할 수 있다.

$p(x\|z)$가 gaussian distribution을 따르면 위의 식처럼 Mean Square Error(MSE)의 형태로 계산된다.

만약 $p(x\|z)$가 Bernoulli distribution을 따르면

$$
    \begin{split}
    Bernoulli(x ; y(z|\theta)) &= 
    \begin{cases}
        y(z|\theta) \\
        1-y(z|\theta) \\
    \end{cases} \\

         &= y(z|\theta)^x \, (1 - y(z|\theta))^{1 - x} 
    \end{split}
$$

이 값들을 아까 구했던 output의 기대값처럼 전개해보면 다음과 같이 Binary Cross Entropy(BCE) 형태의 식이 나온다.

$$
    \begin{split}
        \mathbb{E}_{z \sim q(z|\theta)} \big[ \log p(x|z) \big]  &\approx \frac{1}{N} \sum^{N}_{i=1} \log p(x|z) \\
                    &= \frac{1}{N} \sum^{N}_{i=1} \bigg(x \cdot \log y(z|\theta) + (1-x) \cdot \log \big(1 - y(z|\theta) \big) \bigg) 
    \end{split}
$$

#### (2) Regularization 

두번째 항인 $D\_{KL} (q(z\|\theta) \, \|\| \, p(z) )$는 $q(z\|\theta)$를 $\mathcal{N}(0, 1^2)$인 $p(z)$ 분포에 regularization한다.

gaussian distribution인 $q(z\|\theta), \, p(z)$을 식으로 나타내면 다음과 같다.

$$
    \begin{split}
        q(z|\theta) &= \mathcal{N}(\mu_1, \sigma_1^2) = \frac{1}{\sigma_1 \sqrt{2 \pi}} exp(-\frac{(z - \mu_1)^2}{ 2 \sigma_1^2}) \\
        p(z) &= \mathcal{N}(\mu_2, \sigma_2^2) = \frac{1}{\sigma_2 \sqrt{2 \pi}} exp(-\frac{(z - \mu_2)^2}{ 2 \sigma_2^2})
    \end{split}
$$

$$
    \log q(z|x) = - log \sigma_1 - \frac{1}{2} log 2 \pi - \frac{(z - \mu_1)^2}{2 \sigma_1^2} \\
    \log p(z) = - log \sigma_2 - \frac{1}{2} log 2 \pi - \frac{(z - \mu_2)^2}{2 \sigma_2^2}
$$

계산을 위해 log를 씌운 값을 각각에 대입하여 정리하면 다음과 같다.

$$
    \begin{split}
        D_{KL} (q(z|\theta) \, || \, p(z)) &= \int q(z|\theta) \, \log \frac{q(z|\theta)}{p(z)} \, dz  \\
                &= \int q(z|\theta) \log q(z|\theta) \, dz - \int q(z|\theta) \log p(z) \, dz \\
                &= \int q(z|\theta) \bigg[ log \frac{\sigma_2}{\sigma_1} - \frac{(z - \mu_1)^2}{2 \sigma_1^2} + \frac{(z - \mu_2)^2}{2 \sigma_2^2} \bigg] dz \\
                &= \int q(z|\theta) log \frac{\sigma_2}{\sigma_1} dz 
                    - \int q(z|\theta) \frac{(z - \mu_1)^2}{2 \sigma_1^2} dz
                    + \int q(z|\theta) \frac{(z - \mu_2)^2}{2 \sigma_2^2} dz \\
                &= (1) - (2) + (3)
    \end{split}
$$

먼저 (1)항에 대해서 계산해보면 gaussian distribution의 특성인 전체 확률의 합은 항상 $\int p(x) dx = 1$으로 간단하게 정리된다.

$$
\begin{split}
\left( 1 \right)  \; \int q(z|\theta) \bigg[ log \frac{\sigma_2}{\sigma_1} \bigg] dz &= log \frac{\sigma_2}{\sigma_1} \int q(z|\theta) dz \\
            &= log \frac{\sigma_2}{\sigma_1} \cdot 1
\end{split}
$$

(2) 항은 공통되는 부분을 t로 치환하면 다음과 같이 가우스 적분 형태로 정리된다.

$$
\begin{split}
\left( 2 \right) \; \int q(z|\theta) \cdot \frac{(z - \mu_1)^2}{2 \sigma_1^2} dz &= \int \frac{1}{\sigma_1 \sqrt{ 2 \pi}} 
            \exp \left( \frac{-(z - \mu_1)^2}{ 2 \sigma_1^2} \right) \cdot \frac{(z - \mu_1)^2}{2 \sigma_1^2}dz , \quad t = \frac{z - \mu_1}{ \sigma_1 \sqrt{2} } \\
    &= \frac{1}{\sigma_1 \sqrt{ 2 \pi }} \int t^2 \exp ( -t^2 ) \sigma_1 \sqrt{2} dt  \;, \quad dz = \sigma_1 \sqrt{2} dt \\
    &= \frac{1}{\sqrt{\pi}} \int t^2 \exp(-t^2) dt \\ 
    &= \frac{1}{\sqrt{\pi}} \frac{\sqrt{\pi}}{2} = \frac{1}{2}
\end{split}
$$

가우스 적분 기본 형태는 다음과 같고 부분적분을 사용해서 값을 다음과 같이 구할 수 있다.

$$
    \int^{\infty}_{-\infty} e^{-x^2} \, dx  = \sqrt{\pi}
$$

$$  
    \begin{split}
        \int^{\infty}_{0} x^2 e^{-x^2} dx &= [x \cdot -\frac{1}{2} e^{-x^2}]^{\infty}_{0} + \frac{1}{2} \int^{\infty}_{0} e^{-x^2} dx \\
            &= 0 + \frac{\sqrt{\pi}}{4} \\
        
    \end{split}
$$

(3) 항을 전개 해보면 위와 같이 공통되는 부분이 없어 하나하나 계산해야된다.

$$
\begin{split}
\left( 3 \right) \; \int q(z|\theta) \cdot \frac{(z-\mu_2)^2}{2 \sigma^2_{2}} dz &= \int \frac{1}{\sigma_1 \sqrt{ 2 \pi}}  \exp \left( \frac{-(z - \mu_1)^2}{ 2 \sigma_1^2} \right)
    \cdot \frac{(z-\mu_2)^2}{2 \sigma^2_{2}} dz \\
        &= \frac{1}{\sigma_2^2 \sigma_1 \sqrt{ 2 \pi}} \int \exp \left( \frac{-(z - \mu_1)^2}{ 2 \sigma_1^2} \right) \cdot [ z^2 - 2 \mu_2 z +  \mu_2^2 ] dz \\
        &= \frac{1}{\sigma_2^2 \sigma_1 \sqrt{ 2 \pi}} \left[ (3\text{-}1) - (3\text{-}2) + (3\text{-}3) \right] 
\end{split}
$$

전체를 (3-1), (3-2), (3-3) 항으로 나누어 다시 계산한다.

각각의 항을 전개해보면 다음과 같다.

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

(3-1) 항에서 아까 (2)에서 구한 가우스 적분 $\int t^2 e^{-t^2} dt$에는 값을 대입하고 $\int t e^{-t^2} dt$는 부분 적분으로 계산 가능하지만 기함수 형태이고
좌극한과 우극한이 같으므로 결국 0이된다.

$$
\begin{split}
\left( 3\text{-}2 \right) \; \int \exp \left( \frac{-(z - \mu_1)^2}{ 2 \sigma_1^2} \right) \cdot (2 \mu_2 z) dz 
    &= 2 \mu_2 \int e^{-t^2} (\sigma_1 \sqrt{2} t + \mu_1) \sigma_1 \sqrt{2} \, dt \; , \quad t= \frac{z - \mu_{1}}{\sigma_{1} \sqrt{2}} \\
    &= 4 \sigma_{1}^{2} \mu_2 \int t e^{-t^2} \, dt + 2 \sqrt{2} \sigma_1 \mu_1 \mu_2 \int e^{-t^2} \, dt \\
    &= 4 \sigma_{1}^{2} \mu_2 \cdot 0 + 2 \sqrt{2} \sigma_1 \mu_1 \mu_2 \cdot \sqrt{\pi} \\
    &= 2 \sqrt{2 \pi} \sigma_1 \mu_1 \mu_2
\end{split}
$$

(3-2)는 $\int t e^{-t^2} dt$는 기함수이므로 0을 대입해주면 간단하게 정리된다.

$$
\begin{split}
\left (3 \text{-} 3 \right) \; \int \exp \left( \frac{-(z - \mu_1)^2}{ 2 \sigma_1^2} \right) \cdot \mu^2_2 \, dz 
    &= \mu^2_2 \int e^{-t^2} \sigma_1 \sqrt{2} \, dt  \; , \quad t= \frac{z - \mu_{1}}{\sigma_{1} \sqrt{2}} \\
    &= \mu^2_2 \sigma_1 \sqrt{2} \cdot \sqrt{\pi} =  \mu^2_2 \sigma_1 \sqrt{2 \pi}
\end{split}
$$

(3-3)는 가우스 적분의 기본 형태이므로 아까 구한 값을 대입해서 정리하면된다.

각각의 항을 (3)에 대입해서 정리하면 다음과 같다.

$$
\begin{split}
    \left( 3 \right) \; \int q(z|\theta) \cdot \frac{(z-\mu_2)^2}{2 \sigma^2_{2}} dz &= \frac{1}{\sigma_2^2 \sigma_1 \sqrt{ 2 \pi}} \left[ (3\text{-}1) - (3\text{-}2) + (3\text{-}3) \right] \\
        &= \frac{1}{\sigma_1 \sigma_2^2 \sqrt{2 \pi}} \big[ 
            \sigma^3_1 \sqrt{2 \pi} + \sigma_1 \mu_1^2 \sqrt{2 \pi} -
            2 \sqrt{2 \pi} \sigma_1 \mu_1 \mu_2 +
            \mu^2_2 \sigma_1 \sqrt{2 \pi}
        \big] \\
    &= \frac{\sigma_1^2 + \mu_1^2 - 2\mu_1 \mu_2 + \mu^2_2}{\sigma_2^2} \\
    &= \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{\sigma_2^2}
\end{split}
$$

결국 regularization term인 $D\_{KL} (q(z\|\theta) \, \|\| \, p(z) )$은 각각의 항을 대입해서 정리할 수 있다.

$$
    \begin{split}
        D_{KL} (q(z|\theta) \, || \, p(z)) &= \int q(z|\theta) \, \log \frac{q(z|\theta)}{p(z)} \, dz  \\
                &= (1) - (2) + (3)  \\
                &= \log \frac{\sigma_2}{\sigma_1} - \frac{1}{2} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{\sigma_2^2}
    \end{split}
$$

여기서 $p(z)=\mathcal{N}(\mu\_2, \sigma\_2^2)=\mathcal{N}(0, 1^2)$이므로 대입하면

$$
    \begin{split}
        D_{KL} (q(z|\theta) \, || \, p(z)) &= \log \frac{\sigma_2}{\sigma_1} - \frac{1}{2} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{\sigma_2^2} \\
                &= - \log \sigma_1 - \frac{1}{2} + \sigma_1^2 + \mu_1^2
    \end{split}
$$

ELBO식을 다시 정리해 보면 다음과 같이 실제 계산가능한 값들로 정리된다.

$$ 
\begin{split}
    ELBO &= \mathbb{E}_{z \sim q(z|\theta)} \big[ \log p(x|z) \big] - D_{KL} \big(q(z|\theta) \, || \, p(z) \big) \\
        &= \frac{1}{N} \sum^{N}_{i=1} \left(x \cdot \log y(z|\theta) + (1-x) \cdot \log \big(1 - y(z|\theta) \big) \right)
            - (\sigma_1^2 + \mu_1^2 - \log \sigma_1 - \frac{1}{2})
\end{split}
$$

실제 딥러닝에서는 Loss를 최소화하므로 다음과 같이 설정한다. (Bernoulli distribution일 경우, 만약 Gaussian distribution이면 BCELoss를 MSE로 변경)

$$  
    \begin{split}
    \mathcal{Loss} &= - ELBO \\
                &= - \frac{1}{N} \sum^{N}_{i=1} \left(x \cdot \log y(z|\theta) + (1-x) \cdot \log \big(1 - y(z|\theta) \big) \right)
            + (\sigma_1^2 + \mu_1^2 - \log \sigma_1 - \frac{1}{2}) \\
                &=  BCELoss + Regularziation
    \end{split}
$$

## Reference

[https://hyeongminlee.github.io/post/bnn002_mle_map/](https://hyeongminlee.github.io/post/bnn002_mle_map/)
[https://ratsgo.github.io/generative%20model/2017/12/19/vi/](https://ratsgo.github.io/generative%20model/2017/12/19/vi/)
[https://ratsgo.github.io/generative%20model/2018/01/27/VAE/](https://ratsgo.github.io/generative%20model/2018/01/27/VAE/)

[//]: # (# 초안)

[//]: # ()
[//]: # ($p&#40;x\|z&#41;$ 가우시안 분포인 z에서 원래 데이터인 x로 가는 매핑을 통해 새로운 데이터를 생성하는 분포를 알아야함)

[//]: # ()
[//]: # (데이터를 표현하는 likelihood의 합 $p&#40;x&#41; = \int p&#40;x\|z&#41; p&#40;z&#41; dz$ 이 최대가 되야함 marginal likelihood)

[//]: # ()
[//]: # (사후확률 $p&#40;z\|x&#41;$을 잘 알고있는 $q&#40;z\|x&#41;$에 변분 추론을 통해 근사하려함)

[//]: # ()
[//]: # (근사식)

[//]: # ()
[//]: # ($$)

[//]: # (    D_{KL} &#40;q&#40;z|x&#41; || p&#40;z|x&#41;&#41; = \int q&#40;z|x&#41; \log \frac{q&#40;z|x&#41;}{p&#40;z|x&#41;} dz )

[//]: # ($$)

[//]: # ()
[//]: # (p&#40;z&#41;는 가우시안 )

[//]: # ()
[//]: # ($$)

[//]: # (    p&#40;z|x&#41; = \frac{p&#40;x|z&#41; p&#40;z&#41;}{ p&#40;x&#41; } \\)

[//]: # (    )
[//]: # (    \begin{split})

[//]: # (    D_{KL} &#40;q&#40;z|x&#41; || p&#40;z|x&#41;&#41; &= \int q&#40;z|x&#41; \log \frac {q&#40;z|x&#41; p&#40;x&#41;}{ p&#40;x|z&#41; p&#40;z&#41; } dz \\)

[//]: # (            &= \int q&#40;z|x&#41; \log p&#40;x&#41; dz - \int q&#40;z|x&#41; \log p&#40;x|z&#41; dz + \int q&#40;z|x&#41; log \frac{q&#40;z|x&#41;}{p&#40;z&#41;} dz \\)

[//]: # (            &=  \log p&#40;x&#41; - \mathbb{E}_{z \sim q&#40;z|x&#41;} \big[ log p&#40;x|z&#41; \big] + D_{KL} \big&#40;q&#40;z|x&#41; || p&#40;z&#41; \big&#41;)

[//]: # (    \end{split})

[//]: # ($$)

[//]: # ()
[//]: # ()
[//]: # (likelihood인 $p&#40;x|z&#41;$가 매핑이 잘되면 likelihood와 prior의 적분합인 데이터 $p&#40;x&#41;$를 잘표현하게되고)

[//]: # (marginal likelihood인 $p&#40;x&#41;$의 값이 최대값을 가지면 됨)

[//]: # ()
[//]: # ()
[//]: # (log는 단조함수이므로 $\log p&#40;x&#41;$가 최대면 $p&#40;x&#41;$도 최대)

[//]: # ()
[//]: # ($\log p&#40;x&#41;$를 직접적으로 계산하기는 어려움으로 대신 lower bound가 있다면 lower bound를 커지게 하면 자동적으로 $\log p&#40;x&#41;$도 커짐)

[//]: # ()
[//]: # (여기서 $D\_{KL} \big&#40; q&#40;z\|x&#41; \|\| p&#40;z\|x&#41; \big&#41;$는 kl divergence의 특성상 항상 0보다 큼)

[//]: # ()
[//]: # ($$)

[//]: # (    D_{KL} \big&#40; q&#40;z|x&#41; || p&#40;z|x&#41; \big&#41; \ge 0 \\)

[//]: # (    \log p&#40;x&#41; - \mathbb{E}_{z \sim q&#40;z|x&#41;} \big[ log p&#40;x|z&#41; \big] + D_{KL} \big&#40;q&#40;z|x&#41; || p&#40;z&#41; \big&#41; \ge 0 \\)

[//]: # (    \log p&#40;x&#41; \ge \mathbb{E}_{z \sim q&#40;z|x&#41;} \big[ log p&#40;x|z&#41; \big] - D_{KL} \big&#40;q&#40;z|x&#41; || p&#40;z&#41; \big&#41; \\)

[//]: # (    )
[//]: # (    ELBO&#40;Evidence \; Lower \; Bound&#41; = \mathbb{E}_{z \sim q&#40;z|x&#41;} \big[ log p&#40;x|z&#41; \big] - D_{KL} \big&#40;q&#40;z|x&#41; || p&#40;z&#41; \big&#41;)

[//]: # ($$)

[//]: # ()
[//]: # ()
[//]: # ($$)

[//]: # (    marginal \; log-likelihood \quad \log p&#40;x&#41; \\)

[//]: # (    \begin{split})

[//]: # (        \log p&#40;x&#41; &= D_{KL} \big&#40; q&#40;z|x&#41; || p&#40;z|x&#41; \big&#41; + \mathbb{E}_{z \sim q&#40;z|x&#41;} \big[ \log p&#40;x|z&#41; \big] - D_{KL} \big&#40; q&#40;z|x&#41; || p&#40;z&#41; \big&#41; \\)

[//]: # (            &= D_{KL} \big&#40; q&#40;z|x&#41; || p&#40;z|x&#41; \big&#41; + ELBO)

[//]: # (    \end{split})

[//]: # ($$)

[//]: # ()
[//]: # ()
[//]: # (여기서 ELBO 값이 커지면 Variable Inference 부분인 $D\_{KL} \big&#40; q&#40;z\|x&#41; \|\| p&#40;z\|x&#41; \big&#41;$의 값이 작아짐.)

[//]: # ()
[//]: # (Loss값은 ELBO값을 크게 만들면 됨)

[//]: # ()
[//]: # (ELBO 값중 $\mathbb{E}\_{z \sim q&#40;z\|x&#41;} \big\[ \log p&#40;x\|z&#41; \big\]$을 실제 계산값으로 만들어야됨)

[//]: # ()
[//]: # ($$)

[//]: # (    p&#40;x|z&#41; = \frac{1}{\sigma \sqrt{2 \pi}} \exp \bigg&#40;-\frac{&#40;x-\hat{x}_{mu}&#41;^2}{ 2 \sigma^2 } \bigg&#41; \\)

[//]: # (    )
[//]: # (    \begin{split})

[//]: # (    \log p&#40;x|z&#41; &= -log&#40;\sigma \sqrt{2 \pi}&#41; - \frac{&#40;x-\hat{x}_{mu}&#41;^2}{ 2 \sigma^2 } \\)

[//]: # (    \end{split})

[//]: # ($$)

[//]: # ()
[//]: # (몬테카를로 근사에 의해 다음값으로 근사가능)

[//]: # ()
[//]: # ($$  )

[//]: # (    \begin{split})

[//]: # (    \mathbb{E}_{z \sim q&#40;z|x&#41;} \big[ \log p&#40;x|z&#41; \big] &\approx \frac{1}{N} \sum\log p&#40;x|z&#41; dz \\)

[//]: # (            &= - \frac{1}{N} \sum \bigg&#40; \frac{&#40;x-\hat{x}&#41;^2}{ 2 \sigma^2 } + log&#40;\sigma \sqrt{2 \pi}&#41; \bigg&#41;)

[//]: # (    \end{split})

[//]: # ($$)

[//]: # ()
[//]: # ()
[//]: # ($z$ 는 encoder에 의해 $input \; \hat{x}$를 latent space에 embedding한 값     )

[//]: # ()
[//]: # ($x$ 는 decoder에서 나오는 예측 이미지&#40;normalize된 상태&#41; sigma 는 1)

[//]: # ()
[//]: # ($$)

[//]: # (    \mathbb{E}_{z \sim q&#40;z|x&#41;} \big[ \log p&#40;x|z&#41; \big] \approx - \frac{1}{N} \sum \bigg&#40; \frac{&#40;x-\hat{x}&#41;^2}{ 2 } \bigg&#41;)

[//]: # ($$)

[//]: # ()
[//]: # ($D\_{KL} \big&#40; q&#40;z\|x&#41; \|\| p&#40;z&#41; \big&#41;$ 값 계산)

[//]: # ()
[//]: # ($p&#40;z&#41;$ 는 $\sigma = 1, \mu =0$인 gaussian distribution )

[//]: # ()
[//]: # ()
[//]: # ($$)

[//]: # (    \log q&#40;z|x&#41; = - log \sigma_1 - \frac{1}{2} log 2 \pi - \frac{&#40;z - \mu_1&#41;^2}{2 \sigma_1^2} \\)

[//]: # (    \log p&#40;z&#41; = - log \sigma_2 - \frac{1}{2} log 2 \pi - \frac{&#40;z - \mu_2&#41;^2}{2 \sigma_2^2})

[//]: # ($$)

[//]: # ()
[//]: # ($$)

[//]: # (    \begin{split})

[//]: # (    D_{KL} \big&#40; q&#40;z|x&#41; || p&#40;z&#41; \big&#41; &= \int q&#40;z|x&#41; \log \frac{q&#40;z|x&#41;}{p&#40;z&#41;} dz \\)

[//]: # (        &= \int q&#40;z|x&#41; \bigg[ log \frac{\sigma_2}{\sigma_1} - \frac{&#40;z - \mu_1&#41;^2}{2 \sigma_1^2} + \frac{&#40;z - \mu_2&#41;^2}{2 \sigma_2^2} \bigg] dz)

[//]: # (    \end{split})

[//]: # ($$)

[//]: # ()
[//]: # ($$)

[//]: # (\begin{split})

[//]: # (\left&#40; 1 \right&#41;  \; \int q&#40;z|x&#41; \bigg[ log \frac{\sigma_2}{\sigma_1} \bigg] dz &= log \frac{\sigma_2}{\sigma_1} \int q&#40;z|x&#41; dz \\)

[//]: # (            &= log \frac{\sigma_2}{\sigma_1} \cdot 1)

[//]: # (\end{split})

[//]: # ($$)

[//]: # ()
[//]: # ($$)

[//]: # (\begin{split})

[//]: # (\left&#40; 2 \right&#41; \; \int q&#40;z|x&#41; \cdot \frac{&#40;z - \mu_1&#41;^2}{2 \sigma_1^2} dz &= \int \frac{1}{\sigma_1 \sqrt{ 2 \pi}} )

[//]: # (            \exp \left&#40; \frac{-&#40;z - \mu_1&#41;^2}{ 2 \sigma_1^2} \right&#41; \cdot \frac{&#40;z - \mu_1&#41;^2}{2 \sigma_1^2}dz , \quad t = \frac{z - \mu_1}{ \sigma_1 \sqrt{2} } \\)

[//]: # (    &= \frac{1}{\sigma_1 \sqrt{ 2 \pi }} \int t^2 \exp &#40; -t^2 &#41; \sigma_1 \sqrt{2} dt  \;, \quad dz = \sigma_1 \sqrt{2} dt \\)

[//]: # (    &= \frac{1}{\sqrt{\pi}} \int t^2 \exp&#40;-t^2&#41; dt \\ )

[//]: # (    &= \frac{1}{\sqrt{\pi}} \frac{\sqrt{\pi}}{2} = \frac{1}{2})

[//]: # (\end{split})

[//]: # ($$)

[//]: # ()
[//]: # ()
[//]: # ($$  )

[//]: # (    \begin{split})

[//]: # (        \int^{\infty}_{0} x^2 e^{-x^2} dx &= [x \cdot -\frac{1}{2} e^{-x^2}]^{\infty}_{0} + \frac{1}{2} \int^{\infty}_{0} e^{-x^2} dx \\)

[//]: # (            &= 0 + \frac{\sqrt{\pi}}{4} \\)

[//]: # (        )
[//]: # (    \end{split})

[//]: # ($$)

[//]: # ()
[//]: # (가우스 적분)

[//]: # ()
[//]: # ($$)

[//]: # (    \int^{\infty}_{-\infty} e^{-x^2} dx = \sqrt{\pi} \\)

[//]: # (    \int^{\infty}_{0} x^n e^{-x^m} dx = \frac {1}{m} \Gamma &#40;\frac{n+1}{m}&#41;)

[//]: # ($$)

[//]: # ()
[//]: # (감마 함수)

[//]: # ()
[//]: # ($$)

[//]: # (    \Gamma&#40;s&#41; = \int^{\infty}_{0} e^{-t}t^{s} \frac{dt}{t})

[//]: # ($$)

[//]: # ()
[//]: # ($$)

[//]: # (\begin{split})

[//]: # (\left&#40; 3 \right&#41; \; \int q&#40;z|x&#41; \cdot \frac{&#40;z-\mu_2&#41;^2}{2 \sigma^2_{2}} dz &= \int \frac{1}{\sigma_1 \sqrt{ 2 \pi}}  \exp \left&#40; \frac{-&#40;z - \mu_1&#41;^2}{ 2 \sigma_1^2} \right&#41;)

[//]: # (    \cdot \frac{&#40;z-\mu_2&#41;^2}{2 \sigma^2_{2}} dz \\)

[//]: # (        &= \frac{1}{\sigma_2^2 \sigma_1 \sqrt{ 2 \pi}} \int \exp \left&#40; \frac{-&#40;z - \mu_1&#41;^2}{ 2 \sigma_1^2} \right&#41; \cdot [ z^2 - 2 \mu_2 z +  \mu_2^2 ] dz \\)

[//]: # (\end{split})

[//]: # ($$)

[//]: # ()
[//]: # ($$ )

[//]: # (\begin{split})

[//]: # (\left&#40; 3 \text{-} 1 \right&#41; \; \int \exp \left&#40; \frac{-&#40;z - \mu_1&#41;^2}{ 2 \sigma_1^2} \right&#41; \cdot z^2 dz)

[//]: # (    &= \int e^{-t^2} &#40;\sigma_1 \sqrt{2} t + \mu_1&#41;^2 \sigma_1 \sqrt{2} \, dt  \;,\quad t= \frac{z - \mu_{1}}{\sigma_{1} \sqrt{2}} \\)

[//]: # (    &= \sigma_1 \sqrt{2} \int 2\sigma^2_1 t^2 e^{-t^2} + 2\sqrt{2} \sigma_1 \mu_1 t e^{-t^2} + \mu_1^2 e^{-t^2} \, dt \\)

[//]: # (    &= 2 \sqrt{2} \sigma_1^3 \int t^2 e^{-t^2} \, dt + 4 \sigma_{1}^2 \mu_1 \int t e^{-t^2} \, dt + \sigma_1 \mu_1^2 \sqrt{2} \int e^{-t^2} \, dt \\)

[//]: # (    &= 2 \sqrt{2} \sigma_1^3 \cdot \frac{\sqrt{\pi}}{2} + 4 \sigma_{1}^2 \mu_1 \cdot 0 + \sigma_1 \mu_1^2 \sqrt{2} \cdot \sqrt{\pi} \\)

[//]: # (    &= \sigma^3_1 \sqrt{2 \pi} + \sigma_1 \mu_1^2 \sqrt{2 \pi})

[//]: # (\end{split})

[//]: # ($$)

[//]: # ()
[//]: # ($$)

[//]: # (\begin{split})

[//]: # (\left&#40; 3\text{-}2 \right&#41; \; \int \exp \left&#40; \frac{-&#40;z - \mu_1&#41;^2}{ 2 \sigma_1^2} \right&#41; \cdot &#40;2 \mu_2 z&#41; dz )

[//]: # (    &= 2 \mu_2 \int e^{-t^2} &#40;\sigma_1 \sqrt{2} t + \mu_1&#41; \sigma_1 \sqrt{2} \, dt \; , \quad t= \frac{z - \mu_{1}}{\sigma_{1} \sqrt{2}} \\)

[//]: # (    &= 4 \sigma_{1}^{2} \mu_2 \int t e^{-t^2} \, dt + 2 \sqrt{2} \sigma_1 \mu_1 \mu_2 \int e^{-t^2} \, dt \\)

[//]: # (    &= 4 \sigma_{1}^{2} \mu_2 \cdot 0 + 2 \sqrt{2} \sigma_1 \mu_1 \mu_2 \cdot \sqrt{\pi} \\)

[//]: # (    &= 2 \sqrt{2 \pi} \sigma_1 \mu_1 \mu_2)

[//]: # (\end{split})

[//]: # ($$)

[//]: # ()
[//]: # ($$)

[//]: # (\begin{split})

[//]: # (\left &#40;3 \text{-} 3 \right&#41; \; \int \exp \left&#40; \frac{-&#40;z - \mu_1&#41;^2}{ 2 \sigma_1^2} \right&#41; \cdot \mu^2_2 \, dz )

[//]: # (    &= \mu^2_2 \int e^{-t^2} \sigma_1 \sqrt{2} \, dt  \; , \quad t= \frac{z - \mu_{1}}{\sigma_{1} \sqrt{2}} \\)

[//]: # (    &= \mu^2_2 \sigma_1 \sqrt{2} \cdot \sqrt{\pi} =  \mu^2_2 \sigma_1 \sqrt{2 \pi})

[//]: # (\end{split})

[//]: # ($$)

[//]: # ()
[//]: # ($$  )

[//]: # (\begin{split})

[//]: # (    \left&#40; 3 \right&#41; &= \frac{1}{\sigma_1 \sigma_2^2 \sqrt{2 \pi}} \big[ \left&#40; 3\text{-} 1\right&#41; - \left&#40; 3\text{-} 2\right&#41; + \left&#40; 3\text{-} 3\right&#41; \big] \\)

[//]: # (     &= \frac{1}{\sigma_1 \sigma_2^2 \sqrt{2 \pi}} \big[ )

[//]: # (            \sigma^3_1 \sqrt{2 \pi} + \sigma_1 \mu_1^2 \sqrt{2 \pi} -)

[//]: # (            2 \sqrt{2 \pi} \sigma_1 \mu_1 \mu_2 +)

[//]: # (            \mu^2_2 \sigma_1 \sqrt{2 \pi})

[//]: # (        \big] \\)

[//]: # (    &= \frac{\sigma_1^2 + \mu_1^2 - 2\mu_1 \mu_2 + \mu^2_2}{\sigma_2^2} \\)

[//]: # (    &= \frac{\sigma_1^2 + &#40;\mu_1 - \mu_2&#41;^2}{\sigma_2^2})

[//]: # (    )
[//]: # (\end{split})

[//]: # ($$)

[//]: # ()
[//]: # ($$)

[//]: # (\begin{split})

[//]: # (    D_KL&#40;q&#40;z|x&#41; || p&#40;z&#41;&#41; &= &#40;1&#41; - &#40;2&#41; + &#40;3&#41; \\)

[//]: # (        &= \log \frac{\sigma_2}{\sigma_1} - \frac{1}{2} + \frac{\sigma_1^2 + &#40;\mu_1 - \mu_2&#41;^2}{\sigma_2^2})

[//]: # (\end{split})

[//]: # ($$)

[//]: # ()
[//]: # (여기서 $p&#40;z&#41;$ $\sigma\_2=1, \; \mu\_2=0$인 gaussian 분포 함수임)

[//]: # ()
[//]: # ($$)

[//]: # (    D_KL&#40;q&#40;z|x&#41; || p&#40;z&#41;&#41; = - \log \sigma_1 - \frac{1}{2} + \sigma_1^2 + \mu_1^2 )

[//]: # ($$)

[//]: # ()
[//]: # (그래서 실제 Loss는 ELBO를 최대화하는 것이므로 -ELBO를 최소화하는 것과 같다 식은 다음과 같다.)

[//]: # ()
[//]: # ($$)

[//]: # (\begin{split})

[//]: # (    Loss &= -ELBO \\)

[//]: # (        &= - \mathbb{E}_{z \sim q&#40;z|x&#41;} \big[ \log p&#40;x|z&#41; \big] + D_{KL} \big&#40; q&#40;z|x&#41; || p&#40;z&#41; \big&#41; \\ )

[//]: # (        &= \frac{1}{N} \sum \bigg&#40; \frac{&#40;x-\hat{x}&#41;^2}{ 2 } \bigg&#41; + \sigma_1^2 + \mu_1^2 - \log \sigma_1 - \frac{1}{2} \\)

[//]: # (\end{split})

[//]: # ($$)

[//]: # ()
[//]: # (사실상 ELBO는 recontruction error와 $q&#40;z\|x&#41;$를 gaussian 분포인 $p&#40;z&#41;=\mathcal{N}&#40;0, 1^2&#41;$에 의해 regularization되는 항이 합해짐)

