---
title: "Optimizer 정리"
tags:
  - Pytorch
  - Deep Learning
  - Optimizer
categories:
  - Pytorch Study
date: 2022-11-23
toc: true
toc_sticky: true
toc_icon: "sticky-note"
use_math: true
last_modified_at: 2022-11-23T18:14:36
---


## Optimizer 정리

### Gradient Descent

$$
    \theta_{t+1} = \theta_{t} - \eta \nabla_{\theta} J(\theta_{t})
$$

### Stochastic Gradient Descent

GD에서 전체 batch에서 진행하던걸 mini batch로 자잘한 step으로 움직임

$$
    \theta_{t+1} = \theta_{t} - \eta \nabla_{\theta} J(\theta_{t})
$$


### Momentum

이전 스텝에서의 움직임에대한 관성을 현재 step에 추가함

$$
    v_{t} = \gamma v_{t-1} + \eta \nabla_{\theta} J(\theta_{t}) \\
    \theta_{t+1} = \theta_{t} - v_{t}
$$

### Adagrad

adaptive gradient, gradient가 움직였던 거리를 누적해서 많이 움직인 unit에 대해 제한을 주어 적게 움직이도록 함

$$
    G_{t} = G_{t-1} + (\nabla_{\theta} J(\theta_{t}))^{2} \\

    \theta_{t+1} = \theta_{t} - \frac{\eta}{\sqrt{G_{t} + \epsilon }} \nabla_{\theta} J(\theta_{t})
$$

### RMSProp(Root Mean Square)

adagrad에서 step이 무한히 반복되면 거의 RMS 부분이 무한대로 가서 학습이 진행되지 않음

그래서 이를 지수 평균으로 변경함

$$
    G_{t} = \gamma G_{t-1} + (1 - \gamma) (\nabla_{\theta} J(\theta_{t}))^{2} \\

    \theta_{t+1} = \theta_{t} - \frac{\eta}{ \sqrt{G_{t} + \epsilon } } \nabla_{\theta} J(\theta_{t})
$$

### Adadelta

adagrad에서 학습이 진행되지않은 부분을 지수 평균으로 대체하고
learning rate도 step에 따라 조절되도록 바꿈

$$
    G_{t} = \gamma G_{t-1} + (1 - \gamma) \nabla_{\theta} J(\theta_{t}) \\
    s_{t} = \gamma s_{t-1} + (1 - \gamma) \Delta_{\theta_{t}}^{2} \\
    \Delta_{\theta_{t}} =  \frac{ \sqrt{s_{t} + \epsilon } }{ \sqrt{G_{t}} + \epsilon } \nabla_{\theta} J(\theta_{t}) \\
    \theta_{t+1} = \theta_{t} - \Delta_{\theta_{t}}
$$

### NAG 

Momentum 보다 더 공격적으로 step을 움직임. 

$$
    v_{t} = \gamma v_{t-1} + \eta \nabla_{\theta} J(\theta_{t} - \gamma v_{t-1}) \\ 

    \theta_{t+1} = \theta_{t} - v_{t}
$$

### Adam

RMSProp과 Momentum을 혼합한 optimizer

$$
    m_{t} = \beta_{1} m_{t-1} + (1-\beta_{1}) \nabla_{\theta} J(\theta_{t}) \\
    v_{t} = \beta_{2} v_{t-1} + (1-\beta_{2})(\nabla_{\theta} J(\theta_{t}))^{2} \\

    bias \; correction \; : \; \hat{m}_{t} = \frac{m_{t}}{1 - \beta_{1}^{t}} \\
    bias \; correction \; : \; \hat{v}_{t} = \frac{v_{t}}{1 - \beta_{2}^{t}} \\

    \theta_{t+1} = \theta_{t} - \frac{\eta}{\sqrt{\hat{v}_{t} + \epsilon }} \hat{m}_{t}
$$

bias correction인 $\hat{v}\_{t}$와 $\hat{m}\_{t}$은 $m\_{t}, v\_{t}$가 특정 방향으로 편향되지 않도록 방향을 올바르게 바로 잡아주기 위한
공식이다. 특정 gradient에 심하게 편향되어 업데이트가 이루어지지 않고 평균으로(고르게) weight를 주어서 모든 gradient가 공평하게 영향력을 행사했으면 좋겠다는 그런 뜻(?)이다.

실제로 알고리즘 초반인 $v\_{0}$을 0으로 초기화 하기 때문에 bias correction을 해주지 않으면 0으로 편향되서 제대로 학습되지 않을 것이다.(실제로 해보지는 않음)

Adam 논문에서 해당 내용을 살펴보자 (식의 간편함을 위해 $\nabla\_{\theta} J(\theta\_{t})=g\_{t}$으로 표현한다)

$v\_{t}$는 점화식으로 다음과 같이 표현 가능하다.

$$  
    \begin{split}
    v_{t} = \beta_{2} v_{t-1} &+ (1 - \beta_{2}) \cdot g_{t}^{2} \\ 
    v_{t-1} = \beta_{2} v_{t-2} &+ (1 - \beta_{2}) \cdot g_{t-1}^{2} \\ 
    &... \\
    v_{2} = \beta_{2} v_{1} &+ (1 - \beta_{2}) \cdot g_{2}^{2} \\
    v_{1} = \beta_{2} v_{0} &+ (1 - \beta_{2}) \cdot g_{1}^{2} \\ 
    \end{split} \\
$$

$v\_{0} = 0$이므로 대입하면 다음과 같이 정리된다.


$$
    \begin{split}
    v_{t} &=  (1-\beta_{2}) \cdot \beta_{2}^{t-1} g_{1}^{2}
           + (1-\beta_{2}) \cdot \beta_{2}^{t-2} g_{2}^{2} +  \cdots \\
           &\quad+ (1-\beta_{2}) \cdot \beta_{2}^{2} g_{t-2}^{2} 
           + (1-\beta_{2}) \cdot \beta_{2} g_{t-1}^{2} 
           + (1-\beta_{2}) \cdot g_{t}^{2}
    \end{split}
    \\
    v_{t} = (1 - \beta_{2}) \sum^{t}_{i=1}{\beta_{2}^{t-i} \cdot g_{i}^{2}} \\
$$

$v\_{t}$식의 양변에 expectation을 해주면 다음과 같다.

$$
    \begin{split}
    \mathbb{E}[v_{t}] &= \mathbb{E} \bigg[ (1-\beta_{2}) \sum^{t}_{i=1}{ \beta^{t-i}_{2} \cdot g_{i}^{2} }  \bigg] \\
                    &= \mathbb{E}[g_{t}^{2}] \cdot (1 - \beta_{2}) \sum^{t}_{i=1} \beta_{2}^{t-i} + \zeta \\
                    &= \mathbb{E}[g_{t}^{2}] \cdot (1 - \beta_{2}^{t}) + \zeta
    \end{split}
$$

여기서 $\zeta$값은 매우 작게 유지 되거나 0이다. $v\_{t}$의 평균값을 거의 gradient의 평균값에 근접하도록 하려면 $(1-\beta\_{2}^{t})$을 나눠주면 되는데 
이 값은 위에서 bias correction항에서 본적이 있는 값이다.

솔직히 여기서도 아직 무슨 소리(?)인지 이해가 안되서 google에 검색해보다가 다음과 같은 글을 찾았다.

[[Why is it important to include a bias correction term for the Adam optimizer for Deep Learning?]](https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for){:target="_blank"}

이 글에 의하면 bias correction항 때문에 거의 모든 gradient가 동일한 영향력을 보여준다는 말을 한다.

실제 예시를 들어보면 ($v\_{0} = 0$) 

$$
    \begin{split}
    \hat{v_{1}} &= \frac{0.001 \cdot g_1^{2}}{0.001 \quad\;\;} \\ 
    \hat{v_{2}} &= \frac{0.001 \cdot g_{2}^{2} + 0.001 * 0.999^{1} \cdot g_{1}^{2}}{0.001 \quad\;\; + 0.001 * 0.999^{1} \quad\;\;} \\
    \hat{v_{3}} &= \frac{0.001 \cdot g_{3}^{2} + 0.001 * 0.999^{1} \cdot g_{2}^{2} + 0.001 * 0.999^{2} \cdot g_{1}^{2}}
                    {0.001 \quad\;\; + 0.001 * 0.999^{1} \quad\;\; + 0.001 * 0.999^{2} \quad\;\;} \\
    \hat{v_{4}} &= \frac{0.001 \cdot g_{4}^{2} + 0.001 * 0.999^{1} \cdot g_{3}^{2} + 0.001 * 0.999^{2} \cdot g_{2}^{2} + 0.001 * 0.999^{3} \cdot g_{1}^{2}}
                {0.001 \quad\;\; + 0.001 * 0.999^{1} \quad\;\; + 0.001 * 0.999^{2} \quad\;\; + 0.001 * 0.999^{3} \quad\;\;}
    \end{split}
$$

위의 식에서 보면 gradient의 각항들이 t가 증가함에도 불구하고 모두 일정한 비율을 가지고 있는 것을 알 수 있다. 

만약 $1 - \beta\_{2}^{t}$으로 나눠주지 않으면 다음과 같이 초반 step에서 $v\_{0}=0$항에 거의 99.9%의 영향을 받아 원하지 않은 방향으로 학습이 진행된다.

$$
    \begin{split}
    v_{1} &= 0.999^{1} \cdot 0 + 0.001 \cdot g_{1}^{2} \\
    v_{2} &= 0.999^{2} \cdot 0 + 0.001 * 0.999^{1} \cdot g_{1}^{2} + 0.001 \cdot g_{2}^{2} \\
    &\qquad \qquad \qquad \qquad \cdots \\
    v_{1000} &= 0.3676 \cdot 0 + 0.001 * 0.999^{999} \cdot g_{1}^{2} + \cdots + 0.001 \cdot g_{1000}^{2}
    \end{split}
$$

$t=1000$ 시점이 되어도 $0$에 대한 영향력이 $36.76\%$나 되어 $0$에 대한 편향이 쉽게 없어지지 않는다.

### NAdam

### AdamW

### AdamP
