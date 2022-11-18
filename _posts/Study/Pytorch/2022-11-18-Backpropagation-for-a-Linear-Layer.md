---
title: "Linear Layer 역전파(backpropagation)"
tags:
  - Pytorch
categories:
  - Pytorch Study
date: 2022-11-18
toc: true
toc_sticky: true
toc_icon: "sticky-note"
use_math: true
last_modified_at: 2022-11-18T17:41:49
---

### Linear Layer Backpropagation

linear layer는 input $X$ ($N \times D$)를 입력으로 받고 weight matrix $W$ ($D \times M$)이라 하자
이때 layer의 결과물로 output $Y$ ($N \times M$)가 계산되어 나온다.

실제 예시를 들기 위해 아래처럼 $N=2, \; D=2, \; M=3$이라고 가정한다.

$$
    X = \begin{pmatrix} 
    x_{1,1} & x_{1,2} \\
    x_{2,1} & x_{2,2}
    \end{pmatrix}
    \;\;
    W = \begin{pmatrix}
    w_{1,1} & w_{1,2} & w_{1,3} \\
    w_{2,1} & w_{2,2} & w_{2,3}
    \end{pmatrix}
$$

$$
    \begin{split}
    Y &= XW \\
        &= \begin{pmatrix}
            x_{1,1}w_{1,1} + x_{1,2}w_{2,1} & x_{1,1}w_{1,2} + x_{1,2}w_{2,2} & x_{1,1}w_{1,3} + x_{1,2}w_{2,3} \\
            x_{2,1}w_{1,1} + x_{2,2}w_{2,1} & x_{2,1}w_{1,2} + x_{2,2}w_{2,2} & x_{2,1}w_{1,3} + x_{2,2}w_{2,3}
        \end{pmatrix}
    \end{split}
$$

forward 과정이후 Loss가 계산되고 Loss에 대응되는 back gradient $\frac{\partial L}{\partial Y}$가 역전파로 들어오게 된다.

$\frac{\partial L}{\partial Y}$의 크기는 output $Y$와 동일하게 $N \times M$이고 원소는 다음과 같은 식으로 표현가능하다.

$$
    \frac{\partial L}{\partial Y} = \begin{pmatrix}
    \frac{\partial L}{\partial y_{1,1}} & \frac{\partial L}{\partial y_{1,2}} & \frac{\partial L}{\partial y_{1,3}} \\
    \frac{\partial L}{\partial y_{2,1}} & \frac{\partial L}{\partial y_{2,2}} & \frac{\partial L}{\partial y_{2,3}}
    \end{pmatrix}
$$

딥러닝에서 weight를 업데이트하기 위한 $\frac{\partial L}{\partial W}$과 그다음 back propagation의 gradient 값인 
$\frac{\partial L}{\partial X}$는 체인룰(chain-rule)을 사용해서 다음과 같이 계산이 가능하다.

$$
   \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial X}
    
    \quad\quad 

    \frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial W}
$$

먼저 $\frac{\partial L}{\partial X}$에 대해서 표현하면 다음과 같다.

$$
    X = \begin{pmatrix} 
        x_{1,1} & x_{1,2} \\
        x_{2,1} & x_{2,2}
        \end{pmatrix}
   \implies \frac{\partial L} {\partial X} = 
    \begin{pmatrix}
    \frac{\partial L}{\partial x_{1,1}} & \frac{\partial L}{\partial x_{1,2}} \\
    \frac{\partial L}{\partial x_{2,1}} & \frac{\partial L}{\partial x_{2,2}}
    \end{pmatrix}
$$

여기서 원소 $\frac{\partial L}{\partial x\_{1,1}}$에 대한 계산식은 다음과 같다.

$$
    \frac{\partial L}{\partial x_{1,1}} = 

    \sum^{N}_{i=1} \sum^{M}_{j=1} \frac{\partial L}{\partial y_{i,j}} \frac{\partial y_{i,j}}{\partial x_{1,1}}
    = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial x_{1,1}}
$$

위 식에서 $\frac{\partial Y}{\partial x\_{1,1}}$를 계산하면 

$$
    \frac{\partial Y}{\partial x_{1,1}} = \begin{pmatrix}
    w_{1,1} & w_{1,2} & w_{1,3} \\
    0 & 0 & 0
    \end{pmatrix}
$$

$\frac{\partial Y}{\partial x\_{1,1}}$ 값과 위의 $\frac{\partial L}{\partial Y}$를 이용해서 
$\frac{\partial L}{\partial x\_{1,1}}$를 계산하면 다음과 같다.

$$
    \begin{split}

    \frac{\partial L}{\partial x_{1,1}} &= \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial x_{1,1}} \\
    &=
    \begin{pmatrix}
    \frac{\partial L}{\partial y_{1,1}} & \frac{\partial L}{\partial y_{1,2}} & \frac{\partial L}{\partial y_{1,3}} \\
    \frac{\partial L}{\partial y_{2,1}} & \frac{\partial L}{\partial y_{2,2}} & \frac{\partial L}{\partial y_{2,3}}
    \end{pmatrix}

    \begin{pmatrix}
    w_{1,1} & w_{1,2} & w_{1,3} \\
    0 & 0 & 0
    \end{pmatrix} \\
    
    &= \frac{\partial L}{\partial y_{1,1}} w_{1,1} + 
    \frac{\partial L}{\partial y_{1,2}} w_{1,2} + 
    \frac{\partial L}{\partial y_{1,3}} w_{1,3}
    
    \end{split}
$$

마찬가지로 $x\_{1,2}, \; x\_{2,1}, \;  x\_{2,2}$에 대해서도 똑같이 계산하면 다음과 같다.


$$
    \begin{split}

    \frac{\partial L}{\partial x_{1,2}} &= \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial x_{1,2}} \\
    &=
    \begin{pmatrix}
    \frac{\partial L}{\partial y_{1,1}} & \frac{\partial L}{\partial y_{1,2}} & \frac{\partial L}{\partial y_{1,3}} \\
    \frac{\partial L}{\partial y_{2,1}} & \frac{\partial L}{\partial y_{2,2}} & \frac{\partial L}{\partial y_{2,3}}
    \end{pmatrix}

    \begin{pmatrix}
    w_{2,1} & w_{2,2} & w_{2,3} \\
    0 & 0 & 0
    \end{pmatrix} \\
    
    &= \frac{\partial L}{\partial y_{1,1}} w_{2,1} + 
    \frac{\partial L}{\partial y_{1,2}} w_{2,2} + 
    \frac{\partial L}{\partial y_{1,3}} w_{2,3}
    
    \end{split}
$$

$$
    \begin{split}

    \frac{\partial L}{\partial x_{2,1}} &= \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial x_{2,1}} \\
    &=
    \begin{pmatrix}
    \frac{\partial L}{\partial y_{1,1}} & \frac{\partial L}{\partial y_{1,2}} & \frac{\partial L}{\partial y_{1,3}} \\
    \frac{\partial L}{\partial y_{2,1}} & \frac{\partial L}{\partial y_{2,2}} & \frac{\partial L}{\partial y_{2,3}}
    \end{pmatrix}

    \begin{pmatrix}
    0 & 0 & 0 \\
    w_{1,1} & w_{1,2} & w_{1,3}
    \end{pmatrix} \\
    
    &= \frac{\partial L}{\partial y_{2,1}} w_{1,1} + 
    \frac{\partial L}{\partial y_{2,2}} w_{1,2} + 
    \frac{\partial L}{\partial y_{2,3}} w_{1,3}
    
    \end{split}
$$

$$
    \begin{split}

    \frac{\partial L}{\partial x_{2,2}} &= \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial x_{2,2}} \\
    &=
    \begin{pmatrix}
    \frac{\partial L}{\partial y_{1,1}} & \frac{\partial L}{\partial y_{1,2}} & \frac{\partial L}{\partial y_{1,3}} \\
    \frac{\partial L}{\partial y_{2,1}} & \frac{\partial L}{\partial y_{2,2}} & \frac{\partial L}{\partial y_{2,3}}
    \end{pmatrix}

    \begin{pmatrix}
    0 & 0 & 0 \\
    w_{2,1} & w_{2,2} & w_{2,3}
    \end{pmatrix} \\
    
    &= \frac{\partial L}{\partial y_{2,1}} w_{2,1} + 
    \frac{\partial L}{\partial y_{2,2}} w_{2,2} + 
    \frac{\partial L}{\partial y_{2,3}} w_{2,3}
    
    \end{split}
$$

$\frac{\partial L}{\partial X}$를 위에서 계산된 값으로 표현 하면 다음과 같다.

$$
    \frac{\partial L}{\partial X} = \begin{pmatrix}
        \frac{\partial L}{\partial y_{1,1}} w_{1,1} + \frac{\partial L}{\partial y_{1,2}} w_{1,2} + \frac{\partial L}{\partial y_{1,3}} w_{1,3} 
        &
        \frac{\partial L}{\partial y_{1,1}} w_{2,1} + \frac{\partial L}{\partial y_{1,2}} w_{2,2} + \frac{\partial L}{\partial y_{1,3}} w_{2,3}
        \\
        \frac{\partial L}{\partial y_{2,1}} w_{1,1} + \frac{\partial L}{\partial y_{2,2}} w_{1,2} + \frac{\partial L}{\partial y_{2,3}} w_{1,3}
        &
        \frac{\partial L}{\partial y_{2,1}} w_{2,1} + \frac{\partial L}{\partial y_{2,2}} w_{2,2} + \frac{\partial L}{\partial y_{2,3}} w_{2,3}
    
    \end{pmatrix}
$$

계산된 값에서 규칙성을 토대로 2개의 matrix로 분리가 가능하다. 이를 분리하면 다음과 같다.

$$  
    \begin{split}    

    \frac{\partial L}{\partial X} &= 
     \begin{pmatrix}
    \frac{\partial L}{\partial y_{1,1}} & \frac{\partial L}{\partial y_{1,2}} & \frac{\partial L}{\partial y_{1,3}} \\
    \frac{\partial L}{\partial y_{2,1}} & \frac{\partial L}{\partial y_{2,2}} & \frac{\partial L}{\partial y_{2,3}}
    \end{pmatrix}

    \begin{pmatrix}
    w_{1,1} & w_{2,1} \\
    w_{1,2} & w_{2,2} \\
    w_{1,3} & w_{2,3}
    \end{pmatrix} \\

    &= \frac{\partial L}{\partial Y} W^{T}

    \end{split}
$$

---

마찬가지로 weight $W$에 대해서도 반복해서 구할 수 있다.

$$
     W = \begin{pmatrix}
    w_{1,1} & w_{1,2} & w_{1,3} \\
    w_{2,1} & w_{2,2} & w_{2,3}
    \end{pmatrix} \implies

    \frac{\partial L}{\partial W} = \begin{pmatrix}
    \frac{\partial L}{\partial w_{1,1}}  & \frac{\partial L}{\partial w_{1,2}}  & \frac{\partial L}{\partial w_{1,3}} \\   
    \frac{\partial L}{\partial w_{2,1}}  & \frac{\partial L}{\partial w_{2,2}}  & \frac{\partial L}{\partial w_{2,3}}
    \end{pmatrix}
$$

첫번째 원소 $\frac{\partial L}{\partial w\_{1,1}}$에 대한 계산식은 다음과 같이 체인룰로 표현 가능하다.

$$
    \frac{\partial L}{\partial w_{1,1}} = 

    \sum^{N}_{i=1} \sum^{M}_{j=1} \frac{\partial L}{\partial y_{i,j}} \frac{\partial y_{i,j}}{\partial w_{1,1}}
    = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial w_{1,1}}
$$

여기에 $Y$를 $w\_{1,1}$에 대해 부분 적분한 식을 대입해서 정리하면 다음과 같다.

$$  
    \begin{split}
    \frac{\partial L}{\partial w_{1,1}} 
    &= \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial w_{1,1}} \\
    
    &=
     \begin{pmatrix}
    \frac{\partial L}{\partial y_{1,1}} & \frac{\partial L}{\partial y_{1,2}} & \frac{\partial L}{\partial y_{1,3}} \\
    \frac{\partial L}{\partial y_{2,1}} & \frac{\partial L}{\partial y_{2,2}} & \frac{\partial L}{\partial y_{2,3}}
    \end{pmatrix}

    \begin{pmatrix}
    x_{1,1} & 0 & 0 \\
    x_{2,1} & 0 & 0
    \end{pmatrix}  \\

    &= \frac{\partial L}{\partial y_{1,1}} x_{1,1} + \frac{\partial L}{\partial y_{2,1}} x_{2,1}

    \end{split}
$$

이를 다른 원소에 대해서도 반복계산하여 $\frac{\partial L}{\partial W}$으로 표현하면 다음과 같다.

$$
    \frac{\partial L}{\partial W} = 
    \begin{pmatrix}
    \frac{\partial L}{\partial y_{1,1}} x_{1,1} + \frac{\partial L}{\partial y_{2,1}} x_{2,1}
    &
    \frac{\partial L}{\partial y_{1,2}} x_{1,1} + \frac{\partial L}{\partial y_{2,2}} x_{2,1}
    &
    \frac{\partial L}{\partial y_{1,3}} x_{1,1} + \frac{\partial L}{\partial y_{2,3}} x_{2,1}
    \\
    \frac{\partial L}{\partial y_{1,1}} x_{1,2} + \frac{\partial L}{\partial y_{2,1}} x_{2,2}
    &
    \frac{\partial L}{\partial y_{1,2}} x_{1,2} + \frac{\partial L}{\partial y_{2,2}} x_{2,2}
    &
    \frac{\partial L}{\partial y_{1,3}} x_{1,2} + \frac{\partial L}{\partial y_{2,3}} x_{2,2}

    \end{pmatrix}
$$

이를 각각의 matrix로 분리해서 표현하면 다음과 같이 표현된다.

$$  
    \begin{split}
    \frac{\partial L}{\partial W} &= \begin{pmatrix}
    x_{1,1} & x_{2,1} \\
    x_{1,2} & x_{2,2}
    \end{pmatrix}
    
    \begin{pmatrix}
    \frac{\partial L}{\partial y_{1,1}} & \frac{\partial L}{\partial y_{1,2}} & \frac{\partial L}{\partial y_{1,3}} \\
    \frac{\partial L}{\partial y_{2,1}} & \frac{\partial L}{\partial y_{2,2}} & \frac{\partial L}{\partial y_{2,3}}
    \end{pmatrix}   \\

    &= X^{T} \frac{\partial L}{\partial Y}
    \end{split}
$$


따라서 정리하면 Weight $W$와 input $X$에 대한 gradient는 다음과 같이 간단한 공식으로 표현이 가능하다.

$$
    \frac{\partial L}{\partial W} = X^{T} \frac{\partial L}{\partial Y} 
    \quad\quad
    \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^{T}
$$


### Reference

[[참조] cs231n](http://cs231n.stanford.edu/handouts/linear-backprop.pdf)