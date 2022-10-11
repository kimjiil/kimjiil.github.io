---
title: "수학 공식 테스트(타이틀 변경시 댓글 유지)"
categories:
  - etc
tags:
  - bug
  - math
date: 2022-08-19-13:27:00
toc: true
toc_sticky: true
toc_icon: "sticky-note"
use_math: true
---

# Test

$$
\begin{matrix} a & b \\ c & d \end{matrix}
\begin{pmatrix} a & b \\ c & d \end{pmatrix}
\begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$

$$
\left(\sum_{k=1}^n a_k b_k \right)^2 \leq 
\left( \sum_{k=1}^n a_k^2 \right) 
\left( \sum_{k=1}^n b_k^2 \right)
$$

$$
gcd(a,b) =
    \begin{cases}
      a & \text{if b $\neq$ 0}\\
      gcd(b, a mod b) & \text{if b $\neq$ 0}
    \end{cases}    
$$

$$
    \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)
$$