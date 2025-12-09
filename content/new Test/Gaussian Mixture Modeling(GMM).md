---
imageNameKey: 2024-09-12-09-33
tags:
  - Bayes_Rule
  - Novelty_Detection
  - Probability_and_Statistics
---
### GMM 설명

[[베이즈 정리#Maximum LogLikelihood Estimation, 최대우도법]] 에서 주어진 데이터 $X=\{x_{1}, x_{2}, x_{3}, \cdots , x_{n}\}$  에서 최대 Likelihood를 가지는 정규 분포는 다음과 같은 평균과 분산에서 최대가 된다.
$$
\begin{align}
\hat{\mu} &= \frac{1}{m}\sum\limits^{m}_{i=1}x_{i} \\
\hat{\sigma}^{2} &= \frac{1}{m}\sum\limits^{m}_{i=1}(x_{i}-\hat{\mu})^{2}
\end{align}
$$
#### 예시
아래와 같이 <span style="color:rgb(255, 0, 0)">Labeling된 </span>데이터가 주어져 있을 때 각 Label에 대한 데이터의 평균값($\mu$)과 표준편차($\sigma$)를 계산하여 Likeliehood를 최대로 하는 두 분포를 추정할 수 있다.
![[2024-09-09-19-32_GMM 모델로 스크래치 찾기.png]]

하지만 보통의 경우 데이터에 대한 Label이 주어지지 않기 때문에 Label을 하기 위해선 정규 분포가 필요한데(Likelihood를 비교하여 라벨링 하기 때문) 이를 얻기 위해 데이터의 $\mu, \sigma$ 가 필요하고 이 데이터의 $\mu, \sigma$를 얻기 위해선 Labeling된 데이터가 필요하다.

그래서 랜덤하게 분포를 설정하고 시작해야 된다. 다음과 같이 각 클래스에 대한 평균과 표준편차를 랜덤하게 설정 한다. (파란색이 Class 1, 주황색이 Class 2)
$$
\begin{align}
\mu_{1}=3,\; \sigma_{1}=2.9155 \\
\mu_{2}=10, \; \sigma_{2}=3.9623
\end{align}
$$

![[2024-09-09-19-32_GMM 모델로 스크래치 찾기-1.png]]

각 데이터는 주어진 랜덤한 분포에서 likelihood 비교를 통해 라벨링 되어진다.
![[2024-09-09-19-32_GMM 모델로 스크래치 찾기-2.png | 605]]

![[2024-09-09-19-32_GMM 모델로 스크래치 찾기-3.png]]

위에서 라벨링된 데이터들의 모수($\mu, \sigma$)를 다시 계산할 수 있다. 이 값을 통해 각 그룹에 대한 분포를 다시 그린다.
$$
\begin{align}
	\mu_{1}=4, \; \sigma_{1}=2.1602 \\
	\mu_{2}=21.33, \; \sigma_{2}=7.0048
\end{align}
$$
![[2024-09-09-19-32_GMM 모델로 스크래치 찾기-4.png | 600]]

이렇게 반복적으로 모수를 계산하여 분포를 그리고 라벨링을 반복적으로 하다보면 모수들이 어떤 값에 수렴하게 된다. 이를 그림으로 도식화 하면 다음과 같다.
![[2024-09-09-19-32_GMM 모델로 스크래치 찾기-5.png | 600]]

위의 과정을 실제 데이터에서 수렴하는 영상
![[2024-09-09-19-32_GMM 모델로 스크래치 찾기-1.mp4]]


### Expectation Maximization(EM) 알고리즘
 - 목적 
	 라벨이 없는 데이터가 주어진 상태에서 각 라벨에 대한 Likelihood를 최대로 하는 확률 분포를 구하는 것

라벨을 얻기 위해 분포가 필요하고 분포를 얻기 위해 라벨이 필요한 순환 구조이기 때문에 문제가 어렵다.
- 문제 해결을 위해 데이터셋이 정규 분포라고 가정, 랜덤한 $\mu, \sigma$ 확률 분포 설정 
- 설정된 분포를 통해 라벨링을 진행하고 이 데이터를 통해 다시 확률 분포를 얻는 clustering 과정을 수행

EM 알고리즘은 E-step과 M-step으로 나뉘는데 E-step에서는 변수의 정보, 변수들의 라벨링을 진행하고, M-step에서 업데이트된 라벨을 통해 변수들의 분포를 갱신하는 과정이다.

(E-step) 에서 각 $i, j$ 에 대해, (*여기서 $i$는 데이터의 번호, $j$는 클래스의 번호를 의미)
$$
w^{(i)}_{j} := P(z^{(i)}=j | x^{(i)}; \phi, \mu, \Sigma) 
$$
- $w^{(i)}_{j}$는 $i$번째 데이터가 $j$ 번째 class에 속할 확률을 의미, $z^{(i)}$는 $i$번째 데이터의 라벨을 의미
- 위 식을 풀어 설명하면 $i$번째 데이터 $x^{(i)}$가 다음 파라미터($\phi, \mu, \Sigma$)를 가지는 확률 분포에서 라벨이 $j$일 사후 확률(Posterior)를 $w^{(i)}_{j}$에 업데이트 한다.

(M-step) 파라미터 업데이트
$$
\phi_{j} := \frac{1}{m} \sum\limits^{m}_{i=1}w^{(i)}_{j}
$$
- 여기서 $\phi_{j}$는 $j$ class의 비율을 의미, 즉 전체 데이터에서 $j$ class가 될 확률을 의미함
$$
\begin{align}
\mu_{j} &:= \frac{\sum\limits^{m}_{i=1} w^{(i)}_{j}x^{(i)}} {\sum\limits^{m}_{i=1}w^{(i)}_{j}} \\
\Sigma_{j} &:= \frac{\sum\limits^{m}_{i=1}w^{(i)}_{j}(x^{(i)} - \mu_{j})(x^{(i)} - \mu_{j})^T}{\sum\limits^{m}_{i=1}w^{(i)}_{j}}
\end{align}
$$
#### E-step
식 $w^{(i)}_{j} := P(z^{(i)}=j | x^{(i)}; \phi, \mu, \Sigma)$ 에서 $\phi, \mu, \Sigma$는 이미 주어진 고정 파라미터 이다. 
위 식의 의미는 $x^{(i)}$ 데이터와 파라미터가 주어졌을때 $j$ 라벨에 속할 확률, 즉 사후 확률을 $w^{(i)}_{j}$에 업데이트 한다. 

- 예를 들어 총 3개의 class가 있다고 하고 각각 0, 1, 2 Class에 속할 확률이 0.8, 0.15, 0.05이고 여기서 각 그룹에 속할 확률을 모두 더하면 1이 되어야 된다.
	$$
	\begin{align}
		w^{(i)}_{0} &= P(z^{(i)}=0 | x^{(i)})=0.8 \\
		w^{(i)}_{1} &= P(z^{(i)}=1 | x^{(i)})=0.15 \\
		w^{(i)}_{2} &= P(z^{(i)}=2 | x^{(i)})=0.05 \\
	\end{align}
	$$
사후 확률(Posterior)를 계산하기 위해 베이즈 정리에 따라 다음과 같이 식을 쓸 수 있다.(고정 파라미터 $\phi, \mu, \Sigma$ 는 식에서 제외)
$$
\begin{align}
P(z^{(i)} = j | x^{(i)}) &= \frac{P(x^{(i)} | z^{(i)}= j) P(z^{(i)}=j)}{P(x^{(i)})} \\
&= \frac{P(x^{(i)} | z^{(i)}= j) P(z^{(i)}=j)}{\sum\limits^{n}_{k=1}P(x^{(i)}|z^{(i)} =k) P(z^{(i)}=k)}

\end{align}
$$
$j$ Class분포에서 $x^{(i)}$의 확률(likelihood)와 $j$ Class의 확률인 사전 지식(Prior)를 곱하여 계산할 수 있다.
#### M-step
M-step은 E-step에서 계산된 사후 확률 $w^{(i)}_{j}$값들을 사용해 모수를 추정하는 과정이다.
$\phi$ 는 각 Class에 소속될 평균 확률이다. 아래 그림과 같이 각 데이터 별로 각 그룹에 속할 확률에 대한 평균, Class에 비율을 의미한다.
![[2024-09-12-09-33_Gaussian Mixture Modeling(GMM).png]]

