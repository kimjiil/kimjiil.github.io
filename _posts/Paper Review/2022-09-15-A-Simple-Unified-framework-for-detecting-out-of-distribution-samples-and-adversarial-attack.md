---
title: "[논문 리뷰]A Simple Unified Framework for Detecting Out-Of-Distribution Samples and Adversarial Attack[작성중]"
tags:
  - Abnormal Detection
  - Out-Of-Distribution
  - Adversarial Attack
  - Gaussian Discriminant Analysis
categories:
  - Computer Vision Paper
date: 2022-09-15
toc: true
toc_sticky: true
toc_icon: "sticky-note"
use_math: true
last_modified_at: 2022-09-21T18:19:23
---

### Uncertainty의 유형
1. Out of Distribution Test Data 
   - 학습할 때 한번도 보지 못한 유형의 데이터가 Test에서 사용되는 경우. 예시로 개를 학습한 모델에 대해서 고양이 사진을 
주고 개의 종류를 판별하라고 하는 경우.

<p align="center">
<img src="/assets/images/2022-09-15-A-Simple-Unified-framework-for-detecting-out-of-distribution-samples-and-adversarial-attack/Uncertainty_Type1.png" 
height="70%" width="70%"> </p>

2. Aleatoric
    - 학습 데이터 자체에 노이즈가 많아서 데이터 자체에 문제가 있는 경우. 학습할 때 3가지 유형인 개,소,고양이에 대해 학습한다고 했을때
   고양이 이미지가 심하게 훼손된 데이터셋으로 학습하는 경우 이후 들어오는 고양이 이미지에 대해 제대로 분류하지 못하는 불확실성 발생.
<p align="center">
<img src="/assets/images/2022-09-15-A-Simple-Unified-framework-for-detecting-out-of-distribution-samples-and-adversarial-attack/Uncertainty_Type2.png"
height="70%" width="70%"></p>

3. Epistemic Uncertainty
    - 주어진 데이터셋을 가장 잘 설명할 수 있는 모델을 선택할 때 생기는 불확실성. 아래 그림 처럼 어떤 모델이 해당 데이터셋에 가장 적합한지
   알 수 없어서 생기는 불확실성이다. 3번째 그림이 가장 훈련 데이터에 대해 에러가 적지만 테스트 데이터에 대한 성능은 가장 낮다.
<p align="center">
<img src="/assets/images/2022-09-15-A-Simple-Unified-framework-for-detecting-out-of-distribution-samples-and-adversarial-attack/Uncertainty_Type3.png"
height="80%" width="80%"></p>

[출저] [Gaussian37 블로그](https://gaussian37.github.io/dl-concept-uncertainty_in_deep_learning/)

### 논문 해석

#### [1]Intro
- Deep neural networks`DNNs`는 speech recognition[[1][1_link]], object detection, image classification과 같은 많은 분야의 classification task에서
   높은 정확도를 달성했다.
  - 하지만 예측 불확실성(predictive uncertainty)을 측정하는 것은 여전히 도전 과제로 남아 있다.[[20][20_link], [21][21_link]]
  - 잘 보정된 예측 불확실성을 얻는 것은 실제 시스템[[2]][2_link](self-driving, secure authentication system[[6][6_link], [30][30_link]])에 DNNs을 배포할 때뿐만 아니라 많은 기계 학습 application(active learning[[8][8_link]], novelty detection[[18][18_link]])에서 유용하게 사용되므로 필수적이다.
- DNNs의 예측 불확실성은 적대적이거나 통계적으로 훈련 데이터의 분포로 부터 멀리 떨어진 abnormal sample을 탐지하는 문제와 매우 깊게 연관되어 있다.
- Out-of-distribution`OOD` sample을 탐지하기 위해 최근 연구들은 사후 분포(posterior distribution)[[13][13_link], [21][21_link]]로 부터 얻은 confidence를 이용하고 있다.
  - 예를 들어 Hendrycks & Gimpel[[13][13_link]]는 분류기로 부터 얻은 사후 분포의 최대값을 기준 방법으로 제안하였고, 이것은 DDNs[[21][21_link]]의 input, output으로 처리하여 성능을 향상 시켰다.
  - adversarial sample들을 탐지하기 위해, DNNs의 feature space에서 그 샘플들을 특성화 하기 위한 density estimator를 기반으로 한 confidence score를 제안하였다[[7][7_link]].
  - 가장 최근에는 Ma et al.[[22][22_link]]는 Local Intrinsic Dimensionality`LID`를 제안 했고 LID를 사용하여 효과적으로 test sample들의 특성을 축정할 수 있음을 실험적으로 보여주었다.
  - 하지만, 대부분의 이런 류의 이전 연구들은 전형적으로 OOD와 adversarial sample 둘다에 대해 평가 하지 않는다. 우리가 아는한 2개의 task에 대해 동시에 잘 동작하는 통합 detector는 없다.

- `Contribution`
  - 이 논문에서 간단하지만 매우 효과적인 방법을 제안함. OOD와 adversarial sample을 포함한 abnormal test sample들을 탐지하는데 재학습을 필요로 하지도 않고 어떤 사전 학습된 softmax neural classifier에도 상관 없이 적용 가능함.
  - 아이디어를 요약하자면 거리 기반의 generative classifier의 컨셉을 이용한 DNNs의 feature space에서 test sample의 probability density를 측정하는 것이다.
  - 특히 Gaussian discriminant analysis에서 사후 분포는 softmax classifier와 거의 동일하게 보이기 때문에 class-conditional 가우시안 분포는 사전 훈련된 feature에 적합하다고 가정한다.
  - 이런 가정하에 class-conditinal 분포와의 거리를 나타내는 Mahalanobis score를 사용하여 confidence score를 정의하고, 모델의 파라미터는 실힘적 경험을 바탕으로 훈련 데이터의 covariance와 class mean을 선택했다.
  - 통상의 믿음과는 반대로, 해당 generative classfier를 사용하더라도 softmax classification의 정확도와 비슷한 성능을 유지하였다.

- 제안된 방법의 효율성 및 이점을 증명하기 위해 deep convolutional neural networks의 한 종류인 DensNet[[14][14_link]]과 ResNet[[12][12_link]]을 사용하여 CIFAR-10[[15][15_link]], SVHN[[28][28_link]]
, ImageNet[[5][5_link]], LSUN[[32][32_link]]와 같은 다양한 데이터셋에서 이미지 분류 문제에 대해 학습시킨 모델을 사용하였다.
- 첫번째로 OOD sample들을 탐지하는 문제에서 제안 방법은 State-of-the-art(Sota) 방법인 ODIN[[21][21_link]]의 성능을 뛰어 넘었다.
  - 특히 ODIN보다 OOD sample을 탐지하는 비율인 true negative rate(TNR)의 성능이 45.6%에서 90.9%로 상승했다.(Renset을 CIFAR-10에 대해 학습 하고 LSUN을 OOD Sample로 사용했을때)
  - FGSM[[10][10_link]], BIM[[16][16_link]], DeepFool[[26][26_link]], CW[[3][3_link]]과 같이 4가지 공격 방법의 생성된 adversarial sample을 탐지에서도,
    제안 방법은 Sota 측정 방법인 LID[[22][22_link]]의 성능을 뛰어 넘었다. CW로 부터 생성된 데이터의 TNR 성능이 82.9%에서 95.8%로 향상되었다(CIFAR-10에 대해 학습한 ResNet 모델에서 테스트).

- 제안방법은 노이즈가 포함된 훈련 데이터셋, Random Labels, 매우 적은 데이터 샘플과 같은 극한 상황에서도 hyperparameter를 선택함에 있어서 환경의 영향을 거의 받지 않아 Robust하다.
- 특히 Liang et al.[[21][21_link]]은 OOD sample의 validation set을 사용하여 ODIN의 hyperparameter를 조정했다. 하지만 때때로 OOD의 사전 지식을 얻지 못하는 경우에는 불가능한 방법이다.
  - 제안 방법은 성능을 유지하면서 오직 training sample의 분포만을 사용하여 hyperparameter를 조정한다. 또한 FGSM과 같은 단순한 공격 방법에서 조정된 모델을 더 복잡한 공격 방법인 BIM, DeepFool, CW에서 생성된 adversarial sample을 탐지하는데 사용될 수 있음을 보여준다.

- 마지막으로 제안방법을 사전 학습된 classifier에 점진적으로 new class를 추가하는 class-incremental learning[[29][29_link]]에도 적용 해보았다.
  - new class은 training distribution의 범위 밖에 있기 때문에(OOD sample 과 같음), new class sample을 재학습 없이 우리의 제안 방법을 사용하여 분류할 수 있는 것이 당연히 가능하다.
  - 이점을 이용하여, 모든 클래스의 공유 공분산(tied covariance)을 업데이트하고 new class의 mean값을 단순히 계산함으로써 new class를 수용하는 간단한 방법을 소개한다.
  - 재학습한 softmax classifier와 유클리드 거리 기반의 classifier와 같은 baseline 방법들의 성능을 뛰어 넘음.
  - 이러한 실험적 증거들은 제안 방법이 few-shot learning[[31][31_link]], ensemble learing[[19][19_link]], active learning[[8][8_link]]과 같은 머신 러닝 문제와 관련된 곳에서 적용될 가능성을 가지고 있다고 생각한다.

#### [2] Mahalanobis distance-based score from generative classifier

- softmax classifier을 가진 Deep neural networks(DNNs)가 주어졌을때, Out-of-distribution(OOD)과 adversarial sample과 같은 abnormal sample을 탐지하기위한 단순하지만 효과적인 방법을 제안한다.
- 먼저, Gaussian Discriminant Analysis(GDA)으로 유도된 generaive classifier를 기반으로한 confidence score에 대해 설명하고,
이 모델의 성능을 높이는 추가적인 기술에 대해서도 소개한다.

##### [2.1] Why Mahalanobis distance-based score?

- softmax classifier로부터 generative classifier 유도됨을 증명한다.  Input $x$와 output $y$ 다음과 같다.

$$
x\in\chi, \quad y\in Y=\{1, \cdots, \mathit{C}\}
$$

- 사전 학습된 softmax neural classifier의 posterior도 다음과 같다고 가정한다.

$$
P(y=c|x)=\frac{exp( {\pmb{\mathbb{w}_c^{\top}}} f(x) + b_c )}{ \sum_{c'}{exp( \pmb{\mathbb{w}}_{c'}^{\top} f(x) + b_{c'} )} }
$$

- 여기서 $\pmb{\mathbb{w}_c}$와 $b_c$는 class c에 대한 softmax classifier의 weight와 bias이고 $f(\cdotp)$은 DNNs의 끝에서 2번째 layer의 output 이다.
그런 다음 사전학습한 softmax neural classifier에 대해 어떠한 변형없이, class-conditional distribution이 다변량 가우시안 분포를 따른다고 가정하고 generaitve classifier를 얻는다.
- 공유 공분산 $\Sigma$를 가지는 $\mathit{C}$ class-conditional Gaussian distribution$\mathit{(Likelihood)}$을 다음과 같이 정의한다.

$$
P(f(x)|y=c)= \mathcal{N}(f(x)\,|\, \mu_c, \; \Sigma)
$$

- 여기서 $$\mu_c$$는 $$c\in \{1,\cdots,\mathit{C} \}$$의 다변량 가우시안 분포의 평균 벡터이다. 
- softmax classifier와 GDA사이의 이론적인 연결점을 기반으로 접근한다. 공유 공분산을 가지는 GDA기반의 generative classifier에 의해 정의된 사후 확률 분포(posterior distribution) 추정은 softmax classifier와 동일하다.

---

###### [Supplementary A] 왜 Softmax neural classifier와 class-conditional Gaussian Distribution과 동일한지?

<details>
<summary> <b>[Supplementary A 내용 펼치기]</b> </summary>
<div markdown="1">

- 보통의 softmax classifier의 posterior는 다음과 같음.

$$
    P(y=c|x)= \frac{exp( \pmb{\mathbb{w}_c^{\top}} x + b_c )}
    {\sum_{c'}{exp( \pmb{\mathbb{w}_{c'}^{\top}} x +b_{c'})} }
$$

- x가 주어졌을때 클래스에 대한 다변량 정규 분포(likelihood)는 다음과 같이 추정 가능.

$$
    P(x|y=c)=\mathcal{N}(x| \, \mu_c, \, \Sigma_c) = \frac{1}{ {2\pi^{\frac{d}{2}}|\Sigma|^{\frac{1}{2} }} }
    exp(-\frac{1}{2} (x-\mu_c)^\top \Sigma^{-1} (x-\mu_c))
$$

- 우리는 x일때 c일 확률인 posterior를 구해야하는데 이는 bayes rule 공식 $ posterior=\frac{likelihood \times prior}{evidence} $
으로 계산하면 되는데 이때 $evidence$는 전체 확률의 법칙으로 계산하고 $prior$는 전체에서 c의 비율을 계산하면 된다.

$$
    Posterior \quad P(y=c|x)= \frac{P(x|y=c) \cdotp P(y=c)}{P(x)}
$$

$$
    P(x)= \sum_{c'}{P(x|y=c') \cdotp P(y=c')}, \quad P(y=c)= \frac{\beta_c}{\sum_{c'}{\beta_{c'}}}
$$

$$
    Posterior \quad P(y=c|x)= \frac{P(x|y=c) \cdotp P(y=c)}{\sum_{c'}{P(x|y=c') \cdotp P(y=c')}}
$$

- 이제 $P(y=c)$와 
위의 likelihood $P(x|y=c)$를 
$Posterior$식에 각각 대입하고 계산한다.

$$
    posterior \quad P(y=c|x)= \frac{(2\pi)^{-\frac{d}{2}} |\Sigma_{c}|^{-\frac{1}{2}}
    exp(-\frac{1}{2} (x-\mu_c)^\top \Sigma^{-1}_{c} (x-\mu_c)) \frac{\beta_c}{\sum_{c'}{\beta_{c'}}} }
    {
        \sum_{c'}{(2\pi)^{-\frac{d}{2}}} |\Sigma_{c'}|^{-\frac{1}{2}}  
        exp(-\frac{1}{2} (x-\mu_{c'})^\top \Sigma^{-1}_{c'} (x-\mu_{c'}))
        \frac{\beta_{c'}}{\sum_{c'}{\beta_{c'}}}
    }
$$

- 여기서 classifier는 Linear Discriminant Analysis라고 가정하면 모든 클래스의 공분산은 같다

- 식에 포함된 상수값인
$\left| \Sigma \right|, \; \sum_{c'}{\beta_{c'}}, \; (2\pi)^{-\frac{d}{2}}$들은 분자 분모 약분 되어 사라지고 식을 전개하면 다음과 같이 된다.

$$
    P(y=c|x) = \frac{exp(-\frac{1}{2}(x^{\top} \Sigma^{-1} x - \mu_{c}^{\top} \Sigma^{-1} x - x^{\top} \Sigma^{-1} \mu_{c} + \mu_{c}^{\top} \Sigma^{-1} \mu_{c}) + \ln{\beta_{c}})}
                {\sum_{c'} exp(-\frac{1}{2}(x^{\top} \Sigma^{-1} x - \mu_{c'}^{\top} \Sigma^{-1} x - x^{\top} \Sigma^{-1} \mu_{c'} + \mu_{c'}^{\top} \Sigma^{-1} \mu_{c'}) + \ln{\beta_{c'}})}
$$

- 여기서 
$when \; B \; is \; symmetric \;matrix, \; A \cdotp B \cdotp C = C \cdotp B \cdotp A $을 이용하여
$ x^{\top} \Sigma^{-1} \mu_{c} = \mu_{c}^{\top} \Sigma^{-1} x$ 이므로 식에 대입하면 다음과 같이 정리된다.

$$
    P(y=c|x) = \frac{exp(-\frac{1}{2}(x^{\top} \Sigma^{-1} x - 2\mu_{c}^{\top} \Sigma^{-1} x -  \mu_{c}^{\top} \Sigma^{-1} \mu_{c}) + \ln{\beta_{c}})}
                {\sum_{c'} exp(-\frac{1}{2}(x^{\top} \Sigma^{-1} x - 2\mu_{c'}^{\top} \Sigma^{-1} x + \mu_{c'}^{\top} \Sigma^{-1} \mu_{c'}) + \ln{\beta_{c'}})}
$$

- 위 식에서 공통된 부분인
$ exp(-\frac{1}{2}(x^{\top} \Sigma^{-1} x) )$ 을 따로 빼내어 다음과 같이 약분할 수 있다.

$$
    P(y=c|x) = \frac{exp(-\frac{1}{2}x^{\top} \Sigma^{-1} x) \cdotp exp( \mu_{c}^{\top} \Sigma^{-1} x - \frac{1}{2}\mu_{c}^{\top} \Sigma^{-1} \mu_{c} + \ln{\beta_{c}})}
                {exp(-\frac{1}{2}x^{\top} \Sigma^{-1} x) \cdotp \sum_{c'} exp(  \mu_{c'}^{\top} \Sigma^{-1} x -\frac{1}{2} \mu_{c'}^{\top} \Sigma^{-1} \mu_{c'} + \ln{\beta_{c'}})}
$$

$$
    P(y=c|x) = \frac{exp( \mu_{c}^{\top} \Sigma^{-1} x - \frac{1}{2}\mu_{c}^{\top} \Sigma^{-1} \mu_{c} + \ln{\beta_{c}})}
                {\sum_{c'} exp(  \mu_{c'}^{\top} \Sigma^{-1} x - \frac{1}{2}\mu_{c'}^{\top} \Sigma^{-1} \mu_{c'} + \ln{\beta_{c'}})}
$$

- 위 식에서 
$\pmb{\mathbb{w}}\_{c} = \mu_{c}^{\top} \Sigma^{-1} , \; b_c=\frac{1}{2}\mu_{c}^{\top} \Sigma^{-1} \mu_{c} + \ln{\beta_{c}}$ 이라하고 치환하면 다음과 같이 softmax classifier의 형태가 된다.

$$
    P(y=c|x)=\frac{exp(\pmb{\mathbb{w}}^{\top}_{c}x+b_c)}
                {\sum_{c'} exp(\pmb{\mathbb{w}}^{\top}_{c'}x+ b_{c'})}
$$

</div>
</details>

---

- 그러므로 사전 학습된 softmax classifier $f(x)$의 feature도 역시 class-conditional Gaussian distribution을 따른다고 할 수 있다.

- 사전 학습된 softmax neural classifier로 부터 generative classifier의 parameter를 추정하기 위해서, 실험적으로 training samples 
$\\{(x_1,y_1), \, \cdots, \, (x_N,y_N)\\} $의 공분산과 class mean을 다음과 같이 계산한다.

$$
    \hat{\mu_{c}} = \frac{1}{\mathit{N}_c} \sum_{i:y_{i}=c} f(x_i)), \;
    \hat{\Sigma} = \sum_{c} \sum_{i:y_{i}=c} \big[ (f(x_i)-\hat{\mu_{c}})(f(x_i)-\hat{\mu_{c}})^{\top} \big]
$$

- 여기서 
$\mathit{N}_c$ 는 class label c를 가지는 training sample의 수이다. 이것은 MLE(최대 우도 추정)에서 training sample에 대해 공유 공분산을 가지는
class-conditional Gaussian distribution에 fitting하는 것과 동일하다.

###### Mahalanobis distance-based confidence score

- 위에서 유도된 class-conditional Gaussian distribution을 사용하여 sample x와 가장 가까운 class-conditional distribution 사이의 
**Mahalanobis distance**를 사용한 confidence score $M(x)$를 다음과 같이 정의 한다.

$$
    M(x) = \max_{c}{-(f(x) - \hat{\mu_{c}})^{\top} \Sigma^{-1} (f(x)-\hat{\mu_{c}}) }
$$

---

<details>
<summary> <b>Mahalanobis distance 간단한 설명 펼치기</b> </summary>
<div markdown="1">

- Mahalanobis distance는 어떤 sample x가 분포의 평균값으로 부터 표준 편차의 몇배 만큼 떨어져 있는 비율을 나타낸다.
1-d gaussian distribution 일때는 다음과 같다

$$
    D = \sqrt{\bigg(\frac{x-\mu}{\sigma}\bigg)^2}
$$

- 이 식은 1차 가우스 분포 함수의 자연 상수의 지수 부분과 같다. $g(x) = \frac{1}{\sigma\sqrt{2\pi}} exp(-\frac{1}{2} \big( \frac{x-\mu}{\sigma} \big)^2 )$

- 이를 일반화한 다변량 가우스 분포 함수의 지수 부분으로 나타내면 Mahalanobis distance가 된다.

$$
    Multiple \; gaussian \; g(X) = \frac{1}{ (2\pi)^{\frac{d}{2}} |\Sigma|^{-1} } 
    exp(-\frac{1}{2} (x-\mu)^{\top} \Sigma^{-1} (x-\mu) )
$$

$$
    Mahanobis \; distance = \sqrt{(x-\mu)^{\top} \Sigma^{-1} (x-\mu)}
$$

</div>
</details>

---

- 이 metric은 test sample의 probabilty의 log값을 측정하는 것과 같다. 여기서 주목해야할 부분은 abnormal sample들은 이상치를 탐지하기 위한
이전 연구들[[13][13_link], [21][21_link]]에서 사용된 softmax-based posterior distribution의 **"label-overfitted"**된 output space보다
DNNs의 representation space에서 더 잘 특징화(characterize)할 수 있다는 것이다.

    - softmax classifier는 softmax decision boundary로 부터 멀리 떨어져 있는 abnormal sample에 대해서도 심지어 높은 confidence를 가지는 posterior distribution으로
        confidence score를 얻었기 때문이다.

    - 이러한 단점을 보안 하기 위해 Softmax classifier를 사용하지 않고 DNN feature를 사용하여 adversarial sample을 탐지하는 Feinman et al.[[7][7_link]]
        과 Ma et al.[[22][22_link]]의 연구가 있지만 이러한 연구들은 Mahalanobis distance를 사용하지 않고 단순히 Euclidean distance를 사용하였음.

###### Experimental supports for generative classifiers

- 학습된 DNN의 feature가 Gaussian Distriminant Analysis(GDA) estimation에 도움이 된다는 가설을 평가하기 위해 다음과 같은 식으로 classification accuracy를 측정함.

$$
    \hat{y}(x) = \underset{c}{arg min}(f(x) - \hat{\mu}_{c})^{\top} \hat{\Sigma}^{-1} (f(x) - \hat{\mu_{c}})
$$

<p align="center">
<img src="/assets/images/2022-09-15-A-Simple-Unified-framework-for-detecting-out-of-distribution-samples-and-adversarial-attack/figure_01.png"
height="90%" width="90%"></p>

- 이 식은 uniform class prior를 가진 generative classifier의 posterior distribution을 사용하여 class label을 예측한다는 뜻이다.

  - 흥미롭게도 scratch로 학습한 generative classifier의 성능이 softmax와 같은 discriminative classifier보다 성능이 떨어질것이라는 일반적인 통념과는 다르게
    **Figure 1(b)**를 보면 Mahalanobis distance classifier의 성능(blue bar)이 거의 softmax classifier의 성능(red bar)와 비슷한 것을 볼 수 있다.

- Figure 1(a)는 CIFAR-10 test sample들을 DNN에 통와시켰을 때 마지막 layer의 feature를 t-SNE[[23][23_link]]으로 임배딩한 것을 보여준다.(점의 색깔은 Object의 class를 나타냄)
    
  - embedding space에서 10개의 모든 class가 깔끔하게 분리된 것을 볼 수 있음.

- 추가로, Out-of-distribution sample을 탐지하는데 Mahalanobis distance-based metric이 매우 유용하게 사용될 수 있음을 보여 준다.

- 성능 평가를 위해 test sample x에 대해 Confidence score $M(x)의 값을 계산하고 그 값이 어떤 threshold 이상이면 positive로 판단하는 단순한 detector
를 사용하여 ROC curve를 그림.

  - 데이터로부터 계산된 class mean만을 이용한 Euclidean distance를 비교로 사용했다. Figure 1(c)를 보면 Mahalanobis distance 기반 방법(blue)이
    Euclidean 기반 방법(green)과 maximum of the softmax distribution(red)보다 ROC 성능이 높은 것을 볼 수 있다.

##### [2.2] Calibration techniques

###### Input pre-processing

- In-of-distribution sample과 Out-of-distribution sample을 더 구분이 잘되게 만들기 위해서, test sample에 제한된 작은 noise를 추가하였다.
각각의 test sample $x$에 대해 다음과 같은 small perturbation을 추가하여 전처리된 sample $\hat{x}$를 계산했다.

$$
    \hat{x} = x + \varepsilon sign\big(\nabla_{x}M(x)\big) = x - \varepsilon sign\big(\nabla_{x}(f(x) - \hat{\mu}_{\hat{c}})^{\top}
            \hat{\Sigma}^{-1} (f(x) - \hat{\mu}_{\hat{c}})\big)
$$

- 위 식에서 $\varepsilon$은 noise의 정도를 조절하고 $\hat{c}$는 가장 가까운 class를 나타낸다.

- adversarial attacks[[10][10_link]]와는 다르게 제안된 confidence score가 증가할수록 noise는 생성된다.

---

- Fast Gradient Sign Method(FGSM) 방식에서 아이디어를 얻었으며, Back propagation을 통해 loss를 최소화하도록 학습하는 것을 반대로 이용하여, loss를 증가시키는
방향의 gradient를 계산하여 얻은 극소량의 pertubation을 input에 더해 줌으로써 true label에 대한 softmax score를 낮추어 mis-classification을 유도함

$$
    FGSM \; Method \quad \hat{x} = x + \varepsilon sign(\nabla_{x} J(\theta,x,y))
$$

- FGSM과는 반대로 이 논문에서는 pertubation을 gradient방향으로 빼줌으로써 주어진 input에 대한 confidence score를 높여주는 방향으로 in-distribution sample에 대한 예측을 강화하여
out-of-distribution sample과 더 잘 분리될 수 있도록 도와 주는 역할을 함.

[출저] [hoya12 블로그](https://hoya012.github.io/blog/anomaly-detection-overview-2/)

---    

- 이와 비슷한 방법으로 [[21][21_link]]에서 predict label의 softmax score를 사용한 Noise를 추가함.

###### Feature Ensemble

- 성능을 더 높이기 위해 confidence score를 계산하는데 final feature 뿐만아니라 DNNs의 low-lebel feature들도 추가하는 것을 고려함.

- training data가 주어지면 $\ell$-th hidden feature를 $f_{\ell}(x)$으로 표시하고 그때 들어오는 입력 x에 대하여 class mean과 tied covariance를 각각
$$\hat{\mu}_{\ell,c} \,$$, $$ \; \hat{\Sigma}_{\ell}$$이라고 표시한다.

- 각각의 test sample x에 대해서 다음의 공식으로 $\ell$-th layer의 confidence score를 계산한다.

$$
    M_{\ell}(x) = \max_{c} -(f_{\ell}(x) - \hat{\mu}_{\ell,\hat{c}})^{\top} \hat{\Sigma}^{-1} (f_{\ell}(x) - \hat{\mu}_{\ell,\hat{c}})
$$

- low-feature로 부터 더 많은 상세한 정보를 추출함으로써 confidence score를 더 잘 조정할 수 있는 score를 얻을 수 있다.
- 각 layer에 대해 confidence score를 구하는 알고리즘은 다음과 같다.

> ---
> **Algorithm 1** Computing the Mahalanobis distance-based confidence score
> 
> ---
**Input**: Test sample x, weights of logistic regression detector $\alpha_{\ell}$, noise $\varepsilon$ and parameters of Gaussian distribution
${\hat{\mu_{\ell, \, c}} \; \hat{\Sigma_{\ell}} : \forall \ell, \,c }$
> 
> ---
> Initialize Score vectors: $M(x)=\[M_{\ell} : \forall \ell \]$   
> **for** each layer $\ell \in 1,...,L$ **do**      
> $\quad$Find the closest class: $\hat{c}=\underset{c}{arg min}(f(x) - \hat{\mu}\_{\ell,c} )^\top \hat{\Sigma}\_{\ell}^{-1} (f\_{\ell}(x) - \hat{\mu}\_{\ell,c})$   
> $\quad$Add small noise to test sample: $\hat{x}=x-\epsilon sign\big( \nabla\_{x}(f\_{\ell}(x) - \hat{\mu}\_{\ell,\hat{c}} )^{\top} 
\hat{\Sigma}\_{\ell}^{-1} (f\_{\ell}(x) - \hat{\mu}\_{\ell,\hat{c}} )\big)$     
> $\quad$Computing confidence score: $M\_{\ell}=\max\_{c} - (f\_{\ell}(x) - \hat{\mu}\_{\ell,c})^\top \hat{\Sigma}\_{\ell}^{-1} (f\_{\ell}(x) - \hat{\mu}\_{\ell,c})$   
> **end for**   
> **return** Confidence score for test sample $\sum\_{\ell}\alpha\_{\ell}M\_{\ell}$     
> 
> ---

<p align="center">
<img src="/assets/images/2022-09-15-A-Simple-Unified-framework-for-detecting-out-of-distribution-samples-and-adversarial-attack/figure_02.png"
height="100%" width="100%"></p>


- Figure 2는 CIFAR-10 dataset에 대해 학습한 DenseNet의 basic block을 변경하면서 confidence score를 계산한 ROC curve이다.
SVHN[[28][28_link]], LSUN[[32][32_link]], TinyImageNet과 같은 OOD sample과 DeepFool[[26][26_link]]에 의해 생성된 adversarial sample에 대한 성능을 보여준다.
LSUN, TinyImageNet, DeepFool에서는 final feature와 비교하여 때때로 low-level feature의 성능이 더 나은 경우가 있다.
그래서 성능 향상을 위해 **Algorithm 1**처럼 모든 layer의 feature를 사용하여 confidence score를 계산하고 각 layer의 score는 weight sum을 통하여 전체적인 confidence score를 계산한다.

- validation sample을 사용한 logistic regression detector를 training하면서 각 layer의 weight ratio $\alpha\_{\ell}$을 선택한다.
이러한 score weighted averaging은 몇개의 layer로부터 얻은 score가 유효하지 않을 경우 거의 0에 근접한 weight를 주어 전체적인 성능 하락을 방지 한다.

##### [2.3] Class-Incremental learning using Mahalanobis distance-based score

- 더나아가 Mahalanobis distance-based score를 class-incremental learning task[[29][29_link]]에서도 사용될 수 있다.
  - class-incremental learning은 base class에 대해 사전 학습한 classifier를 new class가 생길때 마다 점진적으로 업데이트하여 new class를 수용함.
  - 이 task는 제한된 메모리로 catastrophic forgetting[[24][24_link]]를 해결해야하기 때문에 매우 어렵다. 이를 해결하기 위해 최근
    연구들은 모델 생성과 데이터 샘플링에 관련된 새로운 학습 방법을 개발하는 방향으로 진행되고 있지만 이러한 학습 방법은 비싼 학습 비용을 발생시킨다.
  - 그래서 제안된 confidence score를 기반으로 복잡한 훈련방법을 사용하지 않는 단순한 classification 방법을 개발했다.
  - 이 방법을 사용하기 위해서는 첫번째 가정으로 기본 class에 대해 충분히 잘 사전 학습된 모델이 있어야하는데 인터넷상에 큰 데이터세트로 학습된 Resnet과 같은 모델등이 많기 때문에 적용에 큰 어려움은 없다.
  - 잘 사전학습된 모델을 사용하는 경우 OOD sample을 잘 탐지할 뿐만아니라 base class로 학습한 represetation이 new class를 잘 특징화 할 수 있으므로 new class를 잘 구별할 수 있다.

- 이러한 점을 기반으로 다음의 공식을 기반으로 class mean과 covariance를 단순히 계산하고 업데이트하므로써 new class를 수용하는 Mahalanobis distance-based classifier를 **Algorithm 2**에서 설명한다.

$$
    \hat{y}(x)=\underset{c}{argmin}\big(f(x) - \hat{\mu}\_{c}  \big)^{\top} \hat{\Sigma}^{-1} \big( f(x) - \hat{\mu}\_{c} \big)
$$

> ---
> 
> **Algorithm 2** Updating Mahalanobis distance-based classifier for class-incremental learning.
> 
> ---
> 
> **Input**: set of samples from a new class ${x\_i: \forall i = 1,...,N\_{C+1}}$, mean and covariance of observed classes
${\hat{\mu}\_{c}: \forall\_{C} = 1,...,C}, \; \hat{\Sigma}$
> 
> ---
> 
> Compute the new class mean:  $\hat{\mu}\_{C+1} \gets \frac{1}{N\_{C+1}} \sum\_{i} f(x\_{i}) $      
> Compute the covariance of the new class: $ \hat{\Sigma}\_{C+1} \gets \frac{1}{N\_{C+1}} \sum\_{i} (f(x\_{i}) -  \hat{\mu}\_{C+1} )^{\top} $        
> Update the shared covariance: $ \hat{\Sigma} \gets \frac{C}{C+1} \hat{\Sigma} + \frac{1}{C+1} \hat{\Sigma}\_{C+1} $       
> **return** Mean and covariance of all classes ${ \hat{\mu}\_{c} : \forall\_{C}=1,...,C+1, \; \hat{\Sigma}  }$     
>
> ---

#### [4] Conclusion

- 단순하지만 OOD와 adversarial에 대한 abnormal test sample을 잘탐지하는 효과적인 방법을 제안.
- 본질적으로, main idea는 LDA 추정 기반의 generative classifier로 부터 나왔으며 새로운 confidence score도 이것을 기반으로 함.
- feature ensemble과 input preprocessing과 같은 calibration technique을 통해 많은 task에 대해 robust한 성능을 가지게함.
  - OOD sample, adversarial attack, class incremental learning
- 이 방법은 training data가 noisy, ranodm label, data sample의 개수가 매우 적은 극한 상황에서도 hyper parameter의 설정이 매우 자유로워 robust하다.


---

### 선형 판별 분석(Linear Discriminant Analysis) 

[참고] [ratgo's blog](https://ratsgo.github.io/machine%20learning/2017/03/21/LDA/)

- LDA(Linear Discriminant Analysis)는 특정한 축(axis)에 사영(projection)한 이후 범주(category) 혹은 클래스를 잘 구분하는 직선인 Decision Boundary를 찾는 것이 목표.
- 2개의 클래스를 잘 구분하기 위해선 클래스들이 사영된 축에서 클래스 사이의 중심(평균)이 서로 멀고 각 클래스의 분산이 작아야 된다.

<p align="center">
<img src="/assets/images/2022-09-15-A-Simple-Unified-framework-for-detecting-out-of-distribution-samples-and-adversarial-attack/LDA_01.png"
height="80%" width="80%"> </p>

---

#### 1. LDA에 대한 첫번째 접근

- $p$차원의 입력 $vector \; x$가 $\mathbf{w}$라는 축에 사영된다고 할때 축 위에서 사영된 1차원의 스칼라 값을 $y$라고 하고
두 클래스 $C\_{1}$과 $C\_{2}$ 각각에 대해 $N\_{1}$, $N\_{2}$개의 데이터가 있다고 하자.

$$
    y = \mathbf{\overrightarrow{w}} \cdotp \overrightarrow{x} = 
    \mathbf{\overrightarrow{w}^{\top}} \overrightarrow{x}
    = \overrightarrow{x}^{\top} \mathbf{\overrightarrow{w}}
    \;, \quad

    \overrightarrow{x} = \begin{bmatrix}
    a_{1} \\
    \vdots \\
    a_{p}
    \end{bmatrix}  
    \quad
    \mathbf{\overrightarrow{w}} = \begin{bmatrix}
    w_{1} \\
    \vdots \\
    w_{p}
    \end{bmatrix}
    \quad 
    \mathbf{\overrightarrow{w}} \; is \; unit \; vector
$$

$$
    C_{1} \; mean : \overrightarrow{m_{1}} = \frac{1}{N_{1}} \sum_{n \in C_{1}} \overrightarrow{x_{n}}
$$

$$
    C_{2} \; mean : \overrightarrow{m_{2}} = \frac{1}{N_{2}} \sum_{n \in C_{2}} \overrightarrow{x_{n}}
$$

- 먼저 사영후 두 클래스의 중심인 평균이 서로 멀어야 된다. 이때 $m\_1$과 $m\_2$를 축에 사영하여 $m\_{1}'$과 $m\_{2}'$를 만들고 그 사이의 거리
$\overline{m\_{1}' m\_{2}'}$가 최대가 되는 축 $\mathbf{w}$를 찾으면 된다.

<p align="center">
<img src="/assets/images/2022-09-15-A-Simple-Unified-framework-for-detecting-out-of-distribution-samples-and-adversarial-attack/LDA_02.png"
width="40%" height="40%">
</p>

- 여기서 축 $\mathbf{w}$에 사영된 $\overline{m\_{1}' m\_{2}'}$의 길이는 다음과 같다. 

$$
    \overline{m_{2}' m_{1}'}  = m_{2}'-m_{1}' = \overrightarrow{\mathbf{w}}^{\top} (\overrightarrow{m_{2}} - \overrightarrow{m_{1}})
    = (\overrightarrow{m_{2}} - \overrightarrow{m_{1}})^{\top}  \overrightarrow{\mathbf{w}}
$$

- 다음으로 사영후 두 클래스 각각의 분산이 작아야 된다. 클래스가 넓은 범위에 걸쳐 사영되면 범위가 겹치는 곳이 많아지므로 클래스를 구분하기 어려워지므로 좁은 범위에 사영되게 하기 위해 분산을 작게한다.
k class의 사영된 값을 $y\_{k}$, 평균 $m\_{k}$, 분산을 $s\_{k}^{2}$ 라고 한다. 이때 분산은 다음과 같이 계산된다.

$$
    s_{k}^{2} = \sum_{n \in C_{k}} (y_{n} - m_{k}')^2
$$

- 이제 축에 사영된 중심의 거리는 최대화 하고 분산은 최소화해야 된다. 다음과 같은 함수를 만들고 이를 최대화 하면 된다.(식을 단순화 하기 위해서 두 개의 클래스만 가정)

$$
    \underset{\mathbf{w}}{Maximize} \; J(\mathbf{w}) = \frac{ (m_{2}'-m_{1}')^{2} }{ s_{2}^2 + s_{1}^2 }
$$ 

- 여기서 $J(\mathbf{w})$ 함수를 최대화하는 $\mathbf{w}$를 찾아야 하므로$\mathbf{w}$에 대한 함수로 만들기 위해 위의 식들을 대입하여 정리하면 다음과 같다.

$$
    m_{2}'-m_{1}' = (\overrightarrow{m_{2}} - \overrightarrow{m_{1}})^{\top}  \overrightarrow{\mathbf{w}} \\
    y_{n} = \overrightarrow{x_{n}}^{\top} \overrightarrow{\mathbf{w}} \;, \quad 
    m_{k}' = \overrightarrow{m_{k}}^{\top} \overrightarrow{\mathbf{w}} \\
    s^{2}_{k} = \sum_{n \in C_{k}} (\overrightarrow{x_{n}}^{\top} \overrightarrow{\mathbf{w}} - \overrightarrow{m_{k}}^{\top} \overrightarrow{\mathbf{w}})^2
    = \sum_{n \in C_{k}} (\overrightarrow{x_{n}}^{\top} \overrightarrow{\mathbf{w}} - \overrightarrow{m_{k}}^{\top} \overrightarrow{\mathbf{w}})^{\top}  (\overrightarrow{x_{n}}^{\top} \overrightarrow{\mathbf{w}} - \overrightarrow{m_{k}}^{\top} \overrightarrow{\mathbf{w}}) \\
    = \sum_{n \in C_{k}} \big( (\overrightarrow{x_{n}}^{\top} - \overrightarrow{m_{k}}^{\top}) \overrightarrow{\mathbf{w}} \big)^{\top} \big( (\overrightarrow{x_{n}}^{\top} - \overrightarrow{m_{k}}^{\top}) \overrightarrow{\mathbf{w}} \big)
    = \sum_{n \in C_{k}} \overrightarrow{\mathbf{w}}^{\top} (\overrightarrow{x_{n}} - \overrightarrow{m_{k}}) (\overrightarrow{x_{n}} - \overrightarrow{m_{k}})^{\top} \overrightarrow{\mathbf{w}}
$$

$$
    J(\mathbf{w}) = \frac{ \big( (\overrightarrow{m_{2}} - \overrightarrow{m_{1}})^{\top}  \overrightarrow{\mathbf{w}} \big)^{\top} \big((\overrightarrow{m_{2}} - \overrightarrow{m_{1}})^{\top}  \overrightarrow{\mathbf{w}} \big) }
                        {\sum_{n \in C_{2}} \overrightarrow{\mathbf{w}}^{\top} (\overrightarrow{x_{n}} - \overrightarrow{m_{2}}) (\overrightarrow{x_{n}} - \overrightarrow{m_{2}})^{\top} \overrightarrow{\mathbf{w}}
                        + \sum_{n \in C_{1}} \overrightarrow{\mathbf{w}}^{\top} (\overrightarrow{x_{n}} - \overrightarrow{m_{1}}) (\overrightarrow{x_{n}} - \overrightarrow{m_{1}})^{\top} \overrightarrow{\mathbf{w}}} \\
    
    = \frac{  \overrightarrow{\mathbf{w}}^{\top} (\overrightarrow{m_{2}} - \overrightarrow{m_{1}})  (\overrightarrow{m_{2}} - \overrightarrow{m_{1}})^{\top}  \overrightarrow{\mathbf{w}} }
            {  \overrightarrow{\mathbf{w}}^{\top} \bigg[ \sum_{n \in C_{2}}  (\overrightarrow{x_{n}} - \overrightarrow{m_{2}}) (\overrightarrow{x_{n}} - \overrightarrow{m_{2}})^{\top} 
            + \sum_{n \in C_{1}} (\overrightarrow{x_{n}} - \overrightarrow{m_{1}}) (\overrightarrow{x_{n}} - \overrightarrow{m_{1}})^{\top} \bigg] \overrightarrow{\mathbf{w}} }
$$

- 여기서 식을 간단하게 하기 위해 식을 치환한다.

$$
    S_{W} = \sum_{n \in C_{2}}  (\overrightarrow{x_{n}} - \overrightarrow{m_{2}}) (\overrightarrow{x_{n}} - \overrightarrow{m_{2}})^{\top} 
            + \sum_{n \in C_{1}} (\overrightarrow{x_{n}} - \overrightarrow{m_{1}}) (\overrightarrow{x_{n}} - \overrightarrow{m_{1}})^{\top} \\
    
    S_{B} = (\overrightarrow{m_{2}} - \overrightarrow{m_{1}})  (\overrightarrow{m_{2}} - \overrightarrow{m_{1}})^{\top} \\
    
    J(\mathbf{w}) = \frac{\overrightarrow{\mathbf{w}}^{\top} S_{B} \overrightarrow{\mathbf{w}}}
                        {\overrightarrow{\mathbf{w}}^{\top} S_{W} \overrightarrow{\mathbf{w}}}
    
$$

- 이제 $J(w)$의 최대값을 구하기 위해 미분을 하여 $J^{\prime}(w)=0$인 $w$값을 찾으면 된다.

$$
\require{cancel}

\big(J(w) (\overrightarrow{\mathbf{w}}^{\top} S_{W} \overrightarrow{\mathbf{w}}) \big)^{\prime} = (\overrightarrow{\mathbf{w}}^{\top} S_{B} \overrightarrow{\mathbf{w}})^{\prime} \\

\cancel{J^{\prime}(w) (\overrightarrow{\mathbf{w}}^{\top} S_{W} \overrightarrow{\mathbf{w}}) }
+ J(w) (\overrightarrow{\mathbf{w}}^{\top} S_{W} \overrightarrow{\mathbf{w}})^{\prime}
= (\overrightarrow{\mathbf{w}}^{\top} S_{B} \overrightarrow{\mathbf{w}})^{\prime} \;, \quad J^{\prime}(w)=0

$$

- 행렬 미분을 이용하여 정리하면 다음과 같다.

$$
    \frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^{\top} \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^{\top})\mathbf{x}= 2\mathbf{A}\mathbf{x}, \quad if \; \mathbf{A} \; is \; symmetric \\

    J(w)(2S_{W} \overrightarrow{\mathbf{w}}) = (2S_{B} \overrightarrow{\mathbf{w}})
$$

- 여기서 $J^{\prime}(w)=0$을 만족하는 w로 고정 되어 있으므로 $J(w)$은 스칼라 값이다. 그래서 식을 다음과 같이 변형하면 고유값 형태의 문제가 된다.

$$
    \lambda = J(w) \; ,\quad S_{W}^{-1} S_{B} = \mathbf{A} \\

    J(w)(S_{W} \overrightarrow{\mathbf{w}}) = (S_{B} \overrightarrow{\mathbf{w}}) \quad \Rightarrow \quad \mathbf{A} \overrightarrow{\mathbf{w}} = \lambda \overrightarrow{\mathbf{w}}

$$

- 즉 두개 클래스의 $vector \; x$들을 $\mathbf{w}$축에 사영했을때 서로의 중심 거리가 최대가 되고 각 클래스의 분산이 최소가 되는 $\mathbf{w}$는 
$(S_{W}^{-1} S_{B})$의 고유 vector이다.

---

#### 2. LDA에 대한 두번째 접근(Bayes rule)

- 클래스 $w\_{1}$, $w\_{2}$가 있고 데이터 x가 있다고 할때 목표인 판별 함수(discriminant function) 
$p(w\_{1}|x)$와 $p(w\_{2}|x)$ 즉, Posterior를 구해야한다. 하지만 이는 실제로 매우 구하기 힘드므로 베이즈 정리를 사용하여 Likelihood와 Prior로 구해야된다.
베이즈 정리에 의해 Posterior는 다음과 같다.

$$
    Prior=p(w_i), \quad Likelihood=p(x|w_i), \quad Evidence=p(x) \\
    Posterior = \frac{Likelihood \times Prior}{Evidence} \;,\quad p(w_i|x) = \frac{p(x|w_i) p(w_i)}{p(x)}  \\

    law \; of \; total \; probability\; :\quad p(x)= \int_{-\infty}^{\infty} p(x|w)p(w)dw \quad or \quad \sum_{i}^{all} p(x|w_i) p(w_i)
$$

- 여기선 두개의 클래스로 가정했으므로 전체 확률의 법칙을 적용하면 다음과 같은 식이 된다.

$$ 
    p(w_1|x) = \frac{p(x|w_1) p(w_1)}{p(x|w_1) p(w_1) + p(x|w_2) p(w_2)} \\
    p(w_2|x) = \frac{p(x|w_2) p(w_2)}{p(x|w_1) p(w_1) + p(x|w_2) p(w_2)}
$$

- 세상에 거의 대부분의 데이터의 분포는 가우스 정규 분포(Gaussian normal distribution)를 따른다. 
현재 우리가 판별하고자 하는 데이터 x도 다변량 정규 분포(Multivariate Gaussian Normal distribution)를 따른다고 가정한다. 

- 이때 다변량 정규 분포의 차원은 d이고 이때 평균 $\mu$과 공분산 $\Sigma$는 다음과 같다.

$$
    \mu_i = \begin{bmatrix}
    m_1 \\
    \vdots \\
    m_d
    \end{bmatrix}
    \; , \quad

    \Sigma_i = \begin{bmatrix}
    \sigma_{1,\,1} && \cdots && \sigma_{1,\,d} \\
    \vdots && \ddots && \vdots \\
    \sigma_{d,\,1} && \cdots && \sigma_{d,\,d}
    \end{bmatrix}
$$

$$
    p(x|w_i) = \frac{1}{(2\pi)^{\frac{d}{2}} |\Sigma_i|^{\frac{1}{2}}} \exp \bigg({ -\frac{1}{2} (x-\mu_i)^{\top}\Sigma_i^{-1}(x-\mu_i)} \bigg)
$$



[1_link]: https://arxiv.org/abs/1512.02595 "Deep Speech 2:End-to-end speech recognition in english and mandarin. In ICML, 2016."

[2_link]: https://arxiv.org/abs/1606.06565 "Amodei, Dario, Olah, Chris, Steinhardt, Jacob, Christiano, Paul, Schulman, John, and Man´e, Dan. Concrete problems in ai safety. arXiv preprint arXiv:1606.06565, 2016."

[3_link]: https://arxiv.org/abs/1705.07263 "Carlini, Nicholas andWagner, David. Adversarial examples are not easily detected: Bypassing ten detection methods. In ACM workshop on AISec, 2017."
[4_link]: https://arxiv.org/abs/1707.08819 "Chrabaszcz, Patryk, Loshchilov, Ilya, and Hutter, Frank. A downsampled variant of imagenet as an alternative to the cifar datasets. arXiv preprint arXiv:1707.08819, 2017."

[5_link]: https://ieeexplore.ieee.org/document/5206848 "Deng, Jia, Dong, Wei, Socher, Richard, Li, Li-Jia, Li, Kai, and Fei-Fei, Li. Imagenet: A large-scale hierarchical image database. In CVPR, 2009."

[6_link]: https://arxiv.org/abs/1707.08945 "Evtimov, Ivan, Eykholt, Kevin, Fernandes, Earlence, Kohno, Tadayoshi, Li, Bo, Prakash, Atul, Rahmati, Amir, and Song, Dawn. Robust physical-world attacks on machine learning models. In CVPR, 2018."

[7_link]: https://arxiv.org/abs/1703.00410 "Feinman, Reuben, Curtin, Ryan R, Shintre, Saurabh, and Gardner, Andrew B. Detecting adversarial samples from artifacts. arXiv preprint arXiv:1703.00410, 2017."

[8_link]: https://arxiv.org/abs/1703.02910 "Gal, Yarin, Islam, Riashat, and Ghahramani, Zoubin. Deep bayesian active learning with image data. In ICML, 2017."

[9_link]: https://arxiv.org/abs/1504.08083 "Girshick, Ross. Fast r-cnn. In ICCV, 2015."

[10_link]: https://arxiv.org/abs/1412.6572 "Goodfellow, Ian J, Shlens, Jonathon, and Szegedy, Christian. Explaining and harnessing adversarial examples. In ICLR, 2015."

[11_link]: https://arxiv.org/abs/1711.00117 "Guo, Chuan, Rana, Mayank, Ciss´e, Moustapha, and van der Maaten, Laurens. Countering adversarial images using input transformations. arXiv preprint arXiv:1711.00117, 2017."

[12_link]: https://arxiv.org/abs/1512.03385 "He, Kaiming, Zhang, Xiangyu, Ren, Shaoqing, and Sun, Jian. Deep residual learning for image recognition. In CVPR, 2016."

[13_link]: https://arxiv.org/abs/1610.02136 "Hendrycks, Dan and Gimpel, Kevin. A baseline for detecting misclassified and out-ofdistribution examples in neural networks. In ICLR, 2017."

[14_link]: https://arxiv.org/abs/1608.06993 "Huang, Gao and Liu, Zhuang. Densely connected convolutional networks. In CVPR, 2017."

[15_link]: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf "Krizhevsky, Alex and Hinton, Geoffrey. Learning multiple layers of features from tiny images. 2009."

[16_link]: https://arxiv.org/abs/1607.02533 "Kurakin, Alexey, Goodfellow, Ian, and Bengio, Samy. Adversarial examples in the physical world. arXiv preprint arXiv:1607.02533, 2016."

[17_link]: https://ieeexplore.ieee.org/document/1640745 "Lasserre, Julia A, Bishop, Christopher M, and Minka, Thomas P. Principled hybrids of generative and discriminative models. In CVPR, 2006."

[18_link]: https://arxiv.org/abs/1804.00722 "Lee, Kibok, Lee, Kimin, Min, Kyle, Zhang, Yuting, Shin, Jinwoo, and Lee, Honglak. Hierarchical novelty detection for visual object recognition. In CVPR, 2018."

[19_link]: https://arxiv.org/abs/1706.03475 "Lee, Kimin, Hwang, Changho, Park, KyoungSoo, and Shin, Jinwoo. Confident multiple choice learning. In ICML, 2017."

[20_link]: https://arxiv.org/abs/1711.09325 "Lee, Kimin, Lee, Honglak, Lee, Kibok, and Shin, Jinwoo. Training confidence-calibrated classifiers for detecting out-of-distribution samples. In ICLR, 2018."

[21_link]: https://www.researchgate.net/publication/317418851_Principled_Detection_of_Out-of-Distribution_Examples_in_Neural_Networks "Liang, Shiyu, Li, Yixuan, and Srikant, R. Principled detection of out-of-distribution examples in neural networks. In ICLR, 2018."

[22_link]: https://arxiv.org/abs/1801.02613 "Ma, Xingjun, Li, Bo, Wang, Yisen, Erfani, Sarah M, Wijewickrema, Sudanthi, Houle, Michael E, Schoenebeck, Grant, Song, Dawn, and Bailey, James. Characterizing adversarial subspaces using local intrinsic dimensionality. In ICLR, 2018."

[23_link]: https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl "Maaten, Laurens van der and Hinton, Geoffrey. Visualizing data using t-sne. Journal of machine learning research, 2008."

[24_link]: https://www.sciencedirect.com/science/article/abs/pii/S0079742108605368 "McCloskey, Michael and Cohen, Neal J. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology of learning and motivation. Elsevier, 1989."

[25_link]: https://hal.inria.fr/hal-00817211/document "Mensink, Thomas, Verbeek, Jakob, Perronnin, Florent, and Csurka, Gabriela. Distance-based image classification: Generalizing to new classes at near-zero cost. IEEE transactions on pattern analysis and machine intelligence, 2013."

[26_link]: https://arxiv.org/abs/1511.04599 "Moosavi Dezfooli, Seyed Mohsen, Fawzi, Alhussein, and Frossard, Pascal. Deepfool: a simple and accurate method to fool deep neural networks. In CVPR, 2016."

[27_link]: http://noiselab.ucsd.edu/ECE228/Murphy_Machine_Learning.pdf "Murphy, Kevin P. Machine learning: a probabilistic perspective. 2012." 

[28_link]: https://research.google/pubs/pub37648/ "Netzer, Yuval, Wang, Tao, Coates, Adam, Bissacco, Alessandro, Wu, Bo, and Ng, Andrew Y. Reading digits in natural images with unsupervised feature learning. In NIPS workshop, 2011."

[29_link]: https://arxiv.org/abs/1611.07725 "Rebuffi, Sylvestre-Alvise and Kolesnikov, Alexander. icarl: Incremental classifier and representation learning. In CVPR, 2017."

[30_link]: https://dl.acm.org/doi/10.1145/2976749.2978392 "Sharif, Mahmood, Bhagavatula, Sruti, Bauer, Lujo, and Reiter, Michael K. Accessorize to a crime: Real and stealthy attacks on state-of-the-art face recognition. In ACM SIGSAC, 2016."

[31_link]: https://arxiv.org/abs/1606.04080 "Vinyals, Oriol, Blundell, Charles, Lillicrap, Tim, Wierstra, Daan, et al. Matching networks for one shot learning. In NIPS, 2016."

[32_link]: https://arxiv.org/abs/1506.03365 "Yu, Fisher, Seff, Ari, Zhang, Yinda, Song, Shuran, Funkhouser, Thomas, and Xiao, Jianxiong. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015."

