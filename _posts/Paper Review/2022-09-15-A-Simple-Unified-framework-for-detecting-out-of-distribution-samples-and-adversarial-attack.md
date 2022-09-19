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
last_modified_at: 2022-09-19T14:39:24
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

---

- 


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

