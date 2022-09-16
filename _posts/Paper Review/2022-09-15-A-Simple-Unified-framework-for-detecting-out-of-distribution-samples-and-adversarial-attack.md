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
last_modified_at: 2022-09-16T13:32:21
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
  - 



[1_link]: https://arxiv.org/abs/1512.02595 "Deep Speech 2:End-to-end speech recognition in english and mandarin. In ICML, 2016."
[2_link]: https://arxiv.org/abs/1606.06565 "Amodei, Dario, Olah, Chris, Steinhardt, Jacob, Christiano, Paul, Schulman, John, and Man´e,
Dan. Concrete problems in ai safety. arXiv preprint arXiv:1606.06565, 2016."
[3_link]: https://arxiv.org/abs/1705.07263 "Carlini, Nicholas andWagner, David. Adversarial examples are not easily detected: Bypassing
ten detection methods. In ACM workshop on AISec, 2017."
[4_link]: https://arxiv.org/abs/1707.08819 "Chrabaszcz, Patryk, Loshchilov, Ilya, and Hutter, Frank. A downsampled variant of imagenet
as an alternative to the cifar datasets. arXiv preprint arXiv:1707.08819, 2017."
[5_link]: https://ieeexplore.ieee.org/document/5206848 "Deng, Jia, Dong, Wei, Socher, Richard, Li, Li-Jia, Li, Kai, and Fei-Fei, Li. Imagenet: A
large-scale hierarchical image database. In CVPR, 2009."
[6_link]: https://arxiv.org/abs/1707.08945 "Evtimov, Ivan, Eykholt, Kevin, Fernandes, Earlence, Kohno, Tadayoshi, Li, Bo, Prakash, Atul,
Rahmati, Amir, and Song, Dawn. Robust physical-world attacks on machine learning models.
In CVPR, 2018."
[7_link]: https://arxiv.org/abs/1703.00410 "Feinman, Reuben, Curtin, Ryan R, Shintre, Saurabh, and Gardner, Andrew B. Detecting adversarial
samples from artifacts. arXiv preprint arXiv:1703.00410, 2017."
[8_link]: https://arxiv.org/abs/1703.02910 "Gal, Yarin, Islam, Riashat, and Ghahramani, Zoubin. Deep bayesian active learning with image
data. In ICML, 2017."
[9_link]: https://arxiv.org/abs/1504.08083 "Girshick, Ross. Fast r-cnn. In ICCV, 2015."
[10_link]: https://arxiv.org/abs/1412.6572 "Goodfellow, Ian J, Shlens, Jonathon, and Szegedy, Christian. Explaining and harnessing adversarial
examples. In ICLR, 2015."
[11_link]: https://arxiv.org/abs/1711.00117 "Guo, Chuan, Rana, Mayank, Ciss´e, Moustapha, and van der Maaten, Laurens. Countering
adversarial images using input transformations. arXiv preprint arXiv:1711.00117, 2017."
[12_link]: https://arxiv.org/abs/1512.03385 "He, Kaiming, Zhang, Xiangyu, Ren, Shaoqing, and Sun, Jian. Deep residual learning for image
recognition. In CVPR, 2016."
[13_link]: https://arxiv.org/abs/1610.02136 "Hendrycks, Dan and Gimpel, Kevin. A baseline for detecting misclassified and out-ofdistribution
examples in neural networks. In ICLR, 2017."
[14_link]: https://arxiv.org/abs/1608.06993 "Huang, Gao and Liu, Zhuang. Densely connected convolutional networks. In CVPR, 2017."
[15_link]: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf "Krizhevsky, Alex and Hinton, Geoffrey. Learning multiple layers of features from tiny images.
2009."
[16_link]: https://arxiv.org/abs/1607.02533 "Kurakin, Alexey, Goodfellow, Ian, and Bengio, Samy. Adversarial examples in the physical
world. arXiv preprint arXiv:1607.02533, 2016."
[17_link]: https://ieeexplore.ieee.org/document/1640745 "Lasserre, Julia A, Bishop, Christopher M, and Minka, Thomas P. Principled hybrids of generative
and discriminative models. In CVPR, 2006."
[18_link]: https://arxiv.org/abs/1804.00722 "Lee, Kibok, Lee, Kimin, Min, Kyle, Zhang, Yuting, Shin, Jinwoo, and Lee, Honglak. Hierarchical
novelty detection for visual object recognition. In CVPR, 2018."
[19_link]: https://arxiv.org/abs/1706.03475 "Lee, Kimin, Hwang, Changho, Park, KyoungSoo, and Shin, Jinwoo. Confident multiple choice
learning. In ICML, 2017."
[20_link]: https://arxiv.org/abs/1711.09325 "Lee, Kimin, Lee, Honglak, Lee, Kibok, and Shin, Jinwoo. Training confidence-calibrated
classifiers for detecting out-of-distribution samples. In ICLR, 2018."
[21_link]: https://www.researchgate.net/publication/317418851_Principled_Detection_of_Out-of-Distribution_Examples_in_Neural_Networks "Liang, Shiyu, Li, Yixuan, and Srikant, R. Principled detection of out-of-distribution examples
in neural networks. In ICLR, 2018."
[22_link]: https://arxiv.org/abs/1801.02613 "Ma, Xingjun, Li, Bo, Wang, Yisen, Erfani, Sarah M, Wijewickrema, Sudanthi, Houle,
Michael E, Schoenebeck, Grant, Song, Dawn, and Bailey, James. Characterizing adversarial
subspaces using local intrinsic dimensionality. In ICLR, 2018."
[23_link]: https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl "Maaten, Laurens van der and Hinton, Geoffrey. Visualizing data using t-sne. Journal of
machine learning research, 2008."
[24_link]: https://www.sciencedirect.com/science/article/abs/pii/S0079742108605368 "McCloskey, Michael and Cohen, Neal J. Catastrophic interference in connectionist networks:
The sequential learning problem. In Psychology of learning and motivation. Elsevier, 1989."
[25_link]: https://hal.inria.fr/hal-00817211/document "Mensink, Thomas, Verbeek, Jakob, Perronnin, Florent, and Csurka, Gabriela. Distance-based
image classification: Generalizing to new classes at near-zero cost. IEEE transactions on
pattern analysis and machine intelligence, 2013."
[26_link]: https://arxiv.org/abs/1511.04599 "Moosavi Dezfooli, Seyed Mohsen, Fawzi, Alhussein, and Frossard, Pascal. Deepfool: a simple
and accurate method to fool deep neural networks. In CVPR, 2016."
[27_link]: http://noiselab.ucsd.edu/ECE228/Murphy_Machine_Learning.pdf "Murphy, Kevin P. Machine learning: a probabilistic perspective. 2012."
[28_link]: https://research.google/pubs/pub37648/ "Netzer, Yuval, Wang, Tao, Coates, Adam, Bissacco, Alessandro, Wu, Bo, and Ng, Andrew Y.
Reading digits in natural images with unsupervised feature learning. In NIPS workshop, 2011."
[29_link]: https://arxiv.org/abs/1611.07725 "Rebuffi, Sylvestre-Alvise and Kolesnikov, Alexander. icarl: Incremental classifier and representation
learning. In CVPR, 2017."
[30_link]: https://dl.acm.org/doi/10.1145/2976749.2978392 "Sharif, Mahmood, Bhagavatula, Sruti, Bauer, Lujo, and Reiter, Michael K. Accessorize to a
crime: Real and stealthy attacks on state-of-the-art face recognition. In ACM SIGSAC, 2016."
[31_link]: https://arxiv.org/abs/1606.04080 "Vinyals, Oriol, Blundell, Charles, Lillicrap, Tim, Wierstra, Daan, et al. Matching networks for
one shot learning. In NIPS, 2016."
[32_link]: https://arxiv.org/abs/1506.03365 "Yu, Fisher, Seff, Ari, Zhang, Yinda, Song, Shuran, Funkhouser, Thomas, and Xiao, Jianxiong.
Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop.
arXiv preprint arXiv:1506.03365, 2015."

