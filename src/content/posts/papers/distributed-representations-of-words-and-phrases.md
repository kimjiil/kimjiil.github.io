---
title: "[word2vec 계보 ③] Distributed Representations of Words and Phrases (negative sampling, 2013.10)"
date: 2026-08-21
category: deep-learning-paper
tags:
  - "NLP"
  - "word2vec"
  - "Word Embedding"
  - "Negative Sampling"
---

<span style="font-size:17pt">
<b>Distributed Representations of Words and Phrases and their Compositionality</b>
</span>

<a href="https://arxiv.org/abs/1310.4546" target="_blank"><b>[PDF]</b></a>
, <b><span style="color: #F2AA4C">Word Embedding</span></b>, Tomáš Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean (Google, NIPS 2013)

### <span style="color: #ffd33d">Summary</span>

**word2vec의 두번째 논문.** [계보 ②](/posts/efficient-estimation-of-word-representations-in-vector-space/)에서 제안한
Skip-gram을 3가지로 개량해서, 오늘날 "word2vec"이라고 부르는 알고리즘(SGNS: Skip-gram with Negative Sampling)을 완성했다.

1. **Negative Sampling**: hierarchical softmax를 버리고, "진짜 (중심, 문맥) 쌍과 노이즈 쌍을 구분하는
이진 분류 $k$개"로 objective를 대체. 더 빠르고 벡터 품질도 더 좋다.
2. **빈번한 단어 Subsampling**: "the" 같은 단어를 확률적으로 건너뛰어 2~10배 가속 + 희귀 단어 벡터 품질 향상.
3. **Phrase 벡터**: "New York Times"처럼 합성이 안 되는 구(phrase)를 통계적으로 찾아 하나의 토큰으로 학습.

두 논문을 합쳐 인용수가 수만 회에 달하는, NLP 역사상 가장 영향력 있는 논문 중 하나다.
"벡터 덧셈으로 의미가 합성된다"(compositionality)는 관찰도 이 논문에서 정리됐다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[1] Skip-gram 복습과 문제 설정</span>

- Skip-gram objective (계보 ②와 동일). 중심 단어로 주변 단어 각각을 예측한다.

$$
    \frac{1}{T}\sum_{t=1}^{T}{\sum_{-c \le j \le c,\; j \ne 0}{\log{p(w_{t+j}|w_t)}}}
    ,\qquad
    p(w_O|w_I) = \frac{\exp\big({v'_{w_O}}^{\top}v_{w_I}\big)}{\sum_{w=1}^{V}{\exp\big({v'_w}^{\top}v_{w_I}\big)}}
$$

- 문제는 여전히 분모다. full softmax의 $\nabla \log p(w_O|w_I)$는 $O(V)$이고, 계보 ②의
hierarchical softmax는 $O(\log V)$까지 줄였지만 **Huffman 트리 구조에 품질이 좌우**되고,
트리라는 자료구조 자체가 구현/병렬화에 부담이다.
- 이 논문의 관점 전환: **애초에 정규화된 확률분포가 필요한가?**
목적이 언어모델이 아니라 좋은 벡터라면, "진짜 쌍을 노이즈에서 구분"만 잘해도 된다.
- 참고로 Skip-gram이 학습하는 것의 직관: $v_{w_I}$와 $v'_{w_O}$의 내적이 로그 동시등장 통계를 근사하도록
밀어붙이는 것이다. 이 직관은 [6]절의 additive compositionality와, 이후 Levy & Goldberg의 PMI 분해 정리로
정확해진다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[2] Negative Sampling (NEG)</span>

#### <span style="color: #4682B4">2.1 Objective</span>

- (중심 $w_I$, 문맥 $w_O$) 쌍 하나에 대해, 노이즈 분포 $P_n(w)$에서 뽑은 $k$개의 가짜 단어와 함께
**$k+1$개의 이진 분류**를 푼다. softmax의 $\log p(w_O|w_I)$ 자리를 다음 식이 대체한다.

$$
    \log{\sigma\big({v'_{w_O}}^{\top}v_{w_I}\big)}
    + \sum_{i=1}^{k}{\mathbb{E}_{w_i \sim P_n(w)}\Big[ \log{\sigma\big(-{v'_{w_i}}^{\top}v_{w_I}\big)} \Big]}
$$

- 진짜 쌍의 내적은 크게(σ→1), 노이즈 쌍의 내적은 작게(σ(−z)→1) 만드는 것이 전부다.
분모 계산이 사라지고 업데이트가 **$O(k)$** 로 끝난다.
- $k$는 작은 데이터에서 5~20, 큰 데이터에서 2~5면 충분하다. ($k=5$면 쌍 하나당 벡터 6개만 업데이트)

#### <span style="color: #4682B4">2.2 노이즈 분포 — 3/4 제곱의 마법</span>

- 노이즈 분포 후보는 unigram $U(w)$, uniform, 그리고 그 사이 어딘가다. 실험 결과
**unigram의 3/4 제곱**이 두 극단을 일관되게 이겼다.

$$
    P_n(w) = \frac{U(w)^{3/4}}{Z}
$$

- 효과의 직관: 빈도 순위는 유지하면서 분포를 평탄화한다 — 고빈도 단어는 살짝 덜, 희귀 단어는 살짝 더
자주 negative로 뽑힌다. 희귀 단어의 출력 벡터도 충분한 "밀어내기" 신호를 받게 하는 조정이다.
- 왜 하필 3/4인지 이론적 설명은 없다(순수 경험적). 하지만 이후 구현들(gensim, fastText)이 전부
이 값을 기본값으로 굳혔고, 추천시스템 등 다른 도메인의 negative sampling에서도 관행처럼 이식됐다.

<details>
<summary> <span style="color: #ffd33d">NEG gradient 유도 (실제 업데이트 식) 펼치기/접기</span> </summary>

- 쌍 하나의 loss를 $L = -\log{\sigma(z_O)} - \sum_{i=1}^{k}{\log{\sigma(-z_i)}}$,
$z_w = {v'_w}^{\top}v_{w_I}$로 쓰자. $\sigma'(z) = \sigma(z)(1-\sigma(z))$와
$1-\sigma(z) = \sigma(-z)$를 쓰면

$$
    \frac{\partial L}{\partial z_O} = -\frac{\sigma(z_O)(1-\sigma(z_O))}{\sigma(z_O)} = \sigma(z_O) - 1
    ,\qquad
    \frac{\partial L}{\partial z_i} = \sigma(z_i)
$$

- 두 경우를 하나로 쓰면, 라벨 $t_w \in \{1, 0\}$ (진짜/노이즈)에 대해 오차항이 $\sigma(z_w) - t_w$인
**로지스틱 회귀의 표준 gradient**다. 업데이트는

$$
    v'_w \leftarrow v'_w - \eta\,\big(\sigma(z_w) - t_w\big)\,v_{w_I}
    \qquad (w \in \{w_O, w_1, ..., w_k\})
$$

$$
    v_{w_I} \leftarrow v_{w_I} - \eta\sum_{w}{\big(\sigma(z_w) - t_w\big)\,v'_w}
$$

- 쌍 하나당 내적 $k+1$번 + 벡터 덧셈 $k+2$번. 분기도 트리도 없어서 구현이 수십 줄이고,
비동기 병렬(HogWild 스타일 lock-free SGD)로도 잘 돈다. 이 단순함이 word2vec 툴킷이
어디서나 돌아가는 이유다. $\blacksquare$

</details>

<hr/> <!-- 수평선 -->

<details>
<summary> <span style="color: #ffd33d">NCE에서 NEG가 유도되는 과정 펼치기/접기</span> </summary>

- 출발점은 NCE(Noise Contrastive Estimation, Gutmann & Hyvärinen 2012; LM 적용은 Mnih & Teh 2012)다.
데이터 쌍 1개와 노이즈 쌍 $k$개가 섞여 있을 때, 어떤 쌍 $(w_I, w)$가 데이터에서 왔을 사후확률은

$$
    P(D=1|w, w_I) = \frac{p(w|w_I)}{p(w|w_I) + k\,P_n(w)}
$$

- NCE는 이 이진 분류의 log likelihood를 최대화하며, $k \rightarrow \infty$에서 원래 softmax LM의
MLE와 일치한다는 이론적 보장이 있다. 이때 $p(w|w_I)$ 자리에 모델의 **비정규화 점수**
$\exp({v'_w}^{\top}v_{w_I})$를 넣고 정규화 상수는 1로 취급해도 잘 동작한다는 것이 알려져 있었다.
- **NEG는 여기서 $k\,P_n(w)$ 항을 통째로 버린 단순화다.** 즉

$$
    P(D=1|w, w_I) = \sigma\big({v'_w}^{\top}v_{w_I}\big)
$$

로 두고 이진 cross-entropy를 최대화한다. 본문 objective가 정확히 이것이다.
- 대가: NCE가 갖던 "softmax MLE로의 수렴" 보장이 사라진다. 즉 **NEG는 더 이상 언어모델 학습이 아니다.**
논문의 입장은 명확하다 — 목적이 벡터 품질이라면 그 보장은 필요 없고, 실험적으로 NEG가 NCE보다
벡터 품질이 좋다. $\blacksquare$

</details>

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[3] Subsampling of Frequent Words</span>

- "the", "in" 같은 초고빈도 단어는 정보량이 적은데 학습 쌍의 대부분을 차지한다.
("France–the" 쌍은 "France–Paris" 쌍보다 훨씬 흔하지만 아무것도 안 가르쳐준다)
- 그리고 방향이 반대인 문제도 있다 — 고빈도 단어의 벡터는 수백만 번의 업데이트 후에는
**더 이상 변하지 않는다.** 그 뒤의 학습량은 순수 낭비다.
- 해법: 각 단어를 다음 확률로 **학습 전에 버린다**. ($f(w)$: 단어 빈도, 임계값 $t \approx 10^{-5}$)

$$
    P(discard\;|\;w) = 1 - \sqrt{\frac{t}{f(w)}}
$$

- $f(w) \le t$인 단어는 보존되고, 그 이상은 빈도의 제곱근에 반비례해 남는다. 빈도 순위는 유지하면서
분포만 평탄화하는 공격적인 heuristic이다.
- 효과가 이중이다:
  1. 학습해야 할 토큰 수 자체가 줄어 **2~10배 가속**.
  2. 고빈도 단어가 시퀀스에서 사라진 자리만큼 **유효 문맥 윈도우가 넓어진다** —
"the"가 빠진 자리 너머의 내용어(content word)들이 서로의 문맥이 되면서, 특히 희귀 단어 벡터의
품질이 정확도로 확인될 만큼 올라간다.
- 수치 예시: $f(\text{"the"}) \approx 0.07$, $t = 10^{-5}$이면 보존 확률 $\sqrt{t/f} \approx 1.2\%$.
"the"의 학습 쌍이 1/80로 줄어든다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[4] 실험 — 방법끼리 비교</span>

#### <span style="color: #4682B4">4.1 세팅</span>

- 데이터: 내부 Google News 코퍼스 **약 10억 단어**. 빈도 5회 미만 단어 제거 후 **vocabulary 692K**.
- 모델: Skip-gram 300차원, 윈도우 $c=5$, 1 epoch. 평가는 계보 ②의 analogy 테스트셋.
- 비교 대상: NEG($k$=5, 15), Hierarchical Softmax(Huffman), NCE($k$=5) × subsampling 유무.

#### <span style="color: #4682B4">4.2 결과 (word analogy 정확도)</span>

| 방법 | subsampling 없음 | subsampling $t=10^{-5}$ |
|---|---|---|
| NEG-5 | 59% 수준 | 61% 수준 |
| **NEG-15** | **61% 수준** | **61% 수준** |
| HS-Huffman | 47% 수준 | 55% 수준 |
| NCE-5 | 53% 수준 | - |

- 관찰 정리:
  - **NEG > NCE**: 이론 보장을 버린 단순화가 벡터 품질에서는 오히려 낫다.
  - **NEG > HS**: 같은 시간이면 NEG가 일관되게 우세. (단 아래 phrase 실험에서는 역전이 있다)
  - **subsampling은 모두에게 이득**이고 학습 시간도 절반 이하로 준다. 특히 HS의 개선폭이 크다.
- 학습 속도 관점: NEG-5 + subsampling 조합이면 10억 단어 코퍼스가 **단일 머신 멀티스레드로 하루 미만**.
계보 ②에서 분산 인프라(DistBelief)가 필요하던 규모가 노트북 수준으로 내려온 것이다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[5] Phrase 학습</span>

#### <span style="color: #4682B4">5.1 Phrase 추출</span>

- "New York Times"는 "New" + "York" + "Times"의 합성으로 의미가 안 나온다. 이런 구를 데이터 기반으로 찾아서
**하나의 토큰으로 치환** 후 같은 알고리즘으로 학습한다.

$$
    score(w_i, w_j) = \frac{count(w_i w_j) - \delta}{count(w_i) \times count(w_j)}
$$

- 함께 등장하는 빈도가 각자 등장 빈도 대비 높으면(임계값 초과) 구로 병합한다. $\delta$는 초저빈도 쌍이
뽑히는 것을 막는 discount 항이다.
- 임계값을 낮춰가며 **2~4회 반복** 실행하면 이미 병합된 구가 다시 병합되어 더 긴 구
("san francisco giants")도 만들어진다.
- 이렇게 학습한 phrase까지 포함한 vocabulary는 수백만 토큰 규모가 된다.

#### <span style="color: #4682B4">5.2 Phrase Analogy 평가</span>

- phrase 버전 analogy 테스트셋을 새로 만들었다. 예: "New York : New York Times = Baltimore : Baltimore Sun",
"Boston : Boston Bruins = Montreal : Montreal Canadiens".
- 3만 개 이상 자주 등장하는 phrase로 구성. 결과 (300차원, 10억 단어):

| 방법 | subsampling 없음 | subsampling $t=10^{-5}$ |
|---|---|---|
| NEG-5 | 24% 수준 | 27% 수준 |
| NEG-15 | 27% 수준 | 42% 수준 |
| **HS-Huffman** | 19% 수준 | **47% 수준** |

- 재밌는 역전: 단어 실험에서 밀리던 **HS가 subsampling과 결합하면 phrase에서는 최강**이 된다.
어떤 방법이 절대 우위가 아니라 태스크/세팅 의존적이라는 정직한 보고.
- 스케일을 밀어붙여 **33B 단어 + 1000차원**으로 학습하면 phrase analogy 정확도가 **72%** 까지 오른다.
데이터 스케일이 여전히 가장 큰 지렛대라는 결론.

#### <span style="color: #4682B4">5.3 Phrase 최근접 이웃 예시</span>

| Phrase | 최근접 이웃 |
|---|---|
| New York Times | Baltimore Sun, Toronto Star, LA Times |
| Steve Ballmer | Larry Page, Sergey Brin (동종 CEO/창업자들) |
| Boston Bruins | Montreal Canadiens, Buffalo Sabres (같은 리그 팀들) |
| Air Canada | Lufthansa, Delta Air Lines |

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[6] Additive Compositionality</span>

- 단어 벡터의 **덧셈이 의미의 AND처럼** 동작하는 현상을 정리했다.

| 벡터 합 | 최근접 결과 |
|---|---|
| vec(Russia) + vec(river) | Volga River |
| vec(Germany) + vec(capital) | Berlin |
| vec(Czech) + vec(currency) | koruna |
| vec(Vietnam) + vec(capital) | Hanoi |
| vec(French) + vec(actress) | Juliette Binoche |

- 왜 되는가에 대한 논문의 설명: Skip-gram의 학습 목표 때문에 단어 벡터는 **그 단어의 문맥 분포의 로그**와
선형적으로 연결된다. 벡터 덧셈은 로그의 덧셈 = **두 문맥 분포의 곱**에 대응하고, 분포의 곱은
"두 단어 모두와 잘 어울리는 문맥"만 남기는 AND 필터처럼 동작한다.
  - "Volga River"는 "Russian"의 문맥과 "river"의 문맥 양쪽에서 자주 등장하므로 곱에서 살아남는다.
- 이 직관은 이듬해 Levy & Goldberg에 의해 정리로 승격된다 (아래 계보 참고).

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[7] 기존 공개 임베딩과의 비교</span>

- 당시 공개돼 있던 임베딩들(Collobert & Weston, Turian, Mnih의 벡터)과 희귀 단어 포함 최근접 이웃을
정성 비교한다. 예를 들어 "redfish", "czarist" 같은 저빈도 단어에서:
  - 기존 임베딩: 관련성이 약한 이웃들이 섞여 나옴.
  - **SGNS(30B 단어 학습, phrase 포함)**: "czarist → czar, tsarist, imperial russia" 식으로 일관된 이웃.
- 포인트는 모델 구조의 우위 주장이 아니라 **"단순한 모델로 30B 단어를 학습할 수 있다는 것 자체"** 가
품질 우위의 원천이라는 것. 기존 방법들은 그 규모의 데이터를 물리적으로 소화할 수 없었다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[8] 계보 — 어디서 왔고 어디로 갔나</span>

#### <span style="color: #4682B4">영향 받은 것</span>

- **← [word2vec 계보 ② (2013.1)](/posts/efficient-estimation-of-word-representations-in-vector-space/)**:
Skip-gram 구조와 "벡터가 목적" 관점을 그대로 가져와서, ②의 hierarchical softmax를 NEG로 교체하고
subsampling/phrase를 얹었다. ②의 "표현력 낮춰서 데이터 키우기" 노선을 한 단계 더 밀어붙인 것.
- **← NCE (Gutmann & Hyvärinen 2012, Mnih & Teh 2012)**: NEG의 이론적 모체. [2]절 접기 증명 참고.
- **← [Mikolov 박사논문 (2012)](/posts/statistical-language-models-based-on-neural-networks/)**:
"softmax 정규화가 병목"이라는 문제의식의 최종 해법이 이 논문의 NEG다.
(class 분해 → hierarchical softmax → 정규화 포기(NEG) 순서로 진화한 셈)

#### <span style="color: #4682B4">후속 연구에 준 영향</span>

- **→ 이론적 재해석 (Levy & Goldberg, NIPS 2014)**: SGNS가 암묵적으로 **shifted PMI 행렬을 분해**하는 것과
동치임이 증명됐다.

$$
    {v'_c}^{\top}v_w = PMI(w, c) - \log{k} = \log{\frac{P(w,c)}{P(w)P(c)}} - \log{k}
$$

  신경망 임베딩과 전통 카운트 기반 방법(LSA 계열)이 같은 뿌리라는 것이 밝혀지면서
임베딩 이론 연구의 문을 열었다. [6]절의 "덧셈 = 분포 곱" 직관도 이 틀에서 정확해진다.
- **→ GloVe (Pennington et al. 2014)**: 전역 co-occurrence 통계를 명시적으로 쓰는 대안으로 등장,
word2vec과 함께 정적 임베딩의 양대 표준이 됐다.
- **→ [word2vec 계보 ④ fastText (2016-17)](/posts/fasttext-subword-information-and-bag-of-tricks/)**:
SGNS의 objective를 **그대로 유지**한 채 단어 벡터를 문자 n-gram 벡터의 합으로 바꾼 직계 후속작.
- **→ everything2vec**: "co-occurrence가 있는 곳엔 SGNS를 쓸 수 있다"는 레시피가 도메인을 넘어 퍼졌다 —
doc2vec(문서), item2vec(추천), node2vec/DeepWalk(그래프), prod2vec(상품) 등.
negative sampling 자체도 추천/검색/contrastive learning의 표준 부품이 됐다.
- **→ 문맥적 임베딩으로**: 정적 벡터의 한계(다의어가 벡터 하나로 뭉개짐)가 ELMo, 그리고
[Transformer](/posts/attention-is-all-you-need/) 기반 BERT/GPT의 문맥적 표현으로 이어지는 동기가 된다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">Reference</span>

- [1] T. Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (ICLR Workshop 2013)
- [2] M. Gutmann & A. Hyvärinen, "Noise-contrastive estimation of unnormalized statistical models" (JMLR 2012)
- [3] A. Mnih & Y. Teh, "A fast and simple algorithm for training neural probabilistic language models" (ICML 2012)
- [4] O. Levy & Y. Goldberg, "Neural Word Embedding as Implicit Matrix Factorization" (NIPS 2014)
- [5] J. Pennington et al., "GloVe: Global Vectors for Word Representation" (EMNLP 2014)
