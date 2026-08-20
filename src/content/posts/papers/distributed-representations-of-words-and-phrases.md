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

**word2vec의 두번째 논문.** [1편](/posts/efficient-estimation-of-word-representations-in-vector-space/)에서 제안한
Skip-gram을 3가지로 개량해서, 오늘날 "word2vec"이라고 부르는 알고리즘(SGNS: Skip-gram with Negative Sampling)을 완성했다.

1. **Negative Sampling**: hierarchical softmax를 버리고, "진짜 (중심, 문맥) 쌍과 노이즈 쌍을 구분하는
이진 분류 $k$개"로 objective를 대체. 더 빠르고 벡터 품질도 더 좋다.
2. **빈번한 단어 Subsampling**: "the" 같은 단어를 확률적으로 건너뛰어 2~10배 가속 + 희귀 단어 벡터 품질 향상.
3. **Phrase 벡터**: "New York Times"처럼 합성이 안 되는 구(phrase)를 통계적으로 찾아 하나의 토큰으로 학습.

두 논문을 합쳐 인용수가 수만 회에 달하는, NLP 역사상 가장 영향력 있는 논문 중 하나다.
"벡터 덧셈으로 의미가 합성된다"(compositionality)는 관찰도 이 논문에서 정리됐다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[1] Skip-gram 복습과 문제 설정</span>

- Skip-gram objective (1편과 동일):

$$
    \frac{1}{T}\sum_{t=1}^{T}{\sum_{-c \le j \le c,\; j \ne 0}{\log{p(w_{t+j}|w_t)}}}
    ,\qquad
    p(w_O|w_I) = \frac{\exp\big({v'_{w_O}}^{\top}v_{w_I}\big)}{\sum_{w=1}^{V}{\exp\big({v'_w}^{\top}v_{w_I}\big)}}
$$

- 문제는 여전히 분모다. full softmax의 $\nabla \log p(w_O|w_I)$ 계산은 $O(V)$이고,
1편의 hierarchical softmax는 이걸 $O(\log V)$로 줄였지만 여전히 트리 구조에 품질이 좌우된다.
- 이 논문의 관점 전환: **애초에 정규화된 확률분포가 필요한가?** 목적이 언어모델이 아니라 좋은 벡터라면,
"진짜 쌍을 노이즈에서 구분"만 잘해도 된다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[2] Negative Sampling (NEG)</span>

- (중심 $w_I$, 문맥 $w_O$) 쌍 하나에 대해, 노이즈 분포 $P_n(w)$에서 뽑은 $k$개의 가짜 단어와 함께
**$k+1$개의 이진 분류**를 푼다.

$$
    \log{\sigma\big({v'_{w_O}}^{\top}v_{w_I}\big)}
    + \sum_{i=1}^{k}{\mathbb{E}_{w_i \sim P_n(w)}\Big[ \log{\sigma\big(-{v'_{w_i}}^{\top}v_{w_I}\big)} \Big]}
$$

- 진짜 쌍의 내적은 크게(σ→1), 노이즈 쌍의 내적은 작게(σ(−z)→1) 만드는 것이 전부다.
분모 계산이 사라지고 업데이트가 **$O(k)$** 로 끝난다.
- $k$는 작은 데이터에서 5~20, 큰 데이터에서 2~5면 충분하다.
- 노이즈 분포는 unigram 분포의 **3/4 제곱** $P_n(w) \propto U(w)^{3/4}$이 uniform이나 unigram 원본보다
일관되게 좋았다 (경험적 선택 — 빈도 순위를 유지하면서 희귀 단어를 살짝 더 자주 뽑는 효과).

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
- 각 단어를 다음 확률로 **학습 전에 버린다**. ($f(w)$: 단어 빈도, 임계값 $t \approx 10^{-5}$)

$$
    P(discard\;|\;w) = 1 - \sqrt{\frac{t}{f(w)}}
$$

- $f(w) \le t$인 단어는 보존되고, 그 이상은 빈도의 제곱근에 반비례해 남는다. 빈도 순위는 유지하면서
분포만 평탄화하는 공격적인 heuristic.
- 효과가 이중이다: (1) 학습량이 줄어 **2~10배 가속**, (2) 고빈도 단어가 사라진 자리만큼 **유효 문맥 윈도우가 넓어져**
희귀 단어 벡터의 품질이 오히려 올라간다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[4] Phrase 학습과 Compositionality</span>

- "New York Times"는 "New" + "York" + "Times"의 합성으로 의미가 안 나온다. 이런 구를 데이터 기반으로 찾아서
**하나의 토큰으로 치환** 후 같은 알고리즘으로 학습한다.

$$
    score(w_i, w_j) = \frac{count(w_i w_j) - \delta}{count(w_i) \times count(w_j)}
$$

- 함께 등장하는 빈도가 각자 등장 빈도 대비 높으면 구로 병합한다. $\delta$는 초저빈도 쌍이 뽑히는 것을 막는
discount. 2~4회 반복 실행하면 더 긴 구("san francisco giants")도 만들어진다.
- **Additive Compositionality**: 단어 벡터의 덧셈이 의미의 AND처럼 동작하는 현상을 정리했다.
  - 예: vec("Russia") + vec("river") ≈ vec("Volga River"), vec("Germany") + vec("capital") ≈ vec("Berlin")
  - 직관: Skip-gram 벡터는 문맥 분포의 로그와 선형적으로 연결되어 있어서, 벡터 합 = 두 문맥 분포의 곱(교집합)에
가까운 의미가 된다.
- Phrase analogy 테스트(예: "New York : New York Times = Baltimore : Baltimore Sun")에서
33B 단어 학습 시 정확도 72%까지 도달.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[5] Results 요약</span>

- 같은 데이터/시간이면 **NEG가 hierarchical softmax보다 analogy 정확도가 높고 빠르다.**
(단 phrase 실험에서 subsampling 결합 시에는 HS도 최상위권으로 올라옴 — 세팅에 따라 갈린다)
- subsampling은 속도 2~10배 + 정확도 향상을 동시에 달성.
- 이 조합(Skip-gram + NEG + subsampling)이 사실상의 표준 word2vec이 되어 공개 구현(word2vec 툴킷)으로 배포됐다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[6] 계보 — 어디서 왔고 어디로 갔나</span>

#### <span style="color: #4682B4">영향 받은 것</span>

- **← [word2vec 1편 (2013.1)](/posts/efficient-estimation-of-word-representations-in-vector-space/)**:
Skip-gram 구조와 "벡터가 목적" 관점을 그대로 가져와서, 1편의 hierarchical softmax를 NEG로 교체하고
subsampling/phrase를 얹었다. 1편의 "표현력 낮춰서 데이터 키우기" 노선을 한 단계 더 밀어붙인 것.
- **← NCE (Gutmann & Hyvärinen 2012, Mnih & Teh 2012)**: NEG의 이론적 모체. 위 접기 증명 참고.
- **← [Mikolov 박사논문 (2012)](/posts/statistical-language-models-based-on-neural-networks/)**:
"softmax 정규화가 병목"이라는 문제의식의 최종 해법이 이 논문의 NEG다.
(class 분해 → hierarchical softmax → 정규화 포기(NEG) 순서로 진화)

#### <span style="color: #4682B4">후속 연구에 준 영향</span>

- **→ 이론적 재해석 (Levy & Goldberg, NIPS 2014)**: SGNS가 암묵적으로 **shifted PMI 행렬
$PMI(w,c) - \log k$를 분해**하는 것과 동치임이 증명됐다. 신경망 임베딩과 전통 카운트 기반 방법(LSA 계열)이
같은 뿌리라는 것이 밝혀지면서 임베딩 이론 연구의 문을 열었다.
- **→ GloVe (Pennington et al. 2014)**: 전역 co-occurrence 통계를 명시적으로 쓰는 대안으로 등장,
word2vec과 함께 정적 임베딩의 양대 표준이 됐다.
- **→ [fastText (2016-17)](/posts/fasttext-subword-information-and-bag-of-tricks/)**: SGNS의 objective를
**그대로 유지**한 채 단어 벡터를 문자 n-gram 벡터의 합으로 바꾼 직계 후속작. (다음 리뷰에서)
- **→ everything2vec**: "co-occurrence가 있는 곳엔 SGNS를 쓸 수 있다"는 레시피가 도메인을 넘어 퍼졌다 —
doc2vec, item2vec(추천), node2vec/DeepWalk(그래프), prod2vec 등.
negative sampling 자체도 추천/검색/contrastive learning의 표준 부품이 됐다.
- **→ 문맥적 임베딩으로**: 정적 벡터의 한계(다의어 처리 불가)가 ELMo, 그리고
[Transformer](/posts/attention-is-all-you-need/) 기반 BERT/GPT의 문맥적 표현으로 이어지는 동기가 된다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">Reference</span>

- [1] T. Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (ICLR Workshop 2013)
- [2] M. Gutmann & A. Hyvärinen, "Noise-contrastive estimation of unnormalized statistical models" (JMLR 2012)
- [3] A. Mnih & Y. Teh, "A fast and simple algorithm for training neural probabilistic language models" (ICML 2012)
- [4] O. Levy & Y. Goldberg, "Neural Word Embedding as Implicit Matrix Factorization" (NIPS 2014)
- [5] J. Pennington et al., "GloVe: Global Vectors for Word Representation" (EMNLP 2014)
