---
title: "[word2vec 계보 ②] Efficient Estimation of Word Representations in Vector Space (word2vec, 2013.1)"
date: 2026-08-21
category: deep-learning-paper
tags:
  - "NLP"
  - "word2vec"
  - "Word Embedding"
  - "CBOW"
  - "Skip-gram"
---

<span style="font-size:17pt">
<b>Efficient Estimation of Word Representations in Vector Space</b>
</span>

<a href="https://arxiv.org/abs/1301.3781" target="_blank"><b>[PDF]</b></a>
, <b><span style="color: #F2AA4C">Word Embedding</span></b>, Tomáš Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean (Google, ICLR 2013 Workshop)

### <span style="color: #ffd33d">Summary</span>

**word2vec의 첫번째 논문.** CBOW와 Skip-gram을 처음 제안했다. 핵심 발상의 전환은 두 가지다.

1. **임베딩을 부산물이 아니라 목표로**: 언어모델을 잘 만들려다 벡터를 얻는 게 아니라,
좋은 단어 벡터 자체를 최대한 싸게 배우는 것이 목적이다. 그래서 품질 평가도 perplexity가 아니라
**단어 유추(analogy) 테스트** — vector("King") − vector("Man") + vector("Woman") ≈ vector("Queen") — 로 한다.
2. **비선형 hidden layer 제거**: [Mikolov 박사논문](/posts/statistical-language-models-based-on-neural-networks/)의
복잡도 분석에 따르면 NNLM의 비용 지배항은 hidden layer 곱셈과 softmax다.
둘 다 제거/완화한 **log-linear 모델**(CBOW, Skip-gram) + hierarchical softmax로,
같은 하드웨어에서 다룰 수 있는 데이터가 수백 배 커진다 (1.6B 단어를 하루 안에 학습).

결과적으로 "표현력이 낮은 모델 × 훨씬 많은 데이터"가 "표현력 높은 모델 × 적은 데이터"를 이긴다는 것을 보였고,
이 트레이드오프 감각이 이후 NLP 전반의 상식이 된다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[1] Model Complexity 프레임워크</span>

- 논문은 모든 모델의 학습 비용을 다음 형태로 통일해서 비교한다. ($E$: epoch 수, $T$: 토큰 수, $Q$: 토큰당 연산량)

$$
    O = E \times T \times Q
$$

- 기존 모델들의 $Q$를 계산해보면 병목이 어디인지 명확해진다. ($N$: 윈도우, $D$: 임베딩 차원, $H$: hidden, $V$: vocab)

| 모델 | $Q$ (토큰당 연산량) | 지배항 |
|---|---|---|
| NNLM | $N \times D + N \times D \times H + H \times V$ | $N \times D \times H$ (hierarchical softmax 적용 시) |
| RNNLM | $H \times H + H \times V$ | $H \times H$ (〃) |
| **CBOW** | $N \times D + D \times \log_2{V}$ | — |
| **Skip-gram** | $C \times (D + D \times \log_2{V})$ | — |

- $H \times V$ 항은 hierarchical softmax로 $H \times \log_2{V}$까지 줄일 수 있으므로($V = 10^6$이어도 $\log_2 V = 20$),
남는 병목은 **비선형 hidden layer**다. 새 모델 둘은 이걸 아예 제거한다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[2] 제안 모델</span>

#### <span style="color: #4682B4">2.1 CBOW (Continuous Bag-of-Words)</span>

- **주변 단어들로 중심 단어를 예측**한다. 앞뒤 $N/2$개씩의 문맥 단어 임베딩을 **평균**내고
(순서 무시 — 그래서 bag-of-words), 그 평균 벡터로 중심 단어를 분류한다.

$$
    \bar{v} = \frac{1}{N}\sum_{-N/2 \le j \le N/2,\; j \ne 0}{v_{w_{t+j}}}
    ,\qquad
    P(w_t\,|\,context) = \frac{\exp\big({v'_{w_t}}^{\top}\bar{v}\big)}{\sum_{w=1}^{V}{\exp\big({v'_w}^{\top}\bar{v}\big)}}
$$

- projection(임베딩 평균) → 출력. 그 사이에 비선형이 없는 **log-linear 모델**이다.
- 입력 임베딩 $v$와 출력 임베딩 $v'$ 두 벌을 쓴다는 것도 포인트. (최종적으로는 $v$를 단어 벡터로 사용)

#### <span style="color: #4682B4">2.2 Skip-gram</span>

- CBOW의 반대 방향. **중심 단어로 주변 단어 각각을 예측**한다.

$$
    \frac{1}{T}\sum_{t=1}^{T}{\sum_{-C \le j \le C,\; j \ne 0}{\log{P(w_{t+j}\,|\,w_t)}}}
    ,\qquad
    P(w_O|w_I) = \frac{\exp\big({v'_{w_O}}^{\top}v_{w_I}\big)}{\sum_{w=1}^{V}{\exp\big({v'_w}^{\top}v_{w_I}\big)}}
$$

- 윈도우 $C$가 클수록 품질이 좋아지지만 비용이 늘어난다. 트릭: 문맥 거리 $c$를 $[1, C]$에서 랜덤 샘플링해서
**가까운 단어일수록 자주 학습**되게 한다 (거리에 반비례하는 가중치 효과).
- 경험적으로 CBOW보다 느리지만 **의미(semantic) 관계와 희귀 단어에 강하다.**
직관: CBOW는 문맥을 평균내며 뭉개지만 Skip-gram은 (중심, 문맥) 쌍 하나하나가 개별 학습 신호다.

#### <span style="color: #4682B4">2.3 Hierarchical Softmax</span>

- 분모의 $V$항 합을 피하기 위해 vocabulary를 이진 트리(여기서는 Huffman 트리)의 잎으로 배치하고,
단어 확률을 루트→잎 경로의 이진 결정 확률 곱으로 정의한다.

<details>
<summary> <span style="color: #ffd33d">hierarchical softmax 정의 + 확률 합이 1이 되는 증명 펼치기/접기</span> </summary>

- $n(w, j)$를 루트에서 단어 $w$까지 경로의 $j$번째 노드, $L(w)$를 경로 길이라 하자.
각 내부 노드가 벡터 $v'_n$을 가지고, 확률은

$$
    P(w|w_I) = \prod_{j=1}^{L(w)-1}{\sigma\Big( \big[\!\big[ n(w,j+1) = ch(n(w,j)) \big]\!\big] \cdot {v'_{n(w,j)}}^{\top} v_{w_I} \Big)}
$$

- $ch(n)$은 $n$의 (임의로 고정한) 왼쪽 자식이고, $[\![x]\!]$는 $x$가 참이면 $+1$, 거짓이면 $-1$이다.
즉 각 내부 노드에서 "왼쪽으로 갈 확률 $\sigma(z)$ / 오른쪽으로 갈 확률 $\sigma(-z)$"의 이진 분기다.
- **합이 1이 되는 이유**: $\sigma(z) + \sigma(-z) = 1$이므로 각 내부 노드에서 두 자식으로 가는 확률 합이 1이다.
잎 전체에 대한 합은 트리를 루트부터 재귀적으로 접으면

$$
    \sum_{w \in leaves}{P(w|w_I)} = \sum_{leaves}\prod_{path}{\sigma(\pm z_n)}
    = \prod_{root}\big(\sigma(z) + \sigma(-z)\big) \cdots = 1 \qquad \blacksquare
$$

- 계산량: 단어 하나당 경로 위의 $L(w)-1$개 노드만 업데이트하면 되고, **Huffman 트리**를 쓰면 빈도 높은 단어의
경로가 짧아져 기대 경로 길이가 $\log_2 V$보다도 작아진다 (엔트로피에 근접).
- 이 구조는 [Mikolov 박사논문](/posts/statistical-language-models-based-on-neural-networks/) 2.3의
class 분해($O(\sqrt{V})$)를 트리 깊이 극한($O(\log V)$)까지 밀어붙인 것이다.

</details>

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[3] 평가 — Analogy Task</span>

- 논문이 직접 만든 **Semantic-Syntactic Word Relationship test set**: 의미 관계 5종(수도-국가, 통화, 성별 등)
8,869문항 + 문법 관계 9종(비교급, 과거형, 복수형 등) 10,675문항.
- "a : b = c : ?"를 벡터 연산 $x = v_b - v_a + v_c$ 후 **cosine 최근접 단어**로 풀고, 정확히 맞아야 정답 처리.
- 주요 결과 (640차원 기준):
  - RNNLM 벡터보다 CBOW/Skip-gram이 **압도적으로 높은 유추 정확도** (특히 문법 관계).
  - **Skip-gram이 의미 관계에서 최강**, CBOW는 문법 관계에 상대적 강점.
  - 차원만 키우거나 데이터만 키우면 수확 체감 — **둘을 같이** 키워야 한다.
  - 1.6B 단어 학습이 하루 안에 끝난다 (DistBelief 분산 학습, Adagrad).
- 이 analogy 평가는 이후 임베딩 논문들의 표준 벤치마크가 된다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[4] 계보 — 어디서 왔고 어디로 갔나</span>

#### <span style="color: #4682B4">영향 받은 것</span>

- **← [Mikolov 박사논문 (2012)](/posts/statistical-language-models-based-on-neural-networks/)**:
  1. $O = E \times T \times Q$ 복잡도 분석은 박사논문의 "병목은 hidden × softmax" 진단을 그대로 계승 —
CBOW/Skip-gram은 그 진단에서 병목 항을 지워서 만든 모델이다.
  2. hierarchical softmax는 박사논문의 class 분해를 이진 트리로 일반화한 것.
  3. "단어 벡터에 규칙성이 있다"는 박사논문/NAACL 2013의 관찰이, 이 논문에서 아예 **평가 지표(analogy)** 로 승격됐다.
- **← Bengio NNLM (2003)**: 임베딩 + log-linear 출력이라는 뼈대. 이 논문은 여기서 비선형만 걷어낸 것에 가깝다.

#### <span style="color: #4682B4">후속 연구에 준 영향</span>

- **→ [word2vec 2편 (2013.10)](/posts/distributed-representations-of-words-and-phrases/)**: 같은 해에 저자들이 직접
Skip-gram을 개량한다 — hierarchical softmax를 **negative sampling**으로 교체하고, subsampling과 phrase 학습을 추가.
(자세한 것은 해당 리뷰에서)
- **→ GloVe (2014)**: "예측 기반(word2vec) vs 카운트 기반(LSA)" 논쟁을 촉발했고, GloVe는 그 절충으로 등장한다.
- **→ 사전학습 임베딩 시대**: "대규모 비지도 코퍼스로 표현을 먼저 배우고 다운스트림에 전이한다"는 워크플로를
NLP의 표준으로 만들었다. 이 흐름이 ELMo → BERT/GPT([Transformer](/posts/attention-is-all-you-need/) 기반)의
"문맥적 임베딩"으로 진화하는데, 그 출발점의 정적(static) 임베딩이 바로 이 논문이다.
- **→ 평가 문화**: analogy 테스트셋과 "벡터 산술" 데모는 임베딩 품질 평가의 표준이자,
표현학습을 대중적으로 알린 상징이 됐다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">Reference</span>

- [1] Y. Bengio et al., "A Neural Probabilistic Language Model" (JMLR 2003)
- [2] T. Mikolov, "Statistical Language Models Based on Neural Networks" (PhD Thesis, 2012)
- [3] T. Mikolov et al., "Linguistic Regularities in Continuous Space Word Representations" (NAACL 2013)
- [4] F. Morin & Y. Bengio, "Hierarchical Probabilistic Neural Network Language Model" (2005)
- [5] T. Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality" (NIPS 2013)
