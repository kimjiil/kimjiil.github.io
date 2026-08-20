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

**word2vec의 두번째 논문.** [계보 ②](/posts/efficient-estimation-of-word-representations-in-vector-space/)의
Skip-gram을 세 방향으로 확장해서, 오늘날 "word2vec"이라 불리는 알고리즘(SGNS)을 완성했다.

1. **Negative Sampling (NEG)**: hierarchical softmax의 단순 대안. NCE를 더 단순화해서
"진짜 (중심, 문맥) 쌍을 노이즈 $k$개에서 구분하는 로지스틱 회귀"로 objective를 바꾼다.
2. **빈번한 단어 Subsampling**: "the" 같은 단어를 확률적으로 건너뛰어 **2~10배 가속** + 희귀 단어
벡터 품질 향상.
3. **Phrase 벡터**: "Air Canada"는 "Air"와 "Canada"의 합성이 아니다 — 통계적으로 구를 찾아
단일 토큰으로 학습하고, phrase analogy 테스트셋(3,218문항)을 새로 만들었다.

여기에 **벡터 덧셈의 합성성**(vec(Russia) + vec(river) ≈ vec(Volga River))에 대한 설명까지 —
두 논문을 합쳐 인용수 수만 회에 달하는, NLP 역사상 가장 영향력 있는 논문 중 하나다.
리뷰는 논문 섹션 구성을 그대로 따라간다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[1] Introduction</span>

- Skip-gram[계보 ②]의 최대 장점 재확인: **dense 행렬곱이 없다.** 최적화된 단일 머신 구현이
**하루에 1000억 단어 이상**을 학습한다.
- 학습된 벡터의 선형 규칙성(vec("Madrid") − vec("Spain") + vec("France") ≈ vec("Paris"))을 다시 짚고,
이 논문의 기여 3가지를 예고한다: subsampling(2~10배 가속 + 희귀 단어 개선), NCE의 단순화 변형(NEG),
그리고 phrase 확장.
- phrase 문제의식이 명확하다: *"'Boston Globe'는 신문이지 'Boston'과 'Globe'의 의미의 자연스러운
조합이 아니다."* — 단어 단위 표현의 본질적 한계. 해법은 단순하게, **데이터 기반으로 구를 찾아
학습 데이터에서 단일 토큰으로 치환**하는 것.
- 마지막 예고: 벡터 **덧셈**만으로 의미있는 합성이 된다는 발견(additive compositionality) — 5장에서 설명.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[2] The Skip-gram Model</span>

- objective 복습. 문맥 크기 $c$ 안의 주변 단어들의 평균 log 확률을 최대화한다.

$$
    \frac{1}{T}\sum_{t=1}^{T}{\sum_{-c \le j \le c,\; j \ne 0}{\log{p(w_{t+j}|w_t)}}}
    ,\qquad
    p(w_O|w_I) = \frac{\exp\big({v'_{w_O}}^{\top}v_{w_I}\big)}{\sum_{w=1}^{W}{\exp\big({v'_w}^{\top}v_{w_I}\big)}}
$$

- 기본형(full softmax)은 $\nabla \log p$ 비용이 vocabulary 크기 $W$($10^5{\sim}10^7$)에 비례해서 비실용적 —
그래서 2.1, 2.2의 두 근사가 나온다.

#### <span style="color: #4682B4">2.1 Hierarchical Softmax</span>

- [계보 ②](/posts/efficient-estimation-of-word-representations-in-vector-space/)에서 쓰던 방법의 정식 서술.
이진 트리의 잎에 단어를 배치하고, 루트→잎 경로의 이진 결정 확률 곱으로 확률을 정의한다.

$$
    p(w|w_I) = \prod_{j=1}^{L(w)-1}{\sigma\Big( \big[\!\big[ n(w,j+1) = ch(n(w,j)) \big]\!\big] \cdot {v'_{n(w,j)}}^{\top} v_{w_I} \Big)}
$$

- 비용은 경로 길이 $L(w_O)$, 평균적으로 $\log{W}$ 이하. 표준 softmax와 달리 단어당 표현이
$v_w$ 하나 + 내부 노드당 $v'_n$ 하나다. (합이 1이 되는 증명과 gradient는
[계보 ② 리뷰의 접기](/posts/efficient-estimation-of-word-representations-in-vector-space/) 참고)
- **트리 구조가 성능을 좌우한다** — Mnih & Hinton의 트리 구축 연구를 인용하며, 이 논문은
빈도 기반 **Huffman 트리**를 쓴다 (빈번한 단어 = 짧은 코드 = 빠른 학습).

#### <span style="color: #4682B4">2.2 Negative Sampling — 이 논문의 대표 기여</span>

- 출발점은 NCE(Noise Contrastive Estimation): "좋은 모델은 로지스틱 회귀로 데이터와 노이즈를
구분할 수 있어야 한다."
- 핵심 문장: *"NCE는 softmax의 log 확률을 근사적으로 최대화하지만, **Skip-gram의 관심사는 고품질 벡터
표현뿐이므로, 벡터 품질이 유지되는 한 NCE를 마음대로 단순화해도 된다.**"* — 그렇게 나온 것이 NEG다.

$$
    \log{\sigma\big({v'_{w_O}}^{\top}v_{w_I}\big)}
    + \sum_{i=1}^{k}{\mathbb{E}_{w_i \sim P_n(w)}\Big[ \log{\sigma\big(-{v'_{w_i}}^{\top}v_{w_I}\big)} \Big]}
$$

- 이 식이 Skip-gram objective의 $\log P(w_O|w_I)$ 자리를 통째로 대체한다. 데이터 쌍 1개당
노이즈 분포 $P_n(w)$에서 뽑은 $k$개의 negative와 함께 이진 분류를 푼다.
- $k$의 실전값: **작은 데이터 5~20, 큰 데이터 2~5.**
- NCE와의 차이 (논문이 명시한 것):
  1. NCE는 노이즈 샘플과 함께 **노이즈 분포의 수치 확률값**도 필요하지만, NEG는 **샘플만** 쓴다.
  2. NCE의 "softmax MLE 근사" 성질은 사라진다 — "우리 응용에는 중요하지 않다."
- **노이즈 분포의 선택**: unigram $U(w)$의 **3/4 제곱** ($U(w)^{3/4}/Z$)이 unigram 원본과 uniform을
**모든 태스크에서** (보고 안 한 언어모델링 포함) 유의미하게 이겼다. 순수 경험적 선택.

<details>
<summary> <span style="color: #ffd33d">NEG gradient 유도 (실제 업데이트 식) 펼치기/접기</span> </summary>

- 쌍 하나의 loss를 $L = -\log{\sigma(z_O)} - \sum_{i=1}^{k}{\log{\sigma(-z_i)}}$,
$z_w = {v'_w}^{\top}v_{w_I}$로 쓰자. $\sigma'(z) = \sigma(z)(1-\sigma(z))$와 $1-\sigma(z) = \sigma(-z)$로

$$
    \frac{\partial L}{\partial z_O} = \sigma(z_O) - 1
    ,\qquad
    \frac{\partial L}{\partial z_i} = \sigma(z_i)
$$

- 라벨 $t_w \in \{1, 0\}$ (진짜/노이즈)로 통일하면 오차항이 $\sigma(z_w) - t_w$인
**로지스틱 회귀의 표준 gradient**다. 업데이트는

$$
    v'_w \leftarrow v'_w - \eta\,\big(\sigma(z_w) - t_w\big)\,v_{w_I}
    ,\qquad
    v_{w_I} \leftarrow v_{w_I} - \eta\sum_{w \in \{w_O, w_{1 \cdots k}\}}{\big(\sigma(z_w) - t_w\big)\,v'_w}
$$

- 쌍 하나당 내적 $k+1$번 + 벡터 갱신 $k+2$개. 트리도 분기도 없어 구현이 수십 줄이고
비동기 멀티스레드(HogWild식)로 잘 돈다 — word2vec 툴킷이 어디서나 도는 이유. $\blacksquare$

</details>

<hr/> <!-- 수평선 -->

#### <span style="color: #4682B4">2.3 Subsampling of Frequent Words</span>

- 문제의 양면:
  1. "France–Paris" 동시등장은 유익하지만 "France–the"는 거의 무익하다 — "the"는 모든 단어와
문장 안에서 동시등장하니까.
  2. 반대 방향도 있다: **고빈도 단어의 벡터는 수백만 예시 후에는 더 변하지 않는다** — 그 뒤는 낭비.
- 해법: 각 단어를 다음 확률로 학습 전에 버린다. ($f(w_i)$: 빈도, 임계값 $t \approx 10^{-5}$)

$$
    P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}
$$

- $f > t$인 단어를 공격적으로 줄이되 **빈도 순위는 보존**하는 식이다. heuristic하게 고른 식이라고
정직하게 밝히면서 — 학습을 가속하고 희귀 단어의 벡터 정확도까지 유의미하게 올린다는 것을
3장에서 실측으로 보인다.
- 수치 감각: $f(\text{"the"}) \approx 0.07$이면 보존 확률 $\sqrt{10^{-5}/0.07} \approx 1.2\%$ —
"the"의 학습 쌍이 1/80로 준다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[3] Empirical Results — 방법끼리 정면 비교</span>

- 셋업: 내부 Google News **10억 단어**, 빈도 5회 미만 제거 → **vocabulary 692K**.
Skip-gram 300차원, 문맥 $c=5$. 평가는 계보 ②의 analogy 태스크.
- 결과 (논문 Table 1 — 학습 시간까지 그대로):

| 방법 | 시간 [분] | Syntactic [%] | Semantic [%] | **Total [%]** |
|---|---|---|---|---|
| NEG-5 | 38 | 63 | 54 | 59 |
| NEG-15 | 97 | 63 | 58 | **61** |
| HS-Huffman | 41 | 53 | 40 | 47 |
| NCE-5 | 38 | 60 | 45 | 53 |
| *이하 $10^{-5}$ subsampling* | | | | |
| NEG-5 | **14** | 61 | 58 | 60 |
| NEG-15 | 36 | 61 | 61 | **61** |
| HS-Huffman | 21 | 52 | 59 | 55 |

- 읽는 포인트:
  - **NEG > NCE** (59 vs 53): 이론 보장을 버린 단순화가 벡터 품질에선 오히려 낫다.
  - **NEG > HS** (59~61 vs 47): 단어 analogy에서는 일관되게 NEG 우세.
  - **subsampling은 전원 이득**: 정확도가 오르면서 시간이 1/3로 준다 (NEG-5: 38분 → 14분).
특히 HS의 semantic이 40 → 59로 급등.
- 흥미로운 부연: "Skip-gram이 선형(log-linear)이라 선형 유추에 유리한 것 아니냐"는 반론에 대해 —
비선형인 RNN 벡터도 데이터가 커지면 이 태스크에서 좋아진다는 [계보 ②] 결과를 들어,
**"비선형 모델도 단어 표현의 선형 구조를 선호한다"** 고 답한다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[4] Learning Phrases</span>

- "New York Times", "Toronto Maple Leafs"는 단일 토큰으로 치환하고 "this is" 같은 흔한 bigram은
그대로 두는 기준: **unigram/bigram 카운트 기반 score.**

$$
    score(w_i, w_j) = \frac{count(w_i w_j) - \delta}{count(w_i) \times count(w_j)}
$$

- $\delta$는 초저빈도 단어들로 구가 만들어지는 것을 막는 discount. 임계값 초과 bigram을 구로 병합하고,
**임계값을 낮추며 2~4 pass 반복**하면 여러 단어짜리 구도 형성된다. (이론상 모든 n-gram으로 학습할 수도
있지만 메모리 때문에 이 방식을 택했다고 명시)
- **Phrase analogy 테스트셋** (3,218문항, 공개): 신문(New York : New York Times = Baltimore : Baltimore Sun),
NHL 팀, NBA 팀, 항공사(Austria : Austrian Airlines = Spain : Spainair), 기업 CEO(Steve Ballmer : Microsoft
= Larry Page : Google)의 5개 카테고리.

#### <span style="color: #4682B4">4.1 Phrase Skip-gram 결과</span>

- 같은 뉴스 데이터(~1B), 300차원, $c=5$. (논문 Table 3)

| 방법 | subsampling 없음 [%] | $10^{-5}$ subsampling [%] |
|---|---|---|
| NEG-5 | 24 | 27 |
| NEG-15 | 27 | 42 |
| **HS-Huffman** | 19 | **47** |

- **재밌는 역전**: 단어에서 밀리던 HS가 subsampling과 결합하면 phrase에서는 최강이 된다.
"subsampling은 빠를 뿐 아니라, 적어도 일부 상황에선 정확도도 올린다"의 재확인이자,
어떤 방법도 절대 우위가 아니라는 정직한 보고.
- 스케일 실험: **33B 단어 + HS + 1000차원 + 문장 전체를 문맥으로** → phrase analogy **72%**.
같은 세팅에서 6B로 줄이면 66% → **"결국 데이터 양이 결정적"**.
- 희귀 phrase의 최근접 이웃 정성 비교(논문 Table 4): HS+subsampling 모델이 가장 그럴듯하다
(예: "Vasco de Gama" → "Italian explorer", "chess master" → "Garry Kasparov").

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[5] Additive Compositionality</span>

- analogy(뺄셈)와 별개로, **원소별 덧셈**만으로도 의미가 합성된다. (논문 Table 5 — 합 벡터의 최근접 토큰)

| 벡터 합 | 최근접 결과 |
|---|---|
| Czech + currency | **koruna**, Check crown, Polish zolty |
| Vietnam + capital | **Hanoi**, Ho Chi Minh City, Viet Nam |
| German + airlines | airline Lufthansa, carrier Lufthansa |
| Russian + river | Moscow, **Volga River**, upriver |
| French + actress | **Juliette Binoche**, Vanessa Paradis |

- **왜 되는가** (논문의 설명을 그대로 따라가면):
  1. 단어 벡터는 softmax 비선형의 **입력과 선형 관계**에 있다.
  2. 벡터는 주변 단어를 예측하도록 학습되므로, **그 단어가 등장하는 문맥의 분포**를 표현한다.
  3. 이 값들은 출력층이 계산하는 확률과 **로그로** 연결되어 있으므로, 두 벡터의 **합**은
두 문맥 분포의 **곱**에 대응한다.
  4. 분포의 곱은 **AND 함수**처럼 동작한다 — 두 벡터 모두가 높은 확률을 주는 단어만 살아남는다.
"Volga River"는 "Russian"과도 "river"와도 같은 문장에 자주 나오므로 곱에서 살아남는다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[6] Comparison to Published Word Representations</span>

- 공개돼 있던 유명 벡터들과 **희귀 단어의 최근접 이웃**을 정성 비교한다. (논문 Table 6 발췌)

| 모델 (학습 시간) | "Redmond" | "ninjutsu" | "capitulate" |
|---|---|---|---|
| Collobert 50d (2개월) | conyers, lubbock | reiki, kohona | abdicate, accede |
| Turian 200d (수 주) | McCarthy, Alston | - (미등재) | - (미등재) |
| Mnih 100d (7일) | Podhurst, Harlang | - | hesitated |
| **Skip-Phrase 1000d (1일)** | **Redmond Wash., Microsoft** | **ninja, martial arts** | **capitulation, capitulated** |

- 30B 단어(기존 대비 2~3 자릿수 많은 데이터)를 **하루**에 학습한 모델이 이웃 품질에서 압도한다.
포인트는 구조 우월성 주장이 아니라 — **"그 규모의 데이터를 소화할 수 있다는 것 자체"** 가 품질의
원천이라는 것.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[7] Conclusion</span>

- 기여 요약: Skip-gram의 phrase 확장 + 선형 구조 실증, 몇 자릿수 큰 데이터 학습(→ 희귀 entity에서
특히 개선), subsampling(속도+희귀 단어 품질), 그리고 NEG("특히 빈번한 단어에 정확한, 극단적으로
단순한 학습법").
- 실전 조언: 최적 세팅은 태스크마다 다르다 — 가장 중요한 결정은 **아키텍처, 벡터 차원, subsampling
비율, 윈도우 크기**.
- 이 논문의 기법들은 CBOW에도 그대로 적용 가능하고, 전부 **오픈소스(word2vec)로 공개** —
"phrase 토큰화 + 벡터 덧셈"의 조합은 최소 비용으로 긴 텍스트를 표현하는 강력한 방법이며
recursive 행렬 연산 계열(Socher)과 상보적이라고 마무리한다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">계보 — 어디서 왔고 어디로 갔나</span>

#### <span style="color: #4682B4">영향 받은 것</span>

- **← [계보 ② (2013.1)](/posts/efficient-estimation-of-word-representations-in-vector-space/)**:
Skip-gram 구조·analogy 평가·Huffman HS를 그대로 이어받아, HS를 NEG로 교체하고
subsampling/phrase를 얹었다. "벡터 품질만 유지되면 뭐든 단순화한다"는 ②의 노선을 한 단계 더 밀었다.
- **← NCE (Gutmann & Hyvärinen 2012, Mnih & Teh 2012)**: NEG의 이론적 모체. 논문 스스로
Collobert & Weston의 ranking(hinge) loss와의 유사성도 언급한다.
- **← [계보 ① 박사논문 (2012)](/posts/statistical-language-models-based-on-neural-networks/)**:
"softmax 정규화가 병목"이라는 문제의식의 종착지가 NEG다.
(class 분해 → hierarchical softmax → **정규화 포기(NEG)** 순서의 진화)

#### <span style="color: #4682B4">후속 연구에 준 영향</span>

- **→ 이론적 재해석 (Levy & Goldberg, NIPS 2014)**: SGNS가 암묵적으로 **shifted PMI 행렬을 분해**하는
것과 동치임이 증명됐다.

$$
    {v'_c}^{\top}v_w = PMI(w, c) - \log{k} = \log{\frac{P(w,c)}{P(w)P(c)}} - \log{k}
$$

  신경망 임베딩과 카운트 기반 방법(LSA 계열)이 같은 뿌리임이 밝혀지며 임베딩 이론 연구가 열렸고,
5장의 "덧셈 = 분포 곱" 직관도 이 틀에서 정확해진다.
- **→ GloVe (2014)**: 전역 co-occurrence를 명시적으로 쓰는 대안으로 등장, 정적 임베딩의 양대 표준.
- **→ [계보 ④ fastText (2016-17)](/posts/fasttext-subword-information-and-bag-of-tricks/)**:
SGNS objective를 그대로 두고 단어 벡터만 문자 n-gram 합으로 바꾼 직계 후속.
- **→ everything2vec**: "co-occurrence가 있으면 SGNS"라는 레시피가 도메인을 넘었다 — doc2vec,
item2vec(추천), node2vec/DeepWalk(그래프) 등. negative sampling 자체도 추천/검색/contrastive
learning의 표준 부품이 됐다.
- **→ 문맥적 임베딩으로**: 정적 벡터의 한계(다의어가 벡터 하나로 뭉개짐)가 ELMo, 그리고
[Transformer](/posts/attention-is-all-you-need/) 기반 BERT/GPT로 이어지는 동기가 된다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">Reference</span>

- [1] T. Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (ICLR Workshop 2013)
- [2] M. Gutmann & A. Hyvärinen, "Noise-contrastive estimation of unnormalized statistical models" (JMLR 2012)
- [3] A. Mnih & Y. Teh, "A fast and simple algorithm for training neural probabilistic language models" (ICML 2012)
- [4] F. Morin & Y. Bengio, "Hierarchical Probabilistic Neural Network Language Model" (AISTATS 2005)
- [5] O. Levy & Y. Goldberg, "Neural Word Embedding as Implicit Matrix Factorization" (NIPS 2014)
- [6] J. Pennington et al., "GloVe: Global Vectors for Word Representation" (EMNLP 2014)
