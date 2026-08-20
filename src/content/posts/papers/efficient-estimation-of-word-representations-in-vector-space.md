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

### <span style="color: #ffd33d">[1] Introduction</span>

- 당시 NLP 시스템 대부분은 단어를 원자적 심볼(vocabulary의 인덱스)로 취급했다. 단어 사이의 유사성 개념이 없다.
  - 이 선택에도 이유는 있었다 — 단순함, 강건함, 그리고 **"많은 데이터 위의 단순한 모델이 적은 데이터 위의
복잡한 모델을 이긴다"** 는 관찰 (n-gram의 성공).
  - 하지만 한계가 명확한 영역이 있다. 음성인식/기계번역용 도메인 데이터는 크기가 제한적이어서,
단순히 데이터를 더 붓는 것으로는 개선이 안 된다. **표현 자체가 일반화를 도와야 한다.**
- 신경망 기반 분산 표현(distributed representation)은 이미 NNLM[1]과
[RNNLM](/posts/statistical-language-models-based-on-neural-networks/)에서 n-gram을 이기고 있었다.
문제는 **비용** — 기존 구조로는 수십억 단어 코퍼스, 수백만 vocabulary를 다룰 수 없었다.
- 이 논문의 목표 선언:
  1. **수십억 단어, 수백만 vocabulary**에서 고품질 단어 벡터를 학습하는 기법을 만든다.
  2. 품질은 "비슷한 단어가 가깝다"를 넘어 **다중의 유사도(multiple degrees of similarity)** 로 측정한다 —
단어는 의미적으로도, 문법적으로도(명사 어미, 시제 등) 동시에 비슷할 수 있다.
  3. **벡터 산술의 규칙성**을 평가 지표로 승격한다: vector("King") − vector("Man") + vector("Woman")이
vector("Queen")에 가장 가까우면 정답.
- 선행 연구 정리: NNLM(Bengio 2003)은 임베딩+비선형 hidden으로 LM을 학습, 이후 "임베딩을 먼저 배우고
NNLM을 그 위에 얹는" 2단계 접근(Collobert & Weston 2008 포함)이 나왔다. 이 논문은 그 첫 단계
"임베딩 학습"만을 떼어내 극단적으로 단순화/확장한 것이다. LSA/LDA 같은 카운트 기반 방법과 비교하면,
LSA는 벡터 산술 규칙성이 약하고 LDA는 큰 데이터에서 비용이 폭발한다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[2] Model Architectures — 복잡도 프레임워크</span>

- 논문은 모든 모델의 학습 비용을 다음 형태로 통일해서 비교한다. ($E$: epoch 수, $T$: 토큰 수, $Q$: 토큰당 연산량)

$$
    O = E \times T \times Q
$$

- 보통 $E = 3{\sim}50$, $T$는 최대 10억 이상. 모든 모델은 SGD + backprop으로 학습한다.
- 목표는 **accuracy를 유지하면서 $Q$를 극단적으로 줄이는 것**이다.

#### <span style="color: #4682B4">2.1 NNLM의 Q</span>

- 구조: 이전 $N$개 단어의 임베딩($D$차원)을 concat → hidden($H$) → softmax($V$).

$$
    Q = N \times D + N \times D \times H + H \times V
$$

- 각 항의 크기감(전형적으로 $N=10$, $D=500{\sim}2000$, $H=500{\sim}1000$):
  - $N \times D$: projection lookup, 미미함.
  - $N \times D \times H$: **hidden layer 곱셈. 전형적 세팅에서 수백만 연산/토큰.**
  - $H \times V$: naive하게는 최대 항이지만, hierarchical softmax로 $H \times \log_2{V}$까지 줄일 수 있다.
- 즉 softmax를 계층화하고 나면 **남는 병목은 비선형 hidden layer**다.

#### <span style="color: #4682B4">2.2 RNNLM의 Q</span>

- [박사논문](/posts/statistical-language-models-based-on-neural-networks/)의 모델. projection layer가 없고
hidden이 recurrent로 연결된다.

$$
    Q = H \times H + H \times V
$$

- 마찬가지로 $H \times V$는 계층화로 $H \times \log_2{V}$가 되고, **병목은 $H \times H$** recurrent 곱셈이다.

#### <span style="color: #4682B4">2.3 병렬 학습 인프라</span>

- Google의 분산 학습 프레임워크 **DistBelief** 위에서 같은 모델의 replica를 50~100개 띄우고,
중앙 파라미터 서버에 비동기 gradient 업데이트를 보내는 방식으로 학습했다.
- optimizer는 **Adagrad** (adaptive learning rate) + mini-batch 비동기 SGD.
- 이 인프라 덕분에 "6B 단어 코퍼스 × 1000차원" 같은 실험이 가능했다. (DistBelief는 이후 TensorFlow의 전신)

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[3] New Log-linear Models</span>

- 설계 원리: **비선형 hidden layer를 포기**하는 대신 데이터를 훨씬 많이 먹인다.
"신경망보다 정밀하게 표현하지는 못해도, 훨씬 많은 데이터를 효율적으로 학습할 수 있는 단순한 모델"이라는
명시적 트레이드오프 선택이다.
- 학습도 2단계 관점: 이 단순한 모델로 벡터를 먼저 배우고, 필요하면 그 벡터 위에 NNLM 등을 얹으면 된다.

#### <span style="color: #4682B4">3.1 CBOW (Continuous Bag-of-Words)</span>

- **주변 단어들로 중심 단어를 예측**한다. 앞 4개 + 뒤 4개의 문맥 단어 임베딩을 **평균**내고
(순서 무시 — 그래서 bag-of-words), 그 평균 벡터로 중심 단어를 분류한다.

$$
    \bar{v} = \frac{1}{N}\sum_{-N/2 \le j \le N/2,\; j \ne 0}{v_{w_{t+j}}}
    ,\qquad
    P(w_t\,|\,context) = \frac{\exp\big({v'_{w_t}}^{\top}\bar{v}\big)}{\sum_{w=1}^{V}{\exp\big({v'_w}^{\top}\bar{v}\big)}}
$$

- NNLM과 비교하면: concat → **평균** (projection 행렬 공유 + 위치 정보 포기), 비선형 hidden → **제거**.
- 토큰당 연산량:

$$
    Q = N \times D + D \times \log_2{V}
$$

- 입력 임베딩 $v$와 출력 임베딩 $v'$ 두 벌을 쓴다는 것도 포인트 (최종적으로 $v$를 단어 벡터로 사용).
- 이름이 "Bag-of-Words"인 이유: 고전 BoW처럼 순서를 버리지만, **연속(continuous) 분산 표현**을 쓴다는 차이.

#### <span style="color: #4682B4">3.2 Continuous Skip-gram</span>

- CBOW의 반대 방향. **중심 단어로 주변 단어 각각을 예측**한다.

$$
    \frac{1}{T}\sum_{t=1}^{T}{\sum_{-C \le j \le C,\; j \ne 0}{\log{P(w_{t+j}\,|\,w_t)}}}
    ,\qquad
    P(w_O|w_I) = \frac{\exp\big({v'_{w_O}}^{\top}v_{w_I}\big)}{\sum_{w=1}^{V}{\exp\big({v'_w}^{\top}v_{w_I}\big)}}
$$

- 윈도우 $C$를 키우면 벡터 품질이 좋아지지만 학습 쌍이 늘어난다. 트릭: 각 중심 단어마다 실제 윈도우를
$[1, C]$에서 **랜덤 샘플링** — 멀리 있는 단어는 확률적으로 덜 뽑히므로, 거리에 반비례하는 가중치를
공짜로 얻는다. ($C = 10$이면 평균 5)
- 토큰당 연산량 ($C$: 최대 거리):

$$
    Q = C \times (D + D \times \log_2{V})
$$

- CBOW vs Skip-gram의 성질 차이 (결과에서 확인됨):
  - CBOW: 학습이 빠르고 **문법(syntactic) 관계에 강함**. 문맥을 평균내므로 smoothing 효과.
  - Skip-gram: 느리지만 **의미(semantic) 관계와 희귀 단어에 강함**. (중심, 문맥) 쌍 하나하나가
개별 학습 신호라 희귀 단어도 자기 몫의 업데이트를 받는다.

#### <span style="color: #4682B4">3.3 Hierarchical Softmax</span>

- 두 모델 모두 분모의 $V$항 합이 남아있는데, 이걸 **Huffman 이진 트리** 기반 hierarchical softmax로
$\log_2$ 스케일로 줄인다. 빈도가 높은 단어일수록 트리에서 짧은 경로를 받으므로 기대 경로 길이는
$\log_2(\text{unigram perplexity})$ 수준까지 내려간다.

<details>
<summary> <span style="color: #ffd33d">hierarchical softmax 정의 + 확률 합 1 증명 + gradient 업데이트 유도 펼치기/접기</span> </summary>

- $n(w, j)$를 루트에서 단어 $w$까지 경로의 $j$번째 노드, $L(w)$를 경로 길이라 하자.
각 내부 노드가 벡터 $v'_n$을 가지고, 확률은

$$
    P(w|w_I) = \prod_{j=1}^{L(w)-1}{\sigma\Big( \big[\!\big[ n(w,j+1) = ch(n(w,j)) \big]\!\big] \cdot {v'_{n(w,j)}}^{\top} v_{w_I} \Big)}
$$

- $ch(n)$은 $n$의 (임의로 고정한) 왼쪽 자식이고, $[\![x]\!]$는 $x$가 참이면 $+1$, 거짓이면 $-1$이다.
즉 각 내부 노드에서 "왼쪽으로 갈 확률 $\sigma(z)$ / 오른쪽으로 갈 확률 $\sigma(-z)$"의 이진 분기다.

- **합이 1이 되는 증명**: $\sigma(z) + \sigma(-z) = 1$이므로 각 내부 노드에서 두 자식으로 가는 확률 합이 1이다.
잎 전체의 합은 트리를 아래에서 위로 접으면서 계산하면 — 형제 잎 두 개의 확률 합은 부모까지의 경로 확률과 같고,
이를 반복하면 루트에서 1이 된다.

$$
    \sum_{w \in leaves}{P(w|w_I)} = 1 \qquad \blacksquare
$$

- **gradient 유도**: 경로의 한 노드에 대한 항을 $L_j = -\log{\sigma(s_j z_j)}$로 쓰자.
($s_j = \pm 1$: 분기 방향, $z_j = {v'_{n_j}}^{\top}v_{w_I}$)

$$
    \frac{\partial L_j}{\partial z_j} = -s_j\,\sigma(-s_j z_j) = \sigma(s_j z_j) - 1 \;\; \text{(부호 } s_j\text{ 반영 형태)}
$$

- 따라서 업데이트는 (learning rate $\eta$):

$$
    v'_{n_j} \leftarrow v'_{n_j} - \eta\,\big(\sigma(s_j z_j)-1\big)\,s_j\, v_{w_I}
    ,\qquad
    v_{w_I} \leftarrow v_{w_I} - \eta\sum_{j}{\big(\sigma(s_j z_j)-1\big)\,s_j\,v'_{n_j}}
$$

- 단어 하나 학습에 **경로 위의 $L(w)-1$개 노드 벡터만** 업데이트하면 된다. $V = 10^6$이어도 20개 내외다.
- 이 구조는 [박사논문](/posts/statistical-language-models-based-on-neural-networks/)의
class 분해($O(\sqrt{V})$)를 트리 깊이 극한($O(\log V)$)까지 밀어붙인 것이다. $\blacksquare$

</details>

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[4] 평가 — Analogy Task</span>

#### <span style="color: #4682B4">4.1 테스트셋 구성</span>

- 논문이 직접 만든 **Semantic-Syntactic Word Relationship test set**:
의미 관계 5종 8,869문항 + 문법 관계 9종 10,675문항. 관계 종류와 예시:

| 유형 | 관계 | 예시 |
|---|---|---|
| Semantic | 수도-국가 | Athens : Greece = Oslo : Norway |
| Semantic | 국가-통화 | Angola : kwanza = Iran : rial |
| Semantic | 주-도시 | Chicago : Illinois = Stockton : California |
| Semantic | 성별 | brother : sister = grandson : granddaughter |
| Syntactic | 형용사-부사 | apparent : apparently = rapid : rapidly |
| Syntactic | 비교급 | great : greater = tough : tougher |
| Syntactic | 최상급 | easy : easiest = lucky : luckiest |
| Syntactic | 현재분사 | think : thinking = read : reading |
| Syntactic | 과거형 | walking : walked = swimming : swam |
| Syntactic | 복수형 | mouse : mice = dollar : dollars |

- 풀이 방식: $x = v_b - v_a + v_c$를 계산하고 **cosine 최근접 단어**를 찾는다.
정확히 그 단어여야 정답 — 동의어도 오답 처리하는 엄격한 채점이라 100%는 사실상 불가능하다.

#### <span style="color: #4682B4">4.2 차원 × 데이터 스케일링</span>

- CBOW로 차원(50~600)과 데이터(24M~783M 토큰)를 격자로 바꿔가며 정확도를 측정했다.
  - **차원만** 키우거나 **데이터만** 키우면 금방 수확 체감이 온다.
  - **둘을 같이** 키워야 정확도가 계속 오른다. (가장 작은 세팅 ~13% → 가장 큰 세팅 60%대 중반)
- "임베딩은 50~100차원이면 충분하다"던 당시 통념을 깨고, **300~600차원 + 대규모 데이터** 조합을
표준으로 만든 실험이다.

#### <span style="color: #4682B4">4.3 아키텍처 비교 (같은 데이터, 640차원)</span>

| 모델 | Semantic 정확도 | Syntactic 정확도 |
|---|---|---|
| RNNLM 벡터 | 9% | 36% |
| NNLM 벡터 | 23% | 53% |
| CBOW | 24% | **64%** |
| **Skip-gram** | **55%** | 59% |

- RNNLM 벡터(계보 ①의 부산물)가 가장 약하고, 새 모델 둘이 NNLM을 이긴다 —
**비선형 hidden 없이도 (오히려 없어서 데이터를 더 먹여서) 벡터 품질이 더 좋다.**
- Skip-gram의 semantic 55%는 NNLM(23%)의 2배가 넘는다. CBOW는 syntactic 특화.
- 학습 시간까지 보면 차이가 더 극적이다: NNLM 계열은 분산 인프라로 며칠 걸리던 것이
CBOW/Skip-gram은 **하루 안에**(같은 데이터) 끝난다.

#### <span style="color: #4682B4">4.4 Epoch vs 데이터 트레이드오프</span>

- 같은 계산 예산이면 "같은 데이터 3 epoch"보다 **"2배 데이터 1 epoch"이 더 좋다.**
- 학습 세팅: 시작 learning rate 0.025를 학습 진행에 따라 **선형 감소**시켜 마지막에 0에 수렴.
이 스케줄과 "epoch을 늘리기보다 데이터를 늘려라"는 지침이 word2vec 툴킷의 기본값으로 이어진다.

#### <span style="color: #4682B4">4.5 Microsoft Sentence Completion Challenge</span>

- 문장에서 빈칸에 들어갈 단어를 5지선다로 고르는 태스크. Skip-gram 단독으로는 48% 수준
(당시 LSA 계열과 비슷)이지만, **RNNLM과 점수를 결합하면 58.9%로 당시 SOTA**를 달성했다.
- Skip-gram이 잡는 정보와 LM이 잡는 정보가 상보적이라는 증거로 제시된다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[5] 학습된 관계의 예시</span>

- 대규모(783M 단어, 300차원) Skip-gram 벡터로 "관계 벡터"를 뽑아 적용한 예시들. (논문 Table 8 발췌)

| 관계 (a : b) | c → 모델의 답 |
|---|---|
| France : Paris | Italy → Rome, Japan → Tokyo, Florida → Tallahassee |
| big : bigger | small → larger, cold → colder |
| Miami : Florida | Baltimore → Maryland, Dallas → Texas |
| Einstein : scientist | Messi → midfielder, Picasso → painter |
| Sarkozy : France | Berlusconi → Italy, Merkel → Germany |
| copper : Cu | zinc → Zn, gold → Au |
| Microsoft : Windows | Google → Android, IBM → Linux |

- 완벽하지 않다는 것도 그대로 보여준다 (Einstein → Messi가 "midfielder"로 연결되는 식의 어긋남).
논문 추정으로 이 용례의 정확도는 60% 수준.
- 응용 제안: 관계 예시 여러 쌍의 차이 벡터를 **평균**내면 관계 벡터가 안정화되어 정확도가 10%p가량 오른다.
out-of-list 단어 찾기, 오탈자 감지 같은 파생 응용도 언급된다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[6] 계보 — 어디서 왔고 어디로 갔나</span>

#### <span style="color: #4682B4">영향 받은 것</span>

- **← [Mikolov 박사논문 (2012)](/posts/statistical-language-models-based-on-neural-networks/)**:
  1. $O = E \times T \times Q$ 복잡도 분석은 박사논문의 "병목은 hidden × softmax" 진단을 그대로 계승 —
CBOW/Skip-gram은 그 진단에서 병목 항을 지워서 만든 모델이다. (2.2의 $Q$식이 곧 박사논문의 RNNLM)
  2. hierarchical softmax는 박사논문의 class 분해를 이진 트리로 일반화한 것.
  3. "단어 벡터에 규칙성이 있다"는 박사논문/NAACL 2013의 관찰이, 이 논문에서 아예 **평가 지표(analogy)** 로 승격됐다.
- **← Bengio NNLM (2003)**: 임베딩 + softmax라는 뼈대. 이 논문은 여기서 비선형만 걷어낸 것에 가깝다.
- **← DistBelief/Adagrad**: 대규모 비동기 분산 학습 인프라가 실험 스케일을 가능하게 했다.

#### <span style="color: #4682B4">후속 연구에 준 영향</span>

- **→ [word2vec 계보 ③ (2013.10)](/posts/distributed-representations-of-words-and-phrases/)**: 같은 해에 저자들이 직접
Skip-gram을 개량한다 — hierarchical softmax를 **negative sampling**으로 교체하고, subsampling과 phrase 학습을 추가.
오늘날 "word2vec"이라 불리는 알고리즘(SGNS)은 그 논문에서 완성된다.
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
- [5] R. Collobert & J. Weston, "A Unified Architecture for Natural Language Processing" (ICML 2008)
- [6] J. Dean et al., "Large Scale Distributed Deep Networks" (DistBelief, NIPS 2012)
- [7] T. Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality" (NIPS 2013)
