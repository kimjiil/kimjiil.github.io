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

1. **임베딩을 부산물이 아니라 목표로**: 언어모델을 잘 만들다 벡터를 얻는 게 아니라, 좋은 단어 벡터
자체를 최대한 싸게 배우는 것이 목적이다. 평가도 perplexity가 아니라 직접 만든 **단어 유추(analogy)
테스트셋** — vector("King") − vector("Man") + vector("Woman") ≈ vector("Queen") — 으로 한다.
2. **비선형 hidden layer 제거**: [계보 ① 박사논문](/posts/statistical-language-models-based-on-neural-networks/)의
복잡도 분석(병목 = $N \times D \times H$ 또는 $H \times H$)을 이어받아, 그 병목 항을 아예 제거한
**log-linear 모델** 둘을 설계했다. 그 결과 1.6B 단어를 하루 안에 학습한다 (기존 NNLM류는 수백M 단어,
50~100차원이 한계였다).

결과적으로 1000차원 Skip-gram(6B 단어)이 analogy 정확도 65.6%로 NNLM(50.8%)을 크게 이기면서 학습은
수 배 빠르다. "표현력이 낮은 모델 × 훨씬 많은 데이터"가 이긴다는 이 트레이드오프 감각이 이후 NLP의
상식이 된다. 리뷰는 논문 섹션 구성을 그대로 따라간다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[1] Introduction</span>

#### <span style="color: #4682B4">1.1 Goals of the Paper</span>

- 당시 NLP 시스템 대부분은 단어를 원자적 단위(vocabulary 인덱스)로 취급했다 — 단어 사이의 유사성 개념이 없다.
  - 이 선택에도 이유는 있다: 단순함, 강건함, 그리고 "많은 데이터의 단순한 모델이 적은 데이터의 복잡한
모델을 이긴다"는 관찰. n-gram은 실제로 수조(trillions) 단어까지 학습되고 있었다.
  - 하지만 한계가 온 영역들이 있다 — ASR의 in-domain 전사 데이터는 수백만 단어 수준, 많은 언어의 MT
코퍼스는 수십억 단어 이하. **"단순 기법의 스케일업으로는 더 못 가는 상황에서는 고급 기법으로 가야 한다."**
- 목표 선언: **수십억 단어 + 수백만 vocabulary**에서 고품질 단어 벡터를 학습하는 기법.
당시까지 그 규모로 학습된 아키텍처는 없었다 (수억 단어 + 50~100차원이 최대).
- 품질의 정의가 이 논문의 독창점이다. "비슷한 단어가 가깝다"를 넘어 **multiple degrees of similarity** —
단어는 의미로도, 문법(어미, 시제)으로도 동시에 비슷할 수 있다.
- 그리고 **벡터 산술**: 이전 연구(NAACL 2013 [20])에서 발견된
vector("King") − vector("Man") + vector("Woman") ≈ vector("Queen") 현상을,
이 논문은 **최적화 대상**으로 승격한다 — "이런 선형 규칙성을 보존하도록 새 아키텍처를 설계하고,
그걸 재는 종합 테스트셋을 만든다."

#### <span style="color: #4682B4">1.2 Previous Work</span>

- 연속 벡터 표현의 역사는 길다 (Hinton의 distributed representation 1986, Elman 1990, Rumelhart backprop).
- **Bengio NNLM (2003)**: linear projection + 비선형 hidden으로 단어 벡터와 언어모델을 **동시에** 학습.
- **Mikolov의 이전 작업 (석사논문 2007, ICASSP 2009)**: 벡터를 **단일 hidden layer 신경망으로 먼저 배우고**,
그 위에 NNLM을 얹는 2단계 방식 — **이 논문은 그 구조의 직접 확장으로, "벡터를 배우는 첫 단계"만
떼어내 극단적으로 단순화한 것이다.** (논문이 스스로 밝히는 출처)
- Collobert & Weston(2008), Turian(2010), Huang(2012), Mnih 등의 공개 벡터들이 존재했지만
전부 학습 비용이 훨씬 비쌌다 — 4.3절에서 이들과 전부 정면 비교한다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[2] Model Architectures — 복잡도 프레임워크</span>

- LSA/LDA 같은 카운트 기반도 있지만, **선형 규칙성 보존은 신경망 계열이 LSA보다 낫고**, LDA는 큰 데이터에서
비용이 폭발하므로 신경망 계열에 집중한다.
- 모든 모델의 학습 비용을 통일된 형태로 정의한다.

$$
    O = E \times T \times Q
$$

- $E$: epoch 수(보통 3~50), $T$: 학습 토큰 수(최대 10억+), $Q$: 모델별 토큰당 연산량.
전부 SGD + backprop으로 학습. **목표: 정확도를 유지하면서 $Q$를 최소화.**

#### <span style="color: #4682B4">2.1 Feedforward NNLM의 Q</span>

- 구조: 이전 $N$개 단어 1-of-V 인코딩 → 공유 projection 행렬로 $N \times D$ projection layer →
비선형 hidden $H$ → 전체 vocabulary 출력.

$$
    Q = N \times D + N \times D \times H + H \times V
$$

- 전형적 수치: $N = 10$, projection $500{\sim}2000$, $H = 500{\sim}1000$.
- 지배항은 $H \times V$지만 이건 **hierarchical softmax로 $\log_2$ 스케일로** 줄일 수 있다. 그러면
**남는 병목은 $N \times D \times H$** — 비선형 hidden layer다.
- 논문이 쓰는 hierarchical softmax는 **Huffman 이진 트리** 기반: 빈도 높은 단어에 짧은 코드가 붙어서
balanced tree의 $\log_2{V}$보다도 적은, 약 $\log_2(\text{unigram perplexity})$개의 출력만 평가하면 된다.
(vocab 100만이면 balanced 대비 약 2배 추가 speedup)
  - 빈도 기반 class가 잘 동작한다는 근거로 [박사논문](/posts/statistical-language-models-based-on-neural-networks/)의
frequency binning을 인용한다 — class 분해($O(\sqrt{V})$) → Huffman 트리($O(\log V)$)로의 진화.

<details>
<summary> <span style="color: #ffd33d">hierarchical softmax 정의 + 확률 합 1 증명 + gradient 업데이트 유도 펼치기/접기</span> </summary>

- $n(w, j)$를 루트에서 단어 $w$까지 경로의 $j$번째 노드, $L(w)$를 경로 길이라 하자.
각 내부 노드가 벡터 $v'_n$을 가지고, 확률은

$$
    P(w|w_I) = \prod_{j=1}^{L(w)-1}{\sigma\Big( \big[\!\big[ n(w,j+1) = ch(n(w,j)) \big]\!\big] \cdot {v'_{n(w,j)}}^{\top} v_{w_I} \Big)}
$$

- $ch(n)$은 $n$의 (고정된) 왼쪽 자식, $[\![x]\!]$는 참이면 $+1$/거짓이면 $-1$.
각 내부 노드에서 "왼쪽 $\sigma(z)$ / 오른쪽 $\sigma(-z)$"의 이진 분기다.
- **합이 1이 되는 증명**: $\sigma(z) + \sigma(-z) = 1$이므로 각 내부 노드에서 두 자식으로 가는 확률 합이 1.
잎에서 루트로 트리를 접으면서 계산하면 형제 잎 확률 합 = 부모까지의 경로 확률이 되고, 반복하면 루트에서 1.

$$
    \sum_{w \in leaves}{P(w|w_I)} = 1 \qquad \blacksquare
$$

- **gradient**: 경로의 한 노드 항을 $L_j = -\log{\sigma(s_j z_j)}$ ($s_j = \pm 1$,
$z_j = {v'_{n_j}}^{\top}v_{w_I}$)로 쓰면 $\frac{\partial L_j}{\partial z_j} = -s_j\,\sigma(-s_j z_j)$이고, 업데이트는

$$
    v'_{n_j} \leftarrow v'_{n_j} + \eta\, s_j\,\sigma(-s_j z_j)\, v_{w_I}
    ,\qquad
    v_{w_I} \leftarrow v_{w_I} + \eta\sum_{j}{s_j\,\sigma(-s_j z_j)\,v'_{n_j}}
$$

- 단어 하나 학습에 경로 위 $L(w)-1$개 노드만 업데이트 — vocab 100만이어도 20개 내외다. $\blacksquare$

</details>

<hr/> <!-- 수평선 -->

#### <span style="color: #4682B4">2.2 RNNLM의 Q</span>

- [계보 ①](/posts/statistical-language-models-based-on-neural-networks/)의 모델. projection layer가 없고
recurrent 행렬이 hidden을 자기 자신과 연결해 short-term memory를 만든다.

$$
    Q = H \times H + H \times V
$$

- $H \times V$는 마찬가지로 hierarchical softmax로 줄이면, **병목은 $H \times H$.**

#### <span style="color: #4682B4">2.3 병렬 학습 (DistBelief)</span>

- Google의 분산 프레임워크 **DistBelief** 위에서 같은 모델의 replica를 100개 이상 띄우고, 중앙 파라미터
서버로 gradient를 비동기 동기화한다. optimizer는 mini-batch async SGD + **Adagrad**.
- 이 인프라가 4.4절의 "6B 단어 × 1000차원" 실험을 가능하게 한다. (DistBelief는 TensorFlow의 전신)

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[3] New Log-linear Models</span>

- 설계 선언이 명확하다: *"복잡도의 대부분은 비선형 hidden layer에서 온다. 신경망을 매력적으로 만드는 게
바로 그것이지만, 우리는 **데이터를 더 정밀하게 표현하지는 못해도 훨씬 많은 데이터를 효율적으로 학습할 수
있는 더 단순한 모델**을 탐색하기로 했다."*
- 구조는 이전 2단계 방식(벡터 먼저, NNLM 나중)의 1단계만 남긴 것이다.

#### <span style="color: #4682B4">3.1 Continuous Bag-of-Words (CBOW)</span>

- NNLM에서 **비선형 hidden을 제거**하고, projection layer를 모든 단어가 **공유**한다(행렬만이 아니라
위치까지) — 즉 문맥 단어들의 벡터가 **평균**되어 하나의 위치로 투영된다. 순서가 사라지므로
bag-of-words인데, 연속 표현을 쓰므로 **Continuous** BOW.
- **미래 단어도 사용한다**: 앞 4개 + 뒤 4개 단어로 가운데 단어를 맞추는 log-linear 분류기가 최적이었다.

$$
    Q = N \times D + D \times \log_2{V}
$$

- NNLM의 $N \times D \times H$ 항이 통째로 사라졌다. 수치로 보면 ($N=8$, $D=640$, $H=640$, $V=10^6$ 기준)
NNLM의 $Q \approx 330$만 → CBOW $Q \approx 1.8$만. **약 180배.**

#### <span style="color: #4682B4">3.2 Continuous Skip-gram</span>

- CBOW의 반대: **현재 단어를 입력으로, 같은 문장 안의 주변 단어들을 각각 예측**한다.
- 범위 $C$를 키우면 벡터 품질이 좋아지지만 비용이 는다. 트릭: 각 단어마다 $R \in [1, C]$를 랜덤으로 뽑아
앞 $R$개 + 뒤 $R$개만 정답으로 쓴다 — **먼 단어일수록 덜 샘플링되므로 거리 반비례 가중치를 공짜로**
얻는다. 실험은 $C = 10$ (평균 $R=5.5$).

$$
    Q = C \times (D + D \times \log_2{V})
$$

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[4] Results</span>

#### <span style="color: #4682B4">4.1 Task Description — Semantic-Syntactic Word Relationship test set</span>

- 기존 논문들의 평가는 "France와 비슷한 단어 표"를 보여주고 직관에 호소하는 수준이었다. 이 논문은
**의미 5종 + 문법 9종, 총 19,544문항**(8,869 + 10,675)의 테스트셋을 직접 만들었다. (논문 Table 1)

| 유형 | 관계 | 예시 |
|---|---|---|
| Semantic | 흔한 수도-국가 | Athens : Greece = Oslo : Norway |
| Semantic | 모든 수도-국가 | Astana : Kazakhstan = Harare : Zimbabwe |
| Semantic | 통화 | Angola : kwanza = Iran : rial |
| Semantic | 주-도시 | Chicago : Illinois = Stockton : California |
| Semantic | 남-여 | brother : sister = grandson : granddaughter |
| Syntactic | 형용사→부사 | apparent : apparently = rapid : rapidly |
| Syntactic | 반의 접두 | possibly : impossibly = ethical : unethical |
| Syntactic | 비교급 | great : greater = tough : tougher |
| Syntactic | 최상급 | easy : easiest = lucky : luckiest |
| Syntactic | 현재분사 | think : thinking = read : reading |
| Syntactic | 국적 형용사 | Switzerland : Swiss = Cambodia : Cambodian |
| Syntactic | 과거형 | walking : walked = swimming : swam |
| Syntactic | 명사 복수 | mouse : mice = dollar : dollars |
| Syntactic | 동사 3인칭 | work : works = speak : speaks |

- 풀이: $X = v_{biggest} - v_{big} + v_{small}$을 계산하고 cosine 최근접 단어를 찾는다 (질문 단어 제외).
- 채점은 **정확히 그 단어여야 정답** — 동의어도 오답. 형태 정보가 입력에 없으므로 100%는 애초에 불가능하다고
명시한다. (이 한계 지적이 [fastText](/posts/fasttext-subword-information-and-bag-of-tricks/)의 출발점이 된다)

#### <span style="color: #4682B4">4.2 Maximization of Accuracy — 차원 × 데이터 격자 실험</span>

- Google News 6B 토큰, vocab 최빈 100만. 먼저 30K vocab 제한 부분집합으로 CBOW의 차원×데이터 격자를
전부 돌렸다. (논문 Table 2, 정확도 %)

| 차원 \ 데이터 | 24M | 49M | 98M | 196M | 391M | 783M |
|---|---|---|---|---|---|---|
| 50 | 13.4 | 15.7 | 18.6 | 19.1 | 22.5 | 23.2 |
| 100 | 19.4 | 23.1 | 27.8 | 28.7 | 33.4 | 32.2 |
| 300 | 23.2 | 29.2 | 35.3 | 38.6 | 43.7 | 45.9 |
| 600 | 24.0 | 30.1 | 36.5 | 40.8 | 46.6 | **50.4** |

- 표의 메시지: **차원만 키우거나(왼쪽 열 아래로) 데이터만 키우면(첫 행 오른쪽으로) 금방 수확 체감**이 오고,
**둘을 같이 키워야**(대각선) 계속 오른다. "데이터는 크게, 차원은 50~100"이라는 당시 관행을 정면 반박.
- 학습 세팅: 3 epoch, 시작 learning rate **0.025를 선형 감소**시켜 마지막에 0 도달.

#### <span style="color: #4682B4">4.3 아키텍처 비교</span>

- **같은 데이터(LDC 320M 단어, 82K vocab), 같은 640차원**으로 4개 아키텍처 정면 비교. 비교 대상 RNNLM은
단일 CPU로 **8주** 걸려 학습된 모델이다. (논문 Table 3)

| 아키텍처 | Semantic [%] | Syntactic [%] | MSR 문법 테스트셋 [%] |
|---|---|---|---|
| RNNLM | 9 | 36 | 35 |
| NNLM | 23 | 53 | 47 |
| CBOW | 24 | **64** | **61** |
| **Skip-gram** | **55** | 59 | 56 |

- 해석 (논문 서술 그대로):
  - RNN 벡터는 주로 문법 쪽에서만 쓸만하다. NNLM이 RNN보다 나은 건 벡터가 비선형 hidden에 직접
연결되기 때문.
  - **CBOW는 문법에서 최강**, 의미는 NNLM 수준. **Skip-gram은 의미에서 압도적**(55 vs 23),
문법도 NNLM보다 낫다.
- **공개 벡터들과의 비교** (논문 Table 4, full vocab 전체 정확도): Collobert-Weston 11.0%,
Turian 2.1%, Mnih 8.8%, Mikolov RNNLM(640d) 24.6%, Huang 12.3% 대비 —
**Skip-gram 300d/783M 단어가 53.3%.** 심지어 저자들의 6B NNLM(100d, 50.8%)보다 좋다.
- **Epoch vs 데이터** (논문 Table 5): 같은 모델로 "3 epoch × 783M"(53.3%, 3일)보다
**"1 epoch × 1.6B"(53.8%, 2일)가 낫다.** → "epoch을 돌리지 말고 데이터를 늘려라."

#### <span style="color: #4682B4">4.4 대규모 분산 학습</span>

- DistBelief에서 replica 50~100개 + Adagrad로 6B 전체를 학습. (논문 Table 6)

| 모델 | 차원 | 데이터 | Sem | Syn | **Total** | 학습 비용 (days × cores) |
|---|---|---|---|---|---|---|
| NNLM | 100 | 6B | 34.2 | 64.5 | 50.8 | 14 × 180 |
| CBOW | 1000 | 6B | 57.3 | 68.9 | 63.7 | 2 × 140 |
| **Skip-gram** | **1000** | 6B | 66.1 | 65.1 | **65.6** | 2.5 × 125 |

- NNLM은 100차원에 14일×180코어. 새 모델들은 **1000차원**을 2~2.5일에 학습하고 정확도도 15%p 높다.
(1000차원 NNLM은 "너무 오래 걸려서 완료 불가"라는 각주가 붙어있다)

#### <span style="color: #4682B4">4.5 Microsoft Sentence Completion Challenge</span>

- [계보 ①](/posts/statistical-language-models-based-on-neural-networks/) 7장에 나왔던 그 벤치마크
(1040문장, 5지선다). Skip-gram 640d를 50M 단어로 학습하고, **빈칸 단어를 입력으로 문장의 나머지 단어들을
예측한 점수 합**으로 문장을 고른다.

| 방법 | 정확도 [%] |
|---|---|
| 4-gram | 39 |
| 평균 LSA 유사도 | 49 |
| Log-bilinear | 54.8 |
| RNNLM 조합 (당시 SOTA) | 55.4 |
| Skip-gram 단독 | 48.0 |
| **Skip-gram + RNNLMs** | **58.9** |

- Skip-gram 단독은 LSA 수준이지만, **RNNLM과 점수가 상보적**이라 가중 결합하면 새 SOTA.
(Skip-gram이 잡는 정보 ≠ LM이 잡는 정보라는 증거)

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[5] Examples of the Learned Relationships</span>

- 최고 모델(Skip-gram 300d/783M)로 관계 벡터를 적용한 예시. (논문 Table 8 발췌)

| 관계 | 예시 1 | 예시 2 | 예시 3 |
|---|---|---|---|
| France − Paris | Italy: Rome | Japan: Tokyo | Florida: Tallahassee |
| big − bigger | small: larger | cold: colder | quick: quicker |
| Miami − Florida | Baltimore: Maryland | Dallas: Texas | Kona: Hawaii |
| Einstein − scientist | Messi: midfielder | Mozart: violinist | Picasso: painter |
| Sarkozy − France | Berlusconi: Italy | Merkel: Germany | Koizumi: Japan |
| copper − Cu | zinc: Zn | gold: Au | uranium: plutonium |
| Microsoft − Windows | Google: Android | IBM: Linux | Apple: iPhone |
| Microsoft − Ballmer | Google: Yahoo | IBM: McNealy | Apple: Jobs |
| Japan − sushi | Germany: bratwurst | France: tapas | USA: pizza |

- 정직한 자평: 이 표를 exact match로 채점하면 약 60% — 틀리는 것도 그대로 보여준다
(Einstein−scientist가 Messi를 "midfielder"로, Google:Yahoo 같은 어긋남).
- 개선 트릭: 관계 예시를 1개가 아니라 **10개의 차이 벡터 평균**으로 만들면 정확도가 **절대 10%p** 오른다.
- 파생 응용: 리스트에서 이질적 단어 고르기(평균 벡터에서 가장 먼 것) — 지능검사 유형 문제.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[6~7] Conclusion & Follow-Up Work</span>

- 결론: 인기있는 신경망 모델(FF/RNN)보다 **훨씬 단순한 구조로 더 좋은 벡터**를 학습할 수 있다.
낮은 복잡도 덕에 더 큰 데이터에서 고차원 벡터가 가능하고, DistBelief면 **1조 단어 + 무제한 vocab**도
가능할 것 — "이전 최고 결과보다 몇 자릿수 큰 규모".
- SemEval-2012 관계 유사도 태스크에서 RNN 벡터가 이미 이전 최고 대비 Spearman 상관 +50%를 만든 사례,
감성분석/패러프레이즈 응용, Knowledge Base 사실 확장/검증과 MT에의 진행 중 실험도 언급.
- **7절 Follow-Up Work** (v3에서 추가): 논문 공개 후 **단일 머신 멀티스레드 C++ 코드 공개** —
"전형적 하이퍼파라미터로 **시간당 수십억 단어**" 학습. 100B 단어로 학습한 140만 개 entity 벡터도 공개.
후속작은 NIPS 2013에 — 그게 [계보 ③](/posts/distributed-representations-of-words-and-phrases/)이다.
- 공개된 그 코드가 바로 `word2vec` 툴킷이고, 임베딩의 대중화는 사실상 이 코드 공개가 만들었다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">계보 — 어디서 왔고 어디로 갔나</span>

#### <span style="color: #4682B4">영향 받은 것</span>

- **← [계보 ① Mikolov 박사논문 (2012)](/posts/statistical-language-models-based-on-neural-networks/)**:
  1. $O = E \times T \times Q$ 복잡도 분석은 박사논문의 병목 진단($N \times D \times H$, $H \times H$,
$H \times V$)의 연장이고, 2.1~2.2절의 NNLM/RNNLM $Q$식이 그대로 그 모델들이다.
  2. Huffman hierarchical softmax는 박사논문의 frequency binning class 분해("빈도만으로 class를 만들어도
잘 된다")를 트리 극한까지 민 것 — 논문이 [16]으로 직접 인용한다.
  3. analogy 평가 자체가 박사논문 벡터의 규칙성 관찰(NAACL 2013 [20])에서 왔고, 4.5절 MSR 문장완성도
박사논문 7장의 벤치마크 재사용이다.
- **← Mikolov 석사논문(2007)/ICASSP 2009**: "벡터를 단순 모델로 먼저 배운다"는 2단계 구조의 1단계가
CBOW/Skip-gram의 직접 원형 (논문 1.2절이 명시).
- **← Bengio NNLM (2003)**, Hinton distributed representation (1986), **← DistBelief/Adagrad** 인프라.

#### <span style="color: #4682B4">후속 연구에 준 영향</span>

- **→ [계보 ③ (2013.10)](/posts/distributed-representations-of-words-and-phrases/)**: 같은 해 저자들이
Skip-gram의 hierarchical softmax를 **negative sampling**으로 교체하고 subsampling/phrase를 추가 —
오늘날의 "word2vec"(SGNS)이 완성된다.
- **→ [계보 ④ fastText (2016-17)](/posts/fasttext-subword-information-and-bag-of-tricks/)**: 4.1절이 스스로
지적한 한계("형태 정보가 입력에 없다")를 문자 n-gram으로 푼 직계 후속. CBOW 구조는 fastText 분류기로
재활용된다.
- **→ GloVe (2014)**: "예측 기반 vs 카운트 기반" 논쟁을 촉발했고 GloVe가 그 절충으로 등장.
- **→ 사전학습-전이 패러다임**: "대규모 비지도 코퍼스로 표현을 먼저 배우고 태스크에 전이"가 NLP 표준
워크플로가 됐다. 정적 벡터의 한계는 ELMo → [Transformer](/posts/attention-is-all-you-need/) 기반
BERT/GPT의 문맥적 임베딩으로 이어진다.
- **→ 평가 문화**: 이 논문의 analogy 테스트셋(공개)이 임베딩 평가의 표준 벤치마크가 됐고,
"King − Man + Woman = Queen"은 표현학습을 대중에게 알린 상징이 됐다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">Reference</span>

- [1] Y. Bengio et al., "A Neural Probabilistic Language Model" (JMLR 2003)
- [2] T. Mikolov, "Statistical Language Models Based on Neural Networks" (PhD Thesis, 2012)
- [3] T. Mikolov et al., "Linguistic Regularities in Continuous Space Word Representations" (NAACL 2013)
- [4] F. Morin & Y. Bengio, "Hierarchical Probabilistic Neural Network Language Model" (AISTATS 2005)
- [5] J. Dean et al., "Large Scale Distributed Deep Networks" (DistBelief, NIPS 2012)
- [6] T. Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality" (NIPS 2013)
- [7] G. Zweig & C.J.C. Burges, "The Microsoft Research Sentence Completion Challenge" (MSR-TR-2011-129)
