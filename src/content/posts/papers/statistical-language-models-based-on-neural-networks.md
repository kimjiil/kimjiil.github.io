---
title: "[word2vec 계보 ①] Statistical Language Models Based on Neural Networks (Mikolov 박사논문, 2012)"
date: 2026-08-21
category: deep-learning-paper
tags:
  - "NLP"
  - "Language Model"
  - "RNN"
  - "word2vec"
---

<span style="font-size:17pt">
<b>Statistical Language Models Based on Neural Networks</b>
</span>

<a href="https://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf" target="_blank"><b>[PDF]</b></a>
, <b><span style="color: #F2AA4C">Language Model</span></b>, Tomáš Mikolov (PhD Thesis, Brno University of Technology, 2012)

### <span style="color: #ffd33d">Summary</span>

word2vec의 저자 Mikolov의 박사논문으로, **RNN 기반 언어모델(RNNLM)** 을 체계화한 문서다.
n-gram(백오프 스무딩) 시대의 통계 언어모델을 신경망으로 대체하면서:

- Elman 스타일의 **simple RNN으로 언어모델을 학습**하면 (truncated) BPTT만으로도 당시 최강이던
modified Kneser-Ney 5-gram을 perplexity에서 크게 이긴다는 것을 보였고,
- softmax 출력층의 $O(V)$ 병목을 **class 기반 분해**로 $O(\sqrt{V})$ 수준까지 줄이는 기법,
dynamic evaluation, 모델 조합 등 실전 기법들을 정리했으며,
- 무엇보다 학습된 **hidden state와 단어 벡터가 문법적/의미적 규칙성을 담는다**는 관찰을 남겼다.

이 마지막 관찰과 "복잡도 병목이 어디에 있는가"에 대한 분석이 다음 해
[word2vec (CBOW/Skip-gram)](/posts/efficient-estimation-of-word-representations-in-vector-space/)으로 직결된다.
word2vec 계보의 출발점이라 이 리뷰 시리즈의 1편으로 읽는 것을 추천.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[1] 배경 — 통계 언어모델과 n-gram의 한계</span>

#### <span style="color: #4682B4">1.1 언어모델이 푸는 문제</span>

- 언어모델의 목표는 단어 시퀀스에 확률을 부여하는 것이다. chain rule로 분해하면
"지금까지의 히스토리가 주어졌을 때 다음 단어의 확률"들의 곱이 된다.

$$
    P(w_1, w_2, ..., w_T) = \prod_{t=1}^{T}{P(w_t \,|\, w_1, ..., w_{t-1})}
$$

- 히스토리 전체를 조건으로 쓰는 것은 불가능하므로(가능한 히스토리의 수가 $V^{t-1}$로 폭발),
어떤 식으로든 히스토리를 **동치류(equivalence class)로 압축**해야 한다. 언어모델의 역사는 사실상
"히스토리를 어떻게 압축할 것인가"의 역사다.
- 평가는 perplexity(PPL)로 한다. 테스트 데이터에 대한 cross-entropy $H$의 지수 형태로,
"모델이 매 단어에서 평균적으로 몇 개의 후보 사이에서 고민하는가"로 읽으면 된다. 낮을수록 좋다.

$$
    PPL = 2^{H} = 2^{-\frac{1}{T}\sum_{t=1}^{T}{\log_2{P(w_t|w_{1 \cdots t-1})}}}
$$

- 단, 논문 전체에서 반복되는 주장: **PPL 개선은 그 자체로는 의미가 없고, 다운스트림(음성인식 WER 등)
개선으로 증명해야 한다.** 실제로 이 논문의 모든 주요 실험은 PPL과 WER을 같이 보고한다.

#### <span style="color: #4682B4">1.2 n-gram과 스무딩</span>

- n-gram 모델은 히스토리를 "마지막 $n-1$개 단어"로 압축한다.

$$
    P(w_t|w_{1 \cdots t-1}) \approx P(w_t|w_{t-n+1 \cdots t-1})
    = \frac{count(w_{t-n+1}, ..., w_t)}{count(w_{t-n+1}, ..., w_{t-1})}
$$

- 카운트가 0인 조합이 무수히 많으므로(4-gram만 되어도 대부분의 조합은 학습 데이터에 없음)
**스무딩**이 필수다. 낮은 차수의 분포와 섞거나(interpolation), 없으면 낮은 차수로 물러난다(back-off).
  - 이 계열의 완성형이 **modified Kneser-Ney(KN) 스무딩**이고, 논문 전체에서 비교 대상 베이스라인은
KN 스무딩된 5-gram(**KN5**)이다.
- n-gram의 근본적 한계 두 가지:
  1. **일반화 불가**: 단어가 원자적 심볼이라서 "cat을 본 문맥"이 "dog"의 확률 추정에 아무 도움이 안 된다.
"party will be on Monday"를 봤어도 "party will be on Friday"의 확률은 오르지 않는다.
  2. **긴 문맥 불가**: $n$을 키우면 파라미터가 지수적으로 늘고 카운트는 더 희소해진다. 실전은 3~5가 한계.
- 이를 보완하는 고전 기법들이 각자 존재했다 — **cache 모델**(최근 나온 단어는 또 나온다),
**class 기반 모델**(단어를 품사/클러스터로 묶어 일반화), **structured LM**(구문 정보 이용),
**maximum entropy 모델**(임의 feature 결합). 논문의 관점: RNN의 hidden state는 이것들이 하던 일을
**상당 부분 하나의 메커니즘으로 포섭**한다. (실험적으로도 RNNLM과 cache를 결합하면 이득이 남아있긴 하지만
n-gram+cache 조합보다 이득 폭이 작다 — 이미 일부를 흡수하고 있다는 간접 증거)

#### <span style="color: #4682B4">1.3 Feedforward NNLM (Bengio 2003)</span>

- NNLM[1]은 히스토리 압축을 **학습된 연속 표현**으로 한다. 마지막 $n-1$개 단어를 각각 $D$차원 벡터로
projection 하고, concat 해서 hidden layer를 거쳐 softmax로 다음 단어를 예측한다.

$$
    y = softmax\Big( V\,\tanh\big( H\,[C_{w_{t-n+1}}; \cdots; C_{w_{t-1}}] + b \big) \Big)
$$

- $C \in \mathbb{R}^{V \times D}$가 공유 임베딩 테이블이다. **비슷한 문맥에서 등장한 단어는 비슷한 벡터를 갖게 되고,
그 벡터를 통해 확률 질량이 자동으로 일반화된다** — n-gram의 한계 1번이 해결된다.
- 하지만 여전히 **고정 윈도우**($n-1$개)라는 한계 2번이 남는다. 그리고 계산량이 커서 당시에는 소규모 데이터에서만
쓸 수 있었다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[2] RNNLM — 네트워크 뜯어보기</span>

#### <span style="color: #4682B4">2.1 구조</span>

- Elman network(simple RNN) 그대로다. 입력은 현재 단어의 one-hot $w(t) \in \mathbb{R}^{V}$과
직전 hidden state $s(t-1) \in \mathbb{R}^{H}$이다.

$$
    \begin{split}
    s(t) &= \sigma\big( U\,w(t) + W\,s(t-1) \big) \\
    y(t) &= softmax\big( V\,s(t) \big)
    \end{split}
$$

- $\sigma$는 sigmoid, $y(t)$가 다음 단어의 확률분포 $P(w_{t+1}|w_t, s(t-1))$이다.
- 파라미터를 하나씩 보면:
  - $U \in \mathbb{R}^{H \times V}$: one-hot과의 곱은 **열 하나를 뽑는 것**이므로 $U$의 각 열이 사실상
단어 임베딩이다. "임베딩 lookup"이라는 현대 표준 구현과 동일하다.
  - $W \in \mathbb{R}^{H \times H}$: recurrent 행렬. 히스토리 압축을 담당한다.
  - $V \in \mathbb{R}^{V \times H}$: 출력(softmax) 행렬. 이게 계산량의 지배항이다. (아래 2.4)
- NNLM과의 결정적 차이: 문맥 윈도우가 고정 $n-1$개가 아니라, $s(t)$가 **이론상 무한한 과거를 재귀적으로 압축**한다.
히스토리 동치류를 사람이 정의하는 게 아니라 데이터가 학습한다.
- hidden 크기는 $H = 30{\sim}500$ 수준 (2012년 스케일). 데이터가 클수록 $H$도 커져야 이득이 유지된다.

#### <span style="color: #4682B4">2.2 학습 — SGD와 Truncated BPTT</span>

- loss는 다음 단어에 대한 cross-entropy $-\log{y_{w_{t+1}}(t)}$이고, plain SGD로 학습한다.
- 학습 스케줄이 소박하지만 구체적으로 적혀 있다 (이후 RNNLM toolkit의 기본값):
  - 초기 learning rate $\alpha = 0.1$, 매 epoch 마다 validation entropy 확인.
  - 개선이 없으면 **learning rate를 절반으로** 줄이며 계속, 또 개선이 없으면 종료. (보통 10~20 epoch 내 수렴)
  - 명시적 regularization은 거의 안 쓴다 — 큰 데이터에서는 overfitting보다 underfitting이 문제라는 입장.
- gradient는 **BPTT(BackPropagation Through Time)** 로 계산하되, 전체 히스토리로 펼치지 않고
$\tau = 5$ 스텝 정도만 펼치는 **truncated BPTT**를 사용한다. 그리고 매 단어마다 unfold 하는 대신
10~20 단어마다 묶어서 backprop 하면(block mode) 계산 효율이 좋아진다.

<details>
<summary> <span style="color: #ffd33d">BPTT gradient 유도 + vanishing/exploding 분석 펼치기/접기</span> </summary>

- 시간축으로 펼친 네트워크에서 $W$는 모든 시점에 공유되므로, loss $L = \sum_t{L_t}$의 gradient는
각 시점 기여분의 합이다.

$$
    \frac{\partial L}{\partial W} = \sum_{t}{\sum_{k \le t}{
        \frac{\partial L_t}{\partial y(t)}
        \frac{\partial y(t)}{\partial s(t)}
        \left( \prod_{j=k+1}^{t}{\frac{\partial s(j)}{\partial s(j-1)}} \right)
        \frac{\partial s(k)}{\partial W}
    }}
$$

- 출력층에서 시작하는 오차는 softmax + cross-entropy 조합이라 깔끔하게 (예측 − 정답)이 된다.

$$
    \frac{\partial L_t}{\partial (Vs(t))} = y(t) - d(t)
    \qquad (d(t): \text{정답 one-hot})
$$

- 시점을 거슬러 올라가는 항이 Jacobian의 곱이다. sigmoid RNN이면

$$
    \frac{\partial s(j)}{\partial s(j-1)} = diag\big( s(j)\odot(1-s(j)) \big)\,W
$$

- 이 Jacobian 곱의 스펙트럼 norm이 1보다 작으면 gradient가 지수적으로 소멸(vanishing), 크면 폭발(exploding)한다.
sigmoid의 미분 최대값이 $1/4$이므로 $\|W\|$가 4 이하이면 소멸 쪽으로 밀리는 구조다.
- 논문의 실전 해법:
  - **exploding** → gradient 성분이 임계값(예: 15)을 넘으면 잘라내는 **clipping**. 이게 없으면 학습이 수시로 발산한다.
  - **vanishing** → 근본 해결은 못 한다. truncated BPTT($\tau=5$)로 "어차피 먼 과거의 gradient는 소멸하니
짧게만 펼치자"는 실용적 타협을 하고, 장거리 정보는 (당시 실험하던) 추가 feature나 cache 결합으로 보충한다.
- 이 미해결 지점이 이후 LSTM 기반 LM(Sundermeyer 2012, Graves 2013)이 표준이 되는 이유고,
더 멀리는 "경로 길이를 아예 $O(1)$로 만들자"는 [Transformer](/posts/attention-is-all-you-need/)의
문제의식으로 이어진다. $\blacksquare$

</details>

<hr/> <!-- 수평선 -->

#### <span style="color: #4682B4">2.3 Vocabulary 처리</span>

- 전체 vocabulary를 그대로 쓰지 않고, 빈도가 임계값 이하인 단어를 전부 `<unk>` 토큰 하나로 병합한다.
(PTB 세팅은 vocab 10K로 고정하는 것이 관례가 됐고, 이 논문의 실험 세팅이 이후 논문들의 표준 벤치마크 세팅으로 굳어졌다)
- `<unk>`의 확률은 rare 단어들의 개수로 나눠서 재분배한다 — n-gram과 공정하게 비교하기 위한 처리.

#### <span style="color: #4682B4">2.4 Class 기반 출력층 — softmax 병목 해결</span>

- 학습/추론 계산량의 지배항은 출력층 $V\,s(t)$의 $H \times V$ 곱이다. ($V$는 수만~수십만)
- 단어를 **빈도 기반으로** $C$개의 class에 배정한다. unigram 확률의 누적 분포를 균등 분할하는
frequency binning이라 별도 클러스터링 학습이 필요 없다. 확률은 2단계로 분해된다.

$$
    P(w_{t+1}|s(t)) = P\big(c(w_{t+1})\,|\,s(t)\big) \times P\big(w_{t+1}\,|\,c(w_{t+1}),\,s(t)\big)
$$

- class 분포($C$개) 하나와, 해당 class 안의 단어 분포(평균 $V/C$개) 하나만 softmax 하면 된다.
학습 시에도 정답 단어가 속한 class의 단어들만 업데이트하면 된다.

<details>
<summary> <span style="color: #ffd33d">최적 class 수가 sqrt(V)임을 증명 + 정확도 손실 논의 펼치기/접기</span> </summary>

- 출력층 계산량은 $f(C) = H\cdot C + H\cdot\frac{V}{C}$에 비례한다. $C$로 미분해서 0으로 두면

$$
    \frac{df}{dC} = H - H\frac{V}{C^2} = 0
    \quad\Rightarrow\quad C^2 = V
    \quad\Rightarrow\quad C = \sqrt{V}
$$

- 이때 계산량은 $2H\sqrt{V}$로, 원래 $HV$ 대비 $\frac{\sqrt{V}}{2}$배 빨라진다.
$V = 10^5$면 약 **158배** 속도 향상이다.
- 공짜는 아니다 — 빈도 기반 class 배정은 의미와 무관한 강제 분해라서 PPL이 약간 나빠진다.
논문 실험 기준으로 전체 softmax 대비 PPL 손실은 수 % 수준, 대신 학습 시간이 수십 배 줄어들므로
같은 시간에 더 큰 모델/데이터를 돌리는 쪽이 항상 이득이었다.
- 이 "출력을 트리/계층으로 쪼갠다"는 아이디어를 극한(이진 트리, 깊이 $\log_2 V$)까지 밀면
[word2vec](/posts/efficient-estimation-of-word-representations-in-vector-space/)의 hierarchical softmax가 된다. $\blacksquare$

</details>

<hr/> <!-- 수평선 -->

#### <span style="color: #4682B4">2.5 Dynamic Evaluation</span>

- 보통 LM은 학습 후 고정된 채 평가되는데, 논문은 **테스트 중에도 방금 처리한 텍스트로 모델을
계속 업데이트**하는 dynamic evaluation을 제안한다. (learning rate를 작게 고정하고 1 pass SGD)
- 효과는 cache 모델과 유사하다 — 문서 안에서 반복되는 주제/고유명사에 빠르게 적응한다.
다만 cache가 "그 단어 자체"만 기억하는 반면, dynamic RNN은 **연속 공간에서 적응**하므로
유사 단어에도 확률이 퍼진다.
- static/dynamic 두 버전은 성질이 달라서 **조합하면 서로 보완**된다. (아래 결과 표)
- 지금 관점에서 보면 test-time adaptation / online learning의 초기 사례다.

#### <span style="color: #4682B4">2.6 모델 조합 (Combination)</span>

- 서로 다른 초기화/하이퍼파라미터로 학습한 RNNLM 여러 개와 n-gram(KN5, cache 포함)을
**선형 보간(linear interpolation)** 으로 섞는다.

$$
    P(w|h) = \sum_{m}{\lambda_m P_m(w|h)}, \qquad \sum_m{\lambda_m} = 1
$$

- 가중치 $\lambda$는 validation에서 EM으로 최적화. 신경망 LM은 실행마다 다른 로컬 미니마에 도달해서
**앙상블 이득이 특히 크다**는 것을 실험으로 보인다.
- 결합 대상: static RNN들 + dynamic RNN들 + KN5 + cache + (maximum entropy 모델 등) — 논문의
최종 수치는 전부 이런 대규모 조합에서 나온다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[3] Results</span>

#### <span style="color: #4682B4">3.1 Penn Treebank (PPL)</span>

- 표준 세팅: PTB 930K 학습 토큰, vocab 10K. 대표 결과 흐름은 다음과 같다. (논문 수치 기준, 근사값)

| 모델 | Perplexity |
|---|---|
| KN5 (baseline) | 141 |
| KN5 + cache | 125 수준 |
| 단일 RNNLM | 124 수준 |
| RNNLM 앙상블 (다수) | 102 수준 |
| **RNN 앙상블 + dynamic + KN5+cache 등 전부 조합** | **80 밑으로** |

- 당시까지 보고된 PTB 최저 PPL을 큰 폭으로 경신했다. 개별 모델의 개선폭보다 **이질적인 모델들의 조합**이
만드는 개선폭이 크다는 것, 그리고 RNNLM이 조합 안에서 가장 큰 기여를 한다는 것이 핵심 메시지.
- 데이터를 늘려가며 측정하면 **n-gram의 이득은 빠르게 포화되는데 RNNLM의 이득은 계속 유지**된다 —
"데이터가 커질수록 신경망 LM의 상대 우위가 커진다"는, 이후 10년을 관통하는 관찰이다.

#### <span style="color: #4682B4">3.2 음성인식 (WER)</span>

- 실제 효용 증명: 음성인식기의 1차 디코딩이 내놓은 n-best/lattice를 RNNLM으로 **rescoring** 한다.
(RNN은 히스토리가 무한이라 lattice에 직접 넣기 어려우므로 n-best 리스트 재채점이 실용적)
- WSJ(Wall Street Journal) 세팅에서 KN 베이스라인 대비 **WER 10~20% 상대 감소**, NIST RT05 등
더 큰 실전 세팅에서도 일관된 개선을 보였다.
- "PPL 개선이 WER 개선으로 이어지는가"라는 오래된 회의론에 대해, 신경망 LM은 **된다**는 것을
보인 초기 사례로 자주 인용된다.

#### <span style="color: #4682B4">3.3 부산물 — RNNLM Toolkit과 단어 벡터</span>

- 논문과 함께 공개한 **RNNLM toolkit**이 이후 연구들의 표준 베이스라인 구현이 됐다.
- 그리고 부산물로 남긴 관찰 하나가 역사를 바꾼다: $U$의 열벡터(단어 임베딩)들을 보면
**문법적/의미적으로 비슷한 단어가 가깝게 모여 있고, 벡터 차이가 관계를 담는다.**
같은 시기 후속 논문(NAACL 2013 [4])에서 이 벡터들로 다음을 보였다.

$$
    v_{king} - v_{man} + v_{woman} \approx v_{queen}
$$

- "LM을 잘 만들기 위한 부산물"이던 벡터가 그 자체로 가치있다는 이 관찰이,
다음 논문에서 **벡터 학습 자체를 목적**으로 삼는 발상 전환으로 이어진다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[4] 계보 — 어디서 왔고 어디로 갔나</span>

#### <span style="color: #4682B4">영향 받은 것</span>

- **← Bengio NNLM (2003)**: "단어 = 연속 벡터" + "신경망으로 LM"이라는 뼈대. 이 논문은 고정 윈도우를
recurrence로 바꿔 히스토리 압축까지 학습하게 만든 것.
- **← Elman (1990)**: simple RNN 구조 자체.
- **← Goodman (2001)**: class 기반 출력 분해로 속도를 버는 아이디어 (2.4의 직접적 선행).
- **← cache/class/maxent 계열 고전 LM들**: RNNLM이 "무엇을 흡수해야 하는가"의 목표 명세 역할.

#### <span style="color: #4682B4">후속 연구에 준 영향</span>

- **→ [word2vec 1편 (2013.1)](/posts/efficient-estimation-of-word-representations-in-vector-space/)**:
이 논문에서 직접 이어지는 부분이 세 가지다.
  1. **복잡도 분석**: 학습 비용의 지배항이 (hidden 곱셈 $H \times H$ + softmax $H \times V$)라는 이 논문의 분석이,
"그럼 비선형 hidden layer를 아예 없애고 log-linear로 가자"는 CBOW/Skip-gram 설계의 출발점이 된다.
word2vec 논문의 복잡도 표에서 RNNLM의 $Q = H \times H + H \times V$가 바로 이 논문의 모델이다.
  2. **계층적 출력**: 2.4의 class 분해($O(\sqrt{V})$)가 Huffman 트리 hierarchical softmax($O(\log V)$)로 일반화된다.
  3. **표현의 규칙성**: 3.3의 관찰이 "임베딩이 부산물이 아니라 목표"라는 관점 전환과
analogy 평가 지표의 탄생으로 이어진다.
- **→ RNN 학습 전반**: gradient clipping은 이후 모든 RNN/LSTM 학습의 기본기가 됐고 (Pascanu 2013이
이론 정리), vanishing 문제의 실증은 LSTM LM 채택과, 더 멀리는 경로 길이 $O(1)$을 내세운
[Transformer](/posts/attention-is-all-you-need/)의 motivation으로 이어진다.
- **→ 벤치마크 문화**: 이 논문의 PTB 전처리 세팅(930K/10K vocab)이 이후 10년간 LM 논문의 표준 벤치마크가 됐고,
"LM 개선은 다운스트림(WER)으로 증명한다"는 실증 스타일도 함께 정착했다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">Reference</span>

- [1] Y. Bengio et al., "A Neural Probabilistic Language Model" (JMLR 2003)
- [2] T. Mikolov et al., "Recurrent neural network based language model" (Interspeech 2010)
- [3] T. Mikolov et al., "Extensions of recurrent neural network language model" (ICASSP 2011)
- [4] T. Mikolov et al., "Linguistic Regularities in Continuous Space Word Representations" (NAACL 2013)
- [5] J. Goodman, "Classes for fast maximum entropy training" (2001)
- [6] R. Pascanu et al., "On the difficulty of training recurrent neural networks" (ICML 2013)
