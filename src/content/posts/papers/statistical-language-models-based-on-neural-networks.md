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

### <span style="color: #ffd33d">[1] 배경 — n-gram의 한계</span>

- 당시 표준 언어모델은 n-gram + 스무딩(Good-Turing, Kneser-Ney)이었다. $P(w_t|w_{t-n+1 \cdots t-1})$을
카운트 기반으로 추정한다.
  - 데이터가 아무리 많아도 **본 적 없는 조합**은 백오프로 뭉개야 하고,
  - 단어를 원자적 심볼로 취급해서 **"cat을 봤으면 dog도 비슷하겠지" 같은 일반화가 원천적으로 불가능**하다.
- Bengio의 NNLM(2003)[1]은 단어를 연속 벡터(distributed representation)로 임베딩해서 이 일반화 문제를 풀었지만,
고정된 윈도우 $n$개 단어만 조건으로 쓸 수 있었다.
- 이 논문의 제안: **recurrent 연결로 문맥 길이 제한 자체를 없애자.** hidden state가 이론상 무한한 과거를 압축한다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[2] RNNLM — 네트워크 뜯어보기</span>

#### <span style="color: #4682B4">2.1 구조</span>

- Elman network(simple RNN) 그대로다. 입력은 현재 단어의 one-hot $w(t) \in \mathbb{R}^{V}$과
직전 hidden state $s(t-1) \in \mathbb{R}^{H}$의 concat이다.

$$
    \begin{split}
    s(t) &= \sigma\big( U\,w(t) + W\,s(t-1) \big) \\
    y(t) &= softmax\big( V\,s(t) \big)
    \end{split}
$$

- $\sigma$는 sigmoid. $y(t)$가 다음 단어의 확률분포 $P(w_{t+1}|w_t, s(t-1))$이다.
- $U \in \mathbb{R}^{H \times V}$의 각 열이 사실상 **단어 임베딩**이다. one-hot과의 곱은 열 하나를 뽑는 것이므로,
"임베딩 lookup + recurrent"라는 지금의 표준 구조와 동일하다.
- hidden 크기는 $H = 30{\sim}500$ 수준 (2012년 스케일).

#### <span style="color: #4682B4">2.2 학습 — Truncated BPTT</span>

- loss는 다음 단어에 대한 cross-entropy이고, gradient는 **BPTT(BackPropagation Through Time)** 로 계산한다.
매 스텝 전체 히스토리로 펼치는 대신 $\tau$ 스텝만 펼치는 truncated BPTT($\tau = 5$ 정도)를 사용한다.

<details>
<summary> <span style="color: #ffd33d">BPTT gradient 유도 펼치기/접기</span> </summary>

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

- 여기서 시점을 거슬러 올라가는 항이 Jacobian의 곱이다. sigmoid RNN이면

$$
    \frac{\partial s(j)}{\partial s(j-1)} = diag\big( s(j)\odot(1-s(j)) \big)\,W
$$

- 이 Jacobian 곱의 norm이 1보다 작으면 지수적으로 소멸(vanishing), 크면 폭발(exploding)한다.
논문은 **gradient의 norm이 임계값을 넘으면 잘라내는 clipping**을 실전 해법으로 사용했고
(exploding 쪽 해결), vanishing은 미해결로 남겨둔다 — 이게 이후 LSTM 채택의 이유가 된다. $\blacksquare$

</details>

<hr/> <!-- 수평선 -->

#### <span style="color: #4682B4">2.3 Class 기반 출력층 — softmax 병목 해결</span>

- 연산량의 지배항은 출력층 $V\,s(t)$의 $H \times V$ 곱이다. ($V$는 수만~수십만)
- 단어를 빈도 기반으로 $C$개의 class로 묶고, 확률을 2단계로 분해한다.

$$
    P(w_{t+1}|s(t)) = P\big(c(w_{t+1})\,|\,s(t)\big) \times P\big(w_{t+1}\,|\,c(w_{t+1}),\,s(t)\big)
$$

- class 분포($C$개) 하나와, 해당 class 안의 단어 분포(평균 $V/C$개) 하나만 계산하면 된다.

<details>
<summary> <span style="color: #ffd33d">최적 class 수가 sqrt(V)임을 증명 펼치기/접기</span> </summary>

- 출력층 계산량은 $f(C) = H\cdot C + H\cdot\frac{V}{C}$에 비례한다. $C$로 미분해서 0으로 두면

$$
    \frac{df}{dC} = H - H\frac{V}{C^2} = 0
    \quad\Rightarrow\quad C^2 = V
    \quad\Rightarrow\quad C = \sqrt{V}
$$

- 이때 계산량은 $2H\sqrt{V}$로, 원래 $HV$ 대비 $\frac{\sqrt{V}}{2}$배 빨라진다.
$V = 10^5$면 약 **158배** 속도 향상이다. $\blacksquare$
- 이 "출력을 트리/계층으로 쪼갠다"는 아이디어를 극한(이진 트리, 깊이 $\log_2 V$)까지 밀면
[word2vec](/posts/efficient-estimation-of-word-representations-in-vector-space/)의 hierarchical softmax가 된다.

</details>

<hr/> <!-- 수평선 -->

#### <span style="color: #4682B4">2.4 실전 기법들</span>

- **Dynamic evaluation**: 테스트 중에도 방금 본 텍스트로 모델을 계속 업데이트 — 도메인 적응 효과.
(지금의 test-time adaptation의 조상격)
- **모델 조합**: RNNLM 여러 개 + n-gram을 선형 보간하면 단일 모델보다 항상 좋다.
- **캐시 모델과의 관계**: RNN의 hidden state가 n-gram에 캐시/스킵그램 feature를 더한 것들을 상당 부분 포섭한다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[3] Results</span>

- **Penn Treebank**: modified Kneser-Ney 5-gram 대비 perplexity 대폭 개선, RNN 앙상블 + 조합으로 당시 보고된 최저 perplexity.
- **음성인식(WSJ, Broadcast News)**: n-gram lattice를 RNNLM으로 rescoring 해서 **WER(단어 오류율)을 유의미하게 감소**
— 신경망 LM이 실제 시스템에서 쓸모있음을 보인 초기 사례.
- 부산물로 공개한 **RNNLM toolkit**이 이후 연구들의 표준 베이스라인이 됐다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[4] 계보 — 어디서 왔고 어디로 갔나</span>

#### <span style="color: #4682B4">영향 받은 것</span>

- **Bengio NNLM(2003)**: "단어 = 연속 벡터" + "신경망으로 LM"이라는 뼈대. 이 논문은 고정 윈도우를
recurrence로 바꾼 것.
- **Elman(1990)의 simple RNN**, Goodman의 class 기반 속도 개선 아이디어.

#### <span style="color: #4682B4">후속 연구에 준 영향</span>

- **→ [word2vec 1편 (2013.1)](/posts/efficient-estimation-of-word-representations-in-vector-space/)**:
이 논문에서 직접 이어지는 부분이 세 가지다.
  1. **복잡도 분석**: 학습 비용의 지배항이 (hidden 곱셈 + softmax)라는 이 논문의 분석이,
"그럼 hidden layer를 아예 없애고 log-linear로 가자"는 CBOW/Skip-gram 설계의 출발점이 된다.
  2. **계층적 출력**: 2.3의 class 분해가 hierarchical softmax로 일반화된다.
  3. **표현의 규칙성**: RNNLM의 단어 벡터가 문법적/의미적 규칙을 담는다는 관찰이
"벡터 학습 자체를 목적으로 삼자"(임베딩이 부산물이 아니라 목표)는 관점 전환을 만든다.
실제로 Mikolov는 같은 시기 "Linguistic Regularities..."(NAACL 2013)에서 RNNLM 벡터로
king − man + woman ≈ queen 을 먼저 보였다.
- **→ RNN 학습 전반**: gradient clipping은 이후 모든 RNN/LSTM 학습의 기본기가 됐고,
vanishing 문제 정리는 LSTM 채택(Sundermeyer 2012, Graves 2013)과 이후
seq2seq → attention → [Transformer](/posts/attention-is-all-you-need/)로 이어지는 흐름의 문제의식을 제공했다.
- **→ 평가 문화**: "LM 개선은 perplexity가 아니라 다운스트림(WER)으로 증명한다"는 실증 스타일.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">Reference</span>

- [1] Y. Bengio et al., "A Neural Probabilistic Language Model" (JMLR 2003)
- [2] T. Mikolov et al., "Recurrent neural network based language model" (Interspeech 2010)
- [3] T. Mikolov et al., "Extensions of recurrent neural network language model" (ICASSP 2011)
- [4] T. Mikolov et al., "Linguistic Regularities in Continuous Space Word Representations" (NAACL 2013)
