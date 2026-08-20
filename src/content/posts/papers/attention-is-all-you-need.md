---
title: "[논문 리뷰]Attention Is All You Need"
date: 2026-08-20
category: deep-learning-paper
tags:
  - "Transformer"
  - "Attention"
  - "NLP"
  - "Seq2Seq"
---

<span style="font-size:17pt">
<b>Attention Is All You Need</b>
</span>

<a href="https://arxiv.org/abs/1706.03762" target="_blank"><b>[PDF]</b></a>
, <b><span style="color: #F2AA4C">Transformer</span></b>, Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google, NeurIPS 2017)

### <span style="color: #ffd33d">Summary</span>

이 논문을 간단하게 요약하면, seq2seq 모델에서 당연하게 쓰이던 **recurrence(RNN)와 convolution을 전부 제거하고
attention 연산만으로** encoder-decoder를 구성한 **Transformer**를 제안한 논문이다.

- RNN의 근본 문제는 $h_t$가 $h_{t-1}$에 의존하는 **순차성** 때문에 시퀀스 내부에서 병렬화가 불가능하다는 것이다.
Transformer는 모든 위치를 한번에 attention으로 연결해서 **전체 시퀀스를 병렬로 계산**하고,
임의의 두 위치 사이의 경로 길이(path length)도 $O(1)$로 만든다.
- 핵심 부품은 3개다: **Scaled Dot-Product Attention** ($\frac{1}{\sqrt{d_k}}$ 스케일링이 왜 필요한지 아래에서 증명),
**Multi-Head Attention** (attention을 $h$개의 저차원 subspace로 쪼개서 병렬 수행),
**Sinusoidal Positional Encoding** (순서 정보가 없는 attention에 위치 정보 주입, 상대 위치가 선형 변환으로 표현됨을 증명).
- WMT14 English→German에서 BLEU **28.4** (당시 SOTA 대비 +2.0 이상), English→French에서 **41.8**을 달성했고,
학습 비용은 8×P100으로 3.5일 — 기존 SOTA들의 수분의 1 수준이다.

이후 BERT, GPT, ViT, 그리고 diffusion model의 UNet 속 attention까지 전부 이 논문의 부품을 쓰고 있으므로
딥러닝 논문 계보에서 가장 영향력이 큰 논문 중 하나다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[1] Intro & Background</span>

- RNN, LSTM[13], GRU는 language modeling, machine translation 같은 sequence modeling에서 SOTA로 확고하게 자리잡고 있었다.
  - recurrent 모델은 입력/출력의 위치를 따라 계산을 나눠서 진행한다. 즉 위치 $t$의 hidden state $h_t$는
이전 hidden state $h_{t-1}$과 현재 입력의 함수로 생성된다.
  - 이런 **본질적인 순차성 때문에 학습 샘플 내부에서 병렬화가 불가능**하고, 이는 메모리 제약 때문에 batch로도
못 묶는 긴 시퀀스에서 치명적이다.
  - factorization trick[21]이나 conditional computation[32] 같은 연구들이 계산 효율을 개선했지만 순차성이라는
근본적인 제약은 그대로 남아있다.
- Attention mechanism 자체는 이미 seq2seq의 필수 부품이었다[2, 19]. 입력/출력의 거리에 상관없이 dependency를
모델링할 수 있게 해주지만, 거의 모든 경우 **RNN에 붙어서** 사용되고 있었다.
- 순차 계산을 줄이려는 다른 시도로 Extended Neural GPU[16], ByteNet[18], ConvS2S[9]가 있는데 전부 CNN 기반이다.
  - 이 모델들은 임의의 두 위치를 연결하는 데 필요한 연산 수가 거리에 따라 증가한다. (ConvS2S는 선형, ByteNet은 로그)
  - 그래서 **멀리 떨어진 위치 사이의 dependency를 배우기 어렵다.**
  - Transformer는 이걸 상수 $O(1)$로 줄인다. (대신 attention-weighted 평균 때문에 유효 해상도가 줄어드는 비용이
있는데, 이건 Multi-Head Attention으로 상쇄한다)
- **Self-attention**(intra-attention)은 한 시퀀스 내부의 위치들끼리 attention 해서 시퀀스의 representation을 계산하는
것으로, reading comprehension, summarization 등에서 이미 사용되고 있었다[4, 27].
- Transformer는 **RNN/CNN 없이 오로지 self-attention만으로** 입력과 출력의 representation을 계산하는 최초의 transduction 모델이다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[2] Model Architecture — 부품 하나하나 뜯어보기</span>

#### <span style="color: #4682B4">2.1 전체 구조 (Encoder-Decoder)</span>

- **Encoder**: 동일한 layer $N=6$개 스택. 각 layer는 2개의 sub-layer로 구성된다.
  1. Multi-Head **Self**-Attention
  2. Position-wise Feed-Forward Network
- **Decoder**: 동일한 layer $N=6$개 스택. 각 layer는 3개의 sub-layer로 구성된다.
  1. **Masked** Multi-Head Self-Attention (미래 위치를 못 보게 masking)
  2. Multi-Head **Cross**-Attention (Query는 decoder에서, Key/Value는 encoder 출력에서)
  3. Position-wise Feed-Forward Network
- 모든 sub-layer에 **residual connection[11] + Layer Normalization[1]** 이 붙는다. (Post-LN 구조)

$$
    \text{output} = LayerNorm\big(x + Sublayer(x)\big)
$$

- residual을 더하기 위해 모든 sub-layer와 embedding의 출력 차원을 $d_{model}=512$로 통일한다.
- 참고: 이후 연구들(GPT-2 등)은 LayerNorm을 sub-layer 앞으로 옮긴 **Pre-LN** ($x + Sublayer(LayerNorm(x))$)을 쓰는데,
깊은 모델에서 학습이 더 안정적이기 때문이다. 원조 논문은 Post-LN이다.

#### <span style="color: #4682B4">2.2 Scaled Dot-Product Attention</span>

- attention은 "Query와 Key의 유사도로 가중치를 만들어 Value를 가중합하는" 연산이다.
Query/Key는 $d_k$차원, Value는 $d_v$차원.

$$
    Attention(Q, K, V) = softmax\left( \frac{QK^{\top}}{\sqrt{d_k}} \right)V
$$

- $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$이면
$QK^{\top}$는 $n \times m$의 유사도 행렬이고, row별 softmax 후 $V$를 곱하면 $n \times d_v$가 나온다.
행렬곱 2번으로 끝나서 고도로 최적화된 GEMM으로 계산된다.
- 기존 attention 두 계열과 비교하면:
  - **Additive attention**[2]: $score(q,k) = v^{\top}\tanh(W_q q + W_k k)$ — 작은 FFN으로 유사도 계산.
  - **Dot-product attention**[19]: $score(q,k) = q^{\top}k$ — 빠르고 메모리 효율적이지만 $d_k$가 크면 성능 하락.
  - 논문은 dot-product를 쓰되 $\frac{1}{\sqrt{d_k}}$로 나눠서 $d_k$가 클 때의 문제를 해결한다.

<details>
<summary> <span style="color: #ffd33d">왜 1/sqrt(d_k)로 나누는가 — 분산 계산 + softmax 포화 증명 펼치기/접기</span> </summary>

- **Step 1 — dot product의 분산이 $d_k$에 비례한다.**
$q, k$의 각 성분이 서로 독립이고 평균 0, 분산 1이라고 가정하면 ($q_i, k_i \sim$ i.i.d., $\mathbb{E}=0$, $Var=1$)

$$
    q \cdot k = \sum_{i=1}^{d_k}{q_i k_i}
$$

- 각 항의 평균과 분산은

$$
    \mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0
    ,\qquad
    Var[q_i k_i] = \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1
$$

- 독립인 항들의 합이므로 분산이 그대로 더해진다.

$$
    \mathbb{E}[q \cdot k] = 0, \qquad Var[q \cdot k] = \sum_{i=1}^{d_k}{1} = d_k
$$

- 즉 $d_k = 64$면 logit의 표준편차가 $8$이나 된다. $\frac{q \cdot k}{\sqrt{d_k}}$로 나누면 분산이 다시 1로 정규화된다.

- **Step 2 — logit이 크면 softmax의 gradient가 사라진다.**
softmax $s_i = \frac{e^{z_i}}{\sum_j{e^{z_j}}}$의 Jacobian은

$$
    \frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)
$$

- logit의 스케일이 커지면 softmax는 최댓값 위치만 1인 one-hot에 가까워진다. 이때
$s_i \approx 1$인 곳은 $s_i(1-s_i) \approx 0$, $s_i \approx 0$인 곳도 $s_i(\cdots) \approx 0$이 되어
**Jacobian의 모든 원소가 0으로 붕괴**한다. 즉 attention 가중치로 gradient가 흐르지 않아 학습이 안 된다.
- 스케일링은 이 포화 영역에 진입하는 것을 막는 장치다. $\blacksquare$

</details>

<hr/> <!-- 수평선 -->

- **Masking**: decoder self-attention에서 위치 $i$가 $j > i$를 참조하지 못하게, softmax 전에 해당 logit을
$-\infty$로 설정한다. (softmax 후 가중치가 정확히 0이 됨) 이것이 auto-regressive 성질을 보존한다.

#### <span style="color: #4682B4">2.3 Multi-Head Attention</span>

- $d_{model}$ 차원 전체로 attention 한 번을 하는 대신, **서로 다른 학습된 projection으로 $h$번 쪼개서** attention을
병렬 수행하고 결과를 concat 한다.

$$
    \begin{split}
    MultiHead(Q, K, V) &= Concat(head_1, ..., head_h)\,W^O
    \\ head_i &= Attention(QW_i^Q,\; KW_i^K,\; VW_i^V)
    \end{split}
$$

$$
    W_i^Q \in \mathbb{R}^{d_{model} \times d_k},\quad
    W_i^K \in \mathbb{R}^{d_{model} \times d_k},\quad
    W_i^V \in \mathbb{R}^{d_{model} \times d_v},\quad
    W^O \in \mathbb{R}^{hd_v \times d_{model}}
$$

- base 모델은 $h=8$, $d_k = d_v = d_{model}/h = 64$를 사용한다.
- **왜 쪼개는가**: 단일 attention은 softmax 가중평균 하나로 정보를 압축해버린다(averaging이 표현력을 깎음).
head를 나누면 각 head가 **서로 다른 representation subspace에서 서로 다른 위치 관계**를 볼 수 있다.
(실제로 학습된 head들을 시각화하면 구문 구조, 장거리 의존성 등 다른 역할을 하는 것이 관찰됨)
- **계산량은 공짜**: head당 차원을 $1/h$로 줄였기 때문에 전체 계산량은 full-dimension 단일 attention과 같다.
- 파라미터 수를 세보면 MHA 하나당 $W^Q, W^K, W^V, W^O$ 각각 $d_{model}^2$ ($h$개 head의 projection을 합치면
$h \times d_{model} \times d_k = d_{model}^2$)이므로 총 $4d_{model}^2 = 4 \times 512^2 \approx 1.05M$개다.

#### <span style="color: #4682B4">2.4 Attention이 쓰이는 3곳</span>

| 위치 | Q | K, V | Mask | 역할 |
|---|---|---|---|---|
| Encoder self-attn | encoder 이전 layer | encoder 이전 layer | ✗ | 입력 문장 내부의 모든 위치 참조 |
| Decoder self-attn | decoder 이전 layer | decoder 이전 layer | ✓ (미래 차단) | 지금까지 생성된 출력 참조 |
| Encoder-Decoder cross-attn | decoder | **encoder 최종 출력** | ✗ | 번역 원문 참조 (기존 seq2seq attention 역할) |

#### <span style="color: #4682B4">2.5 Position-wise Feed-Forward Network</span>

- 각 위치마다 **독립적으로, 동일하게** 적용되는 2-layer MLP.

$$
    FFN(x) = \max(0,\; xW_1 + b_1)\,W_2 + b_2
$$

- 안쪽 차원은 $d_{ff} = 2048$로 $d_{model}$의 4배다. (이 4배 비율도 이후 모델들의 표준이 됨)
- kernel size 1짜리 convolution 2번으로 볼 수도 있다. 파라미터는 layer당 $2 \times 512 \times 2048 \approx 2.1M$개로
사실 attention보다 FFN이 파라미터를 더 많이 먹는다.

#### <span style="color: #4682B4">2.6 Embedding과 Softmax의 Weight 공유</span>

- 입력 embedding, 출력 embedding, softmax 직전 linear 이 3개의 weight 행렬을 **공유**한다[30].
- embedding layer에서는 weight에 $\sqrt{d_{model}}$을 곱해준다.
(공유된 행렬이 softmax용 스케일에 맞춰져 있어서, embedding으로 쓸 때는 positional encoding과 스케일을 맞추기 위한 보정)

#### <span style="color: #4682B4">2.7 Positional Encoding</span>

- attention은 집합 연산이라 **순서 개념이 전혀 없다.** 그래서 위치 정보를 embedding에 직접 더해준다.
- 논문은 주파수가 기하급수적으로 변하는 sin/cos 함수를 사용한다. ($pos$는 위치, $i$는 차원 인덱스)

$$
    PE_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i/d_{model}}} \right)
    ,\qquad
    PE_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{2i/d_{model}}} \right)
$$

- 파장이 $2\pi$부터 $10000 \cdot 2\pi$까지 기하 수열을 이룬다. 낮은 차원은 빠르게 진동(세밀한 위치),
높은 차원은 느리게 진동(거시적 위치)해서 이진수 표현의 연속 버전처럼 동작한다.
- 이 함수를 고른 이유: **상대 위치 $PE_{pos+k}$가 $PE_{pos}$의 선형 변환으로 표현**되기 때문에 모델이 상대적인
위치 관계를 배우기 쉬울 것이라는 가설.

<details>
<summary> <span style="color: #ffd33d">상대 위치가 선형 변환이 되는 증명 (회전 행렬) 펼치기/접기</span> </summary>

- 차원 쌍 $(2i, 2i+1)$ 하나만 보자. 각주파수를 $\omega_i = 10000^{-2i/d_{model}}$로 두면 위치 $pos$의 값은
$(\sin(\omega_i\, pos),\; \cos(\omega_i\, pos))$이다.
- 삼각함수 덧셈정리를 쓰면

$$
    \begin{split}
    \sin(\omega_i(pos+k)) &= \sin(\omega_i pos)\cos(\omega_i k) + \cos(\omega_i pos)\sin(\omega_i k)
    \\ \cos(\omega_i(pos+k)) &= \cos(\omega_i pos)\cos(\omega_i k) - \sin(\omega_i pos)\sin(\omega_i k)
    \end{split}
$$

- 행렬로 묶으면

$$
    \begin{pmatrix} \sin(\omega_i(pos+k)) \\ \cos(\omega_i(pos+k)) \end{pmatrix}
    =
    \underbrace{\begin{pmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{pmatrix}}_{M_i(k)}
    \begin{pmatrix} \sin(\omega_i pos) \\ \cos(\omega_i pos) \end{pmatrix}
$$

- $M_i(k)$는 **$pos$와 무관하게 offset $k$에만 의존하는 회전 행렬**이다. 모든 차원 쌍에 대해 이런 블록들을
대각으로 쌓으면 $PE_{pos+k} = M(k) \cdot PE_{pos}$가 성립한다. $\blacksquare$
- 즉 "$k$칸 이동"이라는 상대 위치 연산이 항상 같은 선형 변환이라서, attention의 linear projection이
상대 위치 관계를 자연스럽게 포착할 수 있다.
- 참고로 이 "위치 = 회전" 아이디어를 끝까지 밀어붙인 것이 요즘 LLM들이 쓰는 RoPE(Rotary Position Embedding)다.

</details>

<hr/> <!-- 수평선 -->

- 학습하는 positional embedding[9]과 비교 실험을 했는데 **성능이 거의 동일**했다. sinusoidal을 선택한 이유는
학습 때 본 것보다 긴 시퀀스에도 외삽(extrapolation)이 가능할 것이라는 기대 때문.

#### <span style="color: #4682B4">2.8 모델 사양 정리</span>

| | $N$ | $d_{model}$ | $d_{ff}$ | $h$ | $d_k=d_v$ | $P_{drop}$ | steps | Params |
|---|---|---|---|---|---|---|---|---|
| base | 6 | 512 | 2048 | 8 | 64 | 0.1 | 100K | 65M |
| big | 6 | 1024 | 4096 | 16 | 64 | 0.3 | 300K | 213M |

#### <span style="color: #4682B4">2.9 파라미터가 어디에 있는지 세보기 (base 기준)</span>

- 65M이 어디에 분포하는지 직접 계산해보면 구조가 더 잘 보인다.

| 부품 | 개수 | 파라미터/개 | 소계 |
|---|---|---|---|
| MHA ($W^Q,W^K,W^V,W^O$) | 18개 (enc self 6 + dec self 6 + cross 6) | $4 \times 512^2 \approx 1.05M$ | ≈ 18.9M |
| FFN ($W_1, W_2$) | 12개 (enc 6 + dec 6) | $2 \times 512 \times 2048 \approx 2.1M$ | ≈ 25.2M |
| Embedding (공유 1벌) | 1개 | $37000 \times 512 \approx 18.9M$ | ≈ 18.9M |
| LayerNorm 등 기타 | - | - | ≈ 1M 미만 |

- 인상과 달리 **attention(29%)보다 FFN(39%)이 더 크고, embedding도 29%나 된다.**
"Transformer의 몸통은 FFN"이라는 사실은 이후 스케일링/MoE 연구(FFN만 sparse하게 키우는)의 배경이 된다.
- weight 공유(2.6)가 없었다면 embedding 계열만 +38M이 됐을 것이다 — 공유가 파라미터 효율에 꽤 기여한다.

#### <span style="color: #4682B4">2.10 Inference — auto-regressive 디코딩</span>

- 학습 때는 정답 시퀀스 전체를 넣고 masking으로 병렬 학습(teacher forcing)하지만,
**추론은 한 토큰씩 순차 생성**이다: `<bos>`에서 시작해 생성된 토큰을 다시 decoder 입력에 붙인다.
- 이때 이전 스텝의 Key/Value는 변하지 않으므로 **캐싱(KV cache)** 하면 스텝당 새 토큰의 Q 계산만 하면 된다.
지금 LLM inference의 KV cache가 바로 이 구조에서 나온 것이다.
- 즉 "Transformer는 학습은 병렬, 생성은 여전히 순차"라는 비대칭이 있다 — RNN의 순차성을 학습에서만
제거한 셈이고, 이 생성 병목은 이후 speculative decoding 등의 연구 주제가 된다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[3] Why Self-Attention</span>

- self-attention을 RNN/CNN과 3가지 기준으로 비교한다: layer당 계산 복잡도, 순차 연산의 최소 횟수(병렬화 가능성),
네트워크 내 장거리 의존성의 최대 경로 길이. ($n$: 시퀀스 길이, $d$: 차원, $k$: 커널 크기, $r$: 제한된 윈도우 크기)

| Layer Type | Complexity per Layer | Sequential Ops | Maximum Path Length |
|---|---|---|---|
| **Self-Attention** | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k{n})$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

- **경로 길이가 짧을수록 장거리 의존성 학습이 쉽다**는 것이 비교의 핵심 논리다. self-attention은 어떤 두 위치든
한 번의 attention으로 직접 연결된다.
- 계산 복잡도는 $n < d$일 때 self-attention이 RNN보다 빠르다. 기계번역에서 쓰는 sentence-piece 시퀀스는
대부분 $n < d = 512$라 실제로 유리하다.
  - 반대로 $n$이 아주 길어지면 $O(n^2)$이 병목이 된다 — 논문도 restricted self-attention(주변 $r$개만 참조)을
언급하고 있고, 이 $O(n^2)$ 문제는 이후 Sparse/Linear attention, FlashAttention 등 수많은 후속 연구를 낳는다.
- 부가 효과로 attention 가중치를 시각화하면 **해석 가능성**도 얻는다. (head별로 구문/의미 역할 분화가 관찰됨)

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[4] Training</span>

- **데이터**: WMT 2014 English-German (4.5M 문장쌍, 37K shared BPE vocab), English-French (36M 문장쌍, 32K word-piece).
  - 비슷한 길이끼리 batch로 묶고, batch당 대략 source 25K + target 25K 토큰.
- **하드웨어**: 8× NVIDIA P100. base는 step당 0.4초 × 100K steps = **12시간**, big은 step당 1.0초 × 300K steps = **3.5일**.
- **Optimizer**: Adam ($\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$)에 다음 learning rate 스케줄을 사용한다.

$$
    lrate = d_{model}^{-0.5} \cdot \min\big( step^{-0.5},\; step \cdot warmup^{-1.5} \big)
$$

- $warmup = 4000$ step까지 선형 증가, 이후 $step^{-0.5}$로 감소하는 스케줄이다. 두 구간이 $step = warmup$에서
정확히 만나도록 $-1.5$ 지수가 맞춰져 있다. (Post-LN Transformer는 초반 learning rate가 크면 발산하기 쉬워서
이 warmup이 사실상 필수라는 것이 이후 연구들에서 밝혀진다)
- **Regularization** 3종:
  1. **Residual Dropout** ($P_{drop}=0.1$): 각 sub-layer 출력에, 그리고 embedding + PE 합에도 적용.
  2. **Label Smoothing** ($\epsilon_{ls}=0.1$)[36]: 정답에 $1-\epsilon$, 나머지에 $\epsilon$을 분배한 soft target을 사용.
     perplexity는 나빠지지만(모델이 더 unsure 해짐) **accuracy와 BLEU는 좋아진다.**
- **Checkpoint Averaging**: 마지막 체크포인트 하나가 아니라 **최근 체크포인트들의 weight를 평균**낸 모델로
평가한다 (base: 마지막 5개, big: 마지막 20개, 10분 간격 저장). 공짜로 BLEU가 오르는 고전 트릭으로,
이후 NMT/LLM 학습(가중치 평균, EMA)에서도 계속 쓰인다.
- **Decoding**: beam search (beam size 4, length penalty $\alpha = 0.6$), 최대 출력 길이는 입력 + 50이되
가능하면 조기 종료.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[5] Results</span>

#### <span style="color: #4682B4">5.1 Machine Translation (WMT 2014)</span>

| Model | EN→DE BLEU | EN→FR BLEU | Training Cost (FLOPs) |
|---|---|---|---|
| ByteNet | 23.75 | - | - |
| ConvS2S | 25.16 | 40.46 | $9.6 \times 10^{18}$ |
| GNMT + RL | 24.6 | 39.92 | $2.3 \times 10^{19}$ |
| ConvS2S Ensemble | 26.36 | 41.29 | $1.2 \times 10^{21}$ |
| **Transformer (base)** | 27.3 | 38.1 | $\mathbf{3.3 \times 10^{18}}$ |
| **Transformer (big)** | **28.4** | **41.8** | $2.3 \times 10^{19}$ |

- big 모델이 **이전의 앙상블 모델들까지 포함해서** 전부 이기고, 학습 비용은 오히려 몇 분의 1 수준이다.
base 모델조차 이전 single 모델 전부를 EN→DE에서 이긴다.
- inference는 beam search (beam size 4, length penalty $\alpha=0.6$) 사용.

#### <span style="color: #4682B4">5.2 Ablation (Table 3)</span>

- base 모델에서 하나씩 바꿔가며 EN→DE dev set(newstest2013)의 perplexity/BLEU 변화를 측정한다. 주요 행 발췌:

| 변경 내용 | PPL (dev) | BLEU (dev) |
|---|---|---|
| **base** ($h=8$) | 4.92 | 25.8 |
| (A) $h=1$ | 5.29 | 24.9 |
| (A) $h=4$ | 5.00 | 25.5 |
| (A) $h=16$ | 4.91 | 25.8 |
| (A) $h=32$ | 5.01 | 25.4 |
| (D) dropout 제거 ($P_{drop}=0$) | 5.77 | 24.6 |
| (D) label smoothing 제거 | **4.67** | 25.3 |
| (E) learned positional embedding | 4.92 | 25.7 |
| **big** | 4.33 | **26.4** |

- 행별로 읽어보면:
  - **head 수 (A)**: 1개면 BLEU가 0.9 떨어지고, 32개로 너무 늘려도 오히려 하락 — $h=8{\sim}16$이 sweet spot.
head가 "여러 관계를 병렬로 본다"는 가설의 정량 근거다.
  - **$d_k$ 축소 (B)**: $d_k$를 줄이면 품질 하락 — Q·K 유사도 계산이 그렇게 만만한 문제가 아니라는 해석.
  - **모델 크기 (C)**: $d_{model}$, $d_{ff}$를 키울수록 일관되게 좋아진다. (스케일링의 초기 증거)
  - **dropout (D)**: 제거하면 PPL 5.77로 명확한 overfitting.
  - **label smoothing (D)**: 제거하면 **PPL은 가장 좋아지는데(4.67) BLEU는 떨어진다(25.3)** —
"PPL과 생성 품질은 다른 지표"임을 보여주는 유명한 행이다.
  - **learned PE (E)**: sinusoidal과 사실상 동일 (4.92/25.7). 그래서 외삽 가능성을 보고 sinusoidal 선택.
- 정리하면 Transformer의 성능은 특정 트릭이 아니라 구조 전체에서 나오고, 크기를 키우면 더 좋아진다.

#### <span style="color: #4682B4">5.3 English Constituency Parsing (일반화 확인)</span>

- 번역이 아닌 구문 분석에도 거의 그대로(4-layer, $d_{model}=1024$) 적용해봤다.
  - WSJ만 학습: F1 91.3 — task-specific 튜닝 없이 대부분의 기존 모델을 이김.
  - semi-supervised: F1 92.7.
- 출력이 입력보다 길고 강한 구조 제약이 있는 task에서도 동작한다는 일반화 증거.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[6] 개인적인 생각</span>

- 이 논문의 부품들이 이후 어디로 이어졌는지 계보를 그려보면:
  - **Encoder만** 떼서 masked LM으로 학습 → BERT
  - **Decoder만** 떼서 auto-regressive LM으로 학습 → GPT 시리즈
  - 이미지를 패치 시퀀스로 취급 → ViT
  - diffusion model의 UNet 안 attention block도 이 논문의 multi-head attention이다.
([Diffusion Models Beat GANs 리뷰](/posts/diffusion-models-beat-gans-on-image-synthesis/)의 [3]장에서 뜯었던
32/16/8 해상도 attention이 정확히 이 부품)
- 지금 시점에서 다시 읽으면 "당시엔 몰랐던 복선"이 많다. ablation의 "키우면 더 좋아진다"는 한 줄이 scaling law로,
sinusoidal PE의 회전 행렬 해석이 RoPE로, $O(n^2)$ 병목이 FlashAttention/linear attention 연구로 이어졌다.
- 수식 자체는 어렵지 않은데(사실상 행렬곱 + softmax가 전부), **"왜 이렇게 설계했는가"** 에 대한 논증
(scaling의 분산 논리, path length 비교, head 분리의 subspace 논리)이 이 논문의 진짜 가치라고 생각한다.
읽을 때 결과 표보다 Section 4 (Why Self-Attention)를 곱씹는 것을 추천.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">Reference</span>

- [1] J. Ba et al., "Layer Normalization" (2016)
- [2] D. Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
- [9] J. Gehring et al., "Convolutional Sequence to Sequence Learning" (ConvS2S, 2017)
- [11] K. He et al., "Deep Residual Learning for Image Recognition" (ResNet, 2016)
- [18] N. Kalchbrenner et al., "Neural Machine Translation in Linear Time" (ByteNet, 2017)
- [19] M. Luong et al., "Effective Approaches to Attention-based Neural Machine Translation" (2015)
- [30] O. Press & L. Wolf, "Using the Output Embedding to Improve Language Models" (2017)
- [36] C. Szegedy et al., "Rethinking the Inception Architecture for Computer Vision" (Label Smoothing, 2016)
