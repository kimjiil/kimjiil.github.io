---
title: "[논문 리뷰]Attention Is All You Need"
date: 2026-08-20
updated: 2026-08-21
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

seq2seq에서 당연하게 쓰이던 **recurrence와 convolution을 전부 제거하고 attention만으로** encoder-decoder를
구성한 **Transformer** 논문. WMT14 English→German BLEU **28.4**(앙상블 포함 기존 최고 대비 +2.0 이상),
English→French **41.8**(단일 모델 신기록)을 8×P100 **3.5일** 학습으로 달성했다 — 기존 SOTA들의 수분의 1 비용.

- RNN의 근본 문제는 $h_t$가 $h_{t-1}$에 의존하는 **순차성** — 시퀀스 내부 병렬화가 불가능하다.
Transformer는 전체 시퀀스를 병렬로 계산하고, 임의의 두 위치 사이 경로 길이도 $O(1)$로 만든다.
- 핵심 부품 3개: **Scaled Dot-Product Attention**($\frac{1}{\sqrt{d_k}}$의 이유를 분산으로 증명),
**Multi-Head Attention**(저차원 subspace $h$개로 쪼개 병렬 attention), **Sinusoidal Positional Encoding**
(상대 위치가 선형 변환이 됨을 증명).
- 재밌는 각주: 저자 8명이 **전원 equal contribution, 나열 순서는 랜덤**이다. "RNN을 self-attention으로
대체하자"는 Jakob의 제안, scaled dot-product/multi-head/파라미터 없는 위치 표현은 Noam의 제안이라고
기여를 명시해뒀다.

이후 BERT, GPT, ViT, diffusion의 UNet 속 attention까지 전부 이 논문의 부품을 쓴다.
리뷰는 논문 섹션 구성(1~7장)을 그대로 따라간다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[1] Introduction & [2] Background</span>

- RNN(LSTM, GRU)은 sequence modeling의 확고한 SOTA였다. 하지만 **본질적인 순차성 때문에 학습 샘플
내부에서 병렬화가 불가능**하고, 이는 (메모리 제약으로 batch도 못 키우는) 긴 시퀀스에서 치명적이다.
factorization trick이나 conditional computation이 효율을 개선했지만 순차성이라는 근본 제약은 그대로다.
- attention은 이미 seq2seq의 필수 부품이었지만[Bahdanau], 거의 모든 경우 **RNN에 붙어서** 쓰였다.
- 순차 계산을 줄이려는 선행 시도들은 전부 CNN 기반이다 — Extended Neural GPU, ByteNet, ConvS2S.
  - 이들은 임의의 두 위치를 연결하는 연산 수가 **거리에 따라 증가**한다 (ConvS2S 선형, ByteNet 로그).
멀리 떨어진 위치의 의존성을 배우기 어렵다.
  - Transformer는 이걸 **상수로** 줄인다. 대신 attention 가중평균 때문에 유효 해상도가 줄어드는
비용이 있는데, 이는 Multi-Head Attention으로 상쇄한다. (3.2.2의 복선)
- self-attention(intra-attention) 자체는 독해, 요약, 함의, 문장 표현 학습에서 이미 쓰이고 있었다.
- 선언: **"Transformer는 sequence-aligned RNN이나 convolution 없이 전적으로 self-attention만으로
입출력 표현을 계산하는 최초의 transduction 모델이다."**

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[3] Model Architecture — 부품 하나하나 뜯어보기</span>

#### <span style="color: #4682B4">3.1 Encoder and Decoder Stacks</span>

- 전체는 표준 encoder-decoder: encoder가 $(x_1,...,x_n) \rightarrow z$, decoder가 $z$에서
$(y_1,...,y_m)$을 **auto-regressive**하게 (이전 출력을 입력에 붙여가며) 생성한다.
- **Encoder**: 동일 layer $N=6$ 스택, 각 layer는 2개 sub-layer — ① multi-head self-attention,
② position-wise FFN.
- **Decoder**: $N=6$ 스택, 각 layer는 3개 sub-layer — ① **masked** self-attention,
② encoder 출력에 대한 multi-head attention(cross), ③ FFN.
  - masking + "출력 embedding을 한 칸 offset"의 조합으로 위치 $i$의 예측이 $i$ 미만의 출력에만
의존하게 보장한다.
- 모든 sub-layer에 **residual connection + LayerNorm** (Post-LN):

$$
    \text{output} = LayerNorm\big(x + Sublayer(x)\big)
$$

- residual을 위해 모든 sub-layer와 embedding의 출력 차원을 $d_{model} = 512$로 통일.
- 참고: 이후 연구들(GPT-2 등)은 LayerNorm을 앞으로 옮긴 **Pre-LN**을 쓴다 — 깊은 모델에서 더 안정적.
원조는 Post-LN이다.

#### <span style="color: #4682B4">3.2 Attention</span>

- attention의 일반 정의: **query와 (key, value) 쌍들을 출력으로 매핑**하는 함수. 출력은 value들의
가중합이고, 가중치는 query와 각 key의 호환성 함수로 계산된다.

**3.2.1 Scaled Dot-Product Attention**

$$
    Attention(Q, K, V) = softmax\left( \frac{QK^{\top}}{\sqrt{d_k}} \right)V
$$

- 기존 두 계열과의 비교 (논문 서술 그대로):
  - **Additive attention**[Bahdanau]: 1-hidden FFN으로 호환성 계산.
  - **Dot-product attention**: scaling 없는 내적. 이론적 복잡도는 additive와 비슷하지만
**고도로 최적화된 행렬곱으로 구현되어 실전에서 훨씬 빠르고 메모리 효율적**이다.
  - $d_k$가 작으면 둘이 비슷하고, $d_k$가 크면 **scaling 없는 dot-product가 additive에 진다** —
그래서 $\frac{1}{\sqrt{d_k}}$를 붙인다.

<details>
<summary> <span style="color: #ffd33d">왜 1/sqrt(d_k)로 나누는가 — 분산 계산 + softmax 포화 증명 펼치기/접기</span> </summary>

- **Step 1 — 내적의 분산이 $d_k$에 비례** (논문 각주 4의 논증을 전개).
$q, k$의 성분이 서로 독립이고 평균 0, 분산 1이라 하자.

$$
    q \cdot k = \sum_{i=1}^{d_k}{q_i k_i}
    ,\qquad
    \mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0
$$

$$
    Var[q_i k_i] = \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1
$$

- 독립 항들의 합이므로 분산이 더해진다: $\mathbb{E}[q \cdot k] = 0$, $Var[q \cdot k] = d_k$.
$d_k = 64$면 logit의 표준편차가 8이나 된다. $\sqrt{d_k}$로 나누면 분산이 1로 정규화된다.

- **Step 2 — logit이 크면 softmax gradient가 사라진다.**
softmax $s_i = \frac{e^{z_i}}{\sum_j{e^{z_j}}}$의 Jacobian은

$$
    \frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)
$$

- logit 스케일이 커지면 softmax는 one-hot에 수렴한다. $s_i \approx 1$인 곳은 $s_i(1-s_i) \approx 0$,
$s_i \approx 0$인 곳도 0 — **Jacobian 전체가 붕괴**해서 attention 가중치로 gradient가 흐르지 않는다.
scaling은 이 포화 영역 진입을 막는 장치다. $\blacksquare$

</details>

<hr/> <!-- 수평선 -->

**3.2.2 Multi-Head Attention**

- $d_{model}$ 차원으로 attention 한 번을 하는 대신, **학습된 projection으로 $h$번 쪼개서** 병렬로
attention 하고 concat 후 다시 projection 한다.

$$
    \begin{split}
    MultiHead(Q, K, V) &= Concat(head_1, ..., head_h)\,W^O
    \\ head_i &= Attention(QW_i^Q,\; KW_i^K,\; VW_i^V)
    \end{split}
$$

$$
    W_i^Q, W_i^K \in \mathbb{R}^{d_{model} \times d_k},\quad
    W_i^V \in \mathbb{R}^{d_{model} \times d_v},\quad
    W^O \in \mathbb{R}^{hd_v \times d_{model}}
$$

- $h = 8$, $d_k = d_v = d_{model}/h = 64$.
- **왜 쪼개는가** (논문 문장 그대로): *"multi-head attention은 서로 다른 위치의 서로 다른 representation
subspace 정보를 동시에 참조하게 한다. 단일 head면 averaging이 이걸 방해한다."* — 2장에서 예고한
"유효 해상도 손실"의 해결책.
- **계산량은 공짜**: head당 차원을 $1/h$로 줄였으므로 전체 비용은 full-dimension 단일 head와 같다.
- 파라미터를 세보면 MHA 하나당 $W^Q, W^K, W^V, W^O$ 합쳐 $4d_{model}^2 = 4 \times 512^2 \approx 1.05M$개.

**3.2.3 Attention이 쓰이는 3곳**

| 위치 | Q | K, V | Mask | 역할 |
|---|---|---|---|---|
| Encoder self-attn | 이전 encoder layer | 이전 encoder layer | ✗ | 입력의 모든 위치 참조 |
| Decoder self-attn | 이전 decoder layer | 이전 decoder layer | ✓ | 현재까지의 출력만 참조 (auto-regressive 보존) |
| Encoder-decoder attn | decoder | **encoder 최종 출력** | ✗ | 기존 seq2seq attention의 역할 |

- masking 구현: softmax 입력에서 불법 연결에 해당하는 값을 $-\infty$로 설정 (softmax 후 정확히 0).

#### <span style="color: #4682B4">3.3 Position-wise Feed-Forward Networks</span>

- 각 위치에 **독립적으로, 동일하게** 적용되는 2-layer MLP (ReLU):

$$
    FFN(x) = \max(0,\; xW_1 + b_1)\,W_2 + b_2
$$

- $d_{ff} = 2048$ ($d_{model}$의 4배 — 이후 표준이 되는 비율). "kernel size 1짜리 convolution 두 번"으로
볼 수도 있다고 논문이 명시. 파라미터는 layer당 $2 \times 512 \times 2048 \approx 2.1M$개 —
**사실 attention보다 FFN이 파라미터를 더 먹는다.**

#### <span style="color: #4682B4">3.4 Embeddings and Softmax</span>

- 입력 embedding, 출력 embedding, softmax 직전 linear의 **weight 3벌을 공유**한다 (Press & Wolf).
embedding으로 쓸 때는 $\sqrt{d_{model}}$을 곱한다.

#### <span style="color: #4682B4">3.5 Positional Encoding</span>

- attention은 순서 개념이 없으므로 위치 정보를 embedding에 **더해준다**. 주파수가 기하수열
($2\pi \rightarrow 10000 \cdot 2\pi$)을 이루는 sin/cos:

$$
    PE_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i/d_{model}}} \right)
    ,\qquad
    PE_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{2i/d_{model}}} \right)
$$

- 선택 이유 (논문의 가설): **고정 offset $k$에 대해 $PE_{pos+k}$가 $PE_{pos}$의 선형 함수**라서
상대 위치 참조를 배우기 쉬울 것.

<details>
<summary> <span style="color: #ffd33d">상대 위치가 선형 변환이 되는 증명 (회전 행렬) 펼치기/접기</span> </summary>

- 차원 쌍 $(2i, 2i+1)$ 하나만 보자. 각주파수 $\omega_i = 10000^{-2i/d_{model}}$로 두면 위치 $pos$의 값은
$(\sin(\omega_i pos), \cos(\omega_i pos))$. 삼각함수 덧셈정리로

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

- $M_i(k)$는 **$pos$와 무관하게 offset $k$에만 의존하는 회전 행렬**. 모든 차원 쌍의 블록을 대각으로
쌓으면 $PE_{pos+k} = M(k) \cdot PE_{pos}$. $\blacksquare$
- "위치 = 회전" 아이디어를 끝까지 밀면 요즘 LLM의 RoPE(Rotary Position Embedding)가 된다.

</details>

<hr/> <!-- 수평선 -->

- 학습된 positional embedding과 비교 실험 결과 **거의 동일**했다 (Table 3 row E). sinusoidal을 고른 건
학습보다 긴 시퀀스로의 **외삽 가능성** 때문.

#### <span style="color: #4682B4">3.6 (보충) 파라미터가 어디에 있는지 세보기 — base 65M의 내역</span>

| 부품 | 개수 | 파라미터/개 | 소계 |
|---|---|---|---|
| MHA | 18개 (enc self 6 + dec self 6 + cross 6) | $4 \times 512^2 \approx 1.05M$ | ≈ 18.9M |
| FFN | 12개 (enc 6 + dec 6) | $2 \times 512 \times 2048 \approx 2.1M$ | ≈ 25.2M |
| Embedding (공유 1벌) | 1개 | $37000 \times 512 \approx 18.9M$ | ≈ 18.9M |
| LayerNorm 등 | - | - | 1M 미만 |

- 인상과 달리 **attention(29%)보다 FFN(39%)이 크고 embedding도 29%**다. "Transformer의 몸통은 FFN"이라는
사실이 이후 스케일링/MoE 연구(FFN만 sparse하게 키우는)의 배경이 된다. weight 공유(3.4)가 없었다면
embedding 계열만 +38M이었을 것.

#### <span style="color: #4682B4">3.7 (보충) Inference — auto-regressive 디코딩</span>

- 학습은 정답 전체 + masking으로 병렬(teacher forcing)이지만 **생성은 한 토큰씩 순차**다.
이전 스텝의 K/V는 변하지 않으므로 **캐싱(KV cache)** 하면 스텝당 새 토큰의 Q만 계산하면 된다 —
지금 LLM inference의 KV cache가 이 구조에서 나왔다.
- 즉 "학습은 병렬, 생성은 순차"라는 비대칭이 남는다. 논문도 결론에서 "생성을 덜 순차적으로 만드는 것"을
연구 목표로 명시한다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[4] Why Self-Attention</span>

- self-attention을 택한 이유를 3가지 기준으로 정당화한다: ① layer당 계산 복잡도, ② 병렬화 가능성
(필요한 최소 순차 연산 수), ③ **장거리 의존성의 최대 경로 길이** — "순방향/역방향 신호가 지나야 하는
경로가 짧을수록 장거리 의존성 학습이 쉽다". (논문 Table 1)

| Layer Type | Complexity per Layer | Sequential Ops | Maximum Path Length |
|---|---|---|---|
| **Self-Attention** | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k{n})$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

- 계산 복잡도: $n < d$일 때 self-attention이 RNN보다 빠르다 — word-piece/BPE 문장 표현은 대부분
$n < d$라 실제로 유리하다.
- $n$이 아주 길면 주변 $r$개만 보는 **restricted self-attention**($O(n/r)$ 경로)을 future work로 언급 —
$O(n^2)$ 병목은 이후 sparse/linear attention, FlashAttention 연구를 낳는다.
- convolution 관련 디테일: $k < n$인 conv 하나는 모든 쌍을 연결하지 못해 $O(n/k)$ 또는 dilated로
$O(\log_k n)$개 layer가 필요하고, separable convolution까지 낮춰도($O(k \cdot n \cdot d + n \cdot d^2)$)
**self-attention + FFN 조합과 같은 복잡도**다 — "그게 바로 우리가 택한 접근"이라는 깔끔한 마무리.
- 부수 효과: attention 분포를 시각화하면 **head별로 다른 역할**(구문/의미 구조 관련 행동)이 관찰된다 —
해석 가능성 (부록의 시각화).

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[5] Training</span>

- **5.1 데이터/배치**: WMT14 EN-DE 4.5M 문장쌍, **BPE, 공유 vocab ~37,000 토큰** / EN-FR 36M 문장쌍,
32,000 word-piece. 비슷한 길이끼리 batch로 묶고, batch당 source 25K + target 25K 토큰.
- **5.2 하드웨어/스케줄**: 8× P100. base는 step당 0.4초 × 100K steps = **12시간**. big은 step당 1.0초
× 300K steps = **3.5일**.
- **5.3 Optimizer**: Adam ($\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$) + 커스텀 스케줄:

$$
    lrate = d_{model}^{-0.5} \cdot \min\big( step^{-0.5},\; step \cdot warmup^{-1.5} \big)
$$

- $warmup = 4000$ step까지 선형 증가 후 $step^{-0.5}$로 감소. 두 구간이 $step = warmup$에서 정확히
만나도록 지수가 맞춰져 있다. (Post-LN Transformer는 초반 lr이 크면 발산하기 쉬워 이 warmup이 사실상
필수임이 이후 연구로 밝혀진다)
- **5.4 Regularization** 3종:
  1. **Residual Dropout** ($P_{drop} = 0.1$): 각 sub-layer 출력에(residual 더하기 전), 그리고
embedding + PE 합에도.
  2. **Label Smoothing** ($\epsilon_{ls} = 0.1$): perplexity는 나빠지지만(모델이 더 unsure해짐)
**accuracy와 BLEU는 좋아진다.**
  3. (6.1에서 추가되는 실전 기법) **Checkpoint Averaging**: base는 10분 간격 저장된 마지막 5개,
big은 마지막 20개 체크포인트의 weight 평균으로 평가. beam search는 beam 4, length penalty
$\alpha = 0.6$, 최대 출력 길이 입력+50 (조기 종료 가능).

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[6] Results</span>

#### <span style="color: #4682B4">6.1 Machine Translation (WMT14, newstest2014)</span>

| Model | EN→DE BLEU | EN→FR BLEU | 학습 비용 (FLOPs, EN-DE) |
|---|---|---|---|
| ByteNet | 23.75 | - | - |
| GNMT + RL | 24.6 | 39.92 | $2.3 \times 10^{19}$ |
| ConvS2S | 25.16 | 40.46 | $9.6 \times 10^{18}$ |
| MoE | 26.03 | 40.56 | $2.0 \times 10^{19}$ |
| GNMT + RL **Ensemble** | 26.30 | 41.16 | $1.8 \times 10^{20}$ |
| ConvS2S **Ensemble** | 26.36 | 41.29 | $7.7 \times 10^{19}$ |
| **Transformer (base)** | 27.3 | 38.1 | $\mathbf{3.3 \times 10^{18}}$ |
| **Transformer (big)** | **28.4** | **41.8** | $2.3 \times 10^{19}$ |

- big이 **앙상블 포함 전부**를 EN→DE에서 +2.0 BLEU 이상으로 이긴다. **base조차 이전의 모든 발표 모델과
앙상블을 이기면서** 학습 비용은 경쟁 모델 대비 수분의 1.
- EN→FR big은 이전 SOTA의 1/4 미만 비용으로 단일 모델 신기록. (이 세팅만 $P_{drop} = 0.1$ 사용)

#### <span style="color: #4682B4">6.2 Model Variations (Table 3 — newstest2013 dev)</span>

- base에서 하나씩 바꿔가며 측정 (checkpoint averaging 없이 beam search만). 주요 행 발췌:

| 변경 | PPL (dev) | BLEU (dev) | params |
|---|---|---|---|
| **base** ($h=8$) | 4.92 | 25.8 | 65M |
| (A) $h=1$ | 5.29 | 24.9 | |
| (A) $h=4$ | 5.00 | 25.5 | |
| (A) $h=16$ | 4.91 | 25.8 | |
| (A) $h=32$ | 5.01 | 25.4 | |
| (B) $d_k=16$ | 5.16 | 25.1 | 58M |
| (C) $N=2$ | 6.11 | 23.7 | 36M |
| (C) $d_{model}=1024$ | 4.66 | 26.0 | 168M |
| (C) $d_{ff}=4096$ | 4.75 | 26.2 | 90M |
| (D) $P_{drop}=0.0$ | 5.77 | 24.6 | |
| (D) $\epsilon_{ls}=0.0$ | **4.67** | 25.3 | |
| (E) learned positional embedding | 4.92 | 25.7 | |
| **big** | 4.33 | **26.4** | 213M |

- 행별 해석 (논문 서술 순서대로):
  - **(A) head 수**: 1개면 BLEU −0.9, 32개도 하락 — $h=8{\sim}16$이 sweet spot. "여러 subspace를
동시에 본다"는 가설의 정량 근거.
  - **(B) $d_k$ 축소**: 품질 하락 — *"호환성 판정은 쉽지 않은 문제고, dot product보다 정교한 호환성
함수가 이득일 수도 있다"* 는 논문의 솔직한 해석.
  - **(C) 크기**: 클수록 좋다. dropout은 overfitting 방지에 매우 유효.
  - **(D) label smoothing 제거**: **PPL은 가장 좋아지는데(4.67) BLEU는 떨어진다(25.3)** —
"PPL과 생성 품질은 다른 지표"임을 보여주는 유명한 행.
  - **(E)**: sinusoidal ≈ learned. 외삽 기대 때문에 sinusoidal 선택.

#### <span style="color: #4682B4">6.3 English Constituency Parsing — 일반화 확인</span>

- 출력이 입력보다 훨씬 길고 강한 구조 제약이 있는 태스크. **4-layer, $d_{model}=1024$** Transformer를
task 특화 튜닝 거의 없이(dropout/lr/beam만 dev에서 선택, 나머지는 EN-DE base 그대로) 적용했다.
추론은 beam 21, $\alpha = 0.3$, 최대 출력 입력+300. (논문 Table 4 발췌)

| Parser | 학습 | WSJ 23 F1 |
|---|---|---|
| Vinyals & Kaiser (2014) | WSJ only | 88.3 |
| Petrov (2006) / Zhu (2013) | WSJ only | 90.4 |
| Dyer RNNG (2016) | WSJ only | **91.7** |
| **Transformer (4layers)** | WSJ only (40K 문장) | 91.3 |
| McClosky (2006) / Vinyals | semi-supervised | 92.1 |
| **Transformer (4 layers)** | semi-supervised (17M 문장) | **92.7** |

- WSJ 40K 문장만으로 BerkeleyParser를 이긴다 — "RNN seq2seq는 소규모 데이터에서 SOTA를 못 냈던"
영역에서의 일반화 증거. semi-supervised에서는 RNNG 계열 빼고 전부 이겼다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[7] Conclusion + 개인적인 생각</span>

- 논문의 결론: 최초의 순수 attention transduction 모델로 번역 SOTA. future work로 **텍스트 외
모달리티(이미지/오디오/비디오)**, **local/restricted attention**, **생성의 비순차화**를 꼽는다 —
셋 다 이후 10년의 연구 지도가 됐다 (ViT/멀티모달, sparse attention/FlashAttention, non-autoregressive
생성과 speculative decoding). 코드는 tensor2tensor로 공개.
- 부품들이 어디로 갔는지 계보를 그려보면:
  - **Encoder만** → BERT, **Decoder만** → GPT 시리즈, 이미지를 패치 시퀀스로 → ViT.
  - diffusion model UNet 안의 attention block도 이 논문의 multi-head attention이다
([Diffusion 계보 ② 리뷰](/posts/diffusion-models-beat-gans-on-image-synthesis/)의 32/16/8 해상도
attention이 정확히 이 부품).
  - [word2vec 계보](/posts/statistical-language-models-based-on-neural-networks/)의 관점에서 보면,
정적 임베딩의 한계(문맥 의존 의미)를 이 구조 위의 사전학습(BERT/GPT)이 이어받았다.
- 지금 다시 읽으면 복선이 많다: ablation의 "키우면 좋아진다"는 scaling law로, sinusoidal PE의
회전 행렬 해석은 RoPE로, restricted attention 언급은 sparse attention 연구로 이어졌다.
- 수식 자체는 어렵지 않다(행렬곱 + softmax가 사실상 전부). 이 논문의 진짜 가치는
**"왜 이렇게 설계했는가"에 대한 논증** — scaling의 분산 논리, path length 비교, head 분리의
subspace 논리 — 이라고 생각한다. 결과 표보다 Section 4를 곱씹는 것을 추천.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">Reference</span>

- [1] J. Ba et al., "Layer Normalization" (2016)
- [2] D. Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
- [9] J. Gehring et al., "Convolutional Sequence to Sequence Learning" (ConvS2S, 2017)
- [11] K. He et al., "Deep Residual Learning for Image Recognition" (ResNet, 2016)
- [18] N. Kalchbrenner et al., "Neural Machine Translation in Linear Time" (ByteNet, 2017)
- [30] O. Press & L. Wolf, "Using the Output Embedding to Improve Language Models" (2017)
- [31] R. Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (BPE, 2016)
- [36] C. Szegedy et al., "Rethinking the Inception Architecture for Computer Vision" (Label Smoothing, 2016)
- [38] Y. Wu et al., "Google's Neural Machine Translation System" (GNMT, 2016)
