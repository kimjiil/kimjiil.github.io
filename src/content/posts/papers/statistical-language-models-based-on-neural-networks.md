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
, <b><span style="color: #F2AA4C">Language Model</span></b>, Tomáš Mikolov (PhD Thesis, Brno University of Technology, 2012, 133p)

### <span style="color: #ffd33d">Summary</span>

word2vec의 저자 Mikolov의 박사논문. **RNN 기반 언어모델(RNNLM)** 을 제안하고, 당시 존재하던 거의 모든
고급 언어모델 기법과 정면 비교/조합해서 다음을 보였다.

- Penn Treebank에서 KN5 baseline perplexity **141.2 → 79.4** (모든 기법 조합, 이후 78.8까지) — 당시까지
보고된 최고 기록의 2배가 넘는 entropy 개선.
- WSJ 음성인식에서 WER **상대 21~24% 감소** — "언어모델 연구 역사상 최고 수준"의 개선폭.
- **학습 데이터가 커질수록 n-gram 대비 개선폭이 오히려 커진다**는 관찰 (기존 고급 기법들은 반대였음).
- 출력층 class 분해, gradient clipping, dynamic evaluation, RNN+ME 조인트 학습(RNNME) 같은
실전 기법들과 오픈소스 RNNLM toolkit.

이 논문의 복잡도 분석(병목은 $H \times H$와 $H \times V$)과 class 분해가 이듬해
[word2vec](/posts/efficient-estimation-of-word-representations-in-vector-space/)의 설계로 직결되므로,
word2vec 계보의 1편으로 읽는다. 리뷰는 논문 챕터 구성(1~9장)을 그대로 따라간다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[Ch.1] Introduction</span>

- 저자의 출발점은 놀랍게도 실용이 아니라 **AI다**. Turing test보다 정밀한 지능 측정으로 Shannon의
"Entropy of printed English"(Shannon game)를 든다 — **"언어를 이해하는 능력 ≈ 문맥에서 다음 단어를
예측하는 능력"** 이라고 가정하면, 언어모델의 품질(entropy)이 곧 지능의 형식적 측정이 된다는 것.
  - Shannon의 실험: 사람은 n-gram보다 훨씬 예측을 잘하고, 문맥이 길어질수록 격차가 벌어진다.
- 물론 텍스트만 읽어서 사람 수준의 이해에 도달하는 건 비현실적으로 어렵다고 인정하면서, 실용 가치
(기계번역, 음성인식)로 연구를 정당화한다.
- 논문의 목표 선언: "여전히 사실상 SOTA로 남아있는 단순 n-gram 모델을 넘어서는 새 기법을 만들고,
표준 데이터셋들에서 광범위한 실증으로 증명한다."
- **Claims of the Thesis** (1.3절, 논문이 주장하는 기여 그대로):
  1. simple RNN 기반 통계 언어모델의 개발
  2. 기본 RNNLM의 확장들 — unigram 빈도 기반 class, **신경망+ME 모델 조인트 학습**,
학습 데이터 정렬에 의한 적응, 테스트 데이터 처리 중 학습(dynamic evaluation)
  3. 실험 재현이 가능한 오픈소스 **RNNLM toolkit**
  4. PTB / WSJ / NIST RT04 / 데이터 압축·기계번역에서의 새 SOTA
  5. NNLM 성능 분석 (hidden 크기, 데이터 증가의 영향)
  6. 전통적 접근의 한계에 대한 논의와 미래 연구 방향

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[Ch.2] Overview of Statistical Language Modeling</span>

#### <span style="color: #4682B4">2.1 평가 — Perplexity와 WER</span>

- 언어모델은 chain rule로 시퀀스에 확률을 부여하고, 평가는 perplexity(PPL)로 한다.

$$
    PPL = 2^{-\frac{1}{K}\sum_{i=1}^{K}{\log_2{P(w_i|w_{1 \cdots i-1})}}}
$$

- "단어 하나를 평균 8bit로 인코딩하면 PPL은 $2^8 = 256$" — cross entropy의 지수 형태다.
- 재밌는 지적: **"PPL 상대 개선률(%)로 보고하는 관행은 잘못됐다.**" 같은 30% PPL 감소라도
시작점에 따라 entropy 감소가 51%~4.7%로 전혀 다르다 (논문 Table 2.1). 그래서 이 논문은
주요 비교를 **entropy 감소율**로 보고한다.

| PPL | 30% 감소 후 | Entropy [bits] | 감소 후 | Entropy 감소율 |
|---|---|---|---|---|
| 2 | 1.4 | 1 | 0.49 | 51% |
| 100 | 70 | 6.64 | 6.13 | 7.7% |
| 2000 | 1400 | 10.97 | 10.45 | 4.7% |

- 언어에 확률을 쓰는 것 자체에 대한 Chomsky의 유명한 반박("문장의 확률이라는 개념은 어떤 해석으로도
전혀 쓸모없다", 1969)을 인용하고 — 실용적 성공(ASR/MT)으로 이미 반박됐다고 받아친다.
(6장에서 이 문장을 직접 실험으로 재반박한다. 아래 참고)
- 이론적 근거로 **Solomonoff의 Algorithmic Probability**까지 끌고 온다: 모든 가능한 모델의 예측을
description length로 가중 평균하는 것이 최적 예측이라는 (계산 불가능한) 이론 —
**뒤에 나올 "모델 전부 조합" 실험의 이론적 정당화**로 쓰인다. Mahoney의 "압축 = 언어모델링" 관점과
Hutter Prize도 같은 맥락에서 소개.
- WER(word error rate)은 $WER = \frac{S+D+I}{N}$ (치환+삭제+삽입). PPL과 WER 각각의 장단점을
목록으로 정리하고, "PPL 2% 개선 + WER 0.3% 개선을 SOTA라고 주장하는 논문들"을 명시적으로 비판한다.

#### <span style="color: #4682B4">2.2 N-gram 모델</span>

- 최대우도 추정 $P(A|H) = \frac{C(HA)}{C(H)}$은 못 본 조합에서 0이 되므로 스무딩이 필수.
Good-Turing(GT)과 **modified Kneser-Ney(KN)** 를 표준으로 사용하며, KN이 일관되게 최강이다.
- n-gram의 진짜 약점을 예문으로 보여준다:
  - 장거리 패턴: `THE SKY ABOVE OUR HEADS IS BLUE` — SKY와 BLUE의 관계는 사이에 뭐가 끼든 유지되지만,
끼는 단어의 변형이 지수적으로 많아 n-gram은 각 변형을 전부 봐야 한다.
  - 단어 유사성: `PARTY WILL BE ON <MONDAY/TUESDAY>`만 보고는 `... ON FRIDAY`에 의미있는 확률을 못 준다.
- 결론: n-gram이 SOTA인 이유는 더 나은 기법이 없어서가 아니라, **더 나은 기법들이 계산적으로 비싸고
개선폭이 애플리케이션 성공에 결정적이지 않았기 때문** — 그래서 이 논문의 상당 부분이 속도 트릭이다.

#### <span style="color: #4682B4">2.3 고급 기법들 개관</span>

논문은 이후 4장에서 전부 실측 비교하므로, 여기서 각 기법의 성격과 비판을 미리 정리한다.

| 기법 | 아이디어 | 논문의 평가 |
|---|---|---|
| Cache LM | 최근 나온 단어는 또 나온다 (동적 n-gram 보간) | PPL은 크게 줄지만(~20%) WER 개선은 의문 — ASR에서는 히스토리 자체가 오염되어 오류가 잠긴다(lock-in) |
| Class 기반 | 단어를 클래스로 묶어 희소성 해소 | 작은 데이터에서 WER에 효과적, 데이터 커지면 이득 소멸 |
| Structured LM | CFG 파스 트리로 문장 구조 모델링 | 장거리 패턴의 정공법이지만 복잡도/모호성/언어 의존성 문제 |
| Decision Tree / Random Forest | 히스토리에 일반적 질문을 하는 트리들 | 좋은 트리 찾기가 어렵고 데이터 커지면 이득 감소 |
| Maximum Entropy (ME) | $P(w\|h) = \frac{e^{\sum_i{\lambda_i f_i(w,h)}}}{Z(h)}$ 임의 feature 결합 | = 로지스틱 회귀. **"hidden layer 없는 신경망"** — 이 관찰이 6장 RNNME의 복선 |
| NNLM | 히스토리를 저차원 연속 공간으로 projection | 자동 클러스터링 + 다단계 유사도. 이 논문의 주인공 |

- Jelinek의 유명한 농담("언어학자를 해고할 때마다 인식률이 올라간다")을 인용하며 — 범용 학습 기법이
task 특화 기법을 이기는 패턴을 지적한다.

#### <span style="color: #4682B4">2.4 실험 세팅 예고</span>

- 재현성에 대한 강한 문제의식: 사설 데이터셋, 약한 baseline, 재현 불가능한 실험을 나열하며 비판하고,
표준 셋업(PTB / WSJ / RT04 / MS Sentence Completion)만 사용 + toolkit 공개를 원칙으로 삼는다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[Ch.3] Neural Network Language Models</span>

#### <span style="color: #4682B4">3.1 Feedforward NNLM (Bengio 2003)</span>

- 구조: 이전 $n-1$개 단어를 1-of-V 인코딩 → 공유 projection 행렬 $P$로 저차원 투영(위치별로 같은 행렬)
→ tanh/sigmoid hidden (100~300) → 전체 vocabulary에 대한 출력층.
  - 수치 감각: $V = 50K$, 5-gram이면 입력이 20만 binary 중 4개만 1. 30차원 projection이면
projection layer는 $30 \times 4 = 120$차원.
- 당시의 비용 현실: Bengio의 원 실험은 **40개 CPU로 1 epoch에 1주일** (14M 단어, 18K vocab,
hidden 60으로 제한). 그래도 5 epoch 후 n-gram 대비 PPL ~20% 개선 — 잠재력은 분명했다.
- Mikolov 자신의 대안 FF 구조(bigram NNLM을 먼저 학습해 그 표현으로 n-gram NNLM 학습)도 소개 —
성능은 Bengio 구조와 거의 동일. (이 "표현을 먼저 배우고 얹는다" 2단계 관점이 word2vec의 씨앗)

#### <span style="color: #4682B4">3.2 RNNLM — 제안 모델</span>

- 핵심 차이: FF는 히스토리가 여전히 "직전 $n-1$개 단어"지만, RNN은 **히스토리의 표현 자체를 데이터에서
학습**한다. hidden state가 전체 과거를 압축하므로 이론상 무제한 문맥.
- 또 하나의 결정적 장점: **가변 위치 패턴**. "히스토리 어딘가에 나왔던 단어"에 의존하는 패턴을
FF로 표현하려면 위치마다 파라미터가 따로 들고, 그 위치별 학습 예시도 따로 필요하다.
RNN은 hidden state에 그냥 기억하면 된다.
- 구조는 Elman network 그대로:

$$
    s(t) = f\big( U\,w(t) + W\,s(t-1) \big)
    ,\qquad
    y(t) = g\big( V\,s(t) \big)
$$

- $f$: sigmoid, $g$: softmax. **bias는 안 쓴다** — 유의미한 개선이 없어서 Occam's razor로 제거했다는
각주가 논문 스타일을 잘 보여준다.
- 1 step 계산량:

$$
    O = H \times H + H \times V = H \times (H + V)
$$

#### <span style="color: #4682B4">3.3 학습 알고리즘 (논문에서 가장 공들인 부분)</span>

- plain SGD + cross entropy. 출력층 오차는 깔끔하게 (정답 − 예측):

$$
    e_o(t) = d(t) - y(t)
$$

- "MSE를 쓰는 건 흔한 실수다 — 돌아는 가지만 entropy/WER 관점에서 suboptimal" 이라고 명시.
- **Learning rate 스케줄**: 시작 $\alpha = 0.1$. validation entropy 개선이 **0.3% 미만**이 되면
그때부터 매 epoch 절반으로 줄이고, 또 개선 없으면 종료. 보통 8~20 epoch.
- 가중치 초기화는 $\mathcal{N}(0, 0.1)$, L2 regularization $\beta = 10^{-6}$ (거의 장식 수준 —
큰 데이터에서는 overfitting보다 underfitting이 문제라는 입장).
- 업데이트 식 (출력층 → hidden → 입력/recurrent 순서로 전부 명시돼 있다):

$$
    V(t+1) = V(t) + s(t)\,e_o(t)^{\top}\alpha - V(t)\beta
$$

$$
    e_h(t) = d_h\big( e_o(t)^{\top}V,\; t \big)
    ,\qquad
    d_{hj}(x, t) = x\,s_j(t)\,(1-s_j(t))
$$

$$
    U(t+1) = U(t) + w(t)\,e_h(t)^{\top}\alpha - U(t)\beta
    ,\qquad
    W(t+1) = W(t) + s(t-1)\,e_h(t)^{\top}\alpha - W(t)\beta
$$

- $w(t)$는 one-hot이므로 $U$ 업데이트는 **활성 뉴런의 열 하나만** 갱신하면 된다 (임베딩 lookup의 학습 버전).

<details>
<summary> <span style="color: #ffd33d">3.3.1 BPTT — 시간 펼침, vanishing/exploding, clipping 펼치기/접기</span> </summary>

- 일반 BP로만 학습하면 네트워크는 "다음 단어 예측"만 최적화할 뿐, **hidden state에 미래에 유용할 정보를
저장하려는 노력을 하지 않는다** — 장거리 정보가 남는 건 설계가 아니라 운이다.
- BPTT의 아이디어: $N$ step 동안 쓰인 RNN은 **hidden layer가 $N$개인 deep feedforward 네트워크**
(recurrent 행렬은 전부 동일)로 볼 수 있다. 이 펼친 네트워크에 일반 gradient descent를 적용한다.
- 오차의 시간 역전파는 재귀식 하나다:

$$
    e_h(t-\tau-1) = d_h\big( e_h(t-\tau)^{\top}W,\; t-\tau-1 \big)
$$

- 펼친 스텝들의 gradient를 **한 번에 합산해서** 업데이트한다 (스텝마다 갱신하면 불안정):

$$
    U(t+1) = U(t) + \sum_{z=0}^{T}{w(t-z)\,e_h(t-z)^{\top}\alpha} - U(t)\beta
    ,\qquad
    W(t+1) = W(t) + \sum_{z=0}^{T}{s(t-z-1)\,e_h(t-z)^{\top}\alpha} - W(t)\beta
$$

- gradient는 시간을 거슬러 가며 빠르게 **소멸(vanish)** 하고 드물게 **폭발(explode)** 한다[Bengio 1994].
그래서 실전은 **truncated BPTT** — 단어 LM은 $\tau \approx 5$ 스텝 펼침이면 충분하다.
(흥미로운 관찰: $\tau=5$로도 네트워크는 5 스텝 이상 정보를 저장하도록 학습될 수 있고,
$\tau=1$(일반 BP)로도 4-gram 수준의 문맥은 배운다)
- 매 단어마다 펼치지 않고 **10~20 단어마다 묶어서**(block mode) 역전파하면 복잡도에서 $T$항이
사실상 사라진다.
- **Exploding 해법 = gradient clipping**: hidden 뉴런에 누적되는 오차 gradient를 $[-15, 15]$로 자른다.
"이것 없이는 큰 데이터에서 RNNLM 학습이 아예 불가능하다." 수치 안정성을 위해 double precision 권장.
- Vanishing은 미해결로 남긴다 — 이 지점이 이후 LSTM 채택, 그리고 경로 길이 $O(1)$을 내세운
[Transformer](/posts/attention-is-all-you-need/)로 이어지는 문제의식이다. $\blacksquare$

</details>

<hr/> <!-- 수평선 -->

#### <span style="color: #4682B4">3.4 확장들</span>

- **3.4.1 Vocabulary truncation**: 희귀 단어를 하나의 클래스로 합치거나(Bengio), 최빈 $S$개만 신경망이
다루고 나머지는 n-gram에 맡기는 shortlist(Schwenk, $S$=2K까지). 빠르지만 **정확도 손실이 크다.**
- **3.4.2 출력층 Factorization (class 분해)** — 이 논문의 대표 속도 트릭.

$$
    P(w_{t+1}|s(t)) = P\big(c_i\,|\,s(t)\big) \times P\big(w_i\,|\,c_i,\,s(t)\big)
$$

<details>
<summary> <span style="color: #ffd33d">Frequency binning class 분해 + 최적 class 수 증명 펼치기/접기</span> </summary>

- Goodman이 ME 모델 가속에 쓰던 트릭의 신경망 이식이다. class 분포($C$개) softmax 하나 +
해당 class 내 단어들($V'$개) softmax 하나만 계산한다.
- **class 배정은 학습 없이 unigram 빈도만으로** 한다(frequency binning): 누적 unigram 확률을
균등 분할. 고빈도 단어는 소수 정예 class에, 희귀 단어는 큰 class에 들어가지만 어차피 드물게 접근된다.
- 출력층 계산량 $f(C) = H \cdot C + H \cdot \frac{V}{C}$를 $C$로 미분하면

$$
    \frac{df}{dC} = H - H\frac{V}{C^2} = 0
    \quad\Rightarrow\quad C = \sqrt{V}
$$

- 최적점에서 $2H\sqrt{V}$ — 원래 $HV$ 대비 $\frac{\sqrt{V}}{2}$배 빨라진다. 실측으로 **15~30배 speedup**,
10만 단어 이상 vocabulary에서는 그 이상. 정확도 손실은 작다 (4장 실측).
- 변형: 빈도에 **제곱근을 씌운 뒤** binning 하면 접근 비용이 더 균형잡혀 추가 speedup (Povey 제안).
- 이 트릭이 [word2vec](/posts/efficient-estimation-of-word-representations-in-vector-space/)의
hierarchical softmax($O(\log V)$)로, 또 [fastText 분류기](/posts/fasttext-subword-information-and-bag-of-tricks/)의
라벨 트리로 이어진다. $\blacksquare$

</details>

<hr/> <!-- 수평선 -->

- **3.4.3 n-gram 근사**: RNNLM에서 **텍스트를 대량 샘플링해서 그 위에 n-gram을 학습**하면, 신경망 코드
한 줄 없이 디코더에 RNN의 지식 일부를 넣을 수 있다. (무한 샘플 + 무한 차수면 이론상 등가)
- **3.4.4 Dynamic evaluation**: 테스트 데이터를 처리하면서 $\alpha = 0.1$ 고정으로 1 pass 학습을 계속한다.
cache와 달리 **연속 공간에서 적응**하므로 유사 단어에도 확률이 퍼진다. 위험: 모호한 데이터가 계속 들어오면
**자기 가중치를 덮어써서 잊어버릴 수 있다** — 그래서 (잊지 않는) static 모델과의 보간이 중요하다.
- **3.4.5 NNLM 조합**: 초기화만 다른 RNN 여러 개의 출력 평균. Solomonoff ALP의 실용 근사라는
해석을 달아둔다. 실전 지침: "가능한 큰 모델을 여러 개 만들어 균등 가중으로 섞어라."

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[Ch.4] PTB에서의 비교와 조합 — 논문의 하이라이트</span>

- 셋업: PTB sections 0-20 학습(930K), 21-22 valid(74K), 23-24 test(82K), vocab 10K, OOV는 전부 `<unk>`.
이 전처리가 이후 10년간 LM 벤치마크 표준이 된다.
- 비교 원칙에 대한 잔소리가 길다(사설 데이터, 약한 baseline, 재현 불가 비판 + "실험은 재현 가능해야 하고
코드를 공개해야 한다"는 제안) — 그리고 실제로 당시 존재하던 기법 대부분을 저자/원저자 구현으로 모아서
**단일 셋업에서 전부 실측**했다. (논문 Table 4.1 발췌)

| 모델 | 단독 PPL | +KN5 | +KN5+cache |
|---|---|---|---|
| 3-gram Good-Turing (GT3) | 165.2 | - | - |
| **KN5 (baseline)** | **141.2** | - | - |
| KN5 + cache | 125.7 | - | - |
| PAQ8o10t (범용 압축기!) | 131.1 | - | - |
| Maximum Entropy 5-gram | 142.1 | 138.7 | 124.5 |
| Random forest LM | 131.9 | 131.3 | 117.5 |
| Structured LM | 146.1 | 125.5 | 114.4 |
| Within/across sentence LM | 116.6 | 110.0 | 108.7 |
| Log-bilinear LM | 144.5 | 115.2 | 105.8 |
| Feedforward NNLM | 140.2 | 116.7 | 106.6 |
| Syntactical NNLM (당시 최고 기록) | 131.3 | 110.0 | 101.5 |
| **RNNLM (제안)** | **124.7** | **105.7** | **97.5** |
| Dynamic RNNLM | 123.2 | 102.7 | 98.0 |
| Static RNN 20개 조합 | 102.1 | 95.5 | 89.4 |
| Dynamic RNN 조합 | 101.0 | 92.9 | 90.0 |

- 읽는 포인트:
  - **단일 모델로 RNNLM이 전부 이긴다** (124.7). 심지어 구문 파서 feature를 쓰는 Syntactical NNLM보다 좋다.
  - **범용 압축기 PAQ가 PPL 131**로 웬만한 LM 기법보다 좋다는 것도 재밌는 발견 — "압축 = LM"의 실증이자,
PAQ의 hash 기반 구조가 6장 RNNME의 모티브가 된다.
  - cache와 결합해도 RNN의 이득이 대부분 남는다(97.5) → **RNN의 개선은 cache류 장거리 정보가 아니라
"짧은 문맥의 더 나은 표현"에서 온다**는 분석. (Bengio의 vanishing gradient 이론과 일치)
- **BPTT ablation** (Figure 4.1): BPTT 스텝 1(일반 BP) → 5~8로 늘리면 PPL이 ~10% 개선.
"모델 4개 평균 vs 4개 조합"과 비교해서, BPTT의 이득은 앙상블로 대체 안 되는 정보라는 것도 확인.
- **조합 실험의 결론들**:
  - NN 계열 전부 조합: 105.8 → dynamic RNN 추가: 100.2 → 전 기법 조합: **93.2**.
  - 최종 전체 조합 (Table 4.5): **PPL 83.5.** 가중치를 보면 RNN 계열이 0.63으로 지배하고,
KN5·ME·log-bilinear·FFNN 등 다수 기법의 가중치가 **0** — RNN이 그들이 담던 정보를 포섭한다.
  - RNN들만 빼고 조합하면 92.0 — 그래도 이전 기록(107)보다 좋지만, RNN이 담는 정보가
다른 어떤 기법으로도 안 잡힌다는 것을 보여준다.
  - greedy 추가 실험 (Table 4.6): dynamic RNN 조합(101.0) → +KN5+cache(90.0) → +static RNN(86.2)
→ +문장경계 LM(84.8) → +랜덤포레스트(84.0). **상위 2~3개 기법이면 거의 끝**이라는 실용 결론.
  - adaptive 보간(가중치를 테스트 중 동적 추정) + 큰 학습률 dynamic RNN까지: **79.4**,
이후 6장의 RNNME까지 넣어 **78.8 (KN5 대비 entropy 11.8% 감소)**.
- 4장 결론의 한 문장이 이후 10년을 예언한다: *"feature와 전문가 지식에 집중하는 기법보다
모델링(학습) 자체에 집중하는 기법이 이긴다."*

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[Ch.5] Wall Street Journal 음성인식 실험</span>

#### <span style="color: #4682B4">5.1 JHU 셋업 — 고급 기법들과 WER 비교</span>

- NYT 37M 토큰 학습, 20K vocab, 100-best rescoring (oracle WER 6.1/9.5%).

| 모델 | Dev WER | Eval WER |
|---|---|---|
| KN5 (baseline) | 12.2% | 17.2% |
| Discriminative LM | 11.5% | 16.9% |
| Joint (structured) LM | - | 16.7% |
| Static RNN | 10.3% | 14.5% |
| Adapted RNN | 9.7% | 14.2% |
| **RNN 3개 보간** | **9.5%** | **13.9%** |

- 경쟁 고급 기법들이 WER 2~3% 상대 감소에 그칠 때 RNN은 **상대 ~20% 감소.**
- **데이터 스케일링 실험** (Table 5.2) — 이 논문에서 가장 중요한 그래프:

| 학습 토큰 | KN5 PPL | +RNN PPL | KN5 WER | +RNN WER | Entropy 개선 | WER 개선 |
|---|---|---|---|---|---|---|
| 223K | 415 | 333 | - | - | 3.7% | - |
| 675K | 390 | 298 | 15.6% | 13.9% | 4.5% | 10.9% |
| 2.2M | 331 | 251 | 14.9% | 12.9% | 4.8% | 13.4% |
| 6.4M | 283 | 200 | 13.6% | 11.7% | 6.1% | 14.0% |
| **37M** | 212 | 133 | 12.2% | 10.2% | **8.7%** | **16.4%** |

- Goodman의 관찰("고급 기법의 이득은 데이터가 커지면 사라진다")과 정반대로,
**RNN의 상대 이득은 데이터가 커질수록 커진다.** 단, hidden 크기도 같이 키워야 한다는 조건이 붙는다.

#### <span style="color: #4682B4">5.2 Kaldi 셋업 — 재현 가능한 최고 기록</span>

- 오픈소스 Kaldi로 1000-best rescoring. 모델은 6장의 RNNME (hash 2G, 1~4gram feature).

| 모델 | Eval92 WER | Eval93 WER |
|---|---|---|
| KN5 (no cutoff) | 12.0% | 16.6% |
| RNNME-10 | 11.9% | 16.3% |
| RNNME-80 | 10.4% | 14.9% |
| RNNME-320 | 9.8% | 14.2% |
| **RNNME 조합 + 적응** | **9.15%** | **13.11%** |

- KN5 대비 **절대 2.9~3.5%p, 상대 21~24% WER 감소** — "통계 언어모델링 분야에서 아마도 최고 기록".
- 문장 단위 정확도로 보면 더 극적: 27.6% → 39.9% (상대 37~45% 향상).
- **hidden 10개짜리 RNNME도 baseline을 이긴다**는 것이 RNNME 구조의 힘 (6장에서 설명).
- n-gram 근사 실험: RNNME-480에서 **150억 단어를 샘플링**해 5-gram을 만들면 WER 12.0→11.4%.
원본 개선의 20~30%만 건지지만, 디코더에 신경망 코드 없이 넣을 수 있다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[Ch.6] 대규모 학습 전략과 RNNME</span>

- 무대: NIST RT04 Broadcast News, IBM Attila 인식기(판별 학습 음향모델), **학습 400M 토큰, vocab 82K** —
"아마도 당시까지 학습된 가장 큰 신경망 언어모델".

#### <span style="color: #4682B4">6.2~6.4 복잡도 줄이기 + 데이터 선택/정렬</span>

- 복잡도 $I \times W \times (H \times H + H \times V)$의 각 항을 줄이는 전략을 하나씩 검토:
epoch 수($I$), 토큰 수($W$), vocab($V$), hidden($H$), 병렬화. ("작은 hidden으로 큰 데이터"는
안 통한다는 것을 뒤에서 보인다)
- **데이터 선택/정렬 (curriculum)**: 전체를 40K 문장씩 560 청크로 나누고, 청크별 2-gram으로 dev PPL을
측정 → PPL 600 이상(노이즈/중복) 청크는 버리고(403M→318M), 나머지를 **out-of-domain부터 in-domain
순서로 정렬**해서 학습한다.
  - 이유가 깔끔하다: online SGD에서 마지막에 본 데이터가 사실상 더 큰 가중치를 갖는다 + incremental
learning(쉬운 것 먼저) 효과.
  - 결과: 표준 shuffled SGD 대비 **PPL 약 10% 추가 감소**, 7 epoch로 수렴.
- 대형 RNN 결과 (Table 6.3): RNN-320이 되어야 KN4(PPL 140)와 비슷해지고, RNN-640은 116 (KN4와 보간 시 102).
lattice rescoring에서 RNN-640: WER 13.11% → 12.05%, 3모델 조합으로 **11.70%** (model M 12.49%보다 좋음).

#### <span style="color: #4682B4">6.6 RNNME — RNN과 Maximum Entropy의 조인트 학습</span>

- 2장의 복선 회수: **ME 모델 = hidden layer 없는 신경망** (입력→출력 직접 연결). 그러면 RNN에
직접 연결(direct connections)을 추가하고 **같은 SGD로 조인트 학습**하면 되지 않나? — 이게 RNNME다.

<details>
<summary> <span style="color: #ffd33d">Hash 기반 class ME 구현 (n-gram feature를 해시로) 펼치기/접기</span> </summary>

- n-gram ME의 full feature 파라미터 수는 $V^N$ — trigram에 $V = 100K$면 $(10^5)^3$으로 불가능하다.
- 그런데 입력 관점에서 보면 "직전 두 단어의 조합"은 **한 시점에 정확히 뉴런 하나만 활성**이다.
따라서 전체 행렬 대신, n-gram 히스토리를 해시 함수로 고정 크기 배열에 매핑한다.

$$
    g(w_{t-2}, w_{t-1}) = \big( w_{t-2} \times P_1 \times P_2 + w_{t-1} \times P_1 \big)\; \%\; SIZE
$$

$$
    P(w|h) = \frac{e^{\sum_{i}{\lambda_i f_i(g(h),\,w)}}}{\sum_{w}{e^{\sum_{i}{\lambda_i f_i(g(h),\,w)}}}}
$$

- **충돌은 감수한다** — 같은 버킷에 여러 히스토리가 겹치면 빈번한 feature가 값을 지배하므로,
작은 해시는 pruned 모델처럼 동작한다. (해시 크기 하나로 메모리↔정확도를 조절하는 다이얼이 된다)
- 이 발상의 직접적 출처는 압축 프로그램 PAQ(Mahoney)다. 그리고 이 해시 트릭은 이후
[fastText](/posts/fasttext-subword-information-and-bag-of-tricks/)의 n-gram 버킷 해싱으로 그대로 이어진다.
- 학습은 RNN 부분과 완전히 동일한 SGD/학습률/정규화 — ME는 그냥 "RNN의 직접 연결 가중치"일 뿐이다.
출력 class 분해도 같이 적용된다. $\blacksquare$

</details>

<hr/> <!-- 수평선 -->

- **해시 크기의 효과** (RNN-80+ME, eval PPL): 해시 0 → 183, $10^6$ → 176, $10^8$ → 136, $10^9$ → **123**
(KN4 보간 시 113). 8GB 메모리를 쓰는 대신 baseline(140)을 크게 이긴다.
- **왜 잘 되는가**: 직접 연결(ME)이 n-gram급 단순 패턴을 담당하므로, **hidden layer는 n-gram이 못 담는
보완 정보에만 집중**할 수 있다. RNN 단독은 단순 패턴 표현에 파라미터를 낭비한다.
  - RNNME-40이 RNN-320과 비슷한 WER — recurrent 가중치를 $40^2$개만 학습하고도 $320^2$개짜리 성능.
  - 후속 실험(1~4gram feature + 8G 해시): RNNME-0(hidden 0!)이 eval PPL 137로 이미 KN4(140)를 이기고,
RNNME-40은 117까지 간다.
- **데이터가 커질 때의 거동** (Figure 6.12): hidden 고정이면 RNN의 이득은 데이터 증가와 함께 줄어들지만,
**RNNME는 완만하게 유지**된다. RNNME-20이 RNN-80보다 낫다. → "수십억 단어 스케일에서는 신경망을
어떤 형태든 n-gram과 조인트로 학습하는 것이 필수" 라는 결론.
- **Chomsky 재반박 실험** (6.6.4): `colorless green ideas sleep furiously`(문법적) vs
`furiously sleep ideas green colorless`(비문). 학습 데이터에 두 문장의 bigram이 하나도 없는데도
RNN-640은 문법적인 문장에 **37,000배 높은 확률**을 준다 (n-gram은 거의 구분 못 함).
"완전히 새로운 문장의 문법성도 통계 모델이 구분할 수 있다"는 직접 증명.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[Ch.7] Additional Experiments — 범용성 증명</span>

- **기계번역**: IWSLT05 중→영 BLEU 48.7 → **51.2** (300-best rescoring), NIST MT05 33.0 → 34.7.
MT 데이터는 괄호 짝맞춤 같은 n-gram이 못 잡는 패턴이 많아 PPL 개선이 ASR보다 크다
(RNNME-160: 110 → 80, dynamic 결합 시 58).
- **데이터 압축**: RNNLM toolkit에 arithmetic coding을 붙여 실제 압축기를 만들었다.
정규화된 뉴스 텍스트 1.7GB 기준:

| 압축기 | bits per character |
|---|---|
| gzip -9 | 2.72 |
| PAQ8o10t -8 (당시 최강 범용 압축기) | 1.28 |
| RNNME-40 | 1.24 |
| **RNNME-200** | **1.21** |

  - 1.21bpc는 **Shannon의 영어 entropy 상한 추정치(1.3bpc)보다 낮다.**
- **Microsoft Sentence Completion Challenge**: 문장의 핵심 단어 하나를 5지선다로 고르는 태스크
(50M 토큰 학습). random 20% / KN5 40.0% / RNNME-300 49.3% / 고빈도 단어 필터링 모델과 조합 **55.4%**
(사람은 91%). — 이 태스크가 이듬해 [word2vec 논문](/posts/efficient-estimation-of-word-representations-in-vector-space/)의
평가에 다시 등장한다.
- **형태가 풍부한 언어**: PTB의 각 토큰에 랜덤 2bit를 붙이는 재밌는 실험 — n-gram은 entropy가 +3bit
나빠지지만 RNN은 +2.5bit만 나빠진다 (클러스터링으로 유사성을 복원). 체코어 강의 ASR에서 KN4 70.7% →
NN LM **75.0%** 단어 정확도.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[Ch.8~9] Towards Intelligent Models & Conclusion</span>

- 8장은 AI 에세이에 가깝다. 요약하면:
  - 신경망의 비선형은 **IF문**이다 — "작고 빨간 것은 사과"는 가중합으로 표현 불가.
깊이가 늘면 IF의 중첩이 늘어 지수적으로 적은 규칙으로 패턴을 표현한다.
  - 그러나 고정 깊이 모델은 "가변 길이 루프/재귀"를 못 담고, RNN도 매 스텝 hidden 전체를 접근해야 해서
"필요할 때만 메모리를 읽는" 실제 프로그램과 다르다.
  - 유전 프로그래밍(GP) 실험담: SGD보다 수십 배 느리지만, **SGD로는 못 배우는 패턴**(1bit를 100 step
기억하기)을 GP는 배운다. SGD+GP 결합을 future work로 제안.
  - `APPLE IS APPLE, BALL IS BALL, ... NOVEL IS ???` — 사람은 `X Y X` 패턴을 즉시 일반화하지만
n-gram/FSM은 원리적으로 불가능. **"점점 복잡해지는 패턴 학습 태스크를 정의하고 제한된 자원으로 풀게
하자"** 는 연구 로드맵 제안. (지금 보면 synthetic benchmark/curriculum 연구의 예고편)
- 9장 결론 (논문이 스스로 꼽은 것):
  - RNN LM은 SGD+BPTT로 안정적으로 학습 가능하고, 단순 class 분해로 크게 가속된다.
  - PTB/WSJ/RT04에서 신기록. **데이터가 커질수록 이득도 커진다** (단, hidden도 커져야).
  - RNNME는 수십억 단어 스케일의 열쇠.
  - 미공개 결과 하나: 같은 정확도 기준으로 RNNLM은 pruning+압축된 n-gram보다 **10배 작다.**
  - future work 중 "character/subword/word/phrase 등 **여러 시간 스케일의 RNN**" 언급 — subword는
[fastText](/posts/fasttext-subword-information-and-bag-of-tricks/)에서, 나머지는 이후 연구들에서 실현된다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">계보 — 어디서 왔고 어디로 갔나</span>

#### <span style="color: #4682B4">영향 받은 것</span>

- **← Bengio NNLM (2003)**: "단어 = 연속 벡터 + 신경망 LM" 뼈대. 이 논문은 고정 윈도우를 recurrence로
바꿔 히스토리 압축까지 학습하게 만들었다.
- **← Elman (1990)**: simple RNN 구조와 인공 문법 학습 실험.
- **← Goodman (2001)**: class 분해 속도 트릭 (3.4.2의 직접 선행) + "기법들을 함께 연구해야 한다"는
방법론적 비판.
- **← Shannon / Solomonoff / Mahoney(PAQ)**: "예측 = 압축 = 지능" 관점과 hash 기반 모델(→RNNME).

#### <span style="color: #4682B4">후속 연구에 준 영향</span>

- **→ [word2vec 계보 ② (2013.1)](/posts/efficient-estimation-of-word-representations-in-vector-space/)**:
세 가지가 직결된다.
  1. **복잡도 분석**: 병목이 $H \times H$(RNN)와 $H \times V$(softmax)라는 이 논문의 분석이,
"비선형 hidden을 아예 제거한 log-linear 모델"(CBOW/Skip-gram)의 설계 근거가 된다.
word2vec 논문의 복잡도 표에 이 논문의 RNNLM이 비교 대상으로 그대로 등장한다.
  2. **계층적 출력**: class 분해($O(\sqrt{V})$)가 Huffman hierarchical softmax($O(\log V)$)로 일반화.
  3. **표현의 규칙성**: 이 논문의 부산물이던 단어 벡터의 규칙성(NAACL 2013에서 king−man+woman≈queen으로
정식화)이 "벡터 학습 자체가 목표"라는 관점 전환과 analogy 평가의 탄생으로 이어진다.
- **→ RNN 학습 실무**: gradient clipping은 모든 RNN/LSTM 학습의 기본기가 됐고(Pascanu 2013이 이론 정리),
vanishing의 실증은 LSTM LM(Sundermeyer 2012)과, 멀리는 경로 길이 $O(1)$의
[Transformer](/posts/attention-is-all-you-need/)로 이어진다.
- **→ 벤치마크/평가 문화**: PTB 930K/10K 셋업이 표준 벤치마크로 굳었고, "PPL이 아니라 entropy로,
그리고 다운스트림(WER)으로 증명하라"는 실증 스타일을 남겼다.
- **→ RNNME의 유산**: "신경망 + 희소 n-gram feature 조인트"는 이후 대규모 시스템들의 wide & deep 계열
설계에서 반복되고, hash 트릭은 [fastText](/posts/fasttext-subword-information-and-bag-of-tricks/)로 이어진다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">Reference</span>

- [1] Y. Bengio et al., "A Neural Probabilistic Language Model" (JMLR 2003)
- [2] T. Mikolov et al., "Recurrent neural network based language model" (Interspeech 2010)
- [3] T. Mikolov et al., "Extensions of recurrent neural network language model" (ICASSP 2011)
- [4] T. Mikolov et al., "Strategies for Training Large Scale Neural Network Language Models" (ASRU 2011)
- [5] T. Mikolov et al., "Linguistic Regularities in Continuous Space Word Representations" (NAACL 2013)
- [6] Y. Bengio et al., "Learning Long-Term Dependencies with Gradient Descent is Difficult" (1994)
- [7] J. Goodman, "A bit of progress in language modeling" (2001)
