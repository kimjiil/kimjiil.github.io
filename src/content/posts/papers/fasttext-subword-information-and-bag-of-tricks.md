---
title: "[word2vec 계보 ④] fastText — Subword Information & Bag of Tricks (2016-17)"
date: 2026-08-21
category: deep-learning-paper
tags:
  - "NLP"
  - "fastText"
  - "Word Embedding"
  - "Subword"
  - "Text Classification"
---

<span style="font-size:17pt">
<b>Enriching Word Vectors with Subword Information / Bag of Tricks for Efficient Text Classification</b>
</span>

<a href="https://arxiv.org/abs/1607.04606" target="_blank"><b>[PDF 1]</b></a>
<a href="https://arxiv.org/abs/1607.01759" target="_blank"><b>[PDF 2]</b></a>
, <b><span style="color: #F2AA4C">Word Embedding</span></b>, Piotr Bojanowski*, Edouard Grave*, Armand Joulin, Tomáš Mikolov (Facebook AI Research, TACL 2017 / EACL 2017)

### <span style="color: #ffd33d">Summary</span>

Mikolov가 Facebook으로 옮겨서 만든 **word2vec의 실질적 후계자, fastText**를 구성하는 두 논문을 함께 리뷰한다.

1. **Enriching Word Vectors with Subword Information (TACL 2017)** — 단어를 원자로 취급하던
[SGNS](/posts/distributed-representations-of-words-and-phrases/)의 한계(형태 정보 무시, OOV 불가)를,
단어를 **문자 n-gram의 가방**으로 표현하고 벡터를 n-gram 벡터의 **합**으로 정의해서 해결한다.
9개 언어에서 word2vec과 형태론 기반 선행 기법들을 이긴다.
2. **Bag of Tricks for Efficient Text Classification (EACL 2017)** — CBOW의 중심 단어를 라벨로 바꾼
선형 분류기 fastText. **10억 단어를 CPU로 10분 안에** 학습하고, 딥러닝 분류기와 대등한 정확도를
**수천~15,000배 빠르게** 얻는다. 31만 개 클래스 분류에서 50만 문장을 1분 안에 처리.

"표현의 기본 단위는 단어가 아니라 subword"라는 방향 제시가 유산이고, BPE/WordPiece 토크나이저를 쓰는
현대 LLM까지 이어진다. 리뷰는 두 논문의 섹션 흐름을 각각 따라간다.

<hr/> <!-- 수평선 -->

## <span style="color: #F2AA4C">Part 1 — Enriching Word Vectors with Subword Information</span>

### <span style="color: #ffd33d">[1] Introduction & Related Work</span>

- word2vec류 모델은 단어마다 독립 벡터를 배정한다 — **파라미터 공유가 없고 단어의 내부 구조를 무시**한다.
  - 프랑스어/스페인어 동사는 활용형이 40개 이상, 핀란드어 명사는 격이 15개. 이런 언어들은 훈련 코퍼스에
드물게 나오거나 아예 안 나오는 형태가 많아 좋은 벡터를 배우기 어렵다.
  - 단어 형성이 규칙을 따르므로 **문자 수준 정보로 개선할 수 있다**는 것이 출발점.
- 관련 연구 정리 (논문 2장):
  - **형태론 기반 표현들** (Luong 2013의 recursive NN, Botha & Blunsom 2014, Qiu 2014 등) —
전부 **형태소 분해가 필요**하다. fastText는 필요 없다는 것이 차별점.
  - 가장 가까운 선행은 **Schütze (1993)**: 문자 4-gram의 SVD 표현을 합해서 단어를 표현 — 아이디어의 원조로
직접 인용한다.
  - **문자 수준 NLP 모델들** (char-RNN/CNN LM, 태깅, 분류)과 NMT의 subword unit(**Sennrich 2016의 BPE**,
Luong & Manning 2016)도 인접 연구로 언급 — subword 시대가 여러 갈래에서 동시에 열리고 있었다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[2] Model</span>

#### <span style="color: #4682B4">2.1 General model — SGNS를 이진 분류로 다시 쓰기</span>

- Skip-gram objective를 복습하되, 이 논문은 [계보 ③](/posts/distributed-representations-of-words-and-phrases/)의
negative sampling을 **이진 로지스틱 loss** 형태로 깔끔하게 다시 쓴다.
로지스틱 loss $\ell(x) = \log(1 + e^{-x})$로 표기하면:

$$
    \sum_{t=1}^{T}{\left[ \sum_{c \in \mathcal{C}_t}{\ell\big(s(w_t, w_c)\big)}
    + \sum_{n \in \mathcal{N}_{t,c}}{\ell\big(-s(w_t, n)\big)} \right]}
$$

- $\mathcal{C}_t$: 문맥 단어들, $\mathcal{N}_{t,c}$: 샘플링된 negative들. 점수 함수가
$s(w_t, w_c) = u_{w_t}^{\top}v_{w_c}$면 이게 정확히 SGNS다.
- **이 논문이 바꾸는 것은 점수 함수 $s$ 하나뿐이다.**

#### <span style="color: #4682B4">2.2 Subword model</span>

- 단어 양끝에 경계 기호 `<`, `>`를 붙이고 $3 \le n \le 6$의 문자 n-gram을 전부 뽑는다. $n=3$ 예시:

```text
where  →  <wh, whe, her, ere, re>  +  특수 시퀀스 <where>
```

- 경계 기호의 역할: `her`(where 내부)와 `<her>`(단어 her)는 **다른 단위** — 접두/접미사가 단어 중간
문자열과 구분된다. 단어 자체(`<where>`)도 단위로 포함해 고빈도 단어는 자기 벡터를 유지한다.
- 단어 $w$의 n-gram 집합을 $\mathcal{G}_w \subset \{1, ..., G\}$라 하면, 점수는 n-gram 벡터 $z_g$들의 합:

$$
    s(w, c) = \sum_{g \in \mathcal{G}_w}{z_g^{\top} v_c}
$$

- *"이 단순한 모델은 단어들 사이에 표현을 공유하게 해서, 희귀 단어의 신뢰할 만한 표현을 배울 수 있게 한다."*
- **메모리 관리 — hashing trick**: n-gram 종류가 폭발하므로 **FNV-1a 해시로 $K = 2 \times 10^6$개
버킷**에 매핑한다. 최종적으로 단어는 "단어 사전 인덱스 + 해시된 n-gram 집합"으로 표현된다.
([계보 ① RNNME](/posts/statistical-language-models-based-on-neural-networks/)의 hash 기반 ME와 같은 트릭 계열)

<details>
<summary> <span style="color: #ffd33d">subword 모델의 gradient — 왜 형태소가 공유되는가 펼치기/접기</span> </summary>

- SGNS gradient([계보 ③ 접기](/posts/distributed-representations-of-words-and-phrases/) 참고)에서 오차항을
$e = \sigma(s(w,c)) - t$ ($t$: 진짜/노이즈 라벨)라 하면, 점수가 합 형태이므로 chain rule에 의해
**같은 gradient가 구성 n-gram 전부에 흘러간다.**

$$
    \frac{\partial L}{\partial z_g} = e \cdot v_c \quad (\forall g \in \mathcal{G}_w)
    ,\qquad
    \frac{\partial L}{\partial v_c} = e \cdot \sum_{g \in \mathcal{G}_w}{z_g}
$$

- "eating"이 학습되면 `eat`, `ing>` 벡터가 갱신되고, 그 벡터는 "eats", "playing"에서 **재사용**된다 —
활용형끼리 파라미터를 공유하는 메커니즘의 전부다.
- 희귀 단어는 자기 벡터의 학습 기회가 적어도, 구성 n-gram들이 다른 단어를 통해 계속 학습되므로
**희귀/미등장 단어에서 이득이 가장 크다.** (5.4절 실험이 이를 확인) $\blacksquare$

</details>

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[3] Experimental Setup</span>

- baseline은 word2vec C 구현의 skipgram/cbow. 최적화는 SGD + **learning rate 선형 감소**
$\gamma_0(1 - \frac{t}{TP})$, **Hogwild 비동기 병렬** (스레드들이 파라미터를 lock 없이 공유 갱신).
- 하이퍼파라미터 (전부 논문 명시값): 300차원, negative 5개 (**unigram 빈도의 제곱근에 비례해 샘플링** —
계보 ③의 3/4제곱과 다른 선택), 윈도우 크기 $c$를 1~5에서 균등 샘플, subsampling 임계값 $10^{-4}$,
min count 5, $\gamma_0$: skipgram 0.025 / cbow·sisg 0.05.
- 속도: subword 합산 때문에 skipgram 대비 **약 1.5배 느림** (105k vs 145k words/sec/thread). C++ 구현 공개.
- 데이터: **위키피디아 9개 언어** (아랍어, 체코어, 독일어, 영어, 스페인어, 프랑스어, 이탈리아어,
루마니아어, 러시아어), 5 pass.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[4] Results — 5가지 실험</span>

#### <span style="color: #4682B4">4.1 인간 유사도 판단 (Spearman 상관)</span>

- 표기: **sisg-** = OOV를 null 벡터로 / **sisg** = OOV를 n-gram 합으로 계산 (Subword Information Skip-Gram).
(논문 Table 1 전체)

| 언어/데이터셋 | sg | cbow | sisg- | **sisg** |
|---|---|---|---|---|
| AR WS353 | 51 | 52 | 54 | **55** |
| DE GUR350 | 61 | 62 | 64 | **70** |
| DE GUR65 | 78 | 78 | **81** | **81** |
| DE ZG222 | 35 | 38 | 41 | **44** |
| EN RW (희귀어) | 43 | 43 | 46 | **47** |
| EN WS353 | **72/73** | | 71 | 71 |
| ES WS353 | 57 | 58 | 58 | **59** |
| FR RG65 | 70 | 69 | **75** | **75** |
| RO WS353 | 48 | 52 | 51 | **54** |
| RU HJ | 59 | 60 | 60 | **66** |

- 관찰 (논문 서술 순서대로):
  1. **영어 WS353 하나만 빼고 전부 sisg 승리.** OOV를 n-gram으로 계산하는 것(sisg)은 null(sisg-)보다
항상 같거나 좋다.
  2. 효과는 **아랍어/독일어/러시아어에서 가장 크다** — 격변화(독일어 4격, 러시아어 6격)와 복합명사
(Tischtennis ↔ Tennis의 문자 유사성을 공유) 때문.
  3. 영어 WS353에서 지는 이유: **흔한 단어들이라 subword 없이도 좋은 벡터가 이미 가능**해서.
반대로 희귀 단어 데이터셋(RW)에서는 이긴다.

#### <span style="color: #4682B4">4.2 Word analogy</span>

- 체코어/독일어/영어/이탈리아어. (논문 Table 2)

| 언어 | 유형 | sg | cbow | **sisg** |
|---|---|---|---|---|
| CS | Semantic | 25.7 | 27.6 | 27.5 |
| CS | Syntactic | 52.8 | 55.0 | **77.8** |
| DE | Semantic | 66.5 | 66.8 | 62.3 |
| DE | Syntactic | 44.5 | 45.0 | **56.4** |
| EN | Semantic | 78.5 | 78.2 | 77.8 |
| EN | Syntactic | 70.1 | 69.9 | **74.9** |
| IT | Semantic | 52.3 | 54.7 | 52.3 |
| IT | Syntactic | 51.5 | 51.8 | **62.7** |

- **문법(syntactic)은 압도적 개선** (체코어 +25%p!), **의미(semantic)는 비슷하거나 하락** (독일어/이탈리아어).
활용형이 n-gram으로 연결되니 당연한 방향이고, semantic 하락은 n-gram 길이 선택과 얽혀있다고
정직하게 분석한다 (4.4절에서 길이를 최적화하면 하락이 줄어든다).

#### <span style="color: #4682B4">4.3 형태론 기반 선행 기법들과 비교</span>

- 공정 비교를 위해 **선행 연구들과 같은 데이터로 재학습**해서 비교한다 (논문 Table 3 발췌):
  - vs Soricut & Och (2015, 접두/접미사 분석 기반): DE GUR350에서 **73 vs 64** — 큰 격차의 원인은
그들이 **명사 합성(compounding)을 모델링하지 못해서**라고 분석.
  - vs Botha & Blunsom (2014): DE GUR350 66 vs 56, EN RW 41 vs 30.
- 형태소 분석기 기반 방법들을 "분석기 없는 단순한 방법"이 이긴다는 것이 포인트.

#### <span style="color: #4682B4">4.4 데이터 크기 / n-gram 길이의 효과</span>

- **데이터 크기 ablation** (위키 1/2/5/10/20/50%): 모든 크기에서 sisg > cbow. 핵심 발견 두 가지:
  1. cbow는 데이터가 늘수록 계속 좋아지지만 **sisg는 빨리 포화**된다.
  2. **아주 작은 데이터에서 sisg가 극적으로 강하다** — DE GUR350에서 **5% 데이터의 sisg(66)가
전체 데이터의 cbow(62)를 이기고**, EN RW에서는 **1% 데이터의 sisg(45)가 전체 cbow(43)를 이긴다.**
"태스크 도메인 데이터는 늘 부족하므로, 적은 데이터에서 배울 수 있다는 것이 큰 실용적 장점."
- **n-gram 길이 ablation** ($n \in \{i,...,j\}$ 격자, 논문 Table 4): 3~6이 언어 전반에서 무난한 최적.
  - **2-gram은 무용** — 경계 기호 하나 + 실제 문자 하나 조합이라 접미사를 못 잡는다.
  - **긴 n-gram(≤5, ≤6 포함)이 중요** — 특히 독일어 복합명사는 긴 시퀀스로만 잡힌다.

#### <span style="color: #4682B4">4.5 언어모델링</span>

- LSTM 650 유닛 LM의 임베딩을 sisg 벡터로 초기화 (CS/DE/ES/FR/RU, Botha & Blunsom 세팅). (논문 Table 5)

| | CS | DE | ES | FR | RU |
|---|---|---|---|---|---|
| CLBL (Botha & Blunsom) | 465 | 296 | 200 | 225 | 304 |
| CANLM (Kim et al.) | 371 | 239 | 165 | 184 | 261 |
| LSTM (초기화 없음) | 366 | 222 | 157 | 173 | 262 |
| LSTM + sg 초기화 | 339 | 216 | 150 | 162 | 237 |
| **LSTM + sisg 초기화** | **312** | **206** | **145** | **159** | **206** |

- sg 대비 perplexity 감소: **체코어 8%, 러시아어 13%** vs 스페인어 3%, 프랑스어 2% —
형태가 풍부한 슬라브어에서 이득이 집중된다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[5] Qualitative Analysis</span>

- **가장 중요한 n-gram 추출** (제거했을 때 벡터가 가장 크게 변하는 n-gram, 논문 Table 6 발췌):

| 언어 | 단어 | 중요 n-gram (순서대로) |
|---|---|---|
| DE | autofahrer (운전자) | `fahr`, `fahrer`, `auto` |
| DE | sprachschule (어학원) | `schul`, `hschul`, `sprach` |
| EN | kindness | `ness>`, `ness`, `kind` |
| EN | starfish | `fish`, `fish>`, `star` |
| EN | unlucky | `<un`, `cky>`, `nlucky` |
| FR | finirais (동사 활용) | `ais>`, `nir`, `fini` |

- **사람이 아는 형태소와 일치한다** — 독일어 복합명사 분해, 영어 접사(-ness, un-), 프랑스어 동사 어미(-ais).
- **OOV 단어의 n-gram 매칭 시각화**: "microcircuit"의 n-gram 중 "chip"과 잘 맞는 것은 `micro`와 `circuit`
두 덩어리로 갈라지고, "rarity ↔ scarceness"에서는 `scarce`↔`rarity`, `-ness`↔`-ity`가 각각 매칭,
"preadolescent"는 `-adolesc-` 부분으로 "young"과 매칭 — **합성이 실제로 형태소 단위로 일어난다.**

<hr/> <!-- 수평선 -->

## <span style="color: #F2AA4C">Part 2 — Bag of Tricks for Efficient Text Classification</span>

### <span style="color: #ffd33d">[1] Introduction & Model</span>

- 배경: 딥러닝 분류기(char-CNN, VDCNN)는 정확하지만 **학습/추론이 느려서** 대규모 데이터에 못 쓴다.
반면 선형 분류기는 "적절한 feature만 있으면" SOTA급이고 대규모로 확장 가능하다는 오랜 관찰이 있다
(Joachims 1998, Wang & Manning 2012).
- 모델: BoW 선형 분류기에 **rank 제약**(저차원 임베딩)을 건 형태. 룩업 테이블 $A$로 단어들을 임베딩해
**평균**내고, 선형 분류기 $B$ → softmax $f$로 분류한다.

$$
    -\frac{1}{N}\sum_{n=1}^{N}{y_n \log\big( f(BAx_n) \big)}
$$

- 논문 스스로 명시한다: *"이 아키텍처는 Mikolov et al.의 **CBOW에서 중심 단어를 라벨로 바꾼 것**과 같다."*
학습은 멀티 CPU 비동기 SGD + 선형 감소 learning rate.
- **부품 1 — Hierarchical Softmax** (클래스가 많을 때): $O(kh) \rightarrow O(h\log_2{k})$.
  - 추론에서도 강력하다: 노드 확률은 항상 부모보다 작으므로, **DFS + 최대값 추적으로 낮은 확률 가지를
통째로 버린다** → 최상위 클래스 탐색이 $O(h\log_2{k})$. binary heap을 쓰면 top-T가 $O(\log T)$ 추가 비용.
  - Goodman(2001)과 Huffman 트리 인용 — [계보 ①](/posts/statistical-language-models-based-on-neural-networks/)
②의 그 부품이 세 번째 재사용되는 순간.
- **부품 2 — n-gram feature + hashing trick**: 순서 정보를 bigram feature로 보충하고,
해시 버킷 **bigram만이면 1000만 개 / 그 이상이면 1억 개**로 메모리를 고정한다 (Weinberger 2009).

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[2] Experiments</span>

#### <span style="color: #4682B4">2.1 Sentiment analysis — 정확도</span>

- Zhang et al. (2015)의 8개 데이터셋 + 동일 평가 프로토콜. fastText는 **hidden 10, 5 epochs**,
lr만 validation에서 {0.05, 0.1, 0.25, 0.5} 중 선택. (논문 Table 1)

| Model | AG | Sogou | DBP | Yelp P. | Yelp F. | Yah. A. | Amz. F. | Amz. P. |
|---|---|---|---|---|---|---|---|---|
| BoW | 88.8 | 92.9 | 96.6 | 92.2 | 58.0 | 68.9 | 54.6 | 90.4 |
| ngrams TFIDF | 92.4 | 97.2 | 98.7 | 95.4 | 54.8 | 68.5 | 52.4 | 91.5 |
| char-CNN | 87.2 | 95.1 | 98.3 | 94.7 | 62.0 | 71.2 | 59.5 | 94.5 |
| char-CRNN | 91.4 | 95.2 | 98.6 | 94.5 | 61.8 | 71.7 | 59.2 | 94.1 |
| VDCNN | 91.3 | 96.8 | **98.7** | 95.7 | **64.7** | **73.4** | **63.0** | **95.7** |
| fastText (h=10) | 91.5 | 93.9 | 98.1 | 93.8 | 60.4 | 72.0 | 55.8 | 91.2 |
| **fastText (h=10, bigram)** | **92.5** | 96.8 | 98.6 | **95.7** | 63.9 | 72.3 | 60.2 | 94.6 |

- bigram 추가로 +1~4%p. **char-CNN/char-CRNN보다 약간 좋고, VDCNN보다 약간 나쁜** 수준.
(Sogou는 trigram까지 쓰면 97.1%까지 오른다는 각주)
- Tang et al. (2015) 프로토콜 비교에서도 LSTM-GRNN(Yelp'13 65.1)에 근접한 64.2 — **사전학습 임베딩도
없이** 낸 수치라고 부연.

#### <span style="color: #4682B4">2.2 학습 시간 — 이 논문의 진짜 결과</span>

- char-CNN/VDCNN은 **Tesla K40 GPU**, fastText는 **CPU 20 스레드**. epoch당 시간 (논문 Table 2 발췌):

| | big char-CNN (GPU) | VDCNN depth 9 (GPU) | **fastText (CPU)** |
|---|---|---|---|
| AG | 3h | 24m | **1초** |
| Yahoo Answers | 1일 | 1h | **5초** |
| Amazon Full | 5일 | 2h45 | **9초** |

- 속도 차이는 데이터가 클수록 벌어져서 **최대 15,000배**. Tang의 GRNN은 단일 CPU 스레드로
epoch당 12시간이었다.

#### <span style="color: #4682B4">2.3 Tag prediction — 스케일 테스트 (YFCC100M)</span>

- 이미지 캡션/제목 → 태그 예측. **학습 9,119만 예시(1.5B 토큰), vocab 297K, 태그 312K개.** (논문 Table 5)

| Model | prec@1 | 학습 시간 | 테스트 시간 |
|---|---|---|---|
| 빈도 baseline | 2.2 | - | - |
| Tagspace (h=50) | 30.1 | 3h 8m | 6h |
| Tagspace (h=200) | 35.6 | 5h 32m | 15h |
| fastText (h=50) | 31.2 | 6m 40s | 48s |
| **fastText (h=200, bigram)** | **46.1** | **13m 38s** | **1m 37s** |

- 같은 hidden이면 Tagspace와 대등, bigram을 더하면 크게 앞선다. 추론은 hierarchical softmax의
가지치기 덕에 **600배** 빠르다 (Tagspace는 클래스 전부의 점수를 계산해야 함).
- 결론 (논문 표현): *"딥 네트워크가 이론상 표현력이 더 높지만, sentiment 분석 같은 단순한 분류 문제가
그걸 평가할 올바른 태스크인지는 불분명하다."* — 이후 "fastText 베이스라인부터 이겨라" 문화의 근거 문장.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">계보 — 어디서 왔고 어디로 갔나</span>

#### <span style="color: #4682B4">영향 받은 것</span>

- **← [계보 ③ SGNS (2013.10)](/posts/distributed-representations-of-words-and-phrases/)**:
subword 논문은 SGNS의 objective(로지스틱 loss 형태로 재서술)를 그대로 두고 **점수 함수 한 줄만**
$u_w^{\top}v_c \rightarrow (\sum z_g)^{\top}v_c$로 바꾼 직계 확장이다. negative 샘플링 분포만
$U^{3/4}$ → $\sqrt{U}$로 조정.
- **← [계보 ② (2013.1)](/posts/efficient-estimation-of-word-representations-in-vector-space/)**:
분류기 논문이 스스로 "CBOW의 라벨 버전"이라 명시한다. "단순 모델 × 속도 × 큰 데이터" 철학도 ②의 직계.
- **← [계보 ① 박사논문 (2012)](/posts/statistical-language-models-based-on-neural-networks/)**:
Goodman class 분해 → Huffman hierarchical softmax의 세 번째 재사용(라벨 트리 + DFS 가지치기),
그리고 hash 트릭(RNNME의 hash ME → n-gram 버킷 해싱).
- **← Schütze (1993)**: 문자 4-gram 합산 표현의 원조 (논문이 직접 "closest to our approach"로 인용).
- **← 형태론 기반 선행들** (Luong 2013, Botha & Blunsom 2014): 분석기가 필요하던 접근을
"분석기 없는 n-gram"으로 대체하며 정면 비교에서 이겼다.

#### <span style="color: #4682B4">후속 연구에 준 영향</span>

- **→ subword 시대의 확립**: 같은 시기 NMT의 BPE(Sennrich 2016)와 함께 "단어 아래로 내려간다"는
방향을 실증했다. 이후 **BERT의 WordPiece, GPT의 BPE**로 이어지는 토크나이저 표준의 개념적 조상이다
(fastText는 n-gram "합산", BPE는 "분절" — 접근은 다르지만 방향이 같다). 현대
[Transformer](/posts/attention-is-all-you-need/) 기반 LLM이 OOV 없이 임의 텍스트를 다루는 방식이 여기서 온다.
- **→ 다국어 임베딩 인프라**: 157개 언어 사전학습 벡터 공개(Grave et al. 2018)로 저자원 언어 NLP의
표준 출발점이 됐다. 검색 자동완성, 오타 강건 매칭, 언어 감지(공식 language-id 모델) 등에서 아직 현역.
- **→ 베이스라인 문화**: "딥러닝 분류 논문은 fastText를 이겨야 한다"는 관행. 산업계 초고속 분류기
(스팸/태깅/의도분류)로 대량 배포.
- **→ 정적 임베딩의 완성**: RNNLM(①) → word2vec(②③) → fastText(④)로 정적 임베딩은 완성형에 도달했고,
남은 한계(문맥 의존 의미)는 ELMo/BERT의 문맥적 임베딩이 이어받는다 — 계보가
[Attention Is All You Need](/posts/attention-is-all-you-need/)로 연결된다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">Reference</span>

- [1] P. Bojanowski et al., "Enriching Word Vectors with Subword Information" (TACL 2017)
- [2] A. Joulin et al., "Bag of Tricks for Efficient Text Classification" (EACL 2017)
- [3] T. Mikolov et al., "Distributed Representations of Words and Phrases..." (NIPS 2013)
- [4] H. Schütze, "Word Space" / character 4-gram SVD (1993)
- [5] R. Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (BPE, ACL 2016)
- [6] X. Zhang et al., "Character-level Convolutional Networks for Text Classification" (NIPS 2015)
- [7] E. Grave et al., "Learning Word Vectors for 157 Languages" (LREC 2018)
