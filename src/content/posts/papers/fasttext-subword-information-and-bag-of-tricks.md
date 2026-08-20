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
[SGNS](/posts/distributed-representations-of-words-and-phrases/)의 한계(형태 정보 무시, OOV 처리 불가)를,
단어 벡터를 **문자 n-gram 벡터들의 합**으로 바꿔서 해결한다. objective는 SGNS 그대로다.
2. **Bag of Tricks for Efficient Text Classification (EACL 2017)** — 같은 구조(임베딩 평균 + 선형 분류기 +
hierarchical softmax)를 텍스트 분류에 적용하면, **딥러닝 모델과 비슷한 정확도를 수천~수만 배 빠르게**
(10억 단어 학습이 CPU에서 10분 수준) 얻을 수 있음을 보인다.

"단어보다 작은 단위(subword)가 표현의 기본 단위가 되어야 한다"는 방향 제시가 이 논문의 유산이고,
이는 BPE/WordPiece 토크나이저를 쓰는 현대 LLM까지 이어진다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[1] 문제의식 — 단어는 원자가 아니다</span>

- [word2vec](/posts/distributed-representations-of-words-and-phrases/)은 단어마다 독립된 벡터를 하나씩 배운다.
  - "eat / eats / eaten / eating"이 **아무 파라미터도 공유하지 않는다.** 형태가 풍부한 언어(체코어, 핀란드어,
독일어, 터키어, 한국어…)에서는 활용형이 수십 개씩 폭발하기 때문에, 각 형태의 등장 횟수가 적어져
벡터 품질이 나빠진다. (프랑스어/스페인어 동사는 활용형이 40개 이상, 핀란드어 명사는 격변화 15가지)
  - 학습 때 못 본 단어(OOV)는 **벡터 자체가 없다.** 실전 시스템에서 치명적이다.
- 형태소를 쓰는 선행 연구들(Luong 2013의 morphological RNN, Botha & Blunsom 2014 등)이 있었지만
**형태소 분석기가 필요**해서 언어마다 별도 자원이 든다.
- fastText의 해법: 단어를 **문자 n-gram의 가방(bag)** 으로 분해해서, n-gram 벡터를 공유 파라미터로 삼는다.
분석기 없이(언어 무관) 형태 정보를 통계적으로 흡수하는 것이 포인트다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[2] Subword Model (TACL 2017) 뜯어보기</span>

#### <span style="color: #4682B4">2.1 단어 분해</span>

- 단어 양끝에 경계 기호 `<`, `>`를 붙이고, $n = 3{\sim}6$의 문자 n-gram을 전부 뽑는다. $n=3$이면

```text
where  →  <where>  →  {<wh, whe, her, ere, re>}  +  단어 자체 <where>
```

- 경계 기호의 역할이 중요하다: `her`(where 내부의 3-gram)와 `<her>`(단어 her 전체)는 **서로 다른 단위**가
되어, 접두/접미사가 단어 중간 문자열과 구분된다.
- 단어 자체(`<where>`)도 하나의 단위로 포함해서, 고빈도 단어는 사실상 자기 전용 벡터를 유지할 수 있다.
- $n$ 범위 ablation: 3~6이 언어 전반에서 무난한 최적. 2-gram은 노이즈가 많고(짧아서 변별력 없음),
접사류가 3~6글자에 주로 걸린다.

#### <span style="color: #4682B4">2.2 Scoring — SGNS에서 바뀌는 곳은 한 줄</span>

- 단어 $w$의 n-gram 집합을 $\mathcal{G}_w \subset \{1, ..., G\}$라 하면, (중심 $w$, 문맥 $c$)의 점수를
n-gram 벡터 $z_g$들의 합으로 정의한다.

$$
    s(w, c) = \sum_{g \in \mathcal{G}_w}{z_g^{\top} v_c}
    = \Big( \sum_{g \in \mathcal{G}_w}{z_g} \Big)^{\top} v_c
$$

- [SGNS](/posts/distributed-representations-of-words-and-phrases/)의 negative sampling objective에서
${v'_{w_O}}^{\top}v_{w_I}$ 자리에 $s$를 넣으면 끝이다. **loss, 노이즈 분포($U^{3/4}$), subsampling 전부 그대로.**

$$
    \log{\sigma\big(s(w_t, w_c)\big)} + \sum_{i=1}^{k}{\mathbb{E}_{n_i \sim P_n}\Big[\log{\sigma\big(-s(w_t, n_i)\big)}\Big]}
$$

<details>
<summary> <span style="color: #ffd33d">subword 모델의 gradient — 왜 형태소가 공유되는가 펼치기/접기</span> </summary>

- SGNS의 gradient 유도([계보 ③ 리뷰의 2절 접기](/posts/distributed-representations-of-words-and-phrases/) 참고)에서
오차항을 $e = \sigma(s(w,c)) - t$ ($t$: 진짜/노이즈 라벨)라 하면, 점수가 합 형태이므로 chain rule에 의해
**같은 gradient가 모든 구성 n-gram에 균등하게 흘러간다.**

$$
    \frac{\partial L}{\partial z_g} = e \cdot v_c \quad (\forall g \in \mathcal{G}_w)
    ,\qquad
    \frac{\partial L}{\partial v_c} = e \cdot \sum_{g \in \mathcal{G}_w}{z_g}
$$

- 결과적으로 "eating"이 학습될 때 n-gram `eat`, `ing>` 벡터가 업데이트되고, 그 벡터들은
"eats", "eaten", "playing"에서도 **재사용**된다 — 이것이 활용형끼리 파라미터를 공유하는 메커니즘의 전부다.
- 희귀 단어일수록 자기 단어 벡터의 업데이트 기회는 적지만, 구성 n-gram들은 다른 단어들을 통해
계속 학습되므로 **희귀/미등장 단어에서 이득이 가장 크다.** $\blacksquare$

</details>

<hr/> <!-- 수평선 -->

- **OOV 처리**: 학습 때 못 본 단어도 n-gram 벡터들의 합으로 즉석에서 벡터를 만들 수 있다.
word2vec에는 없던 능력이다.

#### <span style="color: #4682B4">2.3 Hashing Trick — 메모리 다루기</span>

- 문자 n-gram의 종류는 단어 수보다 훨씬 많아서(수천만 단위) 전부 테이블에 담을 수 없다.
n-gram을 **해시 함수(FNV-1a)로 $K = 2 \times 10^6$개의 버킷에 매핑**해서 벡터 테이블 크기를 고정한다.
- 충돌(서로 다른 n-gram이 같은 버킷)은 감수한다 — 파라미터 공유의 노이즈 버전으로 동작하고,
실험적으로 품질 손실이 작다. (대규모 임베딩을 상수 메모리로 다루는 실전 트릭)
- 구현: 멀티스레드 비동기 SGD(HogWild 스타일), learning rate 선형 감소 — word2vec 툴킷의 학습 루프를
그대로 계승했다. 학습 속도는 SGNS 대비 약 1.5배 느린 수준 (n-gram 합산 비용).

#### <span style="color: #4682B4">2.4 실험 — 유사도/유추/희귀어</span>

- 위키피디아 9개 언어(영어, 독일어, 체코어, 프랑스어, 스페인어, 이탈리아어, 루마니아어, 러시아어, 아랍어)로
학습하고, 인간 평가 유사도(Spearman 상관)와 analogy로 비교한다.
비교 대상: cbow/skipgram(word2vec) vs **sisg**(subword 모델; OOV는 n-gram 합으로 처리).
- 결과 패턴:

| 평가 | 결과 |
|---|---|
| 독일어/체코어/러시아어 유사도 | **sisg가 큰 폭 우위** (형태 풍부 언어에서 이득 최대) |
| 영어 Rare Words(RW) | sisg 우위 (희귀어일수록 subword 이득) |
| 영어 WS353 (일반 유사도) | 비슷하거나 word2vec이 소폭 우위 — subword가 만능은 아님 |
| Analogy — syntactic | **sisg 큰 폭 우위** (활용형이 n-gram으로 연결되므로) |
| Analogy — semantic | 비슷하거나 소폭 하락 (짧은 n-gram이 의미 관계엔 노이즈일 수 있음) |

- **학습 데이터 크기 ablation**: 데이터를 1%, 5%, …로 줄여가며 측정하면 **작은 데이터일수록 sisg의
상대 우위가 커진다.** n-gram 공유가 데이터 효율을 올린다는 직접 증거.
- **언어모델 실험**: subword 임베딩을 LSTM LM의 입력으로 쓰면 체코어/러시아어 등에서 perplexity가
일관되게 개선된다 — 임베딩 교체만으로 다운스트림이 좋아지는 사례.

#### <span style="color: #4682B4">2.5 정성 분석 — 모델이 배운 형태소</span>

- 각 단어에서 **어떤 n-gram이 표현에 가장 중요한지**(제거 시 벡터 변화가 큰 순) 뽑아보면
사람이 아는 형태소와 일치하는 경우가 많다.
  - autofahrer(독일어, 운전자) → `auto`, `fahrer` / anarchy → `narchy`, `chy>` / politeness → `polite`, `ness>`
- OOV 단어의 n-gram 매칭 시각화: 예를 들어 미등장 단어 "scarceness"의 벡터는 학습된 단어 "scarce"의
`scarce` 부분과 "politeness"의 `ness>` 부분에 강하게 반응한다 — 합성이 실제로 형태소 단위로 일어난다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[3] Bag of Tricks (EACL 2017) — 분류기로서의 fastText</span>

#### <span style="color: #4682B4">3.1 모델</span>

- 텍스트 분류의 당시 SOTA는 CNN/RNN 계열(char-CNN, VDCNN 등)이었는데, 학습에 GPU로 몇 시간~며칠이 걸렸다.
- fastText 분류기의 구조는 의도적으로 단순하다. 문서의 단어(+n-gram) 임베딩을 평균내고 선형 분류기 하나로
끝낸다. $N$개 문서에 대한 loss:

$$
    -\frac{1}{N}\sum_{n=1}^{N}{y_n \log\big( f(BAx_n) \big)}
$$

- $x_n$: 문서의 feature(단어, n-gram)들, $A$: 임베딩 lookup 후 **평균**, $B$: 선형 분류기, $f$: softmax.
즉 **CBOW에서 중심 단어 대신 라벨을 예측**하는 구조다. 은닉층이 하나 있는 선형 모델이므로,
feature 공간을 라벨 공간과 공유하는 저차원 공간으로 사상하는 shallow 모델이라 볼 수 있다.
- 속도를 위한 부품 2개:
  1. 라벨이 많을 때(태그 예측: 31만 개)는 **hierarchical softmax** — 계보 ①②의 그 부품(Huffman 트리).
학습 $O(\log K)$, 추론도 트리 가지치기로 최상위 라벨을 $O(\log K)$에 찾는다.
  2. 단어 순서 정보는 **bigram feature + hashing trick**(1000만 버킷)으로 보충한다.
bag-of-words의 순서 손실을 값싸게 만회하는 장치로, sentiment 계열에서 1~4%p를 더 얻는다.

#### <span style="color: #4682B4">3.2 Sentiment 분류 결과</span>

- 8개 표준 데이터셋(AG News, Sogou, DBpedia, Yelp, Yahoo, Amazon 계열)에서 char-CNN, VDCNN 등과 비교:

| Dataset | char-CNN 계열 | VDCNN | fastText (+bigram) |
|---|---|---|---|
| AG News | 90~91% | 91.3% | **92.5%** |
| DBpedia | 98.3% | **98.7%** | 98.6% |
| Yelp Polarity | 94~95% | **95.7%** | 95.7% |
| Yelp Full | 62% 수준 | **64.7%** | 63.9% |
| Amazon Polarity | 94~95% | **95.7%** | 94.6% |
| Amazon Full | 59% 수준 | **63.0%** | 60.2% |

- 정확도는 "대등하거나 소폭 아래" — 그런데 **학습 시간이 차원이 다르다.**

| 모델 | 학습 시간 (규모 큰 데이터셋 기준) |
|---|---|
| char-CNN | GPU로 수 시간~수 일 |
| VDCNN | GPU로 수 시간 |
| **fastText** | **멀티코어 CPU로 수 초~수십 초** |

- 논문 표현 그대로 "3 orders of magnitude faster". 정확도 1~3%p와 수천 배 속도의 교환이면
실무에서는 대부분 fastText가 이긴다.

#### <span style="color: #4682B4">3.3 대규모 태그 예측</span>

- YFCC100M(이미지 캡션/제목 → 태그 31만 개 예측)에서 Tagspace(랭킹 기반 임베딩 모델) 대비
**정확도 우위 + 학습/추론 모두 압도적 속도 우위**를 보인다. (Tagspace가 GPU급 시간을 쓸 때
fastText는 CPU 수 분)
- "라벨이 수십만 개인 극단적 멀티클래스도 hierarchical softmax로 선형 모델이 커버 가능"의 실증.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[4] 계보 — 어디서 왔고 어디로 갔나</span>

#### <span style="color: #4682B4">영향 받은 것</span>

- **← [word2vec 계보 ③ / SGNS (2013.10)](/posts/distributed-representations-of-words-and-phrases/)**:
subword 논문은 SGNS의 **objective/negative sampling/subsampling을 한 글자도 안 바꾸고** 점수 함수만
$v_w^{\top}v_c \rightarrow (\sum z_g)^{\top}v_c$로 바꾼 직계 확장이다. "잘 되는 뼈대는 두고 표현 단위만 바꾼다"는
최소 수정 설계.
- **← [word2vec 계보 ② (2013.1)](/posts/efficient-estimation-of-word-representations-in-vector-space/)**:
분류기 논문의 구조(임베딩 평균 → 선형 → hierarchical softmax)는 **CBOW를 지도학습으로 돌려놓은 것**이고,
"단순한 모델 × 큰 데이터 × 속도"라는 철학도 ②의 직계다.
- **← [Mikolov 박사논문 (2012)](/posts/statistical-language-models-based-on-neural-networks/)**:
class/트리 기반 출력 분해가 hierarchical softmax로 이어져 분류기의 핵심 부품으로 재사용된다.
- **← 형태론 기반 임베딩 선행 연구들** (Luong 2013, Botha & Blunsom 2014): 형태소 분석기가 필요했던 접근을
"분석기 없는 문자 n-gram"으로 단순화한 것이 차별점.

#### <span style="color: #4682B4">후속 연구에 준 영향</span>

- **→ subword 시대의 개막**: "표현의 기본 단위는 단어가 아니라 subword"라는 방향을 실증했다.
같은 시기 NMT 쪽에서 독립적으로 나온 BPE(Sennrich et al. 2016)와 함께, 이후
**BERT의 WordPiece, GPT의 BPE 토크나이저**로 이어지는 표준을 만들었다 — 현대
[Transformer](/posts/attention-is-all-you-need/) 기반 LLM이 OOV 없이 임의 텍스트를 다루는 방식의 개념적 조상이다.
(차이: fastText는 n-gram "합산", BPE는 "분절" — 접근은 다르지만 "단어 아래로 내려간다"는 방향이 같다)
- **→ 다국어 임베딩 인프라**: 157개 언어의 사전학습 fastText 벡터 공개(Grave et al. 2018)로,
저자원 언어 NLP의 사실상 표준 출발점이 됐다. 아직도 가벼운 시스템, 검색 자동완성, 오타 강건 매칭 등에서 현역이다.
- **→ 베이스라인 문화**: "딥러닝 분류 논문은 fastText 베이스라인을 이겨야 한다"는 관행을 만들었고,
산업계에서는 초고속 분류기(스팸/태깅/의도분류/언어감지)로 대량 배포됐다. (공식 language-id 모델이 대표 사례)
- **→ 정적 임베딩의 마지막 세대**: 이 시리즈(RNNLM → word2vec → fastText)로 정적 임베딩은 완성형에 도달했고,
남은 한계(문맥에 따른 의미 변화)는 ELMo/BERT의 **문맥적 임베딩**이 이어받는다. 계보가 여기서
[Attention Is All You Need](/posts/attention-is-all-you-need/) 리뷰로 연결된다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">Reference</span>

- [1] P. Bojanowski et al., "Enriching Word Vectors with Subword Information" (TACL 2017)
- [2] A. Joulin et al., "Bag of Tricks for Efficient Text Classification" (EACL 2017)
- [3] T. Mikolov et al., "Distributed Representations of Words and Phrases..." (NIPS 2013)
- [4] R. Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (BPE, ACL 2016)
- [5] E. Grave et al., "Learning Word Vectors for 157 Languages" (LREC 2018)
- [6] M. Luong et al., "Better Word Representations with Recursive Neural Networks for Morphology" (CoNLL 2013)
