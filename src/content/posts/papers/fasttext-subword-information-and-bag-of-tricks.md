---
title: "[논문 리뷰]fastText — Enriching Word Vectors with Subword Information & Bag of Tricks"
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
독일어, 한국어…)에서는 활용형 폭발 때문에 각 형태의 등장 횟수가 적어 벡터 품질이 나빠진다.
  - 학습 때 못 본 단어(OOV)는 **벡터 자체가 없다.**
- 해결 방향: 단어를 **문자 n-gram의 가방(bag)** 으로 분해해서, n-gram 벡터를 공유 파라미터로 삼는다.
형태소 분석기 없이(언어 무관) 형태 정보를 흡수하는 것이 포인트.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[2] Subword Model (TACL 2017) 뜯어보기</span>

#### <span style="color: #4682B4">2.1 단어 분해</span>

- 단어 양끝에 경계 기호를 붙여 `<where>`로 만들고, $n = 3{\sim}6$의 문자 n-gram을 전부 뽑는다. $n=3$이면

```text
<where>  →  <wh, whe, her, ere, re>  +  특수 시퀀스 <where>
```

- 경계 기호 덕분에 접두/접미사가 구분된다. (`her`와 `<her>`는 서로 다른 단위 — "where 속의 her"와
"단어 her"가 다른 벡터를 가짐)
- 단어 자체(`<where>`)도 하나의 단위로 포함해서, 고빈도 단어는 자기 벡터를 그대로 활용할 수 있게 한다.

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

- **OOV 처리**: 학습 때 못 본 단어도 n-gram들의 벡터 합으로 즉석에서 벡터를 만들 수 있다.
word2vec에는 없던 능력이다.

#### <span style="color: #4682B4">2.3 Hashing Trick — 메모리 다루기</span>

- 문자 n-gram의 종류는 단어 수보다 훨씬 많아서 전부 테이블에 담을 수 없다.
n-gram을 **해시 함수(FNV-1a)로 $K = 2 \times 10^6$개의 버킷에 매핑**해서 벡터 테이블 크기를 고정한다.
- 충돌(서로 다른 n-gram이 같은 버킷)은 감수한다 — 파라미터 공유의 노이즈 버전으로 동작하고,
실험적으로 품질 손실이 작다. (대규모 임베딩을 상수 메모리로 다루는 실전 트릭)

#### <span style="color: #4682B4">2.4 결과</span>

- 단어 유사도 평가: 형태가 풍부한 언어(독일어, 체코어 등)에서 word2vec 대비 큰 폭 향상, 영어도 소폭 향상.
- analogy: **문법(syntactic) 관계에서 큰 향상** (활용형이 n-gram으로 연결되므로 당연한 결과),
의미(semantic) 관계는 비슷하거나 약간 하락 — subword가 만능은 아니라는 정직한 결과.
- 작은 데이터에서 이득이 특히 크다. (n-gram 공유가 데이터 효율을 올려줌)

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[3] Bag of Tricks (EACL 2017) — 분류기로서의 fastText</span>

- 텍스트 분류의 당시 SOTA는 CNN/RNN 계열(char-CNN, VDCNN 등)이었는데, 학습에 GPU로 몇 시간~며칠이 걸렸다.
- fastText 분류기의 구조는 의도적으로 단순하다.

$$
    -\frac{1}{N}\sum_{n=1}^{N}{y_n \log\big( f(BAx_n) \big)}
$$

- $x_n$: 문서의 단어/n-gram feature들, $A$: 임베딩 lookup 후 **평균**, $B$: 선형 분류기, $f$: softmax.
즉 **CBOW에서 중심 단어 대신 라벨을 예측**하는 구조다.
- 속도를 위한 부품 2개:
  1. 라벨 수가 많을 때는 **hierarchical softmax** (word2vec 계보의 그 부품, Huffman 트리) —
inference도 트리 탐색으로 $O(\log K)$.
  2. 단어 순서 정보는 **bigram feature + hashing trick**으로 보충한다. (bag-of-words의 순서 손실을 값싸게 만회)
- 결과: sentiment 분석 8개 데이터셋과 태그 예측(1M 라벨)에서 **깊은 모델들과 대등한 정확도**를,
**수천~수만 배 빠른 학습 속도**(멀티코어 CPU 10분 이내 vs GPU 수시간~수일)로 달성.
- 메시지가 명확하다: "간단한 선형 + 좋은 feature가 아직도 매우 강한 베이스라인이다."
이후 논문들이 fastText를 필수 베이스라인으로 깔게 됐다.

<hr/> <!-- 수평선 -->

### <span style="color: #ffd33d">[4] 계보 — 어디서 왔고 어디로 갔나</span>

#### <span style="color: #4682B4">영향 받은 것</span>

- **← [word2vec 2편 / SGNS (2013.10)](/posts/distributed-representations-of-words-and-phrases/)**:
subword 논문은 SGNS의 **objective/negative sampling/subsampling을 한 글자도 안 바꾸고** 점수 함수만
$v_w^{\top}v_c \rightarrow (\sum z_g)^{\top}v_c$로 바꾼 직계 확장이다. "잘 되는 뼈대는 두고 표현 단위만 바꾼다"는
최소 수정 설계.
- **← [word2vec 1편 (2013.1)](/posts/efficient-estimation-of-word-representations-in-vector-space/)**:
분류기 논문의 구조(임베딩 평균 → 선형 → hierarchical softmax)는 **CBOW를 지도학습으로 돌려놓은 것**이고,
"단순한 모델 × 큰 데이터 × 속도"라는 철학도 1편의 직계다.
- **← [Mikolov 박사논문 (2012)](/posts/statistical-language-models-based-on-neural-networks/)**:
class/트리 기반 출력 분해가 hierarchical softmax로 이어져 분류기의 핵심 부품으로 재사용된다.
- **← 형태론 기반 임베딩 선행 연구들** (Luong 2013의 morpheme RNN 등): 형태소 분석기가 필요했던 접근을
"분석기 없는 문자 n-gram"으로 단순화한 것이 차별점.

#### <span style="color: #4682B4">후속 연구에 준 영향</span>

- **→ subword 시대의 개막**: "표현의 기본 단위는 단어가 아니라 subword"라는 방향을 실증했다.
같은 시기 NMT 쪽에서 독립적으로 나온 BPE(Sennrich et al. 2016)와 함께, 이후
**BERT의 WordPiece, GPT의 BPE 토크나이저**로 이어지는 표준을 만들었다 — 현대
[Transformer](/posts/attention-is-all-you-need/) 기반 LLM이 OOV 없이 임의 텍스트를 다루는 방식의 개념적 조상이다.
- **→ 다국어 임베딩 인프라**: 157개 언어의 사전학습 fastText 벡터 공개로, 저자원 언어 NLP의 사실상 표준
출발점이 됐다. (아직도 가벼운 시스템에서 현역이다)
- **→ 베이스라인 문화**: "딥러닝 논문은 fastText 베이스라인을 이겨야 한다"는 관행을 만들었고,
산업계에서는 초고속 분류기(스팸/태깅/의도분류)로 대량 배포됐다.
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
