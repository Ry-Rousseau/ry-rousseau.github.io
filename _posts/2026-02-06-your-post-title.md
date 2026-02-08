---
tags: Project
---

## 15-Class Framing Detector with Longformer

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Model Architecture Evolution](#3-model-architecture-evolution)
4. [Experimental Runs](#4-experimental-runs)
5. [Phase 1: Silver Training](#5-phase-1-silver-training)
6. [Phase 2: Gold Fine-Tuning](#6-phase-2-gold-fine-tuning)
7. [Key Findings and Ablations](#7-key-findings-and-ablations)
8. [Final Model Performance](#8-final-model-performance)
9. [Limitations and Future Work](#9-limitations-and-future-work)

---

## 1. Overview

Media framing is the process by which communicators select specific aspects of a perceived reality and make them more salient within a text (Card et al. 2015). The predominant classification scheme was introduced in the 2015 Media Frames Corpus, and has since been used for argumentation mining, bias and polarization detection, sentiment analysis, and research into cross-cultural linguistics. Accurate, lightweight frame classifiers have applications in internet media analytics and enable richer social network analysis by tracking the communication tactics of specific users. 

Card et al. (2015) operationalize framing not just as isolated keywords, but as "thematic sets" or "packages" of aligned ideas and assumptions.

| Code | Frame | Description |
|------|-------|-------------|
| 1 | Economic | Costs, benefits, wages, taxes |
| 2 | Capacity & Resources | Staffing, backlog, infrastructure |
| 3 | Morality | Religious, ethical, social duty |
| 4 | Fairness & Equality | Rights, discrimination |
| 5 | Legality & Constitutionality | Laws, court cases |
| 6 | Policy Prescription | Specific proposals |
| 7 | Crime & Punishment | Enforcement, fraud |
| 8 | Security & Defense | Border security, terrorism |
| 9 | Health & Safety | Disease, sanitation |
| 10 | Quality of Life | Community impact |
| 11 | Cultural Identity | Assimilation, language |
| 12 | Public Opinion | Polls, public sentiment |
| 13 | Political | Partisan, elections |
| 14 | External Regulation | International, treaties |
| 15 | Other | Miscellaneous |

Existing methods to automate frame-detection suffer from several limitations:
1) They rely heavily on GenAI prompt-engineering, coercing generative models into classification tasks. This risks the hallucination of non-existing frames, lowers the interpretability of final models (increasing the black-box barrier), and utilizes computationally expensive solutions which are difficult to scale to large-N data.

2) Context length. Existing models are often designed for sentence-level classification, whereas frames develop through complex interplays across hundreds of words of text, and reference past signals from the same text input. A longer context input with appropriate attention handling is essential to pick-up these nuanced interactions. 

3) Training data. Past papers rely on machine-generated classification data, rather than gold-standard human annotations, which can bake-in biases, without appropriately correcting them in fine-tuning. A lack of abundant human annotated data for frame detection remains a serious challenge for work in this area, and the encoding of human-backed reasoning into model weights requires careful correction with this constraint.

I sought to develop a lightweight transformer-based model which addresses these challenges, and would be useful for applied NLP research. The final model uses multi-label classification to detect the presence of each of the 15-frames in input text up to ~1600 words. It outperforms past GenAI approaches in classification accuracy on the Media Frames Corpus, while being 47 times smaller in model size. 

I developed the project in its associated github repo (LINK) and uploaded my final model to HuggingFace. The current version (as of writing) may be iterated upon in future, as I optimize further on the available data and test new architectures, which will reflect in future model updates. 

---

## 2. Training Data

The two best datasets for this task are: the 2015 Media Frames Corpus (MFC) and the 2023 SemEval Task 3 (sub-task 2). Accessibility for the MFC has been greatly reduced (as Lexis Nexis has paywalled document access), leaving about 21% of the original 14,481 articles available to me (via Lexis Uni). The SemEval source is open-access, after messaging the task organizers, and provides 516 additional articles. Together, these generate a gold-standard dataset of 2470 articles, combining labels from over 19 trained annotators. 

### 2.1 Gold Data: Human Annotations

#### Media Frames Corpus (MFC)

Original corpus from Card et al. (2015) containing human-annotated NYT articles on three policy issues.

| Topic | Original MFC | NYT Available | Retrieved | Coverage |
|-------|-------------|---------------|-----------|----------|
| Immigration | 5,500 | 1,443 | 1,370 | 24.9% |
| Smoking | 5,074 | 822 | 821 | 16.2% |
| Same-Sex Marriage | 8,407 | 1,764 | 1,740 | 20.7% |
| **Total** | **18,981** | **4,029** | **3,931** | **20.7%** |

**Annotation Aggregation Strategy:**

Each MFC article has 2+ annotators with span-level annotations. I tested three aggregation methods, and opted to use the union of annotations to maximize the available signal and match the silver data. 

| Strategy | Mean Frames/Article | Match to Silver (4.62)? |
|----------|---------------------|-------------------------|
| Per annotator | 3.18 | Too conservative |
| **Union** | **4.28** | Close match |
| Intersection | 2.14 | Too conservative |

#### SemEval 2023 Task 3 Subtask 2

**Combined Gold Dataset:**

| Source | Articles | Train | Test |
|--------|----------|-------|------|
| MFC | 2,224 | 1,993 | 231 |
| SemEval | 516 | 473 | 43 |
| **Total** | **2,740** | **2,466** | **274** |

**Gold Data Label Distribution:**

| Frame | Count | % of Articles |
|-------|-------|---------------|
| Legality | 1,600 | 58.4% |
| Political | 1,373 | 50.1% |
| Policy Prescription | 1,127 | 41.1% |
| Crime & Punishment | 879 | 32.1% |
| Economic | 870 | 31.8% |
| Quality of Life | 825 | 30.1% |
| Cultural Identity | 737 | 26.9% |
| Public Opinion | 688 | 25.1% |
| Health & Safety | 664 | 24.2% |
| Fairness & Equality | 599 | 21.9% |
| Morality | 590 | 21.5% |
| Security & Defense | 451 | 16.5% |
| External Regulation | 364 | 13.3% |
| Other | 304 | 11.1% |
| Capacity & Resources | 276 | 10.1% |

**Average labels per article:** 4.14

### 2.2 Silver Data: LLM Annotations

**Source:** `copenlu/mm-framing`

This dataset from (CITE) contains ~478,000 news articles with frame labels generated by Mistral-7B-Instruct-v0.3 using prompt engineering. It was cleaned to remove short length articles (<100 words) and non-standard frames. 378,000 examples were retained, 

**Label Generation Method:**
- Model: Mistral-7B-Instruct-v0.3
- Inference: vLLM with temperature=0.2, max_tokens=4000
- Process: Prompted to read text and classify into the 15-frame taxonomy

**Mistral's Performance Against Gold Standard (Media Frames Corpus):**

| Label | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Capacity & Resources | 0.39 | 0.34 | 0.36 |
| Crime | 0.50 | 0.87 | 0.63 |
| Culture | 0.38 | 0.37 | 0.37 |
| Economic | 0.43 | 0.69 | 0.53 |
| Fairness | 0.17 | 0.74 | 0.28 |
| Health | 0.48 | 0.48 | 0.48 |
| Legality | 0.53 | 0.87 | 0.66 |
| Morality | 0.30 | 0.63 | 0.41 |
| Policy | 0.40 | 0.73 | 0.51 |
| Political | 0.68 | 0.53 | 0.60 |
| Public Opinion | 0.32 | 0.55 | 0.40 |
| Quality of Life | 0.28 | 0.36 | 0.31 |
| Regulation | 0.26 | 0.48 | 0.34 |
| Security | 0.30 | 0.45 | 0.36 |
| **Micro Avg** | 0.42 | 0.62 | **0.50** |
| **Macro Avg** | 0.39 | 0.58 | **0.45** |

**Implication:** Our silver-trained models have a ceiling - we're learning to replicate Mistral's labeling behavior, which itself only achieves 0.50 F1 against human annotations.

**Dataset Variables Used:**
- `title` - Article headline
- `gpt_topic` - Consolidated topic (19 categories)
- `text_generic_frame` - List of frame labels
- `maintext` - Full article body (from joined `newsarticles` table)

I consolidated the original unstructured `gpt_topic` into 19 topic categories based on similarity. This ensured the metadata was useful for training a single-label classifier to enable a mixture of experts approach in the final model.

---

3. [Model Architecture](#3-model-architecture-evolution)

## 3. Model Architecture Evolution

### 3.1 Topic Classifier: RoBERTa with Truncation

I found that these models perform best when they adopt a mixture of experts approach. To enable,

**Solution Explored:** Head+Tail truncation strategy
- Head: First 320 tokens (captures title and introduction)
- Tail: Last 190 tokens (captures conclusion)
- Total: 510 tokens (leaving room for special tokens)

**Limitation:** Loses mid-article content where many frames (especially Health, Quality of Life) are discussed.

### 3.2 Domain Specialization Hypothesis

**Question:** Do frames have different meanings across topics?

**Experiment:** Train specialist model on Politics-only articles (~44,000 articles).

**Finding:** Domain-specific training improved Micro F1 (+2.7%) but hurt Macro F1 (-2.0%). The model over-indexed on dominant political frames.

**Insight Validated:** Frame definitions are context-dependent. "Fairness" in politics vs. sports means different things.

### 3.3 Topic Injection Strategy

**Hypothesis:** Instead of separate models per topic, inject topic as a prefix signal.

**Implementation:**
```
TOPIC:{topic}
{title}
{article_text}
```

**Benefit:** Single model handles all topics with domain signal. Ablation showed +2.7% Micro F1 gain from topic injection alone.

### 3.4 Longformer for Full Document Context

**Problem:** RoBERTa's truncation loses critical mid-article framing.

**Solution:** Longformer with 2048-token context window.

**Key Configuration:**
- Global attention on [CLS] token (position 0)
- Global attention on first topic token (position 3)
- Gradient checkpointing for VRAM efficiency
- Batch size 2 with gradient accumulation (effective batch 16)

---




Given this insufficient number of gold training examples for an generalizable classifier, I gather silver, machine-annotated data from the mm-framing 2025 paper, which 

Dealing with nuanced extraction of bias and 'framing,' defined as the feature extraction. GenAI has been used for this task (insert some past papers). But I've gained inherent mistrust of these models for 


Build a lightweight, real-time tool that quantifies "Framing Bias" between news sources by detecting which of 15 political frames each article emphasizes.

**Use Case:** Given two URLs (e.g., CNN vs. Fox News on the same topic), the system:
1. Scrapes article text
2. Classifies into 15 Political Frames
3. Calculates "Frame Delta" between sources

**Example Output:** "Source A focuses on *Morality* (60%), Source B focuses on *Economics* (40%)."

### 1.2 The 15 Media Frames

Based on the Media Frames Corpus taxonomy:

| Code | Frame | Description |
|------|-------|-------------|
| 1 | Economic | Costs, benefits, wages, taxes |
| 2 | Capacity & Resources | Staffing, backlog, infrastructure |
| 3 | Morality | Religious, ethical, social duty |
| 4 | Fairness & Equality | Rights, discrimination |
| 5 | Legality & Constitutionality | Laws, court cases |
| 6 | Policy Prescription | Specific proposals |
| 7 | Crime & Punishment | Enforcement, fraud |
| 8 | Security & Defense | Border security, terrorism |
| 9 | Health & Safety | Disease, sanitation |
| 10 | Quality of Life | Community impact |
| 11 | Cultural Identity | Assimilation, language |
| 12 | Public Opinion | Polls, public sentiment |
| 13 | Political | Partisan, elections |
| 14 | External Regulation | International, treaties |
| 15 | Other | Miscellaneous |

### 1.3 Technical Constraints

- **Hardware:** Single GPU (16GB VRAM)
- **Inference Requirement:** Real-time classification
- **Task Type:** Multi-label text classification

---

