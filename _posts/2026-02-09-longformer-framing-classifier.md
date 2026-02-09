---
title: "Framing Detector with Longformer"
tags: Project
---

## Framing Detector with Transformer Encoder

## Table of Contents

1. [Overview](#1-overview)
2. [Training Data](#2-training-data)
   - [Gold Data: Human Annotations](#21-gold-data-human-annotations)
   - [Silver Data: LLM Annotations](#22-silver-data-llm-annotations)
3. [Model Architecture](#3-model-architecture-evolution)
   - [Topic Classifier: RoBERTa with Truncation](#31-topic-classifier-roberta-with-truncation)
   - [Longformer Classifier](#32-longformer-for-full-document-classification)
4. [Silver Training](#4-silver-training)
5. [Gold Training](#5-gold-training)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Appendix](#appendix)
   - [Future Directions](#future-directions)
   - [Tips and Tricks](#tips-and-tricks)
   - [Past Experiments](#experiments)

---

## 1. Overview

Media framing is the process by which communicators select specific aspects of a perceived reality and make them more salient within a text (Card et al. 2015). The concept was operationalized in the 2015 Media Frames Corpus, which provided a 15-category encoding to be used to describe communication strategy in news articles, and has since been used extensively for measuring argumentation, bias and polarization. Accurate, lightweight frame classifiers have applications in internet media analytics and enable richer social network analysis by tracking the communication tactics of specific users. 

Card et al. (2015) operationalize framing not just as isolated keywords, but as "thematic sets" or "packages" of aligned ideas and assumptions.

Based on the **Media Frames Corpus (MFC)** paper by **Card et al. (2015)**, which draws on the framing dimensions defined by **Boydstun et al. (2014)**, here is the corrected table with the specific descriptions intended by the authors:

| Frame | Description (Card et al., 2015 / Boydstun et al., 2014) |
| :--- | :--- |
| **Economic** | The costs, benefits, or other financial implications of the issue. |
| **Capacity and Resources** | The availability of physical, human, or financial resources, and the capacity of current systems. |
| **Morality** | Any perspective or policy objective/action compelled by religious doctrine, ethics, or social responsibility. |
| **Fairness and Equality** | The balance or distribution of rights, responsibilities, and resources; equality or inequality in the application of laws. |
| **Legality, Constitutionality and Jurisprudence** | The rights, freedoms, and authority of individuals, corporations, and government (focuses on the constraints/freedoms granted via the Constitution or laws). |
| **Policy Prescription and Evaluation** | Discussion of specific policies aimed at addressing problems, and the evaluation of whether certain policies will work or are working. |
| **Crime and Punishment** | The effectiveness and implications of laws and their enforcement (includes breaking laws, loopholes, sentencing, and punishment). |
| **Security and Defense** | Threats to the welfare of the individual, community, or nation (includes protection from not-yet-manifested threats). |
| **Health and Safety** | Health care access/effectiveness, sanitation, disease, mental health, and public safety (e.g., infrastructure safety). |
| **Quality of Life** | Threats and opportunities for the individual's wealth, happiness, and well-being (effects on mobility, ease of routine, community life). |
| **Cultural Identity** | The traditions, customs, or values of a social group in relation to a specific policy issue. |
| **Public Opinion** | Attitudes and opinions of the general public, including polling and demographics. |
| **Political** | Considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters (explicit mentions of partisan maneuvering). |
| **External Regulation and Reputation** | The international reputation or foreign policy of the U.S. (relations with other nations, trade agreements). |
| **Other** | Any coherent group of frames not covered by the above categories. |

Existing methods to automate frame-detection suffer from several limitations:

1) GenAI shortcomings. Several papers use GenAI prompt-engineering, coercing generative models into frame classification tasks. This risks the hallucination of non-existing frames, lowers the interpretability of final models (increasing the black-box barrier), and utilizes computationally expensive solutions which are difficult to scale to large-N data.

2) Context length. Past solutions are often designed for sentence-level classification, whereas frames develop through complex interplays across hundreds of words of text, and reference past signals from earlier in the same input. A longer context with appropriate attention handling is essential to pick-up these nuanced interactions. 

3) Training data. Reliance on machine-generated classification data, rather than gold-standard human annotations, which can bake-in biases, poses another serious risk. A lack of abundant human annotated data for frame detection remains a serious challenge for work in this area, and the encoding of human-backed reasoning into model weights requires careful correction and deliberate fine-tuning.

I sought to develop a lightweight transformer-based model which addresses these challenges, and would be useful for applied NLP research. The final model uses multi-label classification to detect the presence of each of the 15-frames in input text up to ~1600 words. It outperforms past GenAI approaches in classification accuracy on the Media Frames Corpus, while being 47 times smaller in model size (compared to mm-framing's Mistral-7B). 

I developed the project in its associated [repo](https://github.com/Ry-Rousseau/frame-delta) and uploaded my final model to HuggingFace. The current version (as of writing) may be iterated upon in future, as I optimize further on the available data and test new architectures, which will reflect in future model updates. 

---

## 2. Training Data

### 2.1 Gold Data: Human Annotations

The two best datasets for this task are: the 2015 Media Frames Corpus (MFC) and the 2023 SemEval Task 3 (sub-task 2). Accessibility for the MFC has been greatly reduced (as Lexis Nexis has paywalled document access), leaving about 21% of the original 14,481 articles available to me (via Lexis Uni). The SemEval source is open-access, after messaging the task organizers, and provides 516 additional articles. Together, these generate a gold-standard dataset of 2,740 articles, combining labels from 19 trained annotators. 

#### Media Frames Corpus (MFC)

The MFC data records articles within the topics of immigration, same-sex marriage and smoking, spanning 1990-2012. Of the 12 US Newspaper originally encoded, I train on the New York Times articles that are presently accessible through the Lexis Uni portal.  

| Topic | Original MFC | NYT Available | Retrieved | Coverage |
|-------|-------------|---------------|-----------|----------|
| Immigration | 5,500 | 1,443 | 1,370 | 24.9% |
| Smoking | 5,074 | 822 | 821 | 16.2% |
| Same-Sex Marriage | 8,407 | 1,764 | 1,740 | 20.7% |
| **Total** | **18,981** | **4,029** | **3,931** | **20.7%** |

**Annotation Aggregation Strategy:**

Each MFC article has 2+ annotators with span-level annotations. I tested three aggregation methods, and opted to use the union of annotations (recording a frame if either annotator records one) to maximize the available signal. 

| Strategy | Mean Frames/Article | 
|----------|---------------------|
| Per annotator | 3.18 |
| **Union** | **4.28** |
| Intersection | 2.14 |

#### SemEval 2023 Task 3 Subtask 2

Originally designed for competition model development, the SemEval data can be coerced to produce article-level annotations from trained practitioners. It covers articles collected between 2020-2022 on topics ranging from the Ukraine-Russia war, COVID-19, Migration, Abortion and Climate Change. 

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

Given the insufficient number of gold training examples for a useful and accurate classifier, I rely on secondary machine-labelled data to build out the training run.

**Source:** `copenlu/mm-framing`

This dataset from Arora et al. 2025 contains ~478,000 news articles with frame labels generated by an LLM - Mistral-7B-Instruct - using prompt engineering. It was cleaned to remove short length articles (<100 words) and non-standard frames. 378,000 examples were retained, spanning the period May 2023 to April 2024, covering 28 US based news agencies, spanning the political spectrum. Data was hydrated with the python library `newsplease` from base urls.

**Label Generation Method:**
- Model: Mistral-7B-Instruct-v0.3
- Inference: vLLM with temperature=0.2, max_tokens=4000
- Process: Prompted to read text and classify into the 15-frame taxonomy (error-prone)

The Mistral model performed at a mediocre level on the MFC test data, with a Micro F1-score of 0.50. 

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

I used the silver trained model as a baseline for the gold training run, to balance the generalizability of the model with the validity of the human annotated data. 

**Dataset Variables Used:**
- `title` - Article headline
- `gpt_topic` - Consolidated topic (19 categories)
- `text_generic_frame` - List of frame labels
- `maintext` - Full article body (from joined `newsarticles` table)

Finally, I consolidated additional metadata provided by the Mistral model, transforming an unstructured topic classification field into 19 topic categories based on empirical similarity. This enabled the encoding of the 350,000+ examples into discrete topics ranging from politics, sport and environment. 

---

## 3. Model Architecture

### 3.1 Topic Classifier: RoBERTa with Truncation

I found that encoder transformer models (BERT and Longformer) perform better at frame classification after being provided with relevant meta-data. Without any model-level changes, adding the article topic to the beginning of the input tensor bakes in a sizable improvement in test accuracy. Topic injection activates internal weights designed for domain-specific reasoning, generating a +2.7% gain in Micro F1 performance, and represents an easy feature-level improvement. Certain frames experienced greater performance gains over others, largely based on training data availability, but all saw increased performance. 

To leverage this finding, I trained a lightweight roBERTa model to classify unseen text into one of the 19 empirical topics derived from the silver training data. Comparing performance to alternatives (base BERT and Distill BERT), RoBERTa achieved the highest validation accuracy at 76% on 64,000 examples, which was sufficient for the task of assisting the downstream framing classifier.

**Head+Tail truncation strategy**
- Head: First 320 tokens (captures title and introduction)
- Tail: Last 190 tokens (captures conclusion)
- Total: 510 tokens

**Topic Injection Implementation:**
```
TOPIC:{topic}
{title}
{article_text}
```
Topic injection implements a 'soft' mixture of experts approach, giving prior context without the overhead of running multiple 'domain expert' models. 

### 3.2 Longformer for Full Document Classification

To produce the framing classifier, I evaluate a range of models, combining feature-level and architectural findings (see appendix section). I settled on the Longformer, a variant of the RoBERTa model fundamentally designed for long-context inputs (up to 4096 tokens). This model implements sparse attention, which enables longer input while keeping computation comparatively low, and can be run locally with a mid-range GPU. In addition, the Longformer's ability to set specific tokens for global attention directly enables a mixture of experts design with the structured topic injections, allowing the topic token to speak with the rest of the input text during evaluation. This balances the full contextual understanding of smaller BERT models while enabling the necessary input lengths for measuring most social media posts and news articles.

**Key Configuration:**
- Global attention on [CLS] token (position 0)
- Global attention on first topic token (position 3)
- Limit input tensor to 2048 tokens for increase efficiency without loss to most use-cases (handling up to ~1500 words)

---

## 4. Silver Training

I trained `allenai/longformer-base-4096` on the silver dataset for 4 epochs over 72 hours using an NVIDIA A40 (48GB VRAM), holding out 10% for validation and model selection. To address class imbalance, I used binary cross-entropy loss weighted by the inverse frequency of frames in the Mistral-generated labels. This encouraged the model to attend equally to all frames rather than over-predicting dominant categories. I further mitigated imbalance through post-training threshold optimization on the classification layer.

I stopped training at 4 epochs despite observing that validation scores would continue improving marginally, as I wanted to avoid overfitting on the less reliable machine-generated labels.

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Gradient Accumulation | 2 (effective 32) |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Epochs | 4 |

---

## 5. Gold Training

I fine-tuned the silver model on the gold training set from MFC and SemEval. This stage aimed to shift reasoning sufficiently towards the annotators' decision making processes. I relied specifically on focal loss, which is a variation of cross entropy loss designed to focus training on more difficult examples. A down weight penalty is applied to assist the model in ignoring things it already knows well and focus on the examples it is struggling with. I searched over 4 parameters for down weighting settings (settling on a Gamma value of 2) for best performance. In addition, I implemented a learning rate scheduler to make best use of the more limited size of the gold training set, setting LR high early on, waiting for validation plateau, and then slowly decreasing to converge on a minima.

| Parameter | Value |
|-----------|-------|
| Starting Checkpoint | Phase 1 best model (epoch 4) |
| Loss Function | Focal Loss (gamma=2.0) |
| Learning Rate | 2e-5 |
| Epochs | 10 |
| Batch Size | 2 |
| Gradient Accumulation | 8 (effective 16) |
| Validation | 90/10 train/test split |

---

## Evaluation Metrics

**Post-threshold optimization performance**

| Metric | Score |
|--------|-------|
| **Weighted F1** | **0.6863** |
| Micro F1 | 0.6846 |
| Macro F1 | 0.645 |

**Per-Class Report (excluding "Other"):**

| Frame | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Economic | 0.78 | 0.70 | 0.74 | 87 |
| Capacity & Resources | 0.44 | 0.48 | 0.46 | 25 |
| Morality | 0.53 | 0.64 | 0.58 | 61 |
| Fairness & Equality | 0.60 | 0.52 | 0.56 | 63 |
| Legality | 0.83 | 0.79 | 0.81 | 164 |
| Policy Prescription | 0.58 | 0.85 | 0.69 | 110 |
| Crime & Punishment | 0.63 | 0.86 | 0.73 | 87 |
| Security & Defense | 0.49 | 0.72 | 0.58 | 29 |
| Health & Safety | 0.61 | 0.77 | 0.68 | 69 |
| Quality of Life | 0.57 | 0.74 | 0.64 | 96 |
| Cultural Identity | 0.46 | 0.74 | 0.57 | 72 |
| Public Opinion | 0.51 | 0.77 | 0.61 | 81 |
| Political | 0.70 | 0.94 | 0.81 | 134 |
| External Regulation | 0.92 | 0.41 | 0.57 | 29 |

---

## Appendix

### Future Directions

- Improved Lexis-Nexis access to increase scope of gold training data
- Calibration analysis of predicted probabilities, using classification layer to generate continuous measure of frames (e.g. 60% economic vs. 20% policy prescription)
- Semi-supervised refinement using gold model to clean silver data
- Ensemble of Longformer with domain experts to eke out greater domain performance
- Temporal analysis of how frames shift on same topics over time
- Implement knowledge distillation from longformer into lighter weight models, measuring performance trade-off

### Tips and Tricks

Running experiments on smaller models usually scaled very well to larger models in the same architecture. This saved computation and made it easier to predict improvements without running on the full training set (and renting GPUs to do so).

I used RunPod's A40 GPU for the full silver training run, it took approximately 72 hours. I finished with the gold run on my local RTX 4070Ti with 16GB VRAM, optimizing with gradient accumulation and smaller batch size, completing the focal loss grid search and final train over 12 hours.

Using weights and biases to track training runs proved immensely helpful and greatly eased evaluation between experiments.

### Experiments

As a glimpse into underlying reasoning, in certain edge cases, the topic classification model would perform poorly, such as in classifying stories about wildlife hunting as 'crime'. However, on aggregate, these inaccuracies were not found to significantly disrupt downstream model performance. Alternative topic classifiers could be used to optimize this stage and provide more robust classification, but might not affect end performance strongly. 

For the longformer classifier, ideally, a full attention long-context model would be used, as the sliding window of local attention in the sparse attention architecture is non-ideal. However, these are very expensive to run, and grow computation exponentially. This represents a strong challenge with masked language modelling architectures.

Another option I considered was stringing together multiple BERT models in a hierarchical chain. I thought this was excessively complex and failed to integrate the dynamic attention modelling architecture gained by the Longformer model, but it may prove useful if paired with a smart chunking method, breaking a long input into relevant sub-inputs for each BERT.

Finally, owing to a key finding in the winning SemEval 2023 Task 3 sub-task 2 paper, I attempted to run masked language modelling pre-training on the base longformer. This would have taken >48 hours, so I opted not to continue, but has been shown to improve performance on related tasks by 3-4%.

This project remains in its initial version, and will be iterated on in future. I aim to make the model pipeline and predictions more accessible.

Any suggestions for improvements are always welcome! And if anyone knows of how to access the full MFC without the high cost of a full Lexis Nexis subscription, please let me know. 

