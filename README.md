# Cross-Dataset Generalization and Failure Detection in Brain MRI Classification

An exploratory deep learning study of cross-dataset generalization, class-specific failure patterns, and uncertainty-based failure detection in four-class brain MRI classification.

## Abstract

Deep learning models can achieve high classification performance on brain MRI datasets when training and evaluation data originate from the same source. However, performance may degrade when models are evaluated on independently collected data because medical imaging datasets can differ in acquisition conditions, preprocessing, image characteristics, and population composition [1–3].

This exploratory study investigates cross-dataset generalization between two independent brain MRI datasets, BDNeuro-MRI and PMRAM, for four-class classification of glioma, meningioma, pituitary tumor, and no tumor. ImageNet-pretrained EfficientNet-B0 was used for the main bidirectional cross-dataset analysis, while ResNet18 was used in supporting and control experiments. Model failures were further examined using Maximum Softmax Response (MSR), Maximum Logit Score (MLS), and MC-Dropout.

EfficientNet-B0 achieved 97.86% internal accuracy on BDNeuro and 95.96% when evaluated externally on PMRAM. In the reverse direction, the model achieved 96.23% internal accuracy on PMRAM but only 66.29% on BDNeuro. The degradation was strongly class-dependent, with meningioma recall falling to 16.85% in the PMRAM → BDNeuro direction.

Failure-detection experiments showed that MSR, MLS, and MC-Dropout could identify substantial proportions of incorrect external predictions, although error detection involved a trade-off with prediction coverage. Additional failure analysis identified differences in image characteristics associated with classification errors, while a matched training-size control experiment showed that training-set size alone did not explain the observed cross-dataset degradation.

Overall, the results highlight the importance of external validation, class-specific failure analysis, and uncertainty-aware evaluation when assessing the robustness of medical image classifiers.

## 1. Introduction

Deep learning has become widely used in medical image analysis, including classification, detection, and segmentation. However, high performance on an internal test set does not necessarily establish that a model will maintain the same performance when evaluated on independently collected data. External validation is therefore important for assessing the generalizability of medical imaging models [1].

One important challenge is dataset or domain shift. Medical images can vary because of acquisition parameters, imaging systems, preprocessing pipelines, institutional practices, patient populations, and other technical or biological factors [2,3]. These differences can change the input distribution encountered by a model and potentially reduce performance on data from a new source.

Brain tumor MRI classification provides a useful setting for investigating this problem. Although high classification performance is frequently reported within individual datasets, strong internal performance does not by itself demonstrate robustness to independent MRI datasets. Recent research has therefore begun incorporating cross-domain evaluation into brain tumor classification [7].

Prediction reliability is another important consideration. Maximum softmax probability can provide a simple signal for detecting incorrect or out-of-distribution predictions [4]. Maximum-logit-based scoring has also been investigated for out-of-distribution detection [5], while MC-Dropout provides a practical approach for estimating predictive uncertainty through repeated stochastic forward passes [6].

Motivated by these issues, this study investigates four research questions:

**RQ1:** How does brain MRI classification performance change when models are evaluated on an independent dataset?

**RQ2:** Is cross-dataset generalization symmetric, or does performance depend on the direction of transfer?

**RQ3:** Can MSR, MLS, and MC-Dropout identify incorrect predictions under external dataset shift?

**RQ4:** Can the observed directional generalization gap be explained by training-set size alone?

## 2. Related Work

### External Validation and Dataset Shift

External validation is important for estimating whether a medical imaging model can maintain performance beyond the dataset used during development. Systematic reviews have highlighted limitations associated with relying primarily on internal evaluation [1].

Dataset shift is particularly relevant in medical imaging because image formation depends on acquisition processes as well as biological and institutional factors. Variation in physical imaging parameters can produce measurable domain shift [2], while broader healthcare research has shown that naturally occurring data shifts can affect machine-learning performance after development [3].

These observations motivate direct cross-dataset evaluation rather than relying exclusively on internal test accuracy.

### Cross-Dataset Brain MRI Classification

Brain tumor classification studies commonly evaluate deep-learning models on public MRI datasets. However, high within-dataset accuracy does not necessarily demonstrate robustness to independent sources.

Recent work has investigated cross-domain brain tumor classification to examine performance on unseen MRI datasets [7]. The present study extends this perspective by evaluating transfer in both directions:

- **BDNeuro → PMRAM**
- **PMRAM → BDNeuro**

This bidirectional design makes it possible to examine whether cross-dataset generalization depends on the direction of transfer.

### Failure Detection and Uncertainty

Classification accuracy indicates how frequently a model is correct but does not indicate whether incorrect predictions can be recognized.

Maximum Softmax Response provides a simple confidence-based signal for identifying potentially incorrect predictions [4]. Maximum Logit Score provides another score derived directly from classifier outputs [5]. MC-Dropout instead estimates predictive uncertainty through repeated stochastic inference with dropout enabled [6].

In this study, these three approaches are evaluated specifically for detecting classification failures during external cross-dataset evaluation.

## 3. Datasets and Experimental Design

### Datasets

Two independent brain MRI datasets were used.

**BDNeuro-MRI** contains 5,941 preprocessed T1-weighted contrast-enhanced MRI images across four classes [8]:

- Glioma
- Meningioma
- No tumor
- Pituitary tumor

The predefined split used in this study contained:

| Split | Images |
|---|---:|
| Training | 4,160 |
| Validation | 892 |
| Test | 889 |
| **Total** | **5,941** |

**PMRAM** contains the same four diagnostic categories [9]. The original dataset release contains 1,600 raw MRI images, while the prepared experimental dataset used in this project contained **1,410 images**.

A stratified split with `random_state=42` produced:

| Split | Images |
|---|---:|
| Training | 987 |
| Validation | 211 |
| Test | 212 |
| **Experimental dataset** | **1,410** |

When PMRAM served as the external target dataset, all 1,410 images in the prepared experimental set were used for external evaluation.

### Experimental Design

Both datasets were harmonized to the same four labels:

- Glioma
- Meningioma
- No tumor
- Pituitary tumor

The main cross-dataset evaluation was performed in both directions:

**Direction 1:** BDNeuro → PMRAM

**Direction 2:** PMRAM → BDNeuro

Models were first evaluated on the internal test data associated with their training source and then on the independent external dataset.

## 4. Methodology

### Model Architectures

ImageNet-pretrained ResNet18 [10] and EfficientNet-B0 [11] were adapted for four-class classification.

EfficientNet-B0 was used for the main bidirectional classification and failure-analysis results reported in this README. ResNet18 was additionally used in supporting experiments, including the matched training-size control.

### Training Protocol

Images were resized to **224 × 224 pixels** and normalized using ImageNet statistics.

Training augmentation included:

- Random horizontal flip (`p = 0.5`)
- Random rotation (`±10°`)

The main training settings were:

| Parameter | Setting |
|---|---|
| Pretraining | ImageNet |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | Cross-Entropy Loss |
| Batch size | 32 |
| Epochs | 5 |
| Number of classes | 4 |
| Computing device | CPU |

### Failure Detection

Three failure-detection approaches were investigated:

- **MSR:** confidence derived from the maximum softmax probability [4]
- **MLS:** score derived from the maximum pre-softmax logit [5]
- **MC-Dropout:** predictive uncertainty estimated using 20 stochastic forward passes [6]

Failure-detection thresholds were selected **only from the source validation set** using Youden's J statistic and were then applied unchanged to the external target dataset.

Performance was evaluated using:

- AUROC-F
- Error Detection Rate
- Coverage
- Silent Failures

Here, a failure refers to an incorrect classification.

## 5. Results

### 5.1 Cross-Dataset Classification

EfficientNet-B0 achieved strong internal performance on both datasets, but external performance differed substantially depending on the transfer direction.

| Training Dataset | Evaluation Dataset | Evaluation Type | Accuracy |
|---|---|---|---:|
| BDNeuro | BDNeuro | Internal | 97.86% |
| BDNeuro | PMRAM | External | 95.96% |
| PMRAM | PMRAM | Internal | 96.23% |
| PMRAM | BDNeuro | External | 66.29% |

The BDNeuro → PMRAM direction maintained high external performance.

In contrast, PMRAM → BDNeuro showed substantial degradation despite strong internal PMRAM performance, demonstrating a pronounced directional asymmetry in cross-dataset generalization.

The degradation was also class-dependent:

| Class | PMRAM → BDNeuro Recall |
|---|---:|
| Glioma | 83.57% |
| Meningioma | 16.85% |
| No Tumor | 98.22% |
| Pituitary | 73.06% |

Meningioma showed the largest degradation, motivating additional failure analysis.

### 5.2 Failure Detection

MSR, MLS, and MC-Dropout were evaluated for detecting incorrect external predictions.

| Direction | Method | External AUROC-F | Error Detection Rate | Coverage | Silent Failures |
|---|---|---:|---:|---:|---:|
| BDNeuro → PMRAM | MSR | 0.9341 | 0.9123 | 0.7631 | 5 |
| BDNeuro → PMRAM | MLS | 0.9020 | 0.9649 | 0.6433 | 2 |
| BDNeuro → PMRAM | MC-Dropout | 0.9365 | 0.8511 | 0.8291 | 14 |
| PMRAM → BDNeuro | MSR | 0.8536 | 0.8372 | 0.5622 | 326 |
| PMRAM → BDNeuro | MLS | 0.8576 | 0.9096 | 0.4809 | 181 |
| PMRAM → BDNeuro | MC-Dropout | 0.8595 | 0.8507 | 0.5482 | 300 |

All three approaches identified substantial proportions of incorrect external predictions. However, higher error detection could be accompanied by lower prediction coverage, demonstrating a trade-off between detecting failures and retaining predictions.

For BDNeuro → PMRAM, the MC-Dropout results were obtained from a separately trained EfficientNet-B0 model with classifier dropout enabled. Therefore, direct method-to-method comparisons with MSR and MLS in this direction should be interpreted cautiously.

### 5.3 Failure and Dataset-Shift Analysis

Meningioma showed the largest class-specific degradation in the difficult PMRAM → BDNeuro direction.

The two datasets also showed differences in observable image characteristics. For meningioma images, mean brightness and contrast were:

| Dataset | Mean Brightness | Mean Contrast |
|---|---:|---:|
| BDNeuro | 44.60 | 46.15 |
| PMRAM | 58.62 | 53.60 |

Within BDNeuro meningioma samples, incorrectly classified images had lower average contrast than correctly classified images. A Mann–Whitney comparison showed a statistically significant contrast difference (`p < 0.001`), whereas the brightness difference was not statistically significant (`p = 0.1125`).

Controlled contrast modifications further showed that increasing contrast could improve meningioma recall in some settings, but this was accompanied by deterioration in other classes, particularly pituitary.

These observations indicate an association between image characteristics and classification behavior, but they do not establish contrast or brightness as the causal explanation for the cross-dataset performance gap.

### 5.4 Matched Training-Size Control

BDNeuro originally contained 4,160 training images, compared with 987 PMRAM training images. To examine whether training-set size alone explained the directional generalization gap, the BDNeuro training set was reduced to exactly **987 images** using stratified sampling.

A pretrained ResNet18 trained on the matched BDNeuro subset achieved:

| Evaluation | Accuracy | Macro F1 |
|---|---:|---:|
| BDNeuro Internal Test | 90.66% | 90.47% |
| PMRAM External | 84.47% | 83.49% |

Cross-dataset performance remained lower than internal performance even after controlling for the number of training samples.

Meningioma also remained the weakest external class, with a recall of **57.80%**.

Therefore, training-set size alone does not explain the observed cross-dataset performance degradation.

## 6. Discussion

The experiments demonstrate that strong internal brain MRI classification performance does not necessarily translate into equivalent performance on an independent dataset.

The most important observation was the directional asymmetry between the two external evaluations. EfficientNet-B0 trained on BDNeuro maintained high performance on PMRAM, whereas the model trained on PMRAM showed substantial degradation on BDNeuro.

The degradation was also strongly class-dependent. In the difficult PMRAM → BDNeuro direction, meningioma recall fell to 16.85%, while no-tumor recall remained above 98%. This indicates that dataset shift did not affect all diagnostic categories uniformly.

Failure analysis showed associations between image characteristics and classification errors, particularly for meningioma. However, controlled contrast modification did not consistently improve overall performance and introduced trade-offs between classes. The observed shift therefore appears more complex than a single brightness or contrast difference.

MSR, MLS, and MC-Dropout also provided useful signals for identifying incorrect external predictions. Their different combinations of error detection and coverage illustrate why uncertainty-aware evaluation can provide information beyond classification accuracy alone.

Finally, the matched-size experiment showed that reducing the BDNeuro training set to the same size as PMRAM did not eliminate external degradation. Dataset size may contribute to model performance, but it is not sufficient by itself to explain the observed cross-dataset behavior.

Overall, these results emphasize the importance of independent external evaluation, class-specific failure analysis, and prediction-reliability assessment when studying the robustness of medical image classifiers.

## 7. Limitations

This study is exploratory and has several limitations.

First, only two brain MRI datasets were evaluated, so the observed generalization patterns should not be assumed to represent other datasets, scanners, institutions, or patient populations.

Second, the experiments use two-dimensional MRI images rather than complete volumetric examinations and should therefore be interpreted as image-classification experiments rather than patient-level clinical validation.

Third, the datasets may differ in acquisition procedures, preprocessing, image composition, and other unmeasured factors. The current analysis does not isolate all potential sources of domain shift.

Fourth, the brightness and contrast analyses identify associations with model behavior but do not establish causal mechanisms.

Fifth, the matched training-size experiment controls the number of training samples but does not equalize other dataset characteristics such as class distributions or acquisition diversity.

Finally, the experiments use a limited number of architectures and training settings. Additional datasets, repeated experiments, and institution-level external validation would be needed to determine whether the observed patterns generalize more broadly.

## 8. Conclusion and Future Work

This exploratory study investigated cross-dataset generalization and failure detection in four-class brain MRI classification using BDNeuro and PMRAM.

The results revealed substantial directional asymmetry. EfficientNet-B0 generalized strongly from BDNeuro to PMRAM but showed considerable degradation when trained on PMRAM and evaluated on BDNeuro. The degradation was also class-specific, with meningioma showing the largest performance reduction.

MSR, MLS, and MC-Dropout provided useful signals for identifying incorrect external predictions, while additional analysis showed associations between image characteristics and class-specific failures. Matching the training-set sizes did not eliminate cross-dataset degradation, indicating that training-set size alone is insufficient to explain the observed generalization gap.

Future work should extend the analysis to additional independently collected MRI datasets, investigate acquisition- and source-level differences, evaluate repeated training runs, and explore domain-generalization or adaptation strategies.

Overall, this study highlights the importance of moving beyond internal accuracy when evaluating medical image classifiers and provides an exploratory framework combining external validation, failure analysis, and uncertainty-aware evaluation.

## 9. References

[1] A. Aggarwal et al., “Diagnostic Accuracy of Deep Learning in Medical Imaging: A Systematic Review and Meta-Analysis,” *npj Digital Medicine*, vol. 4, 65, 2021. https://doi.org/10.1038/s41746-021-00438-z

[2] O. Kilim et al., “Physical Imaging Parameter Variation Drives Domain Shift,” *Scientific Reports*, vol. 12, 21302, 2022. https://doi.org/10.1038/s41598-022-23990-4

[3] A. Zhang et al., “Shifting Machine Learning for Healthcare from Development to Deployment and from Models to Data,” *Nature Biomedical Engineering*, vol. 6, pp. 1330–1345, 2022. https://doi.org/10.1038/s41551-022-00898-y

[4] D. Hendrycks and K. Gimpel, “A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks,” *International Conference on Learning Representations (ICLR)*, 2017.

[5] D. Hendrycks, S. Basart, M. Mazeika, A. Zou, J. Kwon, M. Mostajabi, J. Steinhardt, and D. Song, “Scaling Out-of-Distribution Detection for Real-World Settings,” *Proceedings of the 39th International Conference on Machine Learning (ICML)*, vol. 162, pp. 8759–8773, 2022.

[6] Y. Gal and Z. Ghahramani, “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning,” *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, vol. 48, pp. 1050–1059, 2016.

[7] Y. Tian, “Trade-Off Analysis of Classical Machine Learning and Deep Learning Models for Robust Brain Tumor Detection: Benchmark Study,” *JMIR AI*, vol. 4, e76344, 2025. https://doi.org/10.2196/76344

[8] M. I. K. Hira et al., “BDNeuro-MRI: A Bangladeshi Clinical Brain Tumor MRI Dataset for Four-Class Deep Learning Classification,” *Mendeley Data*, Version 8, 2026. https://doi.org/10.17632/zwr4ntf94j.8

[9] P. M. S. Mannan, M. Chowdhury, R. Rahman, A. U. Tamim, and M. M. Rahman, “PMRAM: Bangladeshi Brain Cancer - MRI Dataset,” *Mendeley Data*, Version 1, 2024. https://doi.org/10.17632/m7w55sw88b.1

[10] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778, 2016. https://doi.org/10.1109/CVPR.2016.90

[11] M. Tan and Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” *Proceedings of the 36th International Conference on Machine Learning (ICML)*, vol. 97, pp. 6105–6114, 2019.
