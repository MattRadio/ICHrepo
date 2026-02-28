# ICH Detection on Head CT -- Classification with GradCAM Interpretability

Binary intracranial hemorrhage detection from axial head CT slices using a fine-tuned ResNet18 with GradCAM-based clinical evaluation of model attention patterns.

## Overview

This project builds a hemorrhage detection classifier from scratch using the [CT-ICH dataset](https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images) (82 patients, 2501 slices) and evaluates model behavior through the lens of clinical radiology. The goal is not to build a production-ready tool -- it is to demonstrate end-to-end understanding of the classification pipeline and the ability to critically evaluate model output using clinical domain knowledge.

## Why This Matters Clinically

Intracranial hemorrhage detection is one of the most impactful use cases for radiology AI. On overnight call, a single radiologist may be responsible for hundreds of studies. AI-assisted triage -- flagging likely hemorrhage cases for immediate review -- can meaningfully reduce time to diagnosis for emergent findings. Tools like Viz.ai and Aidoc use classification models similar to this one (at much larger scale) to page stroke teams and prioritize worklists.

Understanding how these models work, where they fail, and what they actually "see" is essential for anyone evaluating or implementing AI in clinical practice.

## Architecture and Methods

| Component | Detail |
|-----------|--------|
| Model | ResNet18 (pretrained on ImageNet, fine-tuned) |
| Dataset | CT-ICH: 82 patients, 2501 axial slices (brain window) |
| Split | Patient-level: 65 train / 17 validation (no data leakage) |
| Input | 224 x 224 grayscale JPG (brain window), converted to 3-channel |
| Loss | BCEWithLogitsLoss with 7.5x positive class weight |
| Optimizer | Adam (lr = 1e-4) |
| Epochs | 10 |
| Augmentation | Random horizontal flip, random rotation (10 degrees) |
| Interpretability | GradCAM on final convolutional layer (layer4) |

### Key Design Decisions

**Patient-level split:** Slices from the same CT scan are near-identical. Random slice-level splitting would leak information between train and validation, inflating metrics. All slices from a given patient are in either train or val, never both.

**Class imbalance handling:** Only 12.7% of slices contain hemorrhage. Without correction, the model could achieve 87% accuracy by predicting "no hemorrhage" on every slice. The 7.5x positive weight penalizes missed hemorrhages proportionally, forcing the model to learn hemorrhage features rather than defaulting to the majority class.

**Horizontal flip as augmentation:** Unlike tasks where laterality matters (e.g., left MCA stroke localization), binary hemorrhage detection is side-agnostic. A right-sided subdural looks identical to a left-sided subdural from a detection standpoint. Flipping effectively doubles the training data without creating anatomically implausible images.

**Transfer learning rationale:** ResNet18's early convolutional layers detect universal visual features -- edges, contrast boundaries, textures -- that transfer directly to CT interpretation. Only the final classification layer is replaced. This allows effective training on a small dataset (2501 slices) that would be insufficient to train a deep network from scratch.

## Results

| Metric | Value |
|--------|-------|
| Best AUC | 0.89 |
| Sensitivity | 0.79 |
| Specificity | 0.83 |

### What the Numbers Mean

An AUC of 0.89 indicates reasonable discriminative ability but is insufficient for clinical deployment as a standalone screening tool. Sensitivity of 0.79 means the model misses approximately 1 in 5 hemorrhages -- unacceptable for a triage system where a false negative could delay emergent intervention. Specificity of 0.83 means a 17% false alarm rate, which is tolerable in a triage context (an unnecessary stat read is far less harmful than a missed bleed).

In practice, the classification threshold (set at 0.5 here) would be tuned based on clinical priorities. Lowering it to 0.3 would increase sensitivity at the cost of more false positives -- a tradeoff that is a clinical decision, not an engineering one.

## GradCAM Analysis

GradCAM visualizes which regions of the image most influenced the model's prediction by generating a heatmap from the final convolutional layer.

### What the Model Gets Right

On true positive hemorrhage cases, the model attends broadly to regions containing hyperdense blood. Predictions on obvious hemorrhages are confident (0.96-1.00), and the heatmaps confirm the model is responding to the hemorrhage itself.

![GradCAM Results](results/gradcam_results.png)
*Top row: hemorrhage cases with GradCAM overlay. Bottom row: normal cases. Note model attention on skull base and mastoid air cells in normal cases -- clinically irrelevant structures.*

### What the Model Gets Wrong

**On normal cases:** The model frequently attends to the skull base, temporal bone, and mastoid air cells -- structures that are completely irrelevant to hemorrhage detection. This suggests the model may be learning a shortcut: "skull base anatomy visible = no hemorrhage" rather than truly understanding what hemorrhage looks like. This is a known failure mode called shortcut learning.

**On false negatives (missed hemorrhages):** The model can only detect hemorrhage when it directly visualizes hyperdense blood on a given slice. It has no ability to recognize secondary signs we use clinically to heighten suspicion:

- Pneumocephalus suggesting calvarial disruption
- Cerebral edema or effacement of sulci suggesting mass effect
- Fractures indicating trauma mechanism
- Midline shift from mass effect of adjacent hemorrhage

**False negative attention patterns:** GradCAM on missed hemorrhage cases reveals that the model is often attending to the correct anatomic region -- the heatmap overlaps with the hyperdense blood. The model "sees" the hemorrhage but fails to classify it as positive.

![GradCAM False Negatives](results/gradcam_false_negatives.png)
*Missed hemorrhage cases. Top row: GradCAM overlay showing model attention. Bottom row: raw CT for clinical comparison. The model attends to the correct region but fails to cross the classification threshold.*

This suggests the issue is not one of attention but of confidence calibration: the feature representations for these cases are insufficiently distinct from normal anatomy to cross the decision boundary. Possible explanations include subtle or small hemorrhages below the learned threshold, atypical morphology differing from predominant training examples, and adjacent hyperdense structures (calcified choroid plexus, dense bone) creating similar feature patterns. This is analogous to a trainee who identifies the correct anatomy but lacks the experience to distinguish pathology from normal variant -- and reinforces the need for threshold tuning in clinical deployment.

**Fundamental limitation -- per-slice classification:** This model evaluates each slice in isolation. When we read a head CT, we scroll through the full study and build a mental model: a vertex fracture prompts careful inspection for underlying epidural hemorrhage, even on slices where the hemorrhage itself is subtle. The model cannot perform this kind of cross-slice clinical reasoning. More advanced architectures (CNN + LSTM sequence models) address this by processing the full series of slices as ordered data, approximating how we actually interpret volumetric studies.

## Dataset

The CT-ICH dataset contains 82 patients with the following hemorrhage subtype distribution:

| Subtype | Count | Prevalence |
|---------|-------|------------|
| No hemorrhage | 2183 | 87.3% |
| Epidural | 173 | 6.9% |
| Fracture | 195 | 7.8% |
| Intraparenchymal | 73 | 2.9% |
| Subdural | 56 | 2.2% |
| Intraventricular | 24 | 1.0% |
| Subarachnoid | 18 | 0.7% |

Images are provided as pre-rendered JPGs in brain and bone windows. This project uses brain window images only, as brain window captures the majority of hemorrhage subtypes relevant to detection.

## Limitations

- **Small dataset:** 82 patients from a single source. Performance would likely degrade on data from different institutions, scanner vendors, or acquisition protocols (domain shift).
- **Pre-rendered windows:** Images are JPGs, not raw DICOMs. This prevents applying custom HU windowing (e.g., stacking brain, subdural, and bone windows as separate input channels -- a published technique that improves detection of convexity hemorrhages).
- **Binary classification only:** The model detects "any hemorrhage" without subtype differentiation. Clinical utility would improve with multi-label classification (identifying epidural vs. subdural vs. intraparenchymal, each with different management implications).
- **Per-slice inference:** No volumetric or sequential reasoning across the full CT study. Study-level predictions would require sequence modeling (LSTM/Transformer) on top of per-slice features.
- **Single-reader labels:** Ground truth annotations were not consensus-adjudicated, introducing label noise that limits the performance ceiling.

## Environment

- Python 3.10+
- PyTorch 2.x (CUDA)
- torchvision
- scikit-learn
- pytorch-grad-cam
- Google Colab (T4 GPU)

## References

- Flanders AE, et al. "Construction of a Machine Learning Dataset through Collaboration: The RSNA 2019 Brain CT Hemorrhage Challenge." Radiology: AI. 2020.
- He K, et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
- Selvaraju RR, et al. "Grad-CAM: Visual Explanations from Deep Networks." ICCV 2017.
- Ionescu B, et al. "Accurate and Efficient Intracranial Hemorrhage Detection and Subtype Classification in 3D CT Scans with Convolutional and Long Short-Term Memory Neural Networks." Sensors. 2020.

## Author

Matt -- PGY-3 Diagnostic Radiology Resident, UTHealth Houston. Building AI literacy to bridge clinical radiology and healthcare technology.
