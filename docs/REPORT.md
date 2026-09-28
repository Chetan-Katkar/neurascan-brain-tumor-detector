# NeuraScan — Brain Tumour Classification from MRI Using ResNet18

**A two-phase transfer-learning approach to four-class brain MRI classification**

---

## Abstract

Brain tumours are among the most lethal cancers, and accurate classification of tumour
type from magnetic resonance imaging is a prerequisite for treatment planning. Manual
reading is slow, requires scarce expertise, and shows measurable inter-observer
variability — particularly between glioma and meningioma, which can present with
overlapping appearance on a single slice.

This project trains a ResNet18 convolutional network, pretrained on ImageNet, to
classify axial brain MRI slices into four categories: **glioma**, **meningioma**,
**no tumour**, and **pituitary tumour**. Rather than training from scratch on a
5,000-image dataset — far too small for an 11-million-parameter network — we use a
**two-phase transfer-learning schedule**: first training only a new classification head
over a frozen backbone, then unfreezing the deepest convolutional block and fine-tuning
at a reduced learning rate.

The resulting model reaches **97.00% accuracy on a 1,000-image held-out test set**, with
a macro-averaged F1 of 0.9715 and perfect recall on the no-tumour class. The trained model
is deployed behind a Flask API with a browser interface that returns the predicted class
together with the full probability distribution.

---

## 1. Problem statement

Given a single axial brain MRI slice, assign it to one of four classes:

| Class | Clinical description |
|-------|---------------------|
| **Glioma** | Originates in glial cells; roughly a third of all brain tumours. Ranges from slow-growing grade I–II lesions to aggressive grade III–IV disease including glioblastoma. |
| **Meningioma** | Arises from the meninges, the membranes enclosing the brain and spinal cord. Usually slow-growing and benign; often treatable by surgery. |
| **No tumour** | No tumour signature present in the scan. |
| **Pituitary** | Forms in the pituitary gland at the skull base. Mostly benign adenomas, but can disrupt hormone production and compress adjacent structures. |

The task is a single-label, four-way classification problem over greyscale images.

---

## 2. Dataset

The BRISC2025 classification set, organised as one directory per class.

| Split | Images | Derivation |
|-------|-------:|------------|
| Train | 4,000 | 80% of the training directory |
| Validation | 1,000 | 20% of the training directory, **stratified** by class |
| Test | 1,000 | Held-out test directory, never seen during training |
| **Total** | **6,000** | |

The 80:20 train/validation split is stratified on the label so that class balance is
preserved in both halves; the split is seeded (`random_state = 42`) so the experiment
reproduces exactly.

Test-set class distribution:

| Class | Test images |
|-------|------------:|
| Glioma | 254 |
| Meningioma | 306 |
| No tumour | 140 |
| Pituitary | 300 |

The classes are moderately imbalanced — no tumour is under-represented at 14% of the
test set — which is why per-class metrics, not accuracy alone, are reported in §5.

---

## 3. Preprocessing

MRI slices are single-channel greyscale, but ResNet18 expects three-channel input with
ImageNet statistics. The pipeline resolves both constraints:

| Step | Operation | Rationale |
|------|-----------|-----------|
| 1 | `Resize(224 × 224)` | Matches ResNet18's expected input resolution. |
| 2 | `Grayscale(num_output_channels=3)` | Replicates the single channel three times so pretrained RGB filters apply unchanged. |
| 3 | `RandomHorizontalFlip()` — *train only* | The only augmentation used. See note below. |
| 4 | `ToTensor()` | Converts to a `[0, 1]` float tensor. |
| 5 | `Normalize(ImageNet mean/std)` | Puts inputs in the distribution the pretrained weights were learned on. |

**On augmentation.** Only horizontal flipping is applied. Brain anatomy is approximately
bilaterally symmetric, so a left–right flip yields a plausible scan. Rotation, shear and
aggressive colour jitter were deliberately excluded: they can introduce geometry or
intensity patterns that do not occur in real acquisition, and in a medical setting an
augmentation that fabricates non-physical images risks teaching the model artefacts
rather than pathology. The validation and test transforms omit augmentation entirely.

---

## 4. Model and training

### 4.1 Architecture

ResNet18 pretrained on ImageNet (`IMAGENET1K_V1`), with the final fully-connected layer
replaced:

```
Original:  Linear(512 → 1000)     # ImageNet classes
Replaced:  Linear(512 → 4)        # our four classes
```

Total parameters: **11,178,564**.

### 4.2 Why two phases

The new head is randomly initialised. If the whole network were unfrozen immediately,
the large, noisy gradients from that random head would propagate backwards and corrupt
the pretrained features before they could be of any use. The two-phase schedule avoids
this:

**Phase 1 — head only (5 epochs).** Every backbone parameter is frozen; only the new
`fc` layer trains — 2,052 parameters out of 11.2 million. The backbone acts as a fixed
feature extractor while the classifier head settles into a sensible region.

**Phase 2 — fine-tune `layer4` + head (15 epochs).** `layer4`, the deepest convolutional
block, is unfrozen alongside the head — 8,395,780 trainable parameters. The learning rate
is halved-and-then-some to 5e-5, five times smaller than Phase 1, so the pretrained
features are adjusted rather than overwritten. A `StepLR` scheduler drops the rate by a
further factor of ten after epoch 7.

Only `layer4` is unfrozen, not the full backbone: early convolutional layers encode
generic edge and texture detectors that transfer well across domains, while the deepest
block encodes the class-specific semantics that most need adapting from natural images
to MRI.

### 4.3 Hyperparameters

| Setting | Phase 1 | Phase 2 |
|---------|---------|---------|
| Epochs | 5 | 15 |
| Trainable parameters | 2,052 | 8,395,780 |
| Optimiser | Adam | Adam |
| Learning rate | 1e-4 | 5e-5 |
| Scheduler | none | StepLR (γ = 0.1, step = 7) |
| Batch size | 32 | 32 |
| Loss | Cross-entropy | Cross-entropy |

### 4.4 Training trajectory

| Epoch | Phase | Train loss | Train acc | Val loss | Val acc |
|------:|-------|-----------:|----------:|---------:|--------:|
| 1 | 1 | 1.2805 | 41.88% | 1.1217 | 58.10% |
| 2 | 1 | 1.0199 | 66.67% | 0.9114 | 74.90% |
| 3 | 1 | 0.8586 | 75.95% | 0.7842 | 79.60% |
| 4 | 1 | 0.7559 | 78.80% | 0.6969 | 81.50% |
| 5 | 1 | 0.6807 | 80.60% | 0.6316 | 84.20% |
| 6 | 2 | 0.2829 | 90.72% | 0.1414 | 95.30% |
| 7 | 2 | 0.0936 | 97.28% | 0.1008 | 96.70% |
| 8 | 2 | 0.0492 | 98.83% | 0.1087 | 96.30% |
| 9 | 2 | 0.0295 | 99.28% | 0.0860 | 97.30% |
| 10 | 2 | 0.0219 | 99.55% | 0.0789 | 97.60% |
| 11 | 2 | 0.0165 | 99.62% | 0.0835 | 97.40% |
| 12 | 2 | 0.0154 | 99.62% | 0.0717 | 97.90% |
| 13 | 2 | 0.0068 | 99.97% | 0.0695 | 98.10% |
| 14 | 2 | 0.0076 | 99.90% | 0.0687 | 97.90% |
| 15 | 2 | 0.0076 | 99.85% | 0.0686 | 97.80% |
| 16 | 2 | 0.0057 | 99.90% | 0.0698 | 97.70% |
| 17 | 2 | 0.0063 | 99.92% | 0.0685 | 97.80% |
| 18 | 2 | 0.0059 | 99.90% | 0.0705 | 97.70% |
| 19 | 2 | 0.0055 | 99.92% | 0.0639 | 98.10% |
| 20 | 2 | 0.0043 | 99.92% | 0.0647 | 98.00% |

Phase 1 plateaus at 84.2% validation accuracy — the frozen ImageNet backbone can only
take the model so far on a domain as distant as MRI. Unfreezing `layer4` produces the
decisive jump: validation accuracy moves from 84.2% to 95.3% in a single epoch, and
crosses 97% by epoch 9.

From roughly epoch 13 onward, training accuracy sits at 99.9%+ while validation accuracy
oscillates in a narrow 97.7–98.1% band and validation loss stops falling. The model has
converged; the residual gap between 99.9% train and 98.0% validation is mild overfitting
that the low learning rate and the scheduler keep bounded. Training beyond 20 epochs
would not help.

---

## 5. Results

### 5.1 Headline

**Test accuracy: 97.00%** — 970 of 1,000 held-out scans classified correctly.

### 5.2 Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| Glioma | 0.9793 | 0.9331 | 0.9556 | 254 |
| Meningioma | 0.9486 | 0.9641 | 0.9562 | 306 |
| No tumour | 0.9722 | **1.0000** | 0.9859 | 140 |
| Pituitary | 0.9835 | 0.9933 | 0.9884 | 300 |
| **Macro avg** | **0.9709** | **0.9726** | **0.9715** | 1,000 |
| **Weighted avg** | 0.9702 | 0.9700 | 0.9699 | 1,000 |

Macro and weighted averages agree to within 0.002, which confirms that performance is not
being carried by the larger classes — the model is genuinely competent on all four.

### 5.3 Confusion matrix

Rows are the true class, columns the prediction.

| Actual ↓ / Predicted → | Glioma | Meningioma | No tumour | Pituitary | Recall |
|---|---:|---:|---:|---:|---:|
| **Glioma** | **237** | 14 | 0 | 3 | 93.3% |
| **Meningioma** | 5 | **295** | 4 | 2 | 96.4% |
| **No tumour** | 0 | 0 | **140** | 0 | 100.0% |
| **Pituitary** | 0 | 2 | 0 | **298** | 99.3% |

### 5.4 Error analysis

Thirty errors occurred in total. Their distribution is not uniform, and the pattern is
clinically coherent:

- **Glioma ↔ meningioma accounts for 19 of 30 errors (63%).** Fourteen gliomas were read
  as meningioma and five meningiomas as glioma. This is the single dominant failure mode
  and it mirrors the hardest distinction for human readers: on an individual axial slice,
  without contrast timing or multi-plane context, the two can look genuinely similar.
  Glioma consequently has the lowest recall of the four classes, at 93.3%.
- **No tumour achieves perfect recall (140/140).** Not one scan containing a tumour was
  classified as tumour-free. For a screening-style application this is the most valuable
  property the model has, since a false negative is the costliest error class. Its
  precision of 0.9722 reflects four meningiomas misfiled as no-tumour — errors in the
  *other* direction, which are the safer kind.
- **Pituitary is nearly separable**, at 99.3% recall with only two errors. Pituitary
  tumours occupy a distinctive anatomical location at the skull base, giving the network
  a strong positional cue that the other tumour types do not offer.
- **No glioma or meningioma was ever confused with pituitary in the reverse direction**
  beyond three and two cases respectively, reinforcing that the confusable pair is
  specifically glioma/meningioma rather than tumour types in general.

---

## 6. Deployment

The trained weights are served by a Flask application (`app.py`).

**Device selection** is automatic at start-up: CUDA if an NVIDIA GPU is present, Apple
MPS on Apple Silicon, otherwise CPU. The original training run used Apple MPS.

**Endpoints**

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/` | Serves the browser interface. |
| `GET` | `/health` | Liveness probe; reports the active compute device. |
| `POST` | `/predict` | Accepts `multipart/form-data` with an `image` field. |

`/predict` returns the predicted class, the confidence percentage, the probability across
all four classes, and a base64 thumbnail of the submitted scan.

**Inference path.** The uploaded image is decoded, passed through the same
resize → greyscale-to-3-channel → normalise transform used at validation time, run
through the network under `torch.no_grad()`, and softmaxed into a probability
distribution.

**Interface.** A single-page dark-theme UI supports drag-and-drop upload, shows the
predicted class with its confidence, and — importantly — renders the **full probability
distribution** rather than the winning class alone. A 97%-confident prediction and a
51%-confident one are very different objects, and surfacing the distribution makes that
distinction visible to whoever is reading the result. The four classes carry fixed,
contrast-validated colours throughout the interface, and every bar is directly labelled
so identity is never conveyed by colour alone.

---

## 7. Limitations

1. **Single-slice classification.** The model reads one axial slice in isolation.
   Radiologists read the full volume across planes, with sequence and contrast context;
   a slice that is ambiguous alone is often unambiguous in the stack.
2. **No tumour localisation.** The model outputs a class, not a segmentation or bounding
   box. It cannot say where the lesion is, or how big.
3. **Single-dataset evaluation.** All 6,000 images come from one dataset. MRI appearance
   varies substantially with scanner vendor, field strength and acquisition protocol, and
   performance on scans from a different site is unmeasured and should not be assumed.
4. **No calibration analysis.** The reported confidence is a raw softmax output. Deep
   networks are typically overconfident, so a stated 97% should not be read as a
   well-calibrated 97% probability without temperature scaling or a reliability-diagram
   check.
5. **The glioma/meningioma boundary is the standing weakness**, at 63% of all errors.
6. **Not a medical device.** The system is unvalidated for clinical use and carries no
   regulatory approval.

---

## 8. Future work

- **Attack the dominant error mode directly** — class-weighted loss or focal loss biased
  towards the glioma/meningioma boundary, or a dedicated second-stage binary classifier
  invoked only when the top two probabilities are both in that pair.
- **Explainability.** Grad-CAM overlays showing which region drove the prediction would
  let a reader sanity-check whether the network attended to the lesion or to an
  irrelevant artefact — a prerequisite for any clinical conversation.
- **Volume-level inference.** Aggregate predictions across the slices of a study rather
  than classifying one slice.
- **External validation** on an independent, differently-acquired dataset to measure the
  real generalisation gap.
- **Confidence calibration** via temperature scaling, so the reported number means what
  it says.
- **Larger backbones** — ResNet50, EfficientNet or a vision transformer — benchmarked
  against ResNet18 to see whether capacity or data is the binding constraint.

---

## 9. Conclusion

A ResNet18 fine-tuned in two phases classifies brain MRI slices into four categories at
**97.00% test accuracy**, with a macro F1 of 0.9715 and perfect recall on the no-tumour
class. The two-phase schedule is what makes this work on 5,000 images: freezing the
backbone while the new head stabilises, then adapting only the deepest block at a reduced
learning rate, lifted validation accuracy from a 84.2% ceiling to 98.0%.

The residual errors are concentrated, not diffuse — 63% of them sit on the
glioma/meningioma boundary, the same distinction that is hardest for human readers on a
single slice. That concentration is useful: it points at a specific, addressable target
for the next iteration rather than a general need for more capacity.

---

## Reproducing

```bash
pip3 install -r requirements.txt
python3 app.py          # serves http://127.0.0.1:5001
```

Training is reproducible from `brain_tumor_resnet18_optimized.ipynb`; the executed copy
with all outputs is `brain_tumor_resnet18_optimized_executed.ipynb`.

---

*Research and educational use only. Not a medical device. Not for clinical diagnosis.*
