# RA-Distill

This repository provides the PyTorch implementation of the paper:

> Retrieval-Augmented Reconstruction with Distillation for Robust Skin Lesion Diagnosis under Missing Modalities.

RA-Distill is a retrieval-augmented framework for robust skin lesion diagnosis when one modality is unavailable. Given an incomplete sample, the available modality retrieves similar complete training cases from a retrieval representation bank. The paired missing-modality representations of the retrieved cases are then used as external evidence for query-conditioned missing-modality reconstruction. A full-modality teacher further guides the retrieval-augmented student through logit distillation.

## Repository Structure

```text
RA-Distill-Lesion/
  ra_distill/
    models.py        # teacher model and RA-Distill student
    retrieval.py     # retrieval representation bank and top-k retrieval
    losses.py        # CE + reconstruction + KD objective
    synthetic.py     # example complete multimodal dataset
    train.py         # training utilities
  examples/
    run_example.py   # runnable example
  scripts/
    run_example.sh
  tests/
    test_forward.py
```

## Installation

```bash
git clone https://github.com/Groot-H/RA-Distill-Lesion.git
cd RA-Distill-Lesion
pip install -r requirements.txt
```

## Quick Start

Run the example script:

```bash
bash scripts/run_example.sh
```

or run each missing-modality setting separately:

```bash
python examples/run_example.py --missing clinical
python examples/run_example.py --missing image
```

The example follows the RA-Distill pipeline:

1. train a full-modality teacher on complete image-clinical pairs;
2. build a retrieval representation bank from the teacher encoders and the training set;
3. train an RA-Distill student under a selected missing-modality setting.

## Method Overview

### 1. Full-Modality Teacher and Retrieval Bank

The full-modality teacher receives both image and clinical modalities:

```text
image -> ImageEncoder -> image representation
clinical attributes -> ClinicalEncoder -> clinical representation
[image representation, clinical representation] -> AttentionFusion -> Classifier
```

After teacher training, all complete training samples are encoded by the trained teacher encoders. The paired image and clinical representations form the retrieval representation bank:

```text
Image Representation Bank    = {r_1^I, ..., r_N^I}
Clinical Representation Bank = {r_1^C, ..., r_N^C}
```

The retrieval bank is constructed from the training set of each fold.

### 2. Retrieval-Guided Missing-Modality Reconstruction

For clinical-missing diagnosis, the available image representation is used as the retrieval query:

```text
Query  = available image representation
Keys   = Image Representation Bank
Values = paired Clinical Representation Bank
```

The top-k retrieved clinical representations provide external evidence for reconstructing the clinical representation of the incomplete sample. The image-missing setting is symmetric:

```text
Query  = available clinical representation
Keys   = Clinical Representation Bank
Values = paired Image Representation Bank
```

The missing-modality reconstructor uses a conditional mixture-of-experts design. It
combines the current query representation with each retrieved missing-modality
representation, transforms them with expert MLPs, and aggregates the expert outputs
with the retrieval weights.

### 3. Sample-Aware Prompt and Fusion

RA-Distill generates a sample-aware prompt representation from the available
representation and retrieved missing-modality evidence. The prompt generator combines
the available representation, cross-attended retrieval context, and retrieval-weighted
context. The available representation, reconstructed missing representation, and prompt
representation are then fused by the attention fusion module for diagnosis.

### 4. Training Objective

The RA-Distill student is optimized with:

```text
L = L_CE + lambda_rec * L_rec + lambda_kd * L_KD
```

where:

- `L_CE` is cross-entropy with the ground-truth label;
- `L_rec` is MSE between reconstructed and teacher-encoded target missing representation;
- `L_KD` is logit distillation from the full-modality teacher.

## Data Format

To use a custom dataset, prepare samples with the following fields:

```python
{
    "image": FloatTensor[C, H, W],
    "clinical": FloatTensor[num_clinical_features],
    "label": LongTensor[],
    "sample_id": str,
}
```

Then follow the same pipeline:

1. train `FullModalityTeacher`;
2. build `RetrievalBank.from_teacher(...)` using the training set;
3. train `RADistillStudent` with `missing="clinical"` or `missing="image"`.

## Updates

We will continuously improve and update this repository.
