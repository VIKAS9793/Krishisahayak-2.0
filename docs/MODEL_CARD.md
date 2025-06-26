# **Model Card: KrishiSahayak 2.0 Plant Disease Classifier**

**Date:** June 26, 2025
**Version:** 2.1-wip

## 1. Model Overview

* **Model Name:** KrishiSahayak 2.0 Plant Disease Classifier
* **Version:** 2.1-wip
* **Architecture:** The model is based on the `UnifiedModel` framework, a flexible architecture using `timm` backbones (e.g., EfficientNet). The deployment architecture can be enhanced by the `HybridModel` orchestrator.
* **Purpose:** An offline-first, explainable AI-powered assistant to help farmers, NGOs, and agri-startups with preliminary identification of plant diseases.

## 2. Intended Use

* **Intended Users:** Farmers, NGOs, Agri Startups, and Rural Extension Workers.
* **Intended Use Cases:** To provide an initial diagnostic aid for common crop diseases from leaf images, enabling users to make more informed decisions about seeking professional agronomic advice.
* **Not Intended For:** Use as a sole basis for financial or chemical treatment decisions, or as a replacement for professional agronomists.

## 3. Out-of-Scope Use Cases

This model should **not** be used for diagnosing human/animal diseases, identifying plant species, or automated insurance claim processing.

## 4. Performance Summary

Performance metrics are pending final evaluation runs upon completion of Phase 1 (Robustness Enhancements).

| Metric | Value | Dataset |
| :--- | :--- | :--- |
| **Accuracy** | `[TODO: Pending evaluation]` | Test Set |
| **F1-Score (Macro)** | `[TODO: Pending evaluation]` | Test Set |
| **Adversarial Robustness** | `[TODO: Pending evaluation]` | (As per `robustness_plan.json`) |

* **Dataset Description:** The model is trained on public datasets like PlantVillage and PlantDoc.

## 5. Training Dataset

The model is trained on publicly available datasets of plant leaf images. Preprocessing and augmentation pipelines are managed by the project's `TransformFactory`.

## 6. Model Details

* **Input Format:** A dictionary of tensors, e.g., `{'rgb': torch.Tensor}`.
* **Output Format:** A dictionary containing the top-k predictions with `class` and `probability`.
* **Inference Time:** `[TODO: Pending performance profiling]`

## 7. Ethical Considerations

* **Bias:** Model performance is dependent on the training data and may show bias towards specific crop varieties or geographic regions.
* **Explainability:** The framework is "Explainable-by-default". The architecture includes hooks for Grad-CAM, and SHAP integration is in progress (Task **T1**).
* **Privacy:** The model is trained on public, anonymized data, minimizing privacy risks.

## 8. Known Limitations

A comprehensive list of limitations is maintained in `LIMITATIONS.md` (Task **T3**). Key limitations include sensitivity to out-of-distribution inputs and an inability to distinguish between visually similar diseases and nutrient deficiencies.

## 9. Contact Info

* **Author:** Vikas Sahani
* **Project Link:** [KrishiSahayak 2.0 on GitHub](https://github.com/VIKAS9793/Krishisahayak-2.0)
