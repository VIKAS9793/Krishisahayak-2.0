# **Known Limitations and Scope Boundaries for KrishiSahayak 2.0**

**Version:** 2.1-wip
**Date:** June 26, 2025

This document outlines the known limitations and out-of-scope use cases for the KrishiSahayak model. It is intended to ensure responsible and effective use of the technology.

## 1. Data-Driven Limitations

The model's performance is fundamentally tied to the data it was trained on (e.g., PlantVillage, PlantDoc). Consequently:

* **Geographic & Varietal Bias:** The model may perform poorly on crop varieties, growth stages, or farming conditions not well-represented in the public training datasets.
* **Image Quality Dependency:** Prediction accuracy is highly sensitive to the quality of the input image. Out-of-focus, poorly lit, or oddly angled images may lead to incorrect predictions.

## 2. Scope of Functionality

The model is designed **exclusively** for classifying pre-defined plant diseases from leaf imagery.

* **Exclusively Plant Diseases:** It cannot identify pests, nutrient deficiencies (unless a heuristic rule is triggered, per Task **T4**), or other abiotic stressors.
* **Not a General Plant Identifier:** It cannot be used to identify the species of a healthy plant.
* **Limited Crop Types:** The model is not suitable for identifying diseases in crops it has not been trained on, such as specialized medicinal or ornamental plants.

## 3. Inference and Reliability

* **Not a Substitute for Expert Advice:** The model provides a *preliminary diagnostic aid*, not a definitive diagnosis. All significant agricultural decisions should be confirmed by a certified agronomist.
* **Confidence Scores are not Certainty:** The softmax probability score reflects the model's confidence in its prediction, not the ground truth certainty. Low-confidence warnings (Task **R2**) should be heeded.
