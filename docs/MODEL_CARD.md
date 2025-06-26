# Model Card: KrishiSahayak Plant Disease Classifier

**Date:** June 26, 2025  
**Version:** 0.1.0

## 1. Model Overview

**Model Name:** KrishiSahayak Plant Disease Classifier  
**Version:** 0.1.0  

**Architecture:** The model is based on the UnifiedModel framework, a flexible architecture that uses timm backbones (e.g., EfficientNet) and supports configurable multi-stream inputs (e.g., RGB, Multispectral) with various feature fusion strategies. The deployment architecture can be further enhanced by the HybridModel orchestrator.

**Purpose:** An AI-Powered Crop Health Assistant designed to provide preliminary identification of plant diseases from images, serving as a decision-support tool for farmers and agricultural NGOs.

## 2. Intended Use

**Intended Users:** Farmers, agricultural extension workers, and NGOs, particularly in regions with limited access to agronomic expertise. The system is designed with offline-first use cases in mind.

**Intended Use Cases:**
- As an initial diagnostic aid to identify potential diseases in common crops based on leaf imagery.
- To help users triage crop health issues and decide when to seek professional agronomic advice.
- To be integrated into advisory applications that provide information on potential mitigation strategies based on the identified disease.

**Not Intended For:**
- Use as a sole, final, or automated basis for making financial decisions, such as crop insurance claims or large-scale resource allocation.
- Use as a direct replacement for professional, certified agronomists.
- Real-time, safety-critical control of agricultural machinery.

## 3. Out-of-Scope Use Cases

This model is designed exclusively for plant disease classification. Any use outside of this scope is unsupported and may produce unpredictable, incorrect results. Specifically, the model should not be used for:
- Identifying plant species for non-disease-related purposes.
- Diagnosing human or animal health conditions.
- Analyzing satellite imagery for large-scale land assessment.
- Detecting nutrient deficiencies that mimic disease symptoms, unless explicitly trained for that task.

## 4. Performance Summary

| Metric | Value | Dataset |
|--------|-------|---------|
| Accuracy | [Pending evaluation results] | Test Set |
| F1-Score (Macro) | [Pending evaluation results] | Test Set |
| Average Confidence | [Pending evaluation results] | Test Set |

**Dataset Description:** The model is trained on publicly available datasets of plant leaf images, such as PlantVillage and PlantDoc. These datasets contain thousands of images across multiple crop species and disease categories, captured under varying conditions.

## 5. Training Dataset

**Sources:** The training data is prepared from public datasets, including PlantVillage and PlantDoc, using the project's `prepare.py` script.

**Preprocessing:** The standard preprocessing pipeline, defined in `transforms.py`, includes:
- Resizing all images to a standard resolution (e.g., 224x224 or 256x256).
- Normalization using standard ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for RGB images.
- Conversion to PyTorch tensors.

**Augmentation:** The training pipeline (`TransformFactory`) applies a series of augmentations to improve model generalization, including:
- RandomResizedCrop
- HorizontalFlip & VerticalFlip
- RandomRotate90
- Color jittering via RandomBrightnessContrast and HueSaturationValue.

## 6. Model Details

**Input Format:** The UnifiedModel accepts a dictionary of tensors, where each key corresponds to an input stream (e.g., `{'rgb': tensor, 'ms': tensor}`). For a single-stream model, it can accept a single tensor of shape (B, C, H, W).

**Output Format:** The model outputs a dictionary containing the top-k predictions, where each prediction includes the class name and its corresponding probability.

```json
{
  "predictions": [
    {"class": "Tomato___Late_blight", "probability": 0.98},
    {"class": "Tomato___Early_blight", "probability": 0.015}
  ]
}
```

**Inference Time:** [Pending performance profiling on target hardware.]

**Export Formats:** The primary artifact is the PyTorch Lightning `.ckpt` checkpoint. No other export formats (e.g., ONNX, TorchScript) are currently supported by the provided codebase.

## 7. Ethical Considerations

**Bias:** The model's performance is entirely dependent on the diversity of the training data. It may exhibit bias towards the crop varieties, geographic regions, lighting conditions, and camera types represented in the PlantVillage and PlantDoc datasets. Performance on rare diseases or under-represented crop types may be significantly lower.

**Explainability:** The UnifiedModel architecture includes a `get_feature_maps` method. This enables the use of gradient-based attribution methods (e.g., Grad-CAM) to generate heatmaps that visualize which parts of an image were most influential in a prediction, providing a degree of transparency.

**Privacy & Consent:** The model was trained using publicly available datasets, which do not contain Personally Identifiable Information (PII). Any application built using this model that collects images from users must implement its own robust consent and data privacy policies.

**Fairness:** The model may be less reliable for subsistence farmers or those in regions growing crops not well-represented in the training data, potentially creating an unfair disadvantage. Its use should always be coupled with access to local agronomic expertise.

## 8. Evaluation Protocol

The model is evaluated on a held-out test set created during the data preparation phase using stratified sampling. This ensures the test set reflects the class distribution of the overall dataset. Performance is measured using standard classification metrics, including overall accuracy and macro-averaged F1-score, which provides a balanced measure for datasets with class imbalance.

## 9. Known Limitations

- The model can only identify the disease classes it was trained on. It cannot identify novel or out-of-distribution diseases.
- Performance is sensitive to image quality, lighting conditions, and the growth stage of the plant. Significant variations from the training data can degrade accuracy.
- The model may struggle to differentiate between multiple diseases present on a single leaf or to distinguish between diseases and symptoms of nutrient deficiency.
- The advanced HybridModel's ability to generate synthetic NIR data is an approximation and may not fully capture the detail of true multispectral imagery.

## 10. Version History

**v0.1.0 (June 26, 2025):** Initial release of the production-grade framework, featuring the UnifiedModel architecture, HybridModel orchestrator, and advanced utility modules for distillation and validation.

## 11. Contact Info

**Author:** Vikas Sahani  
**Contact:** [vikassahani17@gmail.com](mailto:vikassahani17@gmail.com)  
**Project Link:** [KrishiSahayak 2.0 on GitHub](https://github.com/VIKAS9793/Krishisahayak-2.0)

## Regulatory & Compliance Summary

**Privacy Status of Training Data:** The model was trained exclusively on publicly available datasets (PlantVillage, PlantDoc) which contain no Personally Identifiable Information (PII).

**Explainability Tools Used:** The model architecture (UnifiedModel) is instrumented with hooks (`get_feature_maps`) that support post-hoc, gradient-based explainability methods like Grad-CAM.

**Fallback Behavior:** Yes. The HybridModel deployment pattern includes a built-in, configurable confidence-based fallback system, allowing it to revert to a simpler, reliable model if the primary model's prediction confidence is low.

**Responsible AI Alignment:** The project's development aligns with key principles of Responsible AI, including a focus on Reliability (fallback behavior, resilient data pipelines), Transparency (explainability hooks, detailed model card), and Fairness (acknowledgment of data-induced biases and limitations).
