# KrishiSahayak 2.0 - Project Roadmap

## Project Overview
- **Version**: 2.1-wip
- **Design Philosophy**: Offline-first, Explainable-by-default, Modular-by-design
- **Scalability**: Cloud-optional for stakeholder-controlled deployments
- **Target Users**: Farmers, NGOs, Agri Startups, Rural Extension Workers

## Development Phases

### Phase 1 – Core Robustness Enhancements

| ID  | Task | Description | Status |
|-----|------|-------------|---------|
| R1  | Add input validation for uploaded images | Check image size, format, corruption, and RGB mode | 🟡 In Progress |
| R2  | Implement softmax threshold fallback logic | Return low-confidence warning if model output < 0.6 | 🟡 In Progress |
| R3  | Enable prediction logging to CSV/JSON | Log input hash, timestamp, confidence, prediction class | 🟡 In Progress |
| R4  | Add adversarial robustness test cases | Use blur, noise, and rotation in test suite | 🟡 In Progress |

### Phase 2 – Trust, Transparency & Explainability

| ID  | Task | Description | Status |
|-----|------|-------------|---------|
| T1  | Integrate SHAP for explainability | Add local SHAP value visualizer for RGB classifier | 🟡 In Progress |
| T2  | Add `model_card.md` | Summarize intended use, limitations, performance, training data | 🟡 In Progress |
| T3  | Add `LIMITATIONS.md` file | Clarify scope boundaries (e.g., not suitable for medicinal crops) | 🟡 In Progress |
| T4  | Implement fallback rule-based layer | Heuristic rules for basic crop disease mapping | 🟡 In Progress |

### Phase 3 – Testing, Versioning, Feedback Loop

| ID  | Task | Description | Status |
|-----|------|-------------|---------|
| V1  | Create `model_config.yaml` | Includes model type, train date, dataset version, accuracy | 🟡 In Progress |
| V2  | Add unit test scripts using pytest | Test prediction, image loading, and fallback logic | 🟡 In Progress |
| V3  | Add user feedback logging | Form-based or CLI-based feedback collection | 🟡 In Progress |
| V4  | Add local annotation CLI tool | Allow users to label/correct images for retraining | 🟡 In Progress |

## Supporting Files

| File | Location |
|------|----------|
| Robustness Plan | `configs/robustness_plan.json` |
| Model Card | `docs/model_card.md` |
| Limitations | `docs/LIMITATIONS.md` |
| Model Config | `configs/model_config.yaml` |
| Annotation Tool | `scripts/annotate.py` |
| Unit Tests | `tests/test_predictor.py` |

## Compliance & Ethics

### Privacy
No private or personal data is used; images are public and anonymized.

### Explainability
Designed for interpretability-first deployment in non-cloud, non-English-first user groups.

### Ethics Alignment
Follows Microsoft Responsible AI checklist and Google model documentation best practices.

## Version History

### v2.1 (Work in Progress)
- Core robustness enhancements
- Trust and transparency features
- Testing and feedback mechanisms

### v2.0
- Initial release of KrishiSahayak 2.0
- Hybrid model architecture
- Offline-first design

---
*Last Updated: June 2025*
