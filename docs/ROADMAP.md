# KrishiSahayak 2.0 - Project Roadmap

## Project Overview
- **Version**: 2.1.0
- **Design Philosophy**: Offline-first, Explainable-by-default, Modular-by-design
- **Scalability**: Cloud-optional for stakeholder-controlled deployments
- **Target Users**: Farmers, NGOs, Agri Startups, Rural Extension Workers

## Development Phases

### Phase 1 – Core Robustness Enhancements (Completed)

| ID  | Task | Description | Status | Implementation Location |
|-----|------|-------------|---------|-------------------------|
| R1  | Add input validation for uploaded images | Check image size, format, corruption, and RGB mode | ✅ Done | `src/krishi_sahayak/inference/data_loader.py` |
| R2  | Implement softmax threshold fallback logic | Return low-confidence warning if model output < 0.6 | ✅ Done | `src/krishi_sahayak/models/utils/confidence.py` |
| R3  | Enable prediction logging to CSV/JSON | Log input hash, timestamp, confidence, prediction class | ✅ Done | `src/krishi_sahayak/inference/handler.py` |
| R4  | Add adversarial robustness test cases | Use blur, noise, and rotation in test suite | ✅ Done | `configs/robustness_plan.json` |

### Phase 2 – Trust, Transparency & Explainability

| ID  | Task | Description | Status | Implementation Location |
|-----|------|-------------|---------|-------------------------|
| T1  | Integrate Grad-CAM for explainability | Add local XAI visualizer for RGB classifier | ✅ Done | `src/krishi_sahayak/inference/predictor.py` (lines 19-46) |
| T2  | Add `MODEL_CARD.md` | Summarize intended use, limitations, performance, training data | ✅ Done | `docs/MODEL_CARD.md` |
| T3  | Add `LIMITATIONS.md` file | Clarify scope boundaries (e.g., not suitable for medicinal crops) | ✅ Done | `docs/LIMITATIONS.md` |
| T4  | Implement fallback rule-based layer | Heuristic rules for basic crop disease mapping | 🔄 Partially Implemented | `src/krishi_sahayak/models/utils/confidence.py` (basic implementation) |

### Phase 3 – Testing, Versioning, Feedback Loop

| ID  | Task | Description | Status | Implementation Location |
|-----|------|-------------|---------|-------------------------|
| V1  | Create `model_config.yaml` | Includes model type, train date, dataset version, accuracy | ✅ Done | `configs/model_config.yaml` |
| V2  | Add unit test scripts using pytest | Test prediction, image loading, and fallback logic | ✅ Done | `tests/` directory |
| V3  | Add user feedback logging | Form-based or CLI-based feedback collection | ❌ Not Started | Not found in codebase |
| V4  | Add local annotation CLI tool | Allow users to label/correct images for retraining | ✅ Done | `src/krishi_sahayak/data/prepare.py` |

---

## Implementation Details

### Core Components
- **Unified Model**: `src/krishi_sahayak/models/core/unified_model.py`
- **Hybrid Model**: `src/krishi_sahayak/models/core/hybrid_model.py`
- **GAN Implementation**: `src/krishi_sahayak/models/gan/`
- **API Layer**: `src/krishi_sahayak/api/main.py`
- **Inference Pipeline**: `src/krishi_sahayak/inference/`

### Configuration
- **Model Config**: `configs/model_config.yaml`
- **Data Augmentation**: `configs/augmentations/plantdoc_augmentations.yaml`
- **Robustness Plan**: `configs/robustness_plan.json`

## Version History

### v2.1.0 (Current)
- Core robustness enhancements (R1-R4)
- Trust and transparency features (T1-T3)
- Testing and feedback mechanisms (V1-V2, V4)
- Updated documentation and model cards

### v2.0.0
- Initial release of KrishiSahayak 2.0
- Hybrid model architecture
- Offline-first design
- Basic API implementation

---
*Last Updated: June 27, 2025*

### Privacy & Ethics
- No private or personal data collection
- All training data from public datasets
- Model cards and limitations documented
- Explainability features integrated

### Note
This roadmap reflects the current implementation status as verified in the codebase. All features marked as completed have been verified in the source code.
