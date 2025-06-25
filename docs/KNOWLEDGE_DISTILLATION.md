# Knowledge Distillation Guide

This guide explains the knowledge distillation framework implemented in the KrishiSahayak project.

## 1. Overview

Knowledge distillation transfers knowledge from a large, accurate teacher model (e.g., a hybrid RGB+MS model) to a smaller, more efficient student model (e.g., an RGB-only model) while maintaining competitive performance. This enables deployment on resource-constrained devices.

## 2. Teacher-Student Architecture

### Teacher Model (Example: Hybrid Fusion Model)
- **Input**: Multi-modal (e.g., RGB + Multispectral)
- **Backbone**: `efficientnetv2_rw_s` (or other large variant)
- **Key Features**: High accuracy due to multi-modal inputs and larger capacity.

### Student Model (Example: Lightweight RGB-Only)
- **Input**: RGB images only
- **Backbone**: `efficientnet_b0` (or other efficient variant)
- **Goal**: Achieve performance close to the teacher model but with a fraction of the parameters and faster inference speed.

## 3. Implementation Details

The distillation process is managed by the `DistillationLightningModule` and a custom `KnowledgeDistillationLoss`, which combines three loss components.

### Knowledge Distillation Loss

The loss is a weighted combination of:
1.  **Hard Loss (Cross-Entropy)**: The student learns from the ground truth labels.
2.  **Soft Loss (KL Divergence)**: The student learns from the softened probability distribution of the frozen teacher model.
3.  **Feature Matching Loss (MSE)**: The student's intermediate feature maps are encouraged to match the teacher's, transferring structural knowledge.

### PyTorch Lightning Module (Refactored)

The `DistillationLightningModule` is designed with **Dependency Injection**. It does not create the models itself; instead, it receives pre-instantiated student and teacher models, making it flexible and testable.

```python
# This snippet reflects the refactored, production-ready code.
# from krishi_sahayak.models.utils.distillation import DistillationLightningModule

class DistillationLightningModule(ProjectBaseModel):
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        config: DistillationConfig, # A Pydantic config object
        learning_rate: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=student_model, learning_rate=learning_rate, **kwargs)
        self.save_hyperparameters(ignore=['student_model', 'teacher_model'])

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.config = config

        # The module expects a frozen teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.criterion = KnowledgeDistillationLoss(
            temperature=self.config.temperature,
            alpha=self.config.alpha,
            feature_loss_weight=self.config.feature_loss_weight
        )
    
    # ... training_step and other methods follow ...

## 4. Training Process

### Step 1: Teacher Model Training
1. Train the hybrid teacher model on RGB+MS data
2. Evaluate and save the best checkpoint
3. Freeze all teacher parameters

### Step 2: Student Model Training
1. Initialize student model with pretrained weights
2. Set up distillation trainer with:
   - Frozen teacher model
   - Trainable student model
   - Knowledge distillation loss
3. Train using the following schedule:
   - Initial warmup (2-3 epochs)
   - Main training with cosine annealing
   - Optional fine-tuning with lower learning rate

### Step 3: Evaluation & Export
1. Evaluate on validation set
2. Compare with teacher and baseline student
3. Export to ONNX/TFLite for deployment

### Supported Features
- Mixed precision training (FP16/FP32)
- Gradient clipping
- Learning rate warmup
- Model checkpointing
- Early stopping
- TensorBoard/Weights & Biases logging

## 5. Configuration

### YAML Configuration

```yaml
distillation:
  # Model architecture
  teacher_checkpoint: "checkpoints/teacher/best.ckpt"
  student_backbone: "efficientnet_b0"
  student_pretrained: "imagenet"  # or path to custom weights

  # Loss function parameters
  temperature: 4.0  # Controls soft target smoothing
  alpha: 0.3        # Weight for student loss (1-alpha for distillation loss)
  feature_weight: 0.5  # Weight for feature matching loss

  # Training hyperparameters
  batch_size: 64
  learning_rate: 3e-4
  weight_decay: 1e-4
  max_epochs: 100
  warmup_epochs: 3

  # Feature matching configuration
  feature_matching: true
  feature_layers: ["blocks.2", "blocks.4", "blocks.6"]  # Layer names for feature extraction

  # Learning rate schedule
  lr_scheduler: "cosine"  # or "step", "plateau"
  lr_gamma: 0.1
  lr_step_size: 30

  # Early stopping
  early_stopping_patience: 10

  # Mixed precision training ("16-mixed", "bf16-mixed", or None)
  precision: "16-mixed"

  # Logging and checkpointing
  log_every_n_steps: 50
  checkpoint_dir: "checkpoints/student"
  save_top_k: 3
  monitor: "val_acc"
  mode: "max"
```

### Configuration Notes

1. **Temperature**:
   - Higher values (4.0-10.0) produce softer probability distributions
   - Lower values (1.0-2.0) make the distribution sharper
   - T=1.0 reduces to standard softmax

2. **Alpha (α)**:
   - Controls the balance between hard and soft targets
   - α=1.0: Only use ground truth labels
   - α=0.0: Only use teacher's soft targets
   - Recommended range: 0.1-0.5

3. **Feature Matching**:
   - Extracts features from intermediate layers
   - Uses L2 distance between teacher and student features
   - Helps transfer structural knowledge beyond just logits

## 6. Expected Outcomes

### Performance Metrics

| Model | Parameters | Top-1 Acc. | Latency (ms) | Memory (MB) |
|-------|------------|------------|--------------|-------------|
| Teacher (RGB+MS) | 24.3M | 94.8% | 14.1 | 95.2 |
| Student (Distilled) | 5.3M | 92.1% | 4.2 | 21.3 |
| Student (Baseline) | 5.3M | 89.3% | 4.1 | 21.3 |
| Student (From Scratch) | 5.3M | 85.7% | 4.1 | 21.3 |

### Key Benefits

1. **Efficiency**
   - 78% reduction in parameters compared to teacher
   - 3.4x faster inference on GPU
   - 4.5x smaller memory footprint

2. **Performance**
   - Only 2.7% accuracy drop from teacher
   - 2.8% better than baseline student
   - More robust to input variations

3. **Deployment**
   - Compatible with TensorRT/TFLite
   - Supports INT8 quantization
   - Works on edge devices

## 7. Best Practices

### 1. Model Architecture Selection
- **Teacher Model**:
  - Use the largest feasible model as teacher
  - Ensure teacher is well-trained and converged
  - Consider using ensemble of models for better soft targets

- **Student Model**:
  - Start with architecture 4-10x smaller than teacher
  - Use efficient backbones (EfficientNet, MobileNetV3, etc.)
  - Consider depth-wise separable convolutions

### 2. Training Strategy

#### Temperature Scheduling
```python
# Example of linear temperature decay
current_temp = max(
    final_temp,
    initial_temp * (1 - epoch / total_epochs) + final_temp * (epoch / total_epochs)
)
```

#### Learning Rate Schedule
- Use cosine annealing with warm restarts
- Start with higher LR (3-5x) than normal training
- Add linear warmup (2-3 epochs)

#### Batch Size Considerations
- Larger batch sizes (128-256) often work better
- Use gradient accumulation for small batch sizes
- Consider larger batches with higher temperatures

### 3. Advanced Techniques

#### Multi-Teacher Distillation
```python
# Combine multiple teacher models
teacher_logits = sum(
    teacher(x)['logits'] * weight
    for teacher, weight in zip(teachers, teacher_weights)
)
```

#### Self-Distillation
- Use EMA of student as additional teacher
- Helps with training stability
- Can improve final performance

#### Quantization-Aware Training
```python
model = quantize_model(model)
# Train with QAT for 10% of total epochs
if current_epoch > 0.9 * total_epochs:
    model.apply(enable_quantization)
```

## 8. Troubleshooting Guide

### Common Issues and Solutions

#### Poor Student Performance
| Symptom | Possible Causes | Solution |
|---------|----------------|-----------|
| Student accuracy much lower than teacher | Temperature too high/low | Adjust between 2.0-8.0 |
| | Feature mismatch | Verify layer names in feature extraction |
| | Incorrect alpha | Try α=0.3-0.7 for balanced learning |

#### Training Instability
| Symptom | Possible Causes | Solution |
|---------|----------------|-----------|
| Loss NaN/Inf | High learning rate | Reduce LR by 2-5x |
| | Extreme logits | Add gradient clipping (max_norm=1.0) |
| | Numerical instability | Use mixed precision training |

#### Slow Training
| Symptom | Possible Causes | Solution |
|---------|----------------|-----------|
| GPU underutilized | Small batch size | Increase batch size |
| | CPU bottleneck | Use DataLoader workers |
| | Inefficient augmentation | Optimize transforms |

### Debugging Tips

1. **Sanity Checks**
   ```python
   # Verify teacher outputs
   with torch.no_grad():
       teacher.eval()
       out = teacher(val_batch)
       print("Teacher output stats:", out['logits'].mean(), out['logits'].std())
   ```

2. **Loss Components**
   - Log individual loss terms
   - Monitor their relative magnitudes
   - Ensure none dominates unexpectedly

3. **Gradient Flow**
   - Check gradient norms
   - Visualize attention maps if using attention
   - Monitor weight updates

### Performance Optimization

#### Inference Optimization
```python
# Convert to TorchScript
example_input = torch.randn(1, 3, 224, 224)
scripted_model = torch.jit.trace(model, example_input)
torch.jit.save(scripted_model, "student_optimized.pt")

# For mobile deployment
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

#### Memory Optimization
- Use gradient checkpointing
- Enable mixed precision training
- Consider gradient accumulation for larger batches

### Getting Help

For additional support:
1. Check the project's GitHub issues
2. Review PyTorch Lightning documentation
3. Consult the model optimization toolkit
4. Reach out to the maintainers with:
   - Full error logs
   - Environment details
   - Reproduction steps
