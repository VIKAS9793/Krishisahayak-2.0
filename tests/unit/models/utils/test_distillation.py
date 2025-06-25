import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# The path to the module under test is now more nested
from krishi_sahayak.models.utils.distillation import (
    KnowledgeDistillationLoss,
    DistillationLightningModule,
    DistillationConfig
)

# --- Fixtures ---

@pytest.fixture
def mock_student_model() -> MagicMock:
    """Creates a mock student model."""
    model = MagicMock(spec=nn.Module)
    model.return_value = torch.randn(2, 10) # Logits output
    return model

@pytest.fixture
def mock_teacher_model() -> MagicMock:
    """Creates a mock teacher model."""
    model = MagicMock(spec=nn.Module)
    model.return_value = torch.randn(2, 10) # Logits output
    # Mock the parameters() method to return a list of Parameters for the requires_grad check
    model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    return model

@pytest.fixture
def distillation_config() -> DistillationConfig:
    """Provides a default DistillationConfig."""
    return DistillationConfig(temperature=3.0, alpha=0.5)

@pytest.fixture
def dummy_batch() -> Dict[str, torch.Tensor]:
    """Provides a dummy data batch."""
    return {"image": torch.randn(2, 3, 32, 32), "target": torch.randint(0, 10, (2,))}


# --- Test Classes ---

class TestKnowledgeDistillationLoss:
    def test_loss_calculation(self):
        """Verify the combined loss is calculated correctly."""
        loss_fn = KnowledgeDistillationLoss(alpha=0.5, temperature=1.0)
        student_logits = torch.randn(2, 10, requires_grad=True)
        teacher_logits = torch.randn(2, 10)
        target = torch.randint(0, 10, (2,))

        loss_dict = loss_fn(student_logits, teacher_logits, target)
        
        assert "loss" in loss_dict
        assert "ce_loss" in loss_dict
        assert "kld_loss" in loss_dict
        
        # Check if total loss is a weighted sum
        expected_loss = (1 - 0.5) * loss_dict['ce_loss'] + 0.5 * loss_dict['kld_loss']
        assert torch.isclose(loss_dict['loss'], expected_loss, atol=1e-6)

    def test_feature_loss_calculation(self):
        """Verify feature loss is added when features are provided."""
        loss_fn = KnowledgeDistillationLoss(feature_loss_weight=10.0)
        student_logits = torch.randn(2, 10)
        teacher_logits = torch.randn(2, 10)
        target = torch.randint(0, 10, (2,))
        student_features = {"layer1": torch.randn(2, 64)}
        teacher_features = {"layer1": torch.randn(2, 64)}

        loss_dict = loss_fn(student_logits, teacher_logits, target, student_features, teacher_features)
        
        assert "feature_loss" in loss_dict
        assert loss_dict["feature_loss"].item() > 0
        assert loss_dict["loss"].item() > loss_dict["feature_loss"].item()


class TestDistillationLightningModule:
    def test_init(self, mock_student_model, mock_teacher_model, distillation_config):
        """Verify successful instantiation and that the teacher model is frozen."""
        # Mock the parent class's __init__ to avoid dealing with its dependencies
        with patch("krishi_sahayak.models.base.BaseModel.__init__") as mock_parent_init:
            module = DistillationLightningModule(
                student_model=mock_student_model,
                teacher_model=mock_teacher_model,
                config=distillation_config,
                learning_rate=1e-3
            )
            mock_parent_init.assert_called_once()

        assert module.student_model is mock_student_model
        assert module.teacher_model is mock_teacher_model
        
        # Check that teacher model was frozen
        mock_teacher_model.eval.assert_called_once()
        for param in mock_teacher_model.parameters():
            assert not param.requires_grad

    def test_training_step(self, mock_student_model, mock_teacher_model, distillation_config, dummy_batch):
        """Verify the training step calls models and logs metrics."""
        # Mock the parent class's __init__
        with patch("krishi_sahayak.models.base.BaseModel.__init__"):
            module = DistillationLightningModule(
                student_model=mock_student_model,
                teacher_model=mock_teacher_model,
                config=distillation_config,
                learning_rate=1e-3
            )
        
        # Mock logging and other helper methods
        module.log = MagicMock()
        module.log_dict = MagicMock()
        module._prepare_batch = MagicMock(return_value=(dummy_batch["image"], dummy_batch["target"]))

        loss = module.training_step(dummy_batch, batch_idx=0)
        
        # Verify models were called
        mock_student_model.assert_called_once_with(dummy_batch["image"])
        mock_teacher_model.assert_called_once_with(dummy_batch["image"])
        
        assert torch.is_tensor(loss)
        module.log.assert_called_once()
        module.log_dict.assert_called_once()
