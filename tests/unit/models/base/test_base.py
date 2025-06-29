import pytest
import torch
import torch.nn as nn
from typing import Dict
from unittest.mock import MagicMock

from krishi_sahayak.models.base.base import BaseModel, BaseModelConfig

# --- A concrete implementation of the BaseModel for testing ---
class ConcreteTestModel(BaseModel):
    """A concrete implementation of BaseModel for testing purposes."""
    # The BaseModel is no longer abstract, as the model is injected.
    # No methods need to be overridden for basic testing.
    pass

# --- Test Fixtures ---
@pytest.fixture
def mock_model() -> MagicMock:
    """Provides a mock nn.Module."""
    model = MagicMock(spec=nn.Module)
    model.return_value = torch.randn(2, 5) # Batch size 2, 5 classes
    return model

@pytest.fixture
def base_config() -> BaseModelConfig:
    """Provides a default BaseModelConfig."""
    return BaseModelConfig(learning_rate=1e-4)

@pytest.fixture
def batch_processor():
    """Provides a simple batch processor function."""
    def _processor(batch):
        return batch['image'], batch['label']
    return _processor

@pytest.fixture
def dummy_batch() -> Dict[str, torch.Tensor]:
    """Provides a dummy data batch."""
    return {
        "image": torch.randn(2, 3, 32, 32),
        "label": torch.randint(0, 5, (2,)),
    }

# --- Test Class ---
class TestBaseModel:
    def test_init_success(self, mock_model, base_config, batch_processor):
        """Verify successful instantiation."""
        instance = ConcreteTestModel(
            model=mock_model, num_classes=5, config=base_config, batch_processor=batch_processor
        )
        assert isinstance(instance, pl.LightningModule)
        assert instance.model is mock_model
        assert instance.batch_processor is batch_processor

    def test_shared_step(self, mock_model, base_config, batch_processor, dummy_batch):
        """Verify that a shared step computes loss and logs metrics correctly."""
        instance = ConcreteTestModel(
            model=mock_model, num_classes=5, config=base_config, batch_processor=batch_processor
        )
        # Mock the logger methods
        instance.log = MagicMock()
        instance.log_dict = MagicMock()

        loss = instance._shared_step(dummy_batch, 'train')

        assert torch.is_tensor(loss)
        mock_model.assert_called_once_with(dummy_batch['image'])
        instance.log.assert_called_with('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        instance.log_dict.assert_called_once()
        
    def test_predict_step(self, mock_model, base_config, batch_processor, dummy_batch):
        """Verify the predict step returns softmax probabilities."""
        instance = ConcreteTestModel(
            model=mock_model, num_classes=5, config=base_config, batch_processor=batch_processor
        )
        predictions = instance.predict_step(dummy_batch, 0)

        assert torch.is_tensor(predictions)
        assert predictions.shape == (2, 5)
        # Check that probabilities sum to 1 for each sample in the batch
        assert torch.allclose(predictions.sum(dim=1), torch.tensor([1.0, 1.0]))

    @pytest.mark.parametrize("optimizer_name", ["Adam", "AdamW", "SGD"])
    def test_configure_optimizers_selection(self, mock_model, batch_processor, optimizer_name):
        """Verify correct optimizer instantiation."""
        config = BaseModelConfig(optimizer=optimizer_name, use_scheduler=False)
        instance = ConcreteTestModel(mock_model, 5, config, batch_processor)
        
        opt_config = instance.configure_optimizers()
        assert isinstance(opt_config['optimizer'], getattr(torch.optim, optimizer_name))

    def test_configure_optimizers_cosine_scheduler_error(self, mock_model, base_config, batch_processor):
        """Verify it raises an error if T_max is missing for cosine scheduler."""
        # Config is valid, but scheduler_params is missing T_max
        config = BaseModelConfig(scheduler_type="cosine", scheduler_params={})
        instance = ConcreteTestModel(mock_model, 5, config, batch_processor)
        
        with pytest.raises(ValueError, match="`T_max` must be provided"):
            instance.configure_optimizers()
