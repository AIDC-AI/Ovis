"""
Shared pytest fixtures and configuration for all tests.
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary."""
    return {
        "model_name": "test_model",
        "model_version": "1.0.0",
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_epochs": 10,
        "device": "cpu",
        "seed": 42,
        "output_dir": "/tmp/test_output"
    }


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Provide a sample tensor for testing."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def sample_text_data() -> list:
    """Provide sample text data for testing."""
    return [
        "This is a test sentence.",
        "Another example for testing.",
        "Machine learning is fascinating."
    ]


@pytest.fixture
def mock_model():
    """Provide a mock model object."""
    model = MagicMock()
    model.forward = MagicMock(return_value=torch.randn(2, 10))
    model.eval = MagicMock()
    model.train = MagicMock()
    model.parameters = MagicMock(return_value=[torch.randn(10, 10)])
    return model


@pytest.fixture
def mock_tokenizer():
    """Provide a mock tokenizer object."""
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(return_value=[101, 2023, 2003, 102])
    tokenizer.decode = MagicMock(return_value="This is decoded text")
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 102
    return tokenizer


@pytest.fixture
def sample_image_path(temp_dir: Path) -> Path:
    """Create a temporary image file for testing."""
    import numpy as np
    from PIL import Image
    
    image_path = temp_dir / "test_image.png"
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(image_path)
    return image_path


@pytest.fixture
def environment_variables():
    """Temporarily set environment variables for testing."""
    original_env = os.environ.copy()
    
    def _set_env(**kwargs):
        os.environ.update(kwargs)
        return os.environ
    
    yield _set_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_api_response():
    """Provide a mock API response."""
    return {
        "status": "success",
        "data": {
            "id": "12345",
            "result": "Test result",
            "metadata": {
                "timestamp": "2024-01-01T00:00:00Z",
                "version": "1.0"
            }
        }
    }


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def capture_logs():
    """Capture log messages during tests."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    yield log_capture
    
    logger.removeHandler(handler)


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "network: mark test as requiring network access"
    )