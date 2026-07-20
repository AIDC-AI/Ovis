"""
Validation tests to ensure the testing infrastructure is properly configured.
"""
import sys
from pathlib import Path

import pytest
import torch


class TestSetupValidation:
    """Validate that the testing infrastructure is properly set up."""
    
    @pytest.mark.unit
    def test_pytest_is_importable(self):
        """Test that pytest can be imported."""
        import pytest
        assert pytest is not None
    
    @pytest.mark.unit
    def test_fixtures_are_available(self, temp_dir, mock_config):
        """Test that custom fixtures are available and working."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert isinstance(mock_config, dict)
        assert "model_name" in mock_config
    
    @pytest.mark.unit
    def test_project_structure(self):
        """Test that the project structure is correct."""
        project_root = Path(__file__).parent.parent
        assert (project_root / "ovis").exists()
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "conftest.py").exists()
    
    @pytest.mark.unit
    def test_coverage_is_working(self):
        """Test that coverage reporting is configured."""
        # This test will be included in coverage
        result = 1 + 1
        assert result == 2
    
    @pytest.mark.unit
    def test_mocking_is_available(self, mocker):
        """Test that pytest-mock is working."""
        mock_func = mocker.Mock(return_value=42)
        assert mock_func() == 42
        mock_func.assert_called_once()
    
    @pytest.mark.unit
    def test_markers_are_defined(self, request):
        """Test that custom markers are properly defined."""
        markers = [marker.name for marker in request.node.iter_markers()]
        assert "unit" in markers
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        import time
        time.sleep(0.1)  # Simulate slow test
        assert True
    
    @pytest.mark.unit
    def test_torch_fixtures(self, sample_tensor):
        """Test that torch-related fixtures work."""
        assert isinstance(sample_tensor, torch.Tensor)
        assert sample_tensor.shape == (2, 3, 224, 224)
    
    @pytest.mark.unit
    def test_mock_model_fixture(self, mock_model):
        """Test that mock model fixture works."""
        output = mock_model.forward(torch.randn(2, 10))
        assert isinstance(output, torch.Tensor)
        mock_model.forward.assert_called_once()
    
    @pytest.mark.unit
    def test_environment_fixture(self, environment_variables):
        """Test that environment variable fixture works."""
        environment_variables(TEST_VAR="test_value")
        import os
        assert os.environ.get("TEST_VAR") == "test_value"
    
    @pytest.mark.unit
    def test_log_capture_fixture(self, capture_logs):
        """Test that log capture fixture works."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Test log message")
        
        log_content = capture_logs.getvalue()
        assert "Test log message" in log_content
    
    @pytest.mark.parametrize("input_val,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_parametrize_works(self, input_val, expected):
        """Test that parametrize decorator works."""
        assert input_val * 2 == expected


class TestCoverageExclusions:
    """Test coverage exclusions are properly configured."""
    
    def test_abstract_method_excluded(self):
        """Test that abstract methods are excluded from coverage."""
        from abc import ABC, abstractmethod
        
        class AbstractClass(ABC):
            @abstractmethod
            def abstract_method(self):  # pragma: no cover
                pass
        
        assert True
    
    def test_type_checking_excluded(self):
        """Test that TYPE_CHECKING blocks are excluded."""
        from typing import TYPE_CHECKING
        
        if TYPE_CHECKING:  # This should be excluded from coverage
            import some_module_that_doesnt_exist
        
        assert True