"""
Tests for coexistence of pseudo-streaming and non-streaming keepalive modes.
"""
import pytest
from src.config import (
    is_pseudo_streaming_model, 
    get_base_model_name, 
    NONSTREAM_KEEPALIVE_ENABLED,
    PSEUDO_STREAMING_HEARTBEAT_INTERVAL
)


class TestCoexistenceLogic:
    """Test that both keepalive modes can coexist properly."""
    
    def test_model_type_detection(self):
        """Test that different model types are detected correctly."""
        # Regular models (should use true streaming)
        regular_models = [
            "gemini-2.5-pro",
            "models/gemini-2.5-flash", 
            "gemini-2.5-pro-search",
            "gemini-2.5-pro-nothinking"
        ]
        
        for model in regular_models:
            assert not is_pseudo_streaming_model(model), f"{model} should not be pseudo-streaming"
        
        # Pseudo-streaming models
        pseudo_models = [
            "gemini-2.5-pro-伪流",
            "models/gemini-2.5-flash-伪流",
            "gemini-2.5-pro-preview-06-05-伪流"
        ]
        
        for model in pseudo_models:
            assert is_pseudo_streaming_model(model), f"{model} should be pseudo-streaming"
    
    def test_request_behavior_matrix(self):
        """Test the behavior matrix for different model types and request types."""
        test_cases = [
            # (model, is_streaming, expected_behavior)
            ("gemini-2.5-pro", True, "true_streaming"),
            ("gemini-2.5-pro", False, "nonstream_keepalive_or_normal"),
            ("gemini-2.5-pro-伪流", True, "pseudo_streaming"), 
            ("gemini-2.5-pro-伪流", False, "nonstream_keepalive_or_normal"),
        ]
        
        for model, is_streaming, expected in test_cases:
            is_pseudo = is_pseudo_streaming_model(model)
            
            if is_streaming and not is_pseudo:
                behavior = "true_streaming"
            elif is_streaming and is_pseudo:
                behavior = "pseudo_streaming"
            else:
                behavior = "nonstream_keepalive_or_normal"
            
            assert behavior == expected, f"Model {model}, streaming={is_streaming} should be {expected}, got {behavior}"
    
    def test_configuration_independence(self):
        """Test that pseudo-streaming and non-streaming keepalive configs are independent."""
        # Pseudo-streaming is controlled by model suffix, not env vars
        assert callable(is_pseudo_streaming_model), "Pseudo-streaming detection should be function-based"
        
        # Non-streaming keepalive is controlled by env var
        assert isinstance(NONSTREAM_KEEPALIVE_ENABLED, bool), "Non-streaming keepalive should be boolean config"
        
        # Pseudo-streaming only has heartbeat interval config
        assert isinstance(PSEUDO_STREAMING_HEARTBEAT_INTERVAL, float), "Pseudo-streaming interval should be float"


class TestModelCoverage:
    """Test that all expected models are covered."""
    
    def test_all_gemini_25_models_have_pseudo_variants(self):
        """Ensure all gemini-2.5 models have pseudo-streaming variants."""
        from src.config import SUPPORTED_MODELS
        
        # Find all base gemini-2.5 models (without variants)
        base_gemini_25 = []
        pseudo_gemini_25 = []
        
        for model in SUPPORTED_MODELS:
            name = model["name"]
            if "gemini-2.5" in name:
                if "-伪流" in name:
                    pseudo_gemini_25.append(name)
                elif not any(suffix in name for suffix in ["-search", "-nothinking", "-maxthinking"]):
                    base_gemini_25.append(name)
        
        # Each base model should have a corresponding pseudo variant
        for base_model in base_gemini_25:
            expected_pseudo = base_model + "-伪流"
            assert expected_pseudo in pseudo_gemini_25, f"Missing pseudo variant for {base_model}"
        
        # Each pseudo variant should have a corresponding base model
        for pseudo_model in pseudo_gemini_25:
            base_model = pseudo_model.replace("-伪流", "")
            assert base_model in base_gemini_25, f"Pseudo variant {pseudo_model} has no base model"


class TestBackwardCompatibility:
    """Test that existing functionality is preserved."""
    
    def test_regular_models_unchanged(self):
        """Test that regular models work the same as before."""
        regular_models = [
            "gemini-2.5-pro",
            "gemini-2.5-flash", 
            "gemini-2.5-pro-search",
            "gemini-2.5-pro-nothinking"
        ]
        
        for model in regular_models:
            # Should not be detected as pseudo-streaming
            assert not is_pseudo_streaming_model(model)
            
            # Base model extraction should work
            base = get_base_model_name(model)
            assert base is not None
            assert len(base) > 0
    
    def test_non_streaming_keepalive_preserved(self):
        """Test that non-streaming keepalive configuration is preserved."""
        # Configuration should exist
        assert hasattr(__import__('src.config', fromlist=['NONSTREAM_KEEPALIVE_ENABLED']), 'NONSTREAM_KEEPALIVE_ENABLED')
        assert hasattr(__import__('src.config', fromlist=['NONSTREAM_KEEPALIVE_INTERVAL']), 'NONSTREAM_KEEPALIVE_INTERVAL')
        
        # Should be independent of pseudo-streaming
        from src.config import NONSTREAM_KEEPALIVE_INTERVAL
        assert isinstance(NONSTREAM_KEEPALIVE_INTERVAL, float)
        assert NONSTREAM_KEEPALIVE_INTERVAL > 0


if __name__ == "__main__":
    pytest.main([__file__])