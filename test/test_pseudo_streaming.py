"""
Tests for pseudo-streaming functionality.
"""
import pytest
from src.config import is_pseudo_streaming_model, get_base_model_name, PSEUDO_STREAMING_HEARTBEAT_INTERVAL


class TestPseudoStreamingDetection:
    """Test pseudo-streaming model detection logic."""
    
    def test_pseudo_streaming_model_detection(self):
        """Test that pseudo-streaming models are correctly identified."""
        test_cases = [
            ("gemini-2.5-pro-伪流", True),
            ("models/gemini-2.5-pro-伪流", True),
            ("gemini-2.5-flash-伪流", True),
            ("gemini-2.5-pro", False),
            ("models/gemini-2.5-pro", False),
            ("gemini-1.5-pro", False),
            ("gemini-2.5-pro-search", False),
            ("gemini-2.5-pro-nothinking", False),
        ]
        
        for model_name, expected in test_cases:
            result = is_pseudo_streaming_model(model_name)
            assert result == expected, f"Model {model_name} should be {expected}, got {result}"
    
    def test_base_model_name_extraction(self):
        """Test that base model names are correctly extracted."""
        test_cases = [
            ("gemini-2.5-pro-伪流", "gemini-2.5-pro"),
            ("models/gemini-2.5-pro-伪流", "models/gemini-2.5-pro"),
            ("gemini-2.5-flash-search-伪流", "gemini-2.5-flash-search"),
            ("gemini-2.5-pro", "gemini-2.5-pro"),
            ("models/gemini-2.5-pro", "models/gemini-2.5-pro"),
        ]
        
        for model_name, expected in test_cases:
            result = get_base_model_name(model_name)
            assert result == expected, f"Base model for {model_name} should be {expected}, got {result}"


class TestPseudoStreamingConfig:
    """Test pseudo-streaming configuration."""
    
    def test_heartbeat_interval_config(self):
        """Test that heartbeat interval is properly configured."""
        assert isinstance(PSEUDO_STREAMING_HEARTBEAT_INTERVAL, float)
        assert PSEUDO_STREAMING_HEARTBEAT_INTERVAL > 0
        assert PSEUDO_STREAMING_HEARTBEAT_INTERVAL <= 10  # Reasonable upper bound


class TestModelGeneration:
    """Test model generation and coverage."""
    
    def test_pseudo_streaming_variants_generation(self):
        """Test that pseudo-streaming variants are generated for all gemini-2.5 models."""
        from src.config import BASE_MODELS, _generate_pseudo_streaming_variants
        
        # Get all gemini-2.5 base models
        gemini_25_models = [
            model for model in BASE_MODELS 
            if "gemini-2.5" in model["name"] and "generateContent" in model["supportedGenerationMethods"]
        ]
        
        # Generate pseudo variants
        pseudo_variants = _generate_pseudo_streaming_variants()
        
        # Check that each gemini-2.5 model has a pseudo variant
        for base_model in gemini_25_models:
            expected_pseudo_name = base_model["name"] + "-伪流"
            found = any(variant["name"] == expected_pseudo_name for variant in pseudo_variants)
            assert found, f"Pseudo variant not found for {base_model['name']}"
        
        # Check that all pseudo variants are for gemini-2.5 models
        for variant in pseudo_variants:
            assert "gemini-2.5" in variant["name"], f"Non-gemini-2.5 pseudo variant: {variant['name']}"
            assert variant["name"].endswith("-伪流"), f"Pseudo variant should end with -伪流: {variant['name']}"


class TestModelNameProcessing:
    """Test model name processing in request flow."""
    
    def test_openai_to_gemini_model_preservation(self):
        """Test that original model names are preserved in OpenAI transformation."""
        # This would require mocking the OpenAI request, simplified test
        test_model = "gemini-2.5-pro-伪流"
        
        # Test the detection logic that would be used
        assert is_pseudo_streaming_model(test_model) == True
        assert get_base_model_name(test_model) == "gemini-2.5-pro"
    
    def test_gemini_native_model_processing(self):
        """Test that native Gemini requests preserve model names."""
        test_model = "gemini-2.5-flash-伪流"
        
        # Test the detection logic for native requests
        assert is_pseudo_streaming_model(test_model) == True
        assert get_base_model_name(test_model) == "gemini-2.5-flash"


if __name__ == "__main__":
    pytest.main([__file__])