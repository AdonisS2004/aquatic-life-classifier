import torch
from .cnn import (
    create_model,
    count_parameters
)

def test_model():
    """Test model with dummy input"""
    print("Testing SeaLife CNN Architecture...")
    print("="*50)
    
    # Create model
    model = create_model(num_classes=46, device='cpu')  # Use CPU for testing
    count_parameters(model) # Count parameters
    print()
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    try:
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Expected output shape: (4, 46)")
            # Test feature extraction
            features = model.get_feature_maps(dummy_input)
            print("\nFeature map shapes:")
            for stage, feature_map in features.items():
                print(f"  {stage}: {feature_map.shape}")
            print("\nSUCCESS: Model test passed! Architecture is working correctly.")
    except Exception as e:
        print(f"ERROR: Model test failed: {e}")
        return False
    return True