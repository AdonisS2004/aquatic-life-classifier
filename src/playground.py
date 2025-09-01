# IMPORTANT: This file is used to play around with this project

import torch
import torch.nn as nn

# Load the model
model = torch.load('models/best_model.pth', map_location='cpu')

# Print the model structure
print("Model keys:", model.keys() if isinstance(model, dict) else "Not a dict")
print("\nModel type:", type(model))

# If it's a state dict
if isinstance(model, dict):
    print("\nState dict keys:")
    for key in model.keys():
        print(f"  {key}: {model[key].shape if hasattr(model[key], 'shape') else type(model[key])}")
    
    # Show some parameter details
    print("\nParameter details:")
    for key, value in model.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")

# If it's a full model
elif hasattr(model, 'state_dict'):
    print("\nModel architecture:")
    print(model)
    
    print("\nModel parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")