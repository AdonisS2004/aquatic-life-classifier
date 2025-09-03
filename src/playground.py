# IMPORTANT: This file is used to play around with this project

# import torch
# import torch.nn as nn

# # Load the model
# model = torch.load('models/best_model.pth', map_location='cpu')

# # Print the model structure
# print("Model keys:", model.keys() if isinstance(model, dict) else "Not a dict")
# print("\nModel type:", type(model))

# # If it's a state dict
# if isinstance(model, dict):
#     print("\nState dict keys:")
#     for key in model.keys():
#         print(f"  {key}: {model[key].shape if hasattr(model[key], 'shape') else type(model[key])}")
    
#     # Show some parameter details
#     print("\nParameter details:")
#     for key, value in model.items():
#         if isinstance(value, torch.Tensor):
#             print(f"  {key}: {value.shape}, dtype={value.dtype}")
#         else:
#             print(f"  {key}: {type(value)}")

# # If it's a full model
# elif hasattr(model, 'state_dict'):
#     print("\nModel architecture:")
#     print(model)
    
#     print("\nModel parameters:")
#     for name, param in model.named_parameters():
#         print(f"  {name}: {param.shape}")


import torch

# Specify the path to your .pth file
file_path = 'models/best_model.pth'

try:
    # Load the contents of the .pth file
    # map_location can be used to load to CPU or a specific GPU
    loaded_content = torch.load(file_path, map_location=torch.device('cpu')) 

    # The loaded_content will often be a dictionary (state_dict)
    # or the entire model object itself, depending on how it was saved.
    print("Contents loaded successfully.")
    
    # If it's a state_dict, you can print its keys to see what's inside
    if isinstance(loaded_content, dict):
        print("Keys in the loaded dictionary (state_dict):")
        for key in loaded_content.keys():
            if key in {"val_acc", "epoch", "history"}:
                print(key, loaded_content[key])

        
        # You can access specific parts, e.g., model weights
        # Example: if 'model_state_dict' is a key, access it like this:
        # model_weights = loaded_content['model_state_dict']

    # If it's an entire model, you might print the model structure
    elif isinstance(loaded_content, torch.nn.Module):
        print("Loaded content is an entire PyTorch model:")
        print(loaded_content)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred while loading the file: {e}")