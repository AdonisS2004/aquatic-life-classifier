import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Primary Convolution building block for Neural Network. Process:
        2D Convolution -> 2D Batch Normalization -> ReLU -> Dropout (Optional)
    
    Explanation:
        - 2D Convolution: Feature detection useful images (Bias=False for reduced complexity with Batch Norm)
        - 2D Batch Normalization: Mathematical normalizer to improve training and prevent vanishing/exploding gradients.
        - ReLU: Introduces Non-Linearity and allows for complex pattern recognition
        - Dropout (Optional): Prevents overfitting
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1,  dropout = 0.0):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout else None
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
        return x
    
class FeatureExtractor(nn.Module):
    """
    (Insert Description)
    """
    def __init__(self):
        raise NotImplementedError
    
class Classifier(nn.Module):
    """
    (Insert Description)
    """
    def __init__(self):
        raise NotImplementedError
    
