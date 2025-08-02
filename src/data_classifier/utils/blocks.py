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
        super(ConvBlock, self).__init__()

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
    Feature extraction. Process:
        - Stage 1: Low-level feature extraction
        - stage 2: Mid-level pattern recognition
        - stage 3: High-level feature abstraction
        - stage 4: Deep feature integration
        - Global Average Pooling
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        # Stage 1: Low-level feature extraction (224x224 -> 56x56)
        self.stage1 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=7, stride=2, padding=3),    # 224x224 -> 112x112
            nn.MaxPool2d(2, stride=2),                               # 112x112 -> 56x56
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1),   # 56x56 -> 56x56
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),   # 56x56 -> 56x56
        )
        
        # Stage 2: Mid-level pattern recognition (56x56 -> 28x28)
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),                               # 56x56 -> 28x28
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),  # 28x28 -> 28x28
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1), # 28x28 -> 28x28
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1), # 28x28 -> 28x28
        )
        
        # Stage 3: High-level feature abstraction (28x28 -> 14x14)
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),                               # 28x28 -> 14x14
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1), # 14x14 -> 14x14
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1), # 14x14 -> 14x14
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1), # 14x14 -> 14x14
        )
        
        # Stage 4: Deep feature integration (14x14 -> 7x7)
        self.stage4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),                               # 14x14 -> 7x7
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1), # 7x7 -> 7x7
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1), # 7x7 -> 7x7
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1, dropout=0.1), # 7x7 -> 7x7
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)  # Flatten to (batch_size, 512)
    
class Classifier(nn.Module):
    """
    Classifier for species we are trying to classify
    """
    def __init__(self, num_classes=46, feature_dim=512):
        super(Classifier, self).__init__()
        
        self.stage1 = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.stage2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.stage3 = nn.Sequential(
            nn.Linear(128, num_classes)  # No activation - will use CrossEntropyLoss
        )
        
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x
