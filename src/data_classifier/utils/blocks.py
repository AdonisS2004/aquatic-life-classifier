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

class ResidualBlock(nn.Module):
    """
    Residual block for better gradient flow and deeper networks
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class FeatureExtractor(nn.Module):
    """
    Improved feature extraction with residual connections and attention mechanisms.
    Process:
        - Stage 1: Low-level feature extraction with residual blocks
        - Stage 2: Mid-level pattern recognition with attention
        - Stage 3: High-level feature abstraction with residual blocks
        - Stage 4: Deep feature integration with attention
        - Global Average Pooling
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stage 1: Low-level feature extraction (56x56 -> 56x56)
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        
        # Stage 2: Mid-level pattern recognition (56x56 -> 28x28)
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128)
            )),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )
        
        # Stage 3: High-level feature abstraction (28x28 -> 14x14)
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, downsample=nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256)
            )),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        )
        
        # Stage 4: Deep feature integration (14x14 -> 7x7)
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2, downsample=nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512)
            )),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        x = self.global_pool(x)
        return x.view(x.size(0), -1)  # Flatten to (batch_size, 512)
    
class Classifier(nn.Module):
    """
    Improved classifier with better regularization and architecture
    """
    def __init__(self, num_classes=46, feature_dim=512):
        super(Classifier, self).__init__()
        
        self.stage1 = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.stage2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.stage3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.stage4 = nn.Sequential(
            nn.Linear(128, num_classes)  # No activation - will use CrossEntropyLoss
        )
        
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
