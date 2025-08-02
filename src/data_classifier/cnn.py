import torch
import torch.nn as nn
import torchsummary as summary
import utils.blocks as blocks
import logging

class AquaticLifeCNN(nn.Module):
    """
    Aquatic Life Classifiying Model
    """
    def __init__(self, num_classes = 46):
        super(AquaticLifeCNN, self).__init__()
        self.num_classes = num_classes
        self.feature_extractor = blocks.FeatureExtractor()
        self.classifier = blocks.Classifier(num_classes=self.num_classes)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def get_feature_maps(self, x):
        """
        Extract intermediate feature maps for visualization
        """
        features = {}
        
        x = self.feature_extractor.stage1(x)
        features['stage1'] = x
        
        x = self.feature_extractor.stage2(x)
        features['stage2'] = x
        
        x = self.feature_extractor.stage3(x)
        features['stage3'] = x
        
        x = self.feature_extractor.stage4(x)
        features['stage4'] = x
        
        return features
    
def create_model(num_classes:int=46, device:str='cuda', logger:logging.Logger = None):
    """
    Factory function to create and initialize the model
    """
    model = AquaticLifeCNN(num_classes=num_classes)
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        if logger: logger.info(f"Model moved to GPU: {torch.cuda.get_device_name()}")
        else: print(f"Model moved to GPU: {torch.cuda.get_device_name()}")
    else: 
        if logger: logger.info("Model running on CPU")
        else: print("Model running on CPU")
    return model

def count_parameters(model, logger:logging.Logger = None):
    """
    Count trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if logger:
        logger.info(f"Total parameters: {total:,}")
        logger.info(f"Trainable parameters: {trainable:,}")
        logger.info(f"Model size: {total * 4 / (1024**2):.2f} MB")  # Assuming float32
    else:
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Model size: {total * 4 / (1024**2):.2f} MB")  # Assuming float32
    
    return total, trainable

def model_summary(logger: logging.Logger = None):
    """
    Print detailed model summary
    """
    model = create_model(num_classes=46, device='cpu')

    if logger:
        logger.info("Detailed Model Architecture:")
        logger.info("="*50)
    else:
        print("Detailed Model Architecture:")
        print("="*50)
    
    # Use torchsummary if available
    try:
        summary(model, (3, 224, 224))
    except ImportError:
        if logger:
            logger.info("Install torchsummary for detailed summary: pip install torchsummary")
            logger.info(model)
        else:
            print("Install torchsummary for detailed summary: pip install torchsummary")
            print(model)