import torch.nn as nn
import torchsummary as summary
import utils.blocks as blocks

class AquaticLifeCNN(nn.Module):
    def __init__(self, num_classes = 46):
        super(AquaticLifeCNN, self).__init__()

        self.num_classes = num_classes
        self.feature_extractor = blocks.FeatureExtractor()
        self.classifier = blocks.Classifier()

    def foward(self, x):
        raise NotImplementedError

    def _initialize_weights(self):
        raise NotImplementedError