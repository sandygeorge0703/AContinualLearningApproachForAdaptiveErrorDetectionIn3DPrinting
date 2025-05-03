from avalanche.models import IncrementalClassifier
from avalanche.models.dynamic_modules import DynamicModule
import torch.nn as nn

class SimpleCNN(DynamicModule):
    """
    Convolutional Neural Network with EWC strategy.
    Initially set for 0 classes.
    """
    def __init__(self, num_classes=0):  # Start with 0 classes; they’ll be added dynamically
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Conv Layer 1
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),           # Conv Layer 2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # Max Pooling 1
            nn.Conv2d(32, 64, kernel_size=3, padding=1),           # Conv Layer 3
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),           # Conv Layer 4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # Max Pooling 2
            nn.Conv2d(64, 64, kernel_size=1, padding=0),           # Conv Layer 5
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),                               # Adaptive Pooling
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(64, num_classes)                             # Classifier
        # )
        
        # 🔁 Use IncrementalClassifier
        self.classifier = IncrementalClassifier(in_features=64, initial_out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Old method to update the classifier for the new number of classes
def update_classifier(model, new_num_classes):
    """
    Updates the model's classifier to support new_num_classes outputs.
    Copies over the weights for the existing classes and randomly initialises the new class.
    """
    # Get the old classifier layer (assumed to be the last module in the classifier)
    old_layer = model.classifier[-1]
    in_features = old_layer.in_features
    old_num_classes = old_layer.out_features

    # Create a new linear layer with updated output dimensions.
    new_layer = nn.Linear(in_features, new_num_classes)

    # Copy existing weights and biases from the old classifier to the new one
    with torch.no_grad():
        new_layer.weight[:old_num_classes] = old_layer.weight
        new_layer.bias[:old_num_classes] = old_layer.bias

    # Replace the old layer in the classifier with the new layer.
    model.classifier[-1] = new_layer
    return model