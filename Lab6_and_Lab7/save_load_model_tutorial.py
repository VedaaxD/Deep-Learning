import torch
import torchvision.models as models

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()
torch.save(model, 'model.pth')
model = torch.load('model.pth', weights_only=False),
# Download a pretrained VGG16.
#
# Save its weights (model_weights.pth).
#
# Create a new VGG16 with random weights.
#
# Load the saved pretrained weights into it.
#
# Switch to eval mode for inference.
#
# Save the full model object (model.pth).
#
# Reload the full model from disk.
# Best practice:
#
# Use state_dict saving/loading if you only care about weights (preferred, more flexible).
#
# Use full model saving/loading if you need exact architecture + weights (less flexible, but convenient).