# transforms_config.py
from torchvision import transforms

# no augmentation - what you've been using so far
transform_none = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# light augmentation - color + mild geometric, no flip
transform_light = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.ToTensor(),
])

# light augmentation + horizontal flip
transform_light_flip = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

# no transformations on validation
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])