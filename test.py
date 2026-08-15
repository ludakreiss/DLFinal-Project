# test_model.py
import torch
from model import GeoCNN
from utils import count_parameters

model = GeoCNN()
print("Total trainable parameters:", count_parameters(model))

dummy_input = torch.randn(4, 3, 224, 224)
output = model(dummy_input)
print("Output shape:", output.shape)