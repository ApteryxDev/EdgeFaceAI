"""Student age/gender model. No FastFace face-attribute weights are loaded."""
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights

class StudentFaceModel(nn.Module):
 def __init__(self,imagenet=True,age_bins=117):
  super().__init__()
  base=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if imagenet else None)
  self.features=base.features;self.pool=base.avgpool
  self.shared=nn.Sequential(nn.Linear(576,256),nn.Hardswish(),nn.Dropout(.2))
  self.gender=nn.Linear(256,2);self.ages=nn.Linear(256,age_bins)
  self.register_buffer('age_values',torch.arange(age_bins,dtype=torch.float32))
 def forward(self,x):
  x=self.shared(torch.flatten(self.pool(self.features(x)),1));gender=self.gender(x);ages=self.ages(x)
  age=(ages.softmax(1)*self.age_values).sum(1)
  return gender,ages,age
