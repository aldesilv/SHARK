import torch
from mlir.ir import Context, Module
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
from shark.shark_inference import SharkInference
from shark.shark_importer import SharkImporter

class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input = torch.randn(1, 3, 32, 32)

mlir_importer = SharkImporter(
    MNIST(),
    (input,),
    frontend="torch",
)

(vision_mlir, func_name), inputs, golden_out = mlir_importer.import_debug(
    tracing_required=True
)

with Context() as ctx:
  module = Module.parse(vision_mlir)
  print(module)

