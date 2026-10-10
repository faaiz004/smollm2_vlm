# Defining the connecter
import torch.nn as nn
import torch
# every neural network is a nn module
class Connector(nn.Module):
  # it is transforming data from clip dim to smollm dim
  def __init__(self, clip_dim = 768, lm_dim = 960):
     super().__init__()
     self.mlp = nn.Sequential(
          nn.Linear(clip_dim, lm_dim),
          nn.GELU(),
          nn.Linear(lm_dim, lm_dim),
     )
     # This is doing the kaiming init the assignments was asking for
     self. _init_weights()
     self.scale_factor = 1.0

  def _init_weights(self):
      for m in self.mlp:
          if isinstance(m, nn.Linear):
              nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
              nn.init.zeros_(m.bias)

  # Now we need to write the forward function
  def forward(self, x):
    return self.mlp(x) * self.scale_factor

# # connector = Connector()
# # connector = connector.to(device).half()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# connector = Connector().to(device)
# print(f"Trainable params: {sum(p.numel() for p in connector.parameters()):,}")
