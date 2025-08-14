import torch.nn as nn
import torch

class LoraInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, dropout_p=0.05, scale=1.0):
        super().__init__()

        # Original linear layer without bias
        self.linear = nn.Linear(in_features, out_features, bias=False)

        # LoRA adaptation layers
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_p)
        
        # LoRA scaling factor
        self.scale = scale

        # Initialization as per standard LoRA practice
        nn.init.normal_(self.lora_down.weight, std=1/r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        result = self.linear(input)

        # Ensure LoRA path runs in float32 regardless of input dtype
        lora_out = self.lora_up(self.lora_down(input.float()))
        lora_out = self.dropout(lora_out) * self.scale

        return result + lora_out.to(result.dtype)
