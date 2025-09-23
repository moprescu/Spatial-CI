import timm
import torch
import json
from pathlib import Path

model_name = "resnet18"
model = timm.create_model(model_name, pretrained=True)

state_dict = model.state_dict()

# Remove the final fully connected layer weights (usually under 'fc.')
filtered_state_dict = {
    k: v for k, v in state_dict.items()
    if not k.startswith("fc.")
}

save_dir = Path(f"./{model_name}_local")
save_dir.mkdir(exist_ok=True)

torch.save(filtered_state_dict, save_dir / "pytorch_model.bin")

config = {
    "architecture": model_name,
    "num_features": model.num_features,
    "pretrained_cfg": model.default_cfg
}
with open(save_dir / "config.json", "w") as f:
    json.dump(config, f)
