import torch
import argparse

torch.serialization.add_safe_globals([argparse.Namespace])

checkpoint_path = "/home/rilyn/project-files/02-pj-cambrians/cambrians-prep/LLaVA-NeXT/spann3r/new_spann3r_2.pth"
save_path = "/home/rilyn/project-files/02-pj-cambrians/cambrians-prep/LLaVA-NeXT/spann3r/new_spann3r_f.pth"

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

if "state_dict" in checkpoint and "model" in checkpoint["state_dict"]:
    new_checkpoint = {
        "args": checkpoint.get("args", {}),  # Preserve args if they exist
        "model": checkpoint["state_dict"]["model"]  # Extract model weights
    }
    torch.save(new_checkpoint, save_path)
    print(f"Saved new checkpoint with 'args' and 'model' to {save_path}")
else:
    print("Error: Could not find 'model' in checkpoint['state_dict']")
