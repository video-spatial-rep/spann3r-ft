# import torch
# import os
# import torch.nn as nn
# from llava.model.builder import load_pretrained_model

# # === Load LLaVA-Video Weights ===
# def load_llava_checkpoint(llava_checkpoint_path):
#     print("🚀 Loading LLaVA-Video-7B-Qwen2...")
#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         model_path=llava_checkpoint_path,
#         model_base=None,
#         model_name=os.path.basename(llava_checkpoint_path),
#         load_8bit=False
#     )
#     print("✅ LLaVA model loaded successfully!")
#     return model.state_dict()

# # === Load Spann3r Weights ===
# def load_spann3r_checkpoint(spann3r_checkpoint_path):
#     print("🚀 Loading Spann3r...")
#     checkpoint = torch.load(spann3r_checkpoint_path, map_location="cpu")
    
#     return checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

# # === Replace Vision Encoder with Projection & Layer Matching ===
# def replace_vision_encoder(spann3r_state_dict, llava_state_dict):
#     """
#     Maps LLaVA vision encoder to Spann3r and applies fixes for mismatched sizes.
#     """
#     llava_vision_keys = {k: v for k, v in llava_state_dict.items() if "vision_tower" in k}

#     # 🔍 Find Spann3r's equivalent ViT keys
#     spann3r_vision_keys = [k for k in spann3r_state_dict.keys() if "vision" in k]
#     print("🔍 Spann3r State Dict Keys:", list(spann3r_state_dict.keys())[:50])

#     print(f"🔍 Spann3r Vision Encoder Keys: {spann3r_vision_keys[:20]}")  # Print first 20 to debug

#     # 🔄 Check if we need a different mapping
#     name_mapping = {
#         k.replace("model.vision_tower.vision_tower.", "model.vision_encoder.")
#         if "model.vision_tower.vision_tower." in k else k
#         for k in llava_vision_keys
#     }

#     # 🔄 Attempt to match keys dynamically
#     for llava_key, spann3r_key in name_mapping.items():
#         if spann3r_key in spann3r_state_dict:
#             spann3r_state_dict[spann3r_key] = llava_state_dict[llava_key]
#         else:
#             print(f"⚠️ Key {spann3r_key} not found in Spann3r! Skipping...")

#     return spann3r_state_dict

# # === Truncate or Extend Transformer Blocks ===
# def match_transformer_blocks(spann3r_state_dict, llava_state_dict):
#     """
#     Truncate or copy layers if Spann3r and LLaVA-Video have different numbers of transformer blocks.
#     """
#     llava_layers = [k for k in llava_state_dict.keys() if "vision_model.encoder.layers" in k]
#     spann3r_layers = [k for k in spann3r_state_dict.keys() if "vision_model.encoder.layers" in k]

#     num_llava_layers = len(set([k.split(".")[4] for k in llava_layers]))
#     num_spann3r_layers = len(set([k.split(".")[4] for k in spann3r_layers]))

#     if num_llava_layers > num_spann3r_layers:
#         print(f"⚠️ LLaVA has {num_llava_layers} layers, Spann3r has {num_spann3r_layers}. Truncating extra layers.")
#         for i in range(num_spann3r_layers, num_llava_layers):
#             layer_key = f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}"
#             if layer_key in llava_state_dict:
#                 del llava_state_dict[layer_key]

#     elif num_llava_layers < num_spann3r_layers:
#         print(f"⚠️ LLaVA has {num_llava_layers} layers, Spann3r has {num_spann3r_layers}. Copying last layer.")
#         last_layer = num_llava_layers - 1
#         for i in range(num_llava_layers, num_spann3r_layers):
#             for param in ["self_attn.k_proj.weight", "self_attn.k_proj.bias", "self_attn.v_proj.weight",
#                           "self_attn.v_proj.bias", "self_attn.q_proj.weight", "self_attn.q_proj.bias",
#                           "self_attn.out_proj.weight", "self_attn.out_proj.bias", "layer_norm1.weight",
#                           "layer_norm1.bias", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight",
#                           "mlp.fc2.bias", "layer_norm2.weight", "layer_norm2.bias"]:
#                 spann3r_key = f"model.vision_encoder.vision_model.encoder.layers.{i}.{param}"
#                 llava_key = f"model.vision_tower.vision_tower.vision_model.encoder.layers.{last_layer}.{param}"
#                 spann3r_state_dict[spann3r_key] = llava_state_dict[llava_key].clone()

#     return spann3r_state_dict

# # === Save Updated Spann3r Model ===
# def save_new_checkpoint(spann3r_state_dict, output_path):
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     print(f"\n💾 Saving modified Spann3r checkpoint to {output_path}...")
#     torch.save({"state_dict": spann3r_state_dict}, output_path)
#     print("✅ New checkpoint saved successfully!")

# # === Main ===
# def main():
#     spann3r_checkpoint_path = "/home/rilyn/project-files/02-pj-cambrians/cambrians-prep/LLaVA-NeXT/spann3r/spann3r.pth"
#     llava_checkpoint_path = "/data_new/rilyn/huggingface/checkpoints/LLaVA-Video-7B-Qwen2"
#     output_checkpoint_path = "/data/rilyn/checkpoints/spann3r/new_spann3r.pth"

#     # Load models
#     llava_state_dict = load_llava_checkpoint(llava_checkpoint_path)
#     spann3r_state_dict = load_spann3r_checkpoint(spann3r_checkpoint_path)

#     # Replace vision encoder and fix mismatches
#     spann3r_state_dict = replace_vision_encoder(spann3r_state_dict, llava_state_dict)
#     spann3r_state_dict = match_transformer_blocks(spann3r_state_dict, llava_state_dict)

#     # Save updated Spann3r
#     save_new_checkpoint(spann3r_state_dict, output_checkpoint_path)

# if __name__ == "__main__":
#     main()

import torch
import os
import torch.nn.functional as F
from llava.model.builder import load_pretrained_model

# === File Paths ===
SPANN3R_CHECKPOINT_PATH = "/home/rilyn/project-files/02-pj-cambrians/cambrians-prep/LLaVA-NeXT/spann3r/spann3r.pth"
LLAVA_CHECKPOINT_PATH = "/data_new/rilyn/huggingface/checkpoints/LLaVA-Video-7B-Qwen2"
OUTPUT_CHECKPOINT_PATH = "/home/rilyn/project-files/02-pj-cambrians/cambrians-prep/LLaVA-NeXT/spann3r/new_spann3r.pth"

# === Load LLaVA-Video Weights ===
def load_llava_checkpoint(llava_checkpoint_path):
    print("🚀 Loading LLaVA-Video-7B-Qwen2...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=llava_checkpoint_path,
        model_base=None,
        model_name=os.path.basename(llava_checkpoint_path),
        load_8bit=False
    )
    print("✅ LLaVA model loaded successfully!")
    return model.state_dict()

# === Load Spann3r Checkpoint ===
def load_spann3r_checkpoint(spann3r_checkpoint_path):
    print("\n🚀 Loading Spann3r checkpoint...")
    checkpoint = torch.load(spann3r_checkpoint_path, map_location="cpu")
    return checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

# === Load Models ===
llava_state_dict = load_llava_checkpoint(LLAVA_CHECKPOINT_PATH)
spann3r_state_dict = load_spann3r_checkpoint(SPANN3R_CHECKPOINT_PATH)

# === Identify Encoder Keys ===
llava_encoder_keys = {k: v for k, v in llava_state_dict.items() if "vision_tower" in k}
spann3r_encoder_keys = {k: v for k, v in spann3r_state_dict.items() if "enc_blocks" in k}

print(f"\n🔍 Found {len(llava_encoder_keys)} LLaVA encoder keys.")
print(f"🔍 Found {len(spann3r_encoder_keys)} Spann3r encoder keys.")

# === Mapping LLaVA Encoder to Spann3r Format ===
name_mapping = {}
for k in llava_encoder_keys:
    mapped_key = (
        k.replace("model.vision_tower.vision_tower.", "dust3r.enc_blocks.")
        .replace("vision_model.encoder.layers", "")
        .replace("layer_norm1", "norm1")
        .replace("layer_norm2", "norm2")
        .replace("self_attn.k_proj", "attn.qkv")
        .replace("self_attn.v_proj", "attn.qkv")
        .replace("self_attn.q_proj", "attn.qkv")
        .replace("self_attn.out_proj", "attn.proj")
        .replace("mlp.fc1", "mlp.fc1")
        .replace("mlp.fc2", "mlp.fc2")
    )
    name_mapping[k] = mapped_key

# === Replace Spann3r Encoder Weights ===
transferred_count = 0
for llava_key, spann3r_key in name_mapping.items():
    if spann3r_key in spann3r_state_dict:
        if llava_state_dict[llava_key].shape == spann3r_state_dict[spann3r_key].shape:
            spann3r_state_dict[spann3r_key] = llava_state_dict[llava_key]
            transferred_count += 1
        else:
            print(f"⚠️ Resizing {spann3r_key}: {spann3r_state_dict[spann3r_key].shape} → {llava_state_dict[llava_key].shape}")
            spann3r_state_dict[spann3r_key] = F.interpolate(
                llava_state_dict[llava_key].unsqueeze(0), 
                size=spann3r_state_dict[spann3r_key].shape[1:], 
                mode="bilinear", 
                align_corners=False
            ).squeeze(0)
            transferred_count += 1
    else:
        print(f"⚠️ Key {spann3r_key} not found in Spann3r! Skipping...")

print(f"\n✅ Successfully transferred {transferred_count} vision encoder weights!")

# === Save Updated Spann3r Checkpoint ===
os.makedirs(os.path.dirname(OUTPUT_CHECKPOINT_PATH), exist_ok=True)
torch.save({"state_dict": spann3r_state_dict}, OUTPUT_CHECKPOINT_PATH)
print(f"💾 Saved updated Spann3r checkpoint to {OUTPUT_CHECKPOINT_PATH}")
