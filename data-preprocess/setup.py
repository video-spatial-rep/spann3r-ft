import torch
import os
import torch.nn.functional as F
from llava.model.builder import load_pretrained_model

# === File Paths ===
SPANN3R_CHECKPOINT_PATH = "/home/rilyn/project-files/02-pj-cambrians/cambrians-prep/LLaVA-NeXT/spann3r/spann3r.pth"
LLAVA_CHECKPOINT_PATH = "/data_new/spatial/huggingface/LLaVA-Video-7B-Qwen2"

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

# === Compare Spann3r & LLaVA Encoder Weights Before Replacing ===
print("\n🔍 Comparing weights before replacement...")

diff_count = 0
for llava_key, spann3r_key in name_mapping.items():
    if spann3r_key in spann3r_state_dict:
        llava_weights = llava_state_dict[llava_key]
        spann3r_weights = spann3r_state_dict[spann3r_key]

        if torch.equal(spann3r_weights, llava_weights):
            print(f"✅ {spann3r_key} and {llava_key} are IDENTICAL!")
        else:
            diff_count += 1
            print(f"❌ {spann3r_key} and {llava_key} are DIFFERENT!")
            print(f"🔍 Spann3r {spann3r_key} weights (first 5 values): {spann3r_weights.view(-1)[:5]}")
            print(f"🔍 LLaVA {llava_key} weights (first 5 values): {llava_weights.view(-1)[:5]}")
            print("------------------------------------------------------")
    else:
        print(f"⚠️ {spann3r_key} not found in Spann3r state dict!")

print(f"\n🔍 Total differing weights: {diff_count}")
