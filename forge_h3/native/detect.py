"""Recognize a MiniMax H3 DiT and read its configuration from tensor shapes.

Ported from ComfyUI comfy/model_detection.py. Works on a loaded state dict or on a safetensors header: shapes are
read through shape_of, so no tensor has to be materialized.
"""

import json
import re

IDENTITY_KEYS = ("video_patch_proj.weight", "audio_patch_proj.weight")


def shape_of(value):
    # a torch tensor, or a safetensors header entry {"dtype", "shape", "data_offsets"}
    if isinstance(value, dict):
        return tuple(value["shape"])
    return tuple(value.shape)


def count_blocks(keys, prefix):
    pattern = re.compile(re.escape(prefix) + r"(\d+)\.")
    found = {int(m.group(1)) for k in keys if (m := pattern.match(k))}
    return max(found) + 1 if found else 0


def is_minimax_h3(state_dict, key_prefix=""):
    return all(f"{key_prefix}{k}" in state_dict for k in IDENTITY_KEYS)


def detect(state_dict, key_prefix="", metadata=None):
    """The MiniMaxH3Model keyword arguments for this checkpoint, or None when it is not an H3 DiT."""
    if not is_minimax_h3(state_dict, key_prefix):
        return None

    def shape(key):
        return shape_of(state_dict[f"{key_prefix}{key}"])

    keys = list(state_dict.keys())
    config = {"image_model": "minimax_h3"}
    config["num_layers"] = count_blocks(keys, f"{key_prefix}blocks.")
    config["token_refiner_num_layers"] = count_blocks(keys, f"{key_prefix}token_refiner.blocks.")
    config["hidden_size"] = shape("video_patch_proj.weight")[0]
    config["latents_dim"] = shape("final_layer.video_out.weight")[0] // 4  # patch 1x2x2
    config["audio_latents_dim"] = shape("final_layer.audio_out.weight")[0]
    config["attention_head_dim"] = shape("blocks.0.attn.q_norm.weight")[0]
    config["num_attention_heads"] = shape("blocks.0.attn.qkv_proj.weight")[0] // (3 * config["attention_head_dim"])
    config["ffn_hidden_size"] = shape("blocks.0.mlp.fc1.weight")[0] // 2
    config["text_dim"] = shape("condition_proj.weight")[1]
    if f"{key_prefix}adaln_t_table" in state_dict:
        # pruned build: the adaln linears span a small shared basis of the time-embedding curve, no time embedder
        grid, basis = shape("adaln_t_table")
        config["adaln_curve_grid"] = grid
        config["time_embed_dim"] = basis
    else:
        proj_in = shape("time_embedder.proj_in.weight")
        config["timestep_input_dim"] = proj_in[1]
        config["time_embed_hidden_size"] = proj_in[0]
        config["time_embed_dim"] = shape("time_embedder.proj_out.weight")[0]
    config["rope_inv_freq_len"] = shape("rope.inv_freq")[0]
    config["gate_compress"] = f"{key_prefix}blocks.0.attn.to_gate_compress.weight" in state_dict  # VSA-trained
    if metadata and "config" in metadata:
        config.update(json.loads(metadata["config"]).get("transformer", {}))
    return config
