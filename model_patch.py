# model_patch.py
import torch
import torch.nn as nn
from lora_layer import LoRALinear

def find_and_patch_linear_layers(
    model,
    target_keywords=None,  # Only Linear files whose names contain these keywords will be patched. None indicates all patches will be patched.
    lora_r=4,
    lora_alpha=1.0,
    lora_dropout=0.0,
    verbose=True
):
    """
    Recursively traverse the model, replacing the matched nn.Linear with LoRALinear.
    By default, only the Linear layer containing target_keywords is patched.
    """
    num_patched = 0
    for name, module in model.named_children():
        # Recursively patch submodules
        patched = find_and_patch_linear_layers(
            module, target_keywords, lora_r, lora_alpha, lora_dropout, verbose=False
        )
        num_patched += patched

        # Replace only nn.Linear
        if isinstance(module, nn.Linear):
            if (target_keywords is None) or any([kw in name for kw in target_keywords]):
                # Record the original parameters
                old_linear = module
                # Create a new LoRA linear layer
                lora_linear = LoRALinear(
                    in_features=old_linear.in_features,
                    out_features=old_linear.out_features,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=old_linear.bias is not None
                )
                # Copy the original weights
                lora_linear.weight.data = old_linear.weight.data.clone()
                if old_linear.bias is not None:
                    lora_linear.bias.data = old_linear.bias.data.clone()
                # Freeze original weights
                lora_linear.weight.requires_grad = False
                if lora_linear.bias is not None:
                    lora_linear.bias.requires_grad = False
                # Some parameters of LoRA are trainable (the default settings are fine).
                # replace
                setattr(model, name, lora_linear)
                num_patched += 1
                if verbose:
                    print(f"[LoRA Patch] Patched: {model.__class__.__name__}.{name}  ({old_linear.in_features}→{old_linear.out_features})")
    return num_patched

def patch_blip1_visual_transformer(model, r=4, alpha=1.0, dropout=0.0, verbose=True):
    """
    Only patch the visual backbone (visual Transformer) of BLIP1.
    Compatible with blip-image-captioning-large (BLIP1)
    """
    # Automatically find the visual backbone（blip1: model.visual_encoder, blip2: model.vision_model, etc）
    visual_attr_candidates = ['visual_encoder', 'vision_model']
    visual_encoder = None
    for attr in visual_attr_candidates:
        if hasattr(model, attr):
            visual_encoder = getattr(model, attr)
            if verbose:
                print(f"[LoRA Patch] Find visual backbone: {attr}")
            break
    assert visual_encoder is not None, "Cannot find visual backbone in model! Please manually set attribute."

    # Typically, the visual backbone is a ViT/Transformer structure, traversing all Attention/Linear attention patches.
    # It is recommended to only patch critical layers such as 'qkv', 'proj', 'fc1', and 'fc2'.
    keywords = ['qkv', 'proj', 'fc1', 'fc2']
    total_patched = find_and_patch_linear_layers(
        visual_encoder, target_keywords=keywords, lora_r=r, lora_alpha=alpha, lora_dropout=dropout, verbose=verbose
    )
    print(f"[LoRA Patch] Total {total_patched} Linear layers patched in {attr}.")

# Example usage:
if __name__ == "__main__":
    from transformers import BlipForConditionalGeneration
    import torch

    # Loading BLIP1 (image captioning large) example
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    print("Before LoRA patch:")
    for n, m in model.visual_encoder.named_modules():
        if isinstance(m, nn.Linear):
            print(n, m)

    patch_blip1_visual_transformer(model, r=8, alpha=16, dropout=0.05)

    print("\nAfter LoRA patch:")
    for n, m in model.visual_encoder.named_modules():
        if isinstance(m, LoRALinear):
            print(n, m)