import torch

from mask2former.modeling.clip_adapter.clip import build_clip_model

model = build_clip_model("ViT-B/16")
#
# maskCLIP_state_dict = torch.jit.load('/home/wjy/.cache/clip/ViT-B-16.pt', map_location="cpu")
# for k, v in maskCLIP_state_dict.items():
#     print(k)