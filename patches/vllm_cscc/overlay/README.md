# Overlay for vllm_cscc — FDU Phase 2+ deep hooks
#
# Contents are copied onto the vllm_cscc tree by scripts/apply_vllm_patches.sh
# then compiled into the wheel.
#
# Current overlays:
#   vllm/v1/attention/backends/fdu_gqa_attn.py  — GQA wrap re-export
#
# Activation (required):
#   USE_FDU_SERVER=1 FDU_PHASE=2 FDU_ENABLE_GQA_OPT=1
#   # or patched wheel with fdu_vllm.activate() in vllm/__init__.py
#
# Do NOT enable FDU_ENABLE_HIP_GRAPH / KV FP8 / defrag until each is wired + gated.
# Never patch batch scheduler from this overlay.
