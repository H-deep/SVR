#!/usr/bin/env python3
import os, argparse, torch, torch.nn as nn
from model_video import build_model2

def strip_dp(state):
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state

class MaskWrapGrouped(nn.Module):
    """
    Export wrapper that:
      - tiles single image 5x along batch (to satisfy group_size=5),
      - runs build_model2,
      - averages the 5 masks to return (B=1,1,H,W).
    """
    def __init__(self, net, group=5):
        super().__init__()
        self.net = net
        self.group = group

    def forward(self, x):
        # x: (1,3,H,W) at export time
        if x.dim() == 3:
            x = x.unsqueeze(0)
        B, C, H, W = x.shape
        # Tile to (group,3,H,W) — same frame repeated
        x5 = x.repeat(self.group, 1, 1, 1)
        _, mask = self.net(x5)         # mask: (group,H,W)
        mask = mask.unsqueeze(1)        # (group,1,H,W)
        mask = mask.mean(dim=0, keepdim=True)  # (1,1,H,W)
        return mask

def parse_args():
    ap = argparse.ArgumentParser("Export SVR mask (build_model2) to ONNX with group tiling")
    ap.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--onnx", required=True, help="Output ONNX path")
    ap.add_argument("--img", nargs=2, type=int, default=[224, 224], help="H W")
    ap.add_argument("--opset", type=int, default=14)
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--simplify", action="store_true")
    return ap.parse_args()

def main():
    a = parse_args()
    os.makedirs(os.path.dirname(a.onnx), exist_ok=True)
    device = torch.device(a.device)

    net = build_model2(device=device).to(device)
    ckpt = torch.load(a.ckpt, map_location=device)
    state = strip_dp(ckpt.get("state_dict", ckpt))
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[export] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        if missing:    print("  missing:", missing[:12], "..." if len(missing) > 12 else "")
        if unexpected: print("  unexpected:", unexpected[:12], "..." if len(unexpected) > 12 else "")

    net.eval()
    torch.set_grad_enabled(False)

    wrapped = MaskWrapGrouped(net, group=5).to(device).eval()
    H, W = a.img
    dummy = torch.randn(1, 3, H, W, device=device)

    print(f"[export] Exporting ONNX -> {a.onnx} (opset={a.opset})")
    torch.onnx.export(
        wrapped, dummy, a.onnx,
        input_names=["image"],
        output_names=["mask"],
        opset_version=a.opset,
        do_constant_folding=True,
        dynamic_axes=None  # static = more TRT-friendly
    )
    print("[export] ONNX saved.")

    if a.simplify:
        try:
            import onnx
            from onnxsim import simplify
            print("[export] simplifying ONNX ...")
            m = onnx.load(a.onnx)
            ms, ok = simplify(m)
            assert ok
            onnx.save(ms, a.onnx)
            print("[export] simplified ONNX saved.")
        except Exception as e:
            print(f"[export] simplifier skipped: {e}")

    print("[export] Done.")

if __name__ == "__main__":
    main()
