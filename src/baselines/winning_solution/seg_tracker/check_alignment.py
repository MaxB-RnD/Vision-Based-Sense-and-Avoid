"""
Alignment Quality Diagnostic
=============================
Run this from your project root:
    python -m src.baselines.winning_solution.seg_tracker.check_alignment

What it checks:
    1. Raw alignment error (MSE of aligned vs current frame background)
    2. Flow confidence map distribution — are weights healthy or collapsed?
    3. Estimated dx/dy/angle magnitudes — are they sane for your flight data?
    4. Visual overlay of aligned prev vs current frame so you can see misalignment

Usage:
    - Set FLIGHT_DIR to a folder of grayscale frames from one of your flights
    - Run and look at the printed stats + saved images in /tmp/alignment_check/
"""

import os
import sys
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, saves to file
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(__file__))

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Point this at a folder containing sequential grayscale frames from one flight.
FLIGHT_DIR   = "data/part3/Images/part354be71509266480ca13ae3fec487af47"
OUTPUT_DIR   = "tmp/alignment_check"
MAX_FRAMES   = 242      # how many consecutive frame pairs to check
MODELS_DIR   = "configs/models/baselines/winning_solution"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_transform_model(models_dir, device):
    from src.baselines.winning_solution.seg_tracker import models_transformation
    model = models_transformation.TrEstimatorTsm(
        cfg=dict(base_model_name="resnet34", weight_scale=3),
        pretrained=False
    )
    ckpt = torch.load(
        f"{models_dir}/030_tr_tsn_rn34_w3_crop_borders_255_0.pth",
        map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def estimate_transform_and_heatmap(model, prev_img, img, device):
    """Returns (dx, dy, angle, heatmap_np, offsets_np, aligned_prev)"""
    from src.baselines.winning_solution.seg_tracker import offset_grid_to_transform, common_utils

    crop_w = 2048
    crop_h = 1280
    h, w = img.shape
    y0 = 512
    x0 = (w - crop_w) // 2

    # Guard against images smaller than expected crop
    if y0 + crop_h > h or x0 + crop_w > w or x0 < 0:
        print(f"  ⚠️  Frame size {w}x{h} is too small for crop {crop_w}x{crop_h} at y0={y0}, x0={x0}")
        print(f"      This means estimate_transformation_full will silently crop wrong regions.")
        return None

    img_crop      = img[y0:y0 + crop_h, x0:x0 + crop_w]
    img_crop_prev = prev_img[y0:y0 + crop_h, x0:x0 + crop_w]

    t_cur  = torch.from_numpy(img_crop / 255.0).float().to(device)
    t_prev = torch.from_numpy(img_crop_prev / 255.0).float().to(device)

    with torch.no_grad():
        with torch.amp.autocast(device.type):
            heatmap, offsets = model(t_prev[None, :], t_cur[None, :])
        heatmap = heatmap.detach().float().cpu().numpy()
        offsets = offsets.detach().float().cpu().numpy()

    crop_w_pts = crop_w // 32
    crop_h_pts = crop_h // 32
    prev_points = np.zeros((2, crop_h_pts, crop_w_pts), dtype=np.float32)
    prev_points[0, :, :] = np.arange(16, crop_w, 32)[None, :]
    prev_points[1, :, :] = np.arange(16, crop_h, 32)[:, None]
    prev_points = prev_points[..., 2:-2, 2:-2]

    points_offset = np.array([x0, y0])[:, None]
    cur_points = prev_points + offsets[0]

    tr, err = offset_grid_to_transform.offset_grid_to_transform(
        prev_frame_points=prev_points.reshape(2, -1) + points_offset,
        cur_frame_points=cur_points.reshape(2, -1) + points_offset,
        points_weight=heatmap[0].reshape(-1) ** 2
    )

    # Apply the alignment
    prev_aligned = cv2.warpAffine(
        prev_img, tr[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
    )

    return tr, heatmap[0, 0], offsets[0], prev_aligned


def alignment_mse(prev_aligned, img):
    """MSE over background pixels (centre crop to avoid border artifacts)"""
    h, w = img.shape
    margin = 64
    cur   = img[margin:h-margin, margin:w-margin].astype(np.float32)
    prev  = prev_aligned[margin:h-margin, margin:w-margin].astype(np.float32)
    return float(np.mean((cur - prev) ** 2))


def run_diagnostic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Loading transform model from {MODELS_DIR}...")
    model = load_transform_model(MODELS_DIR, device)

    # Load frames
    frame_files = sorted([
        f for f in os.listdir(FLIGHT_DIR)
        if os.path.isfile(os.path.join(FLIGHT_DIR, f))
    ])[:MAX_FRAMES + 1]

    if len(frame_files) < 2:
        print(f"ERROR: Need at least 2 frames in {FLIGHT_DIR}")
        return

    print(f"Processing {len(frame_files)-1} frame pairs from {FLIGHT_DIR}\n")
    print(f"{'Pair':<6} {'MSE_aligned':>12} {'MSE_raw':>10} {'Improvement':>12} {'dx':>8} {'dy':>8} {'conf_mean':>10} {'conf_max':>10}")
    print("-" * 80)

    mse_aligned_all = []
    mse_raw_all     = []

    for i in range(1, len(frame_files)):
        prev_path = os.path.join(FLIGHT_DIR, frame_files[i-1])
        cur_path  = os.path.join(FLIGHT_DIR, frame_files[i])

        prev_img = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
        img      = cv2.imread(cur_path,  cv2.IMREAD_GRAYSCALE)

        if prev_img is None or img is None:
            print(f"  Pair {i}: could not load images, skipping")
            continue

        result = estimate_transform_and_heatmap(model, prev_img, img, device)
        if result is None:
            continue

        tr, heatmap, offsets, prev_aligned = result

        mse_a = alignment_mse(prev_aligned, img)
        mse_r = alignment_mse(prev_img, img)  # raw unaligned baseline

        mse_aligned_all.append(mse_a)
        mse_raw_all.append(mse_r)

        dx = tr[0, 2]
        dy = tr[1, 2]
        conf_mean = float(heatmap.mean())
        conf_max  = float(heatmap.max())

        improvement = (mse_r - mse_a) / (mse_r + 1e-6) * 100

        print(f"{i:<6} {mse_a:>12.2f} {mse_r:>10.2f} {improvement:>11.1f}% {dx:>8.2f} {dy:>8.2f} {conf_mean:>10.4f} {conf_max:>10.4f}")

        # Save overlay image for first 5 pairs
        if i <= 5:
            # Red = current, Green = aligned previous, overlap = yellow = well aligned
            overlay = np.stack([
                img,
                prev_aligned,
                img
            ], axis=2)
            out_path = os.path.join(OUTPUT_DIR, f"overlay_pair_{i:03d}.png")
            cv2.imwrite(out_path, overlay)

            # Save heatmap
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].imshow(heatmap, cmap='hot')
            axes[0].set_title(f'Flow confidence map\nmean={conf_mean:.4f} max={conf_max:.4f}')
            axes[0].axis('off')

            # Flow magnitude
            flow_mag = np.sqrt(offsets[0]**2 + offsets[1]**2)
            im = axes[1].imshow(flow_mag, cmap='viridis')
            axes[1].set_title(f'Flow magnitude\ndx={dx:.1f} dy={dy:.1f}')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"heatmap_pair_{i:03d}.png"), dpi=100)
            plt.close()

    print("-" * 80)
    if mse_aligned_all:
        avg_improvement = (np.mean(mse_raw_all) - np.mean(mse_aligned_all)) / (np.mean(mse_raw_all) + 1e-6) * 100
        print(f"\nSummary:")
        print(f"  Avg MSE aligned : {np.mean(mse_aligned_all):.2f}")
        print(f"  Avg MSE raw     : {np.mean(mse_raw_all):.2f}")
        print(f"  Avg improvement : {avg_improvement:.1f}%")
        print(f"\nInterpretation:")
        if avg_improvement > 30:
            print("  ✅ Alignment is working well (>30% MSE reduction)")
        elif avg_improvement > 10:
            print("  ⚠️  Alignment is marginal (10-30% MSE reduction) — retraining recommended")
        else:
            print("  ❌ Alignment is poor (<10% MSE reduction) — retraining is likely the main issue")

        print(f"\nVisual overlays saved to: {OUTPUT_DIR}")
        print("  In the overlay images: yellow = well aligned, red/green fringing = misalignment")
        print("  In the heatmap images: bright regions = where the model is confident about flow")
        print("  If confidence is concentrated only in one corner or is near-uniform, alignment is unreliable")


if __name__ == "__main__":
    run_diagnostic()