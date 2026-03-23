"""ADP rollout visualization helpers, aligned with LeRobot-style heatmap overlays."""

from typing import Any, Dict, Optional

import numpy as np


def get_adp_overlay_state(model: Any) -> Optional[Dict[str, Any]]:
    """Extract the latest ADP visualization payload from the model."""
    getter = getattr(model, "get_adp_visualization_state", None)
    state = getter() if callable(getter) else getattr(model, "_adp_visualization_state", None)
    if not state:
        return None

    heatmap_grids = state.get("last_heatmap_grids")
    if not heatmap_grids:
        return None

    heatmap_grid = np.array(heatmap_grids[0], copy=False)
    mask_grids = state.get("last_heatmap_mask_grids")
    heatmap_mask_grid = None
    if mask_grids:
        heatmap_mask_grid = np.array(mask_grids[0], copy=False)

    return {
        "heatmap_grid": heatmap_grid,
        "heatmap_mask_grid": heatmap_mask_grid,
        "heatmap_mask_mode": state.get("last_heatmap_mask_mode"),
    }


def render_adp_overlay(img: np.ndarray, overlay_state: Optional[Dict[str, Any]], cfg: Any) -> np.ndarray:
    """Render LeRobot-style heatmap overlay on a frame."""
    if overlay_state is None or not bool(getattr(cfg, "visualize_pruning", False)):
        return img

    overlay_mode = str(getattr(cfg, "overlay_mode", "heatmap")).lower()
    if overlay_mode not in {"heatmap", "heatmap_plain", "heatmap_kept_only"}:
        return img

    heatmap_grid = overlay_state.get("heatmap_grid")
    heatmap_mask_grid = overlay_state.get("heatmap_mask_grid")
    heatmap_mask_mode = overlay_state.get("heatmap_mask_mode")
    if overlay_mode == "heatmap_plain":
        heatmap_mask_grid = None
        heatmap_mask_mode = None
    elif overlay_mode == "heatmap_kept_only":
        heatmap_mask_mode = "keep_only"

    return _apply_heatmap_overlay(
        img,
        heatmap_grid,
        alpha=float(getattr(cfg, "overlay_alpha", 0.5)),
        heatmap_threshold=float(getattr(cfg, "overlay_heatmap_threshold", 0.0)),
        mask_grid=heatmap_mask_grid,
        mask_mode=heatmap_mask_mode,
        prune_darken=float(getattr(cfg, "overlay_prune_darken", 0.5)),
        prune_color=_parse_rgb_color(getattr(cfg, "overlay_pruned_color", "0,0,255")),
        prune_alpha=int(getattr(cfg, "overlay_pruned_alpha", 120)),
        prune_stripe_gap=int(getattr(cfg, "overlay_prune_stripe_gap", 4)),
        plain_style=overlay_mode == "heatmap_plain",
    )


def _parse_rgb_color(value: Any) -> np.ndarray:
    """Parse an RGB color from a string like '0,0,255' or a length-3 sequence."""
    if value is None:
        return np.array([0, 0, 255], dtype=np.float32)

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        try:
            parts = list(value)
        except TypeError:
            parts = ["0", "0", "255"]

    if len(parts) != 3:
        parts = ["0", "0", "255"]

    try:
        parsed = [float(part) for part in parts]
    except (TypeError, ValueError):
        parsed = [0.0, 0.0, 255.0]

    return np.clip(np.asarray(parsed, dtype=np.float32), 0.0, 255.0)


def _apply_heatmap_overlay(
    img,
    heatmap_grid,
    region_scores=None,
    alpha=0.5,
    heatmap_threshold: float = 0.0,
    mask_grid=None,
    mask_mode="pruned",
    prune_darken=0.5,
    prune_color=None,
    prune_alpha=120,
    prune_stripe_gap=4,
    plain_style: bool = False,
):
    """Overlay a continuous heatmap on an image, matching LeRobot's rollout visualization style."""
    if heatmap_grid is None:
        return img

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return img

    img_np = img.astype(np.uint8, copy=False)
    height, width = img_np.shape[:2]

    hm_img = Image.fromarray(np.asarray(heatmap_grid, dtype=np.float32), mode="F")
    hm_up = np.array(hm_img.resize((width, height), resample=Image.NEAREST))

    def _jet_colormap(values):
        r = np.clip(1.5 - np.abs(values - 0.75) * 4.0, 0.0, 1.0)
        g = np.clip(1.5 - np.abs(values - 0.5) * 4.0, 0.0, 1.0)
        b = np.clip(1.5 - np.abs(values - 0.25) * 4.0, 0.0, 1.0)
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    hm_vis = np.clip(hm_up, 0.0, 1.0)
    heat_source = hm_vis if plain_style else np.clip(hm_up, 0.0, 1.0)
    heat_rgb = _jet_colormap(heat_source)
    heat_blended = (
        img_np.astype(np.float32) * (1.0 - alpha) + heat_rgb.astype(np.float32) * alpha
    ).astype(np.uint8)

    if heatmap_threshold > 0.0:
        heatmap_active_mask = hm_vis >= heatmap_threshold
        blended = img_np.copy()
        blended[heatmap_active_mask] = heat_blended[heatmap_active_mask]
    else:
        heatmap_active_mask = np.ones((height, width), dtype=bool)
        blended = heat_blended

    if mask_grid is not None:
        mask_arr = mask_grid.astype(np.uint8) if mask_grid.dtype != np.uint8 else mask_grid
        mask_img = Image.fromarray(mask_arr, mode="L").resize((width, height), resample=Image.NEAREST)
        mask_up = np.array(mask_img)
        marked_mask = (mask_up != 0) if mask_mode == "reused" else (mask_up == 0)
        if mask_mode != "keep_only":
            marked_mask &= heatmap_active_mask

        if mask_mode == "keep_only":
            blended = blended.copy()
            blended[marked_mask] = img_np[marked_mask]
        elif mask_mode != "reused":
            blended_f = blended.astype(np.float32)
            blended_f[marked_mask] *= prune_darken
            if prune_color is not None and prune_alpha > 0:
                prune_alpha_norm = np.clip(float(prune_alpha) / 255.0, 0.0, 1.0)
                blended_f[marked_mask] = (
                    blended_f[marked_mask] * (1.0 - prune_alpha_norm) + prune_color * prune_alpha_norm
                )
            blended = blended_f.astype(np.uint8)

        if mask_mode != "keep_only" and prune_stripe_gap > 0:
            yy, xx = np.mgrid[:height, :width]
            stripe = ((xx + yy) % prune_stripe_gap) < max(1, prune_stripe_gap // 3)
            stripe_mask = marked_mask & stripe
            blended_f2 = blended.astype(np.float32)
            blended_f2[stripe_mask] = blended_f2[stripe_mask] * 0.6 + 255.0 * 0.4
            blended = blended_f2.astype(np.uint8)

    if region_scores is None:
        return blended

    try:
        import torch

        if torch.is_tensor(region_scores):
            region_scores = region_scores.detach().cpu().numpy()
    except Exception:
        pass

    region_scores = np.asarray(region_scores, dtype=np.float32)
    total = float(region_scores.sum())
    if total > 0:
        region_scores = region_scores / total

    region_h, region_w = heatmap_grid.shape
    cell_w = width / max(region_w, 1)
    cell_h = height / max(region_h, 1)
    visible_region_mask = None
    if heatmap_threshold > 0.0:
        visible_region_mask = np.clip(heatmap_grid, 0.0, 1.0) >= heatmap_threshold
    kept_region_mask = None
    if mask_mode == "keep_only" and mask_grid is not None:
        keep_region_img = Image.fromarray(mask_arr, mode="L").resize((region_w, region_h), resample=Image.NEAREST)
        kept_region_mask = np.array(keep_region_img) != 0

    pil_img = Image.fromarray(blended)
    draw = ImageDraw.Draw(pil_img)
    score_font_size = max(7, min(12, int(min(cell_w, cell_h) * 0.38)))

    def _load_font(size):
        for font_name in ("DejaVuSansMono.ttf", "DejaVuSans.ttf", "Arial.ttf"):
            try:
                return ImageFont.truetype(font_name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    score_font = _load_font(score_font_size)
    for row in range(region_h):
        for col in range(region_w):
            if visible_region_mask is not None and not visible_region_mask[row, col]:
                continue
            if kept_region_mask is not None and not kept_region_mask[row, col]:
                continue
            x = int((col + 0.03) * cell_w)
            y = int((row + 0.03) * cell_h)
            text = f"{region_scores[row, col]:.3f}"
            draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=score_font)
            draw.text((x, y), text, fill=(255, 255, 255), font=score_font)
    return np.array(pil_img)
