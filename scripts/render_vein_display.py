from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from train_tiny_unet import IMAGE_EXTS, TinyUNet, imread_unicode


DEFAULT_WEIGHTS = "runs/tiny_unet/data1_final_v2_light_seed42_160_bc8_best.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render hand-vein predictions as display-friendly images.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", default="datasets/split_hand_vein/test/images")
    parser.add_argument("--out", default="runs/tiny_unet/final_display_visualizations")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--alpha", type=float, default=0.38, help="Maximum vein overlay opacity.")
    parser.add_argument("--min-area", type=int, default=18, help="Remove predicted components smaller than this area.")
    parser.add_argument("--connect-veins", action="store_true", help="Visually bridge nearby vein endpoints.")
    parser.add_argument("--bridge-distance", type=float, default=32.0, help="Maximum endpoint distance for visual bridging.")
    parser.add_argument("--bridge-thickness", type=int, default=2, help="Line thickness used for bridged vein segments.")
    parser.add_argument("--bridge-alpha", type=float, default=0.55, help="Strength of bridged segments in the display mask.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of images to render; 0 means all.")
    return parser.parse_args()


def image_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        return False
    encoded.tofile(str(path))
    return True


def enhance_hand_image(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2RGB)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.03, beta=2)
    return enhanced


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def skeletonize(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary)

    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    current = binary.copy()
    while cv2.countNonZero(current) > 0:
        eroded = cv2.erode(current, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(current, opened))
        current = eroded
    return skeleton


def endpoint_direction(skeleton: np.ndarray, point: tuple[int, int], radius: int = 8) -> np.ndarray | None:
    x, y = point
    h, w = skeleton.shape
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    ys, xs = np.where(skeleton[y0:y1, x0:x1] > 0)
    if len(xs) < 3:
        return None

    coords = np.column_stack([xs + x0, ys + y0]).astype(np.float32)
    center = np.array([x, y], dtype=np.float32)
    distances = np.linalg.norm(coords - center, axis=1)
    farthest = coords[int(np.argmax(distances))]
    vector = center - farthest
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return None
    return vector / norm


def skeleton_endpoints(skeleton: np.ndarray) -> list[tuple[int, int]]:
    binary = (skeleton > 0).astype(np.uint8)
    neighbor_count = cv2.filter2D(binary, -1, np.ones((3, 3), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.where((binary > 0) & (neighbor_count == 2))
    return list(zip(xs.tolist(), ys.tolist()))


def line_clear_enough(mask: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], distance: float) -> bool:
    canvas = np.zeros_like(mask, dtype=np.uint8)
    cv2.line(canvas, p1, p2, 255, 1, cv2.LINE_AA)
    dilated_line = cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    support = int(((dilated_line > 0) & (mask > 0)).sum())
    line_area = int((dilated_line > 0).sum())
    if line_area == 0:
        return False
    support_ratio = support / line_area
    return support_ratio > 0.10 or distance <= 18


def bridge_endpoints(mask: np.ndarray, skeleton: np.ndarray, max_distance: float, thickness: int) -> np.ndarray:
    endpoints = skeleton_endpoints(skeleton)
    if len(endpoints) < 2:
        return np.zeros_like(mask, dtype=np.uint8)

    directions = {point: endpoint_direction(skeleton, point) for point in endpoints}
    candidates: list[tuple[float, int, int]] = []
    for i, p1 in enumerate(endpoints):
        d1 = directions[p1]
        if d1 is None:
            continue
        p1_arr = np.array(p1, dtype=np.float32)
        for j in range(i + 1, len(endpoints)):
            p2 = endpoints[j]
            d2 = directions[p2]
            if d2 is None:
                continue
            p2_arr = np.array(p2, dtype=np.float32)
            vector = p2_arr - p1_arr
            distance = float(np.linalg.norm(vector))
            if distance < 4 or distance > max_distance:
                continue
            unit = vector / distance
            facing_score = min(float(np.dot(d1, unit)), float(np.dot(d2, -unit)))
            if facing_score < 0.25:
                continue
            if not line_clear_enough(mask, p1, p2, distance):
                continue
            candidates.append((distance, i, j))

    bridges = np.zeros_like(mask, dtype=np.uint8)
    used: set[int] = set()
    for _, i, j in sorted(candidates, key=lambda item: item[0]):
        if i in used or j in used:
            continue
        p1, p2 = endpoints[i], endpoints[j]
        cv2.line(bridges, p1, p2, 255, thickness, cv2.LINE_AA)
        used.add(i)
        used.add(j)
    return bridges


def postprocess_probability(
    prob: np.ndarray,
    threshold: float,
    min_area: int,
    connect_veins: bool = False,
    bridge_distance: float = 32.0,
    bridge_thickness: int = 2,
    bridge_alpha: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (prob >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = remove_small_components(mask, min_area)
    thin = skeletonize(mask)
    bridges = np.zeros_like(mask, dtype=np.uint8)
    if connect_veins:
        bridges = bridge_endpoints(mask, thin, bridge_distance, bridge_thickness)
        connected = cv2.bitwise_or(mask, bridges)
        connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel, iterations=1)
        thin = cv2.bitwise_or(skeletonize(connected), bridges)
        mask = connected

    soft_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0, sigmaY=2.0)
    soft_line = cv2.GaussianBlur(thin, (0, 0), sigmaX=1.2, sigmaY=1.2)
    display_alpha = np.maximum(soft_mask.astype(np.float32) / 255.0, soft_line.astype(np.float32) / 180.0)
    if connect_veins and np.any(bridges):
        soft_bridge = cv2.GaussianBlur(bridges, (0, 0), sigmaX=1.5, sigmaY=1.5)
        display_alpha = np.maximum(display_alpha, (soft_bridge.astype(np.float32) / 255.0) * bridge_alpha)
    display_alpha = np.clip(display_alpha, 0.0, 1.0)
    return mask, display_alpha


def render_overlay(rgb: np.ndarray, alpha_mask: np.ndarray, max_alpha: float) -> np.ndarray:
    base = enhance_hand_image(rgb)
    vein_color = np.array([42, 78, 105], dtype=np.float32)
    shadow_color = np.array([22, 48, 62], dtype=np.float32)
    alpha = (alpha_mask * max_alpha)[..., None]

    shaded = base.astype(np.float32) * (1.0 - alpha * 0.35) + shadow_color * (alpha * 0.35)
    colored = shaded * (1.0 - alpha) + vein_color * alpha
    return np.clip(colored, 0, 255).astype(np.uint8)


def render_vein_only(alpha_mask: np.ndarray) -> np.ndarray:
    background = np.full((*alpha_mask.shape, 3), 232, dtype=np.float32)
    vein_color = np.array([32, 72, 96], dtype=np.float32)
    alpha = np.clip(alpha_mask * 0.95, 0.0, 1.0)[..., None]
    image = background * (1.0 - alpha) + vein_color * alpha
    return np.clip(image, 0, 255).astype(np.uint8)


def add_title(rgb: np.ndarray, title: str) -> np.ndarray:
    canvas = cv2.copyMakeBorder(rgb, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (35, 35, 35), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    source = Path(args.source).resolve()
    out_root = Path(args.out).resolve()
    overlay_dir = out_root / "display_overlay"
    vein_only_dir = out_root / "display_vein_only"
    compare_dir = out_root / "comparison"
    connected_compare_dir = out_root / "comparison_connected"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    vein_only_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)
    if args.connect_veins:
        connected_compare_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(weights_path, map_location="cpu")
    imgsz = int(checkpoint["imgsz"])
    model = TinyUNet(base_channels=int(checkpoint["base_channels"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    paths = image_paths(source)
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No images found: {source}")

    for image_path in paths:
        bgr = imread_unicode(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        original_size = (rgb.shape[1], rgb.shape[0])
        resized = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1)[None].float() / 255.0

        with torch.no_grad():
            prob = torch.sigmoid(model(tensor))[0, 0].numpy()
        prob = cv2.resize(prob, original_size, interpolation=cv2.INTER_LINEAR)
        _, raw_alpha_mask = postprocess_probability(prob, args.threshold, args.min_area)
        _, alpha_mask = postprocess_probability(
            prob,
            args.threshold,
            args.min_area,
            connect_veins=args.connect_veins,
            bridge_distance=args.bridge_distance,
            bridge_thickness=args.bridge_thickness,
            bridge_alpha=args.bridge_alpha,
        )

        raw_overlay = render_overlay(rgb, raw_alpha_mask, args.alpha)
        overlay = render_overlay(rgb, alpha_mask, args.alpha)
        vein_only = render_vein_only(alpha_mask)
        compare = np.concatenate(
            [
                add_title(rgb, "original"),
                add_title(overlay, "display overlay"),
                add_title(vein_only, "vein only"),
            ],
            axis=1,
        )
        connected_compare = np.concatenate(
            [
                add_title(rgb, "original"),
                add_title(raw_overlay, "original display"),
                add_title(overlay, "connected display"),
                add_title(vein_only, "connected vein only"),
            ],
            axis=1,
        )

        stem = image_path.stem
        if not imwrite_unicode(overlay_dir / f"{stem}_display_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
            raise ValueError(f"Could not write overlay for {image_path}")
        if not imwrite_unicode(vein_only_dir / f"{stem}_display_vein_only.png", cv2.cvtColor(vein_only, cv2.COLOR_RGB2BGR)):
            raise ValueError(f"Could not write vein-only image for {image_path}")
        if not imwrite_unicode(compare_dir / f"{stem}_display_compare.png", cv2.cvtColor(compare, cv2.COLOR_RGB2BGR)):
            raise ValueError(f"Could not write comparison for {image_path}")
        if args.connect_veins:
            compare_path = connected_compare_dir / f"{stem}_connected_compare.png"
            if not imwrite_unicode(compare_path, cv2.cvtColor(connected_compare, cv2.COLOR_RGB2BGR)):
                raise ValueError(f"Could not write connected comparison for {image_path}")

    print(f"Rendered {len(paths)} image(s)")
    print(f"display_overlay: {overlay_dir}")
    print(f"display_vein_only: {vein_only_dir}")
    print(f"comparison: {compare_dir}")
    if args.connect_veins:
        print(f"comparison_connected: {connected_compare_dir}")


if __name__ == "__main__":
    main()
