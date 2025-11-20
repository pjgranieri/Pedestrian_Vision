import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def findDepth(image: np.ndarray, tuple_list: list[tuple[str, bool, np.ndarray]], midas_model: str = "DPT_Large", debug: bool = False) -> list[tuple[str, bool, float | None]]:
    """
    1. Run MiDaS on an RGB image to get depth map
    2. For each mask in tuple_list, compute average depth within the mask
    3. Drop mask and append depth to each tuple
    
    Parameters
    ----------
    image : np.ndarray
        Input image (RGB color space).

    tuple_list : list[tuple]
        Each tuple: ("sign type", present_flag, mask)
        
    midas_model : str
        MiDaS model type. Options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
        Default is "DPT_Large".

    debug : bool
        Show debug output
        Default is False.

    Returns
    -------
    list[tuple]
        Each tuple with depth appended: ("sign type", present_flag, depth)
    """

    # -------- DEBUG: ENTERED DEPTH FUNCTION --------
    if debug:
        print(f"\033[34m[INFO]\033[0m Finding depth using MiDaS.")

    # -------- Get Device for PyTorch --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if debug:
        print(f"\033[34m[INFO]\033[0m Using device: {device}.")

    # -------- Load MiDaS model --------
    if debug:
        print(f"\033[34m[INFO]\033[0m Loading MiDaS model: {midas_model}...", end="")

    midas = torch.hub.load("intel-isl/MiDaS", midas_model)
    midas.to(device).eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", verbose = False)
    transform = (
        midas_transforms.dpt_transform
        if midas_model in ["DPT_Hybrid", "DPT_Large"]
        else midas_transforms.small_transform
    )

    if debug:
        print(f"\033[34m[INFO]\033[0m MiDaS model {midas_model} loaded.")

    # -------- Run MiDaS --------
    if debug:
        print(f"\033[34m[INFO]\033[0m Running MiDaS depth estimation...", end="")

    input_batch = transform(image)
    if input_batch.ndim == 3:
        input_batch = input_batch.unsqueeze(0)
    input_batch = input_batch.to(device)

    with torch.no_grad():
        pred = midas(input_batch)
        if pred.ndim == 4 and pred.shape[1] == 1:
            depth = pred.squeeze(1)
        else:
            depth = pred
        depth = depth.squeeze(0).cpu().numpy()

    if debug:
        print("Done.")

    # -------- Normalize Depth to [0, 1] --------
    if debug:
        print(f"\033[34m[INFO]\033[0m Normalizing depth map...", end="")

    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    dh, dw = depth_norm.shape

    if debug:
        print("Done.")

    # -------- Compute Depth Value For Each Mask --------
    if debug:
        debug_masks_colors = {
            "street_sign":  (0.0, 0.0, 1.0),  # blue
            "stop sign":    (0.0, 1.0, 0.0),  # green
            "walk signal":  (1.0, 1.0, 1.0),  # white
            "stop signal":  (1.0, 1.0, 1.0),  # white
            "crosswalk":    (0.0, 1.0, 1.0),  # cyan
        }
        debug_masks = []
        debug_depth_rgb = plt.cm.plasma(depth_norm)[:, :, :3]
        debug_overlay_rgb = debug_depth_rgb.copy()
        debug_alpha = 0.5
        debug_warn_msgs = []
        debug_total_idx = len(tuple_list)

    results = []

    for idx, (sign_type, present_flag, mask) in enumerate(tuple_list, start=1):
        
        if debug:
            print(f"\r\033[34m[INFO]\033[0m Computing average depth for each mask...[{idx}/{debug_total_idx}]", end="", flush=True)

        # Skip if sign not present
        if not present_flag:
            if debug:
                debug_warn_msgs.append(f"\033[33m[WARN]\033[0m Sign {idx} ({sign_type}) marked as not present. Skipping mask.")
            continue

        # Skip if mask is invalid
        if mask is None or not isinstance(mask, np.ndarray):
            if debug:
                debug_warn_msgs.append(f"\033[33m[WARN]\033[0m Mask {idx} is invalid or missing.")
            continue

        # -------- Prepare Binary Mask --------
        mask_bin = mask > 0

        # Resize mask to match depth map
        if mask_bin.shape != (dh, dw):
            mask_bin = cv2.resize(mask_bin.astype(np.uint8),
                                  (dw, dh),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            
        # Check if mask is empty
        if not np.any(mask_bin):
            if debug:
                debug_warn_msgs.append(f"\033[33m[WARN]\033[0m Mask {idx} ({sign_type}) is empty.")
            continue

        # -------- DEBUG: GET MASK COLOR FOR OVERLAY --------
        if debug:
            color = np.array(debug_masks_colors.get(sign_type, (0, 0, 0)), dtype=np.float32)  # default black
            debug_overlay_rgb[mask_bin] = (
                (1 - debug_alpha) * debug_overlay_rgb[mask_bin]
                + debug_alpha * color
            )

        # -------- Find Average Depth within Mask --------
        avg_depth = float(depth_norm[mask_bin].mean())
        results.append((sign_type, present_flag, avg_depth))

    # -------- DEBUG: PRINT WARNING MESSAGES --------
    if debug:
        print("")
        for msg in debug_warn_msgs:
            print(msg)

    # -------- DEBUG: SHOW DEPTH MAP WITH MASKS --------
    if debug:
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(debug_depth_rgb)
        plt.title("Original Depth Map")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(debug_overlay_rgb)
        plt.title("Depth Map with Mask Overlays")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return results