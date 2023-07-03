from itertools import product

import cv2
import numpy as np


#### Generation of heatmaps ####
def generate_gaussian_heatmaps(
    heatmap_size,
    keypoints,
    keypoints_visible,
    sigma,
):
    """Generate gaussian heatmaps of keypoints.

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)
        sigma (float or List[float]): A list of sigma values of the Gaussian
            heatmap for each instance. If sigma is given as a single float
            value, it will be expanded into a tuple

    Returns:
        tuple:
        - heatmaps (np.ndarray): The generated heatmap in shape
            (K, H, W) where [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape
            (N, K)
    """

    N, K, _ = keypoints.shape
    W, H = heatmap_size

    heatmaps = np.zeros((K, H, W), dtype=np.float32)
    keypoint_weights = keypoints_visible.copy()

    if isinstance(sigma, (int, float)):
        sigma = (sigma,) * N

    for n in range(N):
        # 3-sigma rule
        radius = sigma[n] * 3

        # xy grid
        gaussian_size = 2 * radius + 1
        x = np.arange(0, gaussian_size, 1, dtype=np.float32)
        y = x[:, None]
        x0 = y0 = gaussian_size // 2

        for k in range(K):
            # skip unlabled keypoints
            if keypoints_visible[n, k] < 0.5:
                continue

            # get gaussian center coordinates
            mu = (keypoints[n, k] + 0.5).astype(np.int64)

            # check that the gaussian has in-bounds part
            left, top = (mu - radius).astype(np.int64)
            right, bottom = (mu + radius + 1).astype(np.int64)

            if left >= W or top >= H or right < 0 or bottom < 0:
                keypoint_weights[n, k] = 0
                continue

            # The gaussian is not normalized,
            # we want the center value to equal 1
            gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma[n] ** 2))

            # valid range in gaussian
            g_x1 = max(0, -left)
            g_x2 = min(W, right) - left
            g_y1 = max(0, -top)
            g_y2 = min(H, bottom) - top

            # valid range in heatmap
            h_x1 = max(0, left)
            h_x2 = min(W, right)
            h_y1 = max(0, top)
            h_y2 = min(H, bottom)

            heatmap_region = heatmaps[k, h_y1:h_y2, h_x1:h_x2]
            gaussian_regsion = gaussian[g_y1:g_y2, g_x1:g_x2]

            _ = np.maximum(heatmap_region, gaussian_regsion, out=heatmap_region)

    return heatmaps, keypoint_weights


def generate_unbiased_gaussian_heatmaps(
    heatmap_size,
    keypoints,
    keypoints_visible,
    sigma,
):
    """Generate gaussian heatmaps of keypoints using `Dark Pose`_.

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)

    Returns:
        tuple:
        - heatmaps (np.ndarray): The generated heatmap in shape
            (K, H, W) where [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape
            (N, K)

    .. _`Dark Pose`: https://arxiv.org/abs/1910.06278
    """

    N, K, _ = keypoints.shape
    W, H = heatmap_size

    heatmaps = np.zeros((K, H, W), dtype=np.float32)
    keypoint_weights = keypoints_visible.copy()

    # 3-sigma rule
    radius = sigma * 3

    # xy grid
    x = np.arange(0, W, 1, dtype=np.float32)
    y = np.arange(0, H, 1, dtype=np.float32)[:, None]

    for n, k in product(range(N), range(K)):
        # skip unlabled keypoints
        if keypoints_visible[n, k] < 0.5:
            continue

        mu = keypoints[n, k]
        # check that the gaussian has in-bounds part
        left, top = mu - radius
        right, bottom = mu + radius + 1

        if left >= W or top >= H or right < 0 or bottom < 0:
            keypoint_weights[n, k] = 0
            continue

        gaussian = np.exp(-((x - mu[0]) ** 2 + (y - mu[1]) ** 2) / (2 * sigma**2))

        _ = np.maximum(gaussian, heatmaps[k], out=heatmaps[k])

    return heatmaps, keypoint_weights


def get_heatmap_maximum(heatmaps: np.ndarray):
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps, np.ndarray), "heatmaps should be numpy.ndarray"
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, f"Invalid shape {heatmaps.shape}"

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    y_locs, x_locs = np.unravel_index(np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.0] = -1

    if B:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)

    return locs, vals


def refine_keypoints(keypoints: np.ndarray, heatmaps: np.ndarray) -> np.ndarray:
    """Refine keypoint predictions by moving from the maximum towards the
    second maximum by 0.25 pixel. The operation is in-place.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - heatmap size: [W, H]

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        heatmaps (np.ndarray): The heatmaps in shape (K, H, W)

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    for n, k in product(range(N), range(K)):
        x, y = keypoints[n, k, :2].astype(int)

        if 1 < x < W - 1 and 0 < y < H:
            dx = heatmaps[k, y, x + 1] - heatmaps[k, y, x - 1]
        else:
            dx = 0.0

        if 1 < y < H - 1 and 0 < x < W:
            dy = heatmaps[k, y + 1, x] - heatmaps[k, y - 1, x]
        else:
            dy = 0.0

        keypoints[n, k] += np.sign([dx, dy], dtype=np.float32) * 0.25

    return keypoints


def refine_keypoints_dark(
    keypoints: np.ndarray, heatmaps: np.ndarray, blur_kernel_size: int
) -> np.ndarray:
    """Refine keypoint predictions using distribution aware coordinate
    decoding. See `Dark Pose`_ for details. The operation is in-place.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - heatmap size: [W, H]

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        heatmaps (np.ndarray): The heatmaps in shape (K, H, W)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)

    .. _`Dark Pose`: https://arxiv.org/abs/1910.06278
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    # modulate heatmaps
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.maximum(heatmaps, 1e-10, heatmaps)
    np.log(heatmaps, heatmaps)

    for n, k in product(range(N), range(K)):
        x, y = keypoints[n, k, :2].astype(int)
        if 1 < x < W - 2 and 1 < y < H - 2:
            dx = 0.5 * (heatmaps[k, y, x + 1] - heatmaps[k, y, x - 1])
            dy = 0.5 * (heatmaps[k, y + 1, x] - heatmaps[k, y - 1, x])

            dxx = 0.25 * (
                heatmaps[k, y, x + 2] - 2 * heatmaps[k, y, x] + heatmaps[k, y, x - 2]
            )
            dxy = 0.25 * (
                heatmaps[k, y + 1, x + 1]
                - heatmaps[k, y - 1, x + 1]
                - heatmaps[k, y + 1, x - 1]
                + heatmaps[k, y - 1, x - 1]
            )
            dyy = 0.25 * (
                heatmaps[k, y + 2, x] - 2 * heatmaps[k, y, x] + heatmaps[k, y - 2, x]
            )
            derivative = np.array([[dx], [dy]])
            hessian = np.array([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy**2 != 0:
                hessianinv = np.linalg.inv(hessian)
                offset = -hessianinv @ derivative
                offset = np.squeeze(np.array(offset.T), axis=0)
                keypoints[n, k, :2] += offset
    return keypoints


def refine_keypoints_dark(
    keypoints: np.ndarray, heatmaps: np.ndarray, blur_kernel_size: int
) -> np.ndarray:
    """Refine keypoint predictions using distribution aware coordinate
    decoding. See `Dark Pose`_ for details. The operation is in-place.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - heatmap size: [W, H]

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        heatmaps (np.ndarray): The heatmaps in shape (K, H, W)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)

    .. _`Dark Pose`: https://arxiv.org/abs/1910.06278
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    # modulate heatmaps
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.maximum(heatmaps, 1e-10, heatmaps)
    np.log(heatmaps, heatmaps)

    for n, k in product(range(N), range(K)):
        x, y = keypoints[n, k, :2].astype(int)
        if 1 < x < W - 2 and 1 < y < H - 2:
            dx = 0.5 * (heatmaps[k, y, x + 1] - heatmaps[k, y, x - 1])
            dy = 0.5 * (heatmaps[k, y + 1, x] - heatmaps[k, y - 1, x])

            dxx = 0.25 * (
                heatmaps[k, y, x + 2] - 2 * heatmaps[k, y, x] + heatmaps[k, y, x - 2]
            )
            dxy = 0.25 * (
                heatmaps[k, y + 1, x + 1]
                - heatmaps[k, y - 1, x + 1]
                - heatmaps[k, y + 1, x - 1]
                + heatmaps[k, y - 1, x - 1]
            )
            dyy = 0.25 * (
                heatmaps[k, y + 2, x] - 2 * heatmaps[k, y, x] + heatmaps[k, y - 2, x]
            )
            derivative = np.array([[dx], [dy]])
            hessian = np.array([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy**2 != 0:
                hessianinv = np.linalg.inv(hessian)
                offset = -hessianinv @ derivative
                offset = np.squeeze(np.array(offset.T), axis=0)
                keypoints[n, k, :2] += offset
    return keypoints


def refine_keypoints_dark_udp(
    keypoints: np.ndarray, heatmaps: np.ndarray, blur_kernel_size: int
) -> np.ndarray:
    """Refine keypoint predictions using distribution aware coordinate decoding
    for UDP. See `UDP`_ for details. The operation is in-place.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - heatmap size: [W, H]

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        heatmaps (np.ndarray): The heatmaps in shape (K, H, W)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)

    .. _`UDP`: https://arxiv.org/abs/1911.07524
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    # modulate heatmaps
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50.0, heatmaps)
    np.log(heatmaps, heatmaps)

    heatmaps_pad = np.pad(heatmaps, ((0, 0), (1, 1), (1, 1)), mode="edge").flatten()

    for n in range(N):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(K, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints[n] -= np.einsum("imn,ink->imk", hessian, derivative).squeeze()

    return keypoints


def gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate heatmap distribution with Gaussian.

    Note:
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape

    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps
