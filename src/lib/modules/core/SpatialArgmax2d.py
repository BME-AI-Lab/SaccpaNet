import torch
from torch import nn


class HardArgmax2d(nn.Module):
    def __init__(self, normalized_coordinates):
        super().__init__()
        self.normalized_coordinates = normalized_coordinates

    def forward(self, input):
        heatmaps = input
        assert (
            heatmaps.ndim == 3 or heatmaps.ndim == 4
        ), f"Invalid shape {heatmaps.shape}"
        if len(heatmaps.shape) == 4:
            B, C, H, W = heatmaps.shape
            heatmaps_flatten = heatmaps.reshape(B * C, -1)
        elif len(heatmaps.shape) == 3:
            C, H, W = heatmaps.shape
            B = None
            heatmaps_flatten = heatmaps.reshape(C, -1)
        # argmax
        value, index = heatmaps_flatten.max(dim=-1)
        x_locs, y_locs = index % W, index // W
        locs = torch.stack([x_locs, y_locs], dim=-1)
        locs[value <= 0] = -1
        if B:
            locs = locs.reshape(B, C, 2)
            # value = value.reshape(B, C)
        if self.normalized_coordinates:
            locs = locs / torch.tensor([H, W], device=locs.device, dtype=locs.dtype)
        return locs


class SpatialSoftArgmax2d(nn.Module):
    r"""Creates a module that computes the Spatial Soft-Argmax 2D
    of a given input heatmap.

    Returns the index of the maximum 2d coordinates of the give map.
    The output order is x-coord and y-coord.

    Arguments:
        normalized_coordinates (Optional[bool]): wether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`

    Examples::
        >>> input = torch.rand(1, 4, 2, 3)
        >>> m = tgm.losses.SpatialSoftArgmax2d()
        >>> coords = m(input)  # 1x4x2
        >>> x_coord, y_coord = torch.chunk(coords, dim=-1, chunks=2)
    """

    def __init__(self, normalized_coordinates) -> None:
        super(SpatialSoftArgmax2d, self).__init__()
        self.normalized_coordinates = normalized_coordinates
        self.eps: float = 1e-6

    def create_meshgrid(self, x, normalized_coordinates):
        assert len(x.shape) == 4, x.shape
        _, _, height, width = x.shape
        _device, _dtype = x.device, x.dtype
        if normalized_coordinates:
            xs = torch.linspace(-1.0, 1.0, width, device=_device, dtype=_dtype)
            ys = torch.linspace(-1.0, 1.0, height, device=_device, dtype=_dtype)
        else:
            xs = torch.linspace(0, width - 1, width, device=_device, dtype=_dtype)
            ys = torch.linspace(0, height - 1, height, device=_device, dtype=_dtype)
        return torch.meshgrid(ys, xs)  # pos_y, pos_x

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError(
                "Input input type is not a torch.Tensor. Got {}".format(type(input))
            )
        if not len(input.shape) == 4:
            raise ValueError(
                "Invalid input shape, we expect BxCxHxW. Got: {}".format(input.shape)
            )
        # unpack shapes and create view from input tensor
        batch_size, channels, height, width = input.shape
        x: torch.Tensor = input.view(batch_size, channels, -1)

        # compute softmax with max substraction trick
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
        exp_x_sum = 1.0 / (exp_x.sum(dim=-1, keepdim=True) + self.eps)

        # create coordinates grid
        pos_y, pos_x = self.create_meshgrid(input, self.normalized_coordinates)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        # compute the expected coordinates
        expected_y: torch.Tensor = torch.sum(
            (pos_y * exp_x) * exp_x_sum, dim=-1, keepdim=True
        )
        expected_x: torch.Tensor = torch.sum(
            (pos_x * exp_x) * exp_x_sum, dim=-1, keepdim=True
        )
        output: torch.Tensor = torch.cat([expected_x, expected_y], dim=-1)
        return output.view(batch_size, channels, 2)  # BxNx2
