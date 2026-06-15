import torch


class AvgPool2dEfficient(torch.nn.Module):
    def _init_(self, kernel_size: int, stride: int = None):
        """
        Implements 2D Average Pooling using torch operations (efficient approach).

        Args:
            kernel_size: Size of the pooling window.
            stride: Step size for the sliding window. Defaults to kernel_size.
        """
        super()._init_()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size  # Default stride = kernel size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Average Pooling.

        Args:
            inputs: Tensor of shape [batch, channels, height, width].

        Returns:
            Tensor after pooling.
        """
        B, C, H, W = inputs.shape

        # Reorganizar el tensor en bloques de tamaño kernel_size x kernel_size
        inputs = inputs.unfold(2, self.kernel_size, self.stride).unfold(
            3, self.kernel_size, self.stride
        )

        # Ahora inputs tiene forma [B, C, out_H, out_W, kernel_size, kernel_size]
        # Tomamos la media en las dimensiones de kernel
        pooled = inputs.mean(dim=(-1, -2))

        return pooled

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        out_H = (H - self.kernel_size) // self.stride + 1
        out_W = (W - self.kernel_size) // self.stride + 1
        outputs = torch.zeros((B, C, out_H, out_W), device=inputs.device)
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.kernel_size
                w_end = w_start + self.kernel_size
                window = inputs[:, :, h_start:h_end, w_start:w_end]
                outputs[:, :, i, j] = torch.mean(window, dim=(2, 3))