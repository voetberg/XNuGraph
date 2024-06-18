from typing import Iterable
import torch


class DynamicLinearDecoder(torch.nn.Module):
    def __init__(
        self,
        in_shape,
        out_shape,
        input_function,
        loss_function,
        extra_metrics,
        planes=("u", "v", "y"),
        device="cpu",
    ) -> None:
        super().__init__()

        self.planes = planes

        self.input_function = input_function
        self.loss_function = loss_function
        self.device = device
        self.decoder = torch.nn.ModuleDict()
        for plane in planes:
            self.decoder[plane] = self.class_decoder(in_shape, out_shape)

        self.metrics = (
            extra_metrics if isinstance(extra_metrics, Iterable) else [extra_metrics]
        )

    def class_decoder(self, in_shape, out_shape):
        return torch.nn.Linear(*in_shape, out_shape).to(device=self.device)

    def forward(self, m):
        decoded = {}
        for p in self.planes:
            decoded[p] = torch.max(self.decoder[p](m[p]), axis=1).values
        return decoded

    def loss(self, y, y_hat):
        return self.loss_function(y, y_hat)
