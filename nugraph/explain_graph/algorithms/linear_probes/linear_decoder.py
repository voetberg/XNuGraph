import torch


class DynamicLinearDecoder(torch.nn.Module):
    def __init__(
        self,
        in_shape,
        input_function,
        loss_function,
        planes=("u", "v", "y"),
        num_classes=5,
    ) -> None:
        super().__init__()

        self.planes = planes
        self.num_classes = num_classes

        self.input_function = input_function
        self.loss_function = loss_function

        self.decoder = torch.nn.ModuleDict()
        for plane in planes:
            self.decoder[plane] = self.class_decoder(in_shape)

        self.softmax = torch.nn.functional.softmax

    def class_decoder(self, in_shape):
        return torch.nn.Linear(*in_shape, self.num_classes)

    def forward(self, x):
        step_output = self.input_function(x)
        return {
            plane: self.softmax(self.decoder[plane](step_output[plane]))
            for plane in self.planes
        }

    def loss(self, y, y_hat):
        return self.loss_function(y, y_hat)
