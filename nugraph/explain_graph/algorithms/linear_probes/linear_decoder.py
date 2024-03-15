import torch


class StaticLinearDecoder:
    def __init__(self, in_shape, planes, num_classes) -> None:
        self.decoder = {}
        self.planes = planes
        self.num_classes = num_classes

        for plane in planes:
            self.decoder[plane] = self.class_decoder(in_shape)

    def class_decoder(self, in_shape):
        if not isinstance(in_shape, int):
            in_shape = in_shape
        return torch.eye(*in_shape)

    def forward(self, x):
        return {
            plane: torch.matmul(x[plane], self.decoder[plane]).squeeze(dim=-1)
            for plane in self.planes
        }

    def classes(self, x):
        return {
            plane: torch.argmax(torch.nn.functional.softmax(x[plane]), dim=-1)
            for plane in x.keys()
        }

    def probablity(self, x):
        return {plane: torch.nn.functional.sigmoid(x[plane]) for plane in x.keys()}


class DynamicLinearDecoder(StaticLinearDecoder, torch.nn.Module):
    def __init__(self, in_shape, planes, num_classes) -> None:
        StaticLinearDecoder.__init__(self, in_shape, planes, num_classes)
        torch.nn.Module.__init__(self)

        self.decoder = torch.nn.ModuleDict()
        for plane in planes:
            self.decoder[plane] = self.class_decoder(in_shape)

    def class_decoder(self, in_shape):
        if not isinstance(in_shape, int):
            in_shape = in_shape
        return torch.nn.Linear(*in_shape, self.num_classes)

    def forward(self, x):
        return {plane: self.decoder[plane](x[plane]) for plane in self.planes}
