import torch

class LinearDecoder: 
    def __init__(self, in_shape, planes, num_classes) -> None:
        self.decoder = {}
        self.planes = planes 
        self.num_classes = num_classes

        for plane in planes: 
            self.decoder[plane] = self.class_decoder(in_shape)

    def class_decoder(self, in_shape): 
        return torch.ones(in_shape) 
    
    def forward(self, x): 
        return {
            plane: torch.matmul(x[plane], self.decoder[plane]).squeeze(dim=-1) 
            for plane in self.planes
            } 
    
    def classes(self, x): 
        return {
            plane: torch.argmax(torch.nn.functional.softmax(x[plane]), dim=-1) for plane in x.keys()
        }
    
    def probablity(self, x):
        return {
            plane: torch.nn.functional.sigmoid(x[plane]) for plane in x.keys()
        }