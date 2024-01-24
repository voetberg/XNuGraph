import torch 

class FeatureLoss: 
    def __init__(self, feature:str, planes:list=['u', 'v', 'y']) -> None:
        self.func = {
            "tracks": FeatureLoss._tracks, 
            "hipmip": FeatureLoss._hipmip
        }[feature]
        self.planes = planes 

    def loss(self, y_hat, y): 
        loss = [self.func(y_hat[plane], y[plane]) 
            for plane in self.planes
        ]
        return loss
    
    @staticmethod
    def _tracks(x, label): 
        """
            Binary classification for if a hit is part of a track (hip/mip) or not
        """
        binary = torch.Tensor([0, 1])
        x_binary_mask = torch.isin(torch.argmax(torch.nn.functional.softmax(x, dim=-1), dim=-1), binary).long()
        x_binary_mask = torch.stack([x_binary_mask for _ in range(x.shape[-1])]).swapaxes(0,-1).swapaxes(0,1)
        label_binary_mask = torch.isin(label['y_semantic'], binary).long()
 
        x_binary = x * x_binary_mask
        label_binary = label['y_semantic'] * label_binary_mask

        loss = torch.nn.CrossEntropyLoss()(x_binary[:,:,0], label_binary)
        return loss

    @staticmethod
    def _hipmip(x, label): 
        pass 