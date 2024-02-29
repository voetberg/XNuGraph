import torch 

class FeatureLoss: 
    def __init__(self, feature:str, planes:list=['u', 'v', 'y']) -> None:
        self.planes = planes 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.func = {
            "tracks": self._tracks, 
            "hipmip": self._hipmip
        }[feature]

    def loss(self, y_hat, y): 
        loss = [self.func(y_hat[plane], y[plane]) 
            for plane in self.planes
        ]
        return loss
    
    def _tracks(self, x, label): 
        """
            Binary classification for if a hit is part of a track (hip/mip) or not
        """
        binary = torch.tensor([0, 1], device=self.device)
        x_binary_mask = torch.isin(torch.argmax(torch.nn.functional.softmax(x, dim=-1), dim=-1), binary).long()
        x_binary_mask = torch.stack([x_binary_mask for _ in range(x.shape[-1])]).swapaxes(0,-1).swapaxes(0,1)
        label_binary_mask = torch.isin(label['y_semantic'], binary).long()
 
        x_binary = x * x_binary_mask
        label_binary = label['y_semantic'] * label_binary_mask    

        loss = torch.nn.CrossEntropyLoss()(x_binary[:,:,0], label_binary)
        return loss

    def _hipmip(self, x, label): 
        """
            Verifies the model can tell the difference between two different track type hits
            Constructs a binary mask of hip or mip, and then tells the difference between the two
        """
        track_filter = torch.where(
            torch.tensor([True], device=self.device), 
            torch.isin(label['y_semantic'], torch.tensor([0,1], device=self.device)), 
            other=torch.tensor([torch.nan], device=self.device)) # Pick if in either track class
        y_hat = x[:,:,0] * torch.stack([track_filter for _ in range(x.size(1))]).mT
        y = label['y_semantic'] * track_filter
        y = y.type(torch.LongTensor).to(torch.device(self.device))

        loss = torch.nn.CrossEntropyLoss()(y_hat, y)
        return loss