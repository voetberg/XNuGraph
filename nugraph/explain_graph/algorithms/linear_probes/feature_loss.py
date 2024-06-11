import torch
from nugraph.data.feature_generation import FeatureGeneration

class FeatureLoss:
    def __init__(self, feature: str, planes: list = ["u", "v", "y"], device=None) -> None:
        self.planes = planes
        if device is None: 
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else: 
            self.device = device

        self.included_features = {
            "tracks": self._tracks, 
            "hipmip": self._hipmip,
            "node_slope": self._node_slope, 
            "michel_conservation": self._michel_required
        }

        for index, feat in enumerate(['wire', 'peak', 'integral', 'rms']): 
            self.included_features[feat] = lambda y_hat, y: self._node_feature(y_hat, y, index)

        self.func = self.included_features[feature]

    def loss(self, y_hat, y):
        loss = 0
        for p in self.planes: 
            loss += self.func(y_hat[p], y[p])
        return loss/len(self.planes)

    def _tracks(self, x, label):
        """
        Binary classification for if a hit is part of a track (hip/mip) or not
        """
        binary = torch.tensor([0, 1], device=self.device)
        x_binary_mask = torch.isin(
            torch.argmax(torch.nn.functional.softmax(x, dim=-1), dim=-1), binary
        ).long()
        x_binary_mask = (
            torch.stack([x_binary_mask for _ in range(x.shape[-1])])
            .swapaxes(0, -1)
            .swapaxes(0, 1)
        )
        label_binary_mask = torch.isin(label["y_semantic"], binary).long()

        x_binary = x * x_binary_mask
        label_binary = label["y_semantic"] * label_binary_mask

        if len(x_binary.shape) == 3:
            x_binary = x_binary[:, :, 0]
        loss = torch.nn.CrossEntropyLoss()(x_binary, label_binary)
        return loss

    def _hipmip(self, x, label):
        """
        Verifies the model can tell the difference between two different track type hits
        Constructs a binary mask of hip or mip, and then tells the difference between the two
        """
        track_filter = torch.where(
            torch.tensor([True], device=self.device),
            torch.isin(label["y_semantic"], torch.tensor([0, 1], device=self.device)),
            other=torch.tensor([torch.nan], device=self.device),
        )  # Pick if in either track class
        if len(x.shape) == 3:
            x = x[:, :, 0]
        else:
            x = x.expand(-1, 5)  # Cheating :0
        y_hat = x * torch.stack([track_filter for _ in range(x.size(1))]).mT
        y = label["y_semantic"] * track_filter
        y = y.type(torch.LongTensor).to(torch.device(self.device))
        try:
            loss = torch.nn.CrossEntropyLoss()(y_hat, y)
        except IndexError:
            print(y_hat, y)
        return loss

    def _node_slope(self, x, label): 
        """
        Predict the node slope and compare it to the truth
        """
        y = FeatureGeneration().node_slope(label)
        y_hat = x
        return torch.nn.MSELoss()(y_hat, y)

    def _node_feature(self, x, label, feature_index): 
        """
        X is a prediction of a node feature 
        """
        y = label.collect('x')[:, feature_index]
        y_hat = x
        return torch.nn.MSELoss(y_hat, y)

    def _michel_required(self, x, label): 
        """
        If there is a michel, there must be a mip track 

        """

        michel_index = 3 
        mip_index = 0

        def michel_ratio(labels): 
            n_michel = labels * torch.where(
                torch.tensor([1], device=self.device),
                torch.isin(labels, torch.tensor([michel_index], device=self.device)),
                other=torch.tensor([0], device=self.device))
            n_michel = torch.sum(n_michel)

            n_mip = labels * torch.where(
                torch.tensor([1], device=self.device),
                torch.isin(labels, torch.tensor([mip_index], device=self.device)),
                other=torch.tensor([0], device=self.device))
            n_mip = torch.sum(n_mip)
            try: 
                n_mip = 1/n_mip 
            except DivisionByZeroError: 
                n_mip = torch.NaN 

            return 1/(n_michel - n_mip)

        y_hat = torch.argmax(torch.nn.functional.softmax(x, dim=-1), dim=-1)
        y_hat_loss = michel_ratio(y_hat)

        return y_hat_loss

    def _michel_energy(self, x, label):
        """
        Michel is within a known mass - so there is a low energy range in which it can be
        Just look at all the hit integral of specifically michel 
            - Energy and momentum is conserved, there's a vague linear relationship 
        """
        pass
