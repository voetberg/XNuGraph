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

    def _tracks(self, y_hat, label):
        """
        Binary classification for if a hit is part of a track (hip/mip) or not
        """
        binary = torch.tensor([0, 1], device=self.device)
        label_binary_mask = torch.isin(label["y_semantic"], binary).float()

        label_binary = label["y_semantic"] * label_binary_mask
        y_hat = y_hat.squeeze().type(torch.float) * label_binary_mask

        loss = torch.nn.CrossEntropyLoss()(y_hat, label_binary)
        return loss

    def _hipmip(self, y_hat, label):
        """
        Verifies the model can tell the difference between two different track type hits
        Constructs a binary mask of hip or mip, and then tells the difference between the two
        """
        track_filter = torch.where(
            torch.tensor([True], device=self.device),
            torch.isin(label["y_semantic"], torch.tensor([0, 1], device=self.device)),
            other=torch.tensor([torch.nan], device=self.device),
        )  # Pick if in either track class

        y_hat = y_hat.squeeze().type(torch.float) * track_filter
        y = label["y_semantic"] * track_filter
        y = y.type(torch.float).to(torch.device(self.device))

        loss = torch.nn.CrossEntropyLoss()(y_hat, y)

        return loss

    def _node_slope(self, y_hat, label): 
        """
        Predict the node slope and compare it to the truth
        """

        positions = label["pos"]
        m = torch.tensor(positions[:, 1] / positions[:, 0])
        y = m.unsqueeze(-1)
        return torch.nn.MSELoss()(y_hat, y)

    def _node_feature(self, y_hat, label, feature_index): 
        """
        X is a prediction of a node feature 
        """
        y = label['x'][:, feature_index]
        return torch.nn.MSELoss()(y_hat.squeeze().float(), y.float())

    def _michel_required(self, y_hat, label): 
        """
        If there is a michel, there must be a mip track 

        """
        michel_index = 3 
        mip_index = 0

        def michel_ratio(labels): 
            n_michel = labels * torch.where(
                torch.isin(labels, torch.tensor([michel_index], device=self.device)).bool(),
                 torch.tensor([1], device=self.device).float(),
                other=torch.tensor([0], device=self.device).float())
            n_michel = torch.sum(n_michel)

            n_mip = labels * torch.where(
                torch.isin(labels, torch.tensor([mip_index], device=self.device)).bool(),
                torch.tensor([1], device=self.device).float(),
                other=torch.tensor([0], device=self.device).float())
            n_mip = torch.sum(n_mip)
            try: 
                n_mip = 1/n_mip 
            except ZeroDivisionError: 
                n_mip = torch.NaN 

            return 1/(n_michel - n_mip)

        true_michel_ratio = michel_ratio(label['y_semantic']).float()
        y = torch.concat([torch.tensor([true_michel_ratio]) for _ in range(y_hat.shape[0])]).to(self.device)
        return torch.nn.MSELoss()(y_hat.squeeze().float(), y)

    def _michel_energy(self, x, label):
        """
        Michel is within a known mass - so there is a low energy range in which it can be
        Just look at all the hit integral of specifically michel 
            - Energy and momentum is conserved, there's a vague linear relationship 
        """
        pass
