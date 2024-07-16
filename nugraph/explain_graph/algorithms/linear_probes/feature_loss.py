import torch
import numpy as np
import os


class FeatureLoss:
    def __init__(
        self, feature: str, planes: list = ["u", "v", "y"], device=None
    ) -> None:
        self.planes = planes
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.included_features = {
            "tracks": self._tracks,
            "hipmip": self._hipmip,
            "node_slope": self._node_slope,
            "michel_conservation": self._michel_required,
            "michel_energy": self._michel_energy,
        }

        for index, feat in enumerate(["wire", "peak", "integral", "rms"]):
            self.included_features[feat] = lambda y_hat, y: self._node_feature(
                y_hat, y, index
            )

        self.func = self.included_features[feature]
        self.michel_distribution = MichelDistribution()

    def loss(self, y_hat, y):
        loss = 0
        for p in self.planes:
            loss += self.func(y_hat[p], y[p])
        return loss / len(self.planes)

    def _tracks(self, y_hat, label):
        """
        Binary classification for if a hit is part of a track (hip/mip) or not
        """

        def binary_mask(array):
            return torch.isin(
                torch.argmax(
                    torch.nn.functional.softmax(array.float(), dim=-1), dim=-1
                ),
                torch.tensor([0, 1], device=self.device),
            ).float()

        predict_binary_mask = binary_mask(y_hat)
        predict_binary_mask = (
            (torch.stack([predict_binary_mask for _ in range(y_hat.shape[-1])]))
            .swapaxes(0, -1)
            .float()
        )

        y_hat_binary = y_hat * predict_binary_mask
        label_binary = label * binary_mask(label)

        loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(
            y_hat_binary, label_binary.long()
        )
        return loss

    def _hipmip(self, y_hat, label):
        """
        Verifies the model can tell the difference between two different track type hits
        Constructs a binary mask of hip or mip, and then tells the difference between the two
        """
        track_filter = torch.where(
            torch.tensor([True], device=self.device),
            torch.isin(label, torch.tensor([0, 1], device=self.device)),
            other=torch.tensor([torch.nan], device=self.device),
        )  # Pick if in either track class
        y = label * track_filter
        y = y.type(torch.LongTensor).to(torch.device(self.device))
        y_hat = y_hat * torch.stack([track_filter for _ in range(y_hat.size(1))]).mT
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
        y = label["x"][:, feature_index]
        return torch.nn.MSELoss()(y_hat.squeeze().float(), y.float())

    def _michel_required(self, y_hat, label):
        """
        If there is a michel, there must be a mip track

        """
        base_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(y_hat, label)

        michel_index = 3
        mip_index = 0

        include_indices = torch.isin(
            label, torch.tensor([michel_index, mip_index], device=self.device)
        )
        if len(include_indices != 0):
            y = label[include_indices]
            y_hat = y_hat[include_indices]
            return (base_loss + torch.nn.CrossEntropyLoss()(y_hat, y)) / 2
        else:
            return base_loss

    def _michel_energy(self, y_hat, label):
        """
        Michel is within a known mass - so there is a low energy range in which it can be
        Just look at all the hit integral of specifically michel
            - Energy and momentum is conserved, there's a vague linear relationship

        Uses the histogram binning method from Vitor
        """
        # Michel Energy Regularization
        # Get the integral of all michel hits within an event and sum them. Then, use this sum to predict the deposited
        # energy according to a linear relation that I've derived from the h5 dataset. Note that we don't even need to
        # use `edep`, we can use the regularization with the integral directly since they are related by a constant.

        # Hyperparams to tune
        # pdf_amp is a adjustment on the pdf reading
        pdf_amp = 10
        michel_index = 3
        michel_reg_cte = 1e-2
        edep_michel = 0.0
        michel_reg_loss = 0.0

        base_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(y_hat, label.y_semantic)

        # Finding the indices of the entries that truly correspond to michel electrons
        michel_indices = torch.nonzero(torch.argmax(y_hat, dim=1) == michel_index)

        # Getting the `integral` feature of the nodes that the semantic decoder labeled as michel
        integral_michel = torch.sum(
            label.x[michel_indices, 2]
        )  # Integral is the third feature

        # Finding the deposited energy from that `integral`
        edep_michel += integral_michel * 0.00580717

        if edep_michel > 0:
            pdf_value = self.michel_distribution.get_pdf_value(edep_michel)
            michel_reg_loss += michel_reg_cte * (1 - pdf_amp * pdf_value)

        return base_loss + michel_reg_loss


class MichelDistribution:
    def __init__(self, distribution: str = "landau") -> None:
        path = os.path.dirname(__file__)
        distribution_paths = f"{path}/michel_energy_distribution.npz"
        data = np.load(distribution_paths)

        if distribution == "landau":
            self.pdf = data["landau_pdf"]
            self.bin_center = data["landau_bins_center"]
        elif distribution == "data":
            self.pdf = data["data_pdf"]
            self.bin_center = data["data_bins_center"]
        else:
            raise ValueError(f"Cannot initialize distribution {distribution}")

        self.distribution = distribution

    def get_pdf_value(self, edep):
        """Get the PDF from the relevant distribution, which can be `data`, `landau`, and `double_peaked`"""
        closest_idx = np.argmin(np.abs(edep - self.bin_center))
        pdf = self.pdf[closest_idx]
        return pdf
