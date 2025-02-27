from typing import Any
import torch
import scipy.stats as stats


def guassian_kernel(u, sigma=0.1):
    return torch.exp(-0.5 * (u / sigma).pow(2))


class MutualInformation:
    """
    mutual information such that out = KL Divergence(P_x,y||P_x X P_y)
    Using the joint entropy definition: H_x1 + H_x2 - H_x1x2, where H is the shannon entropy (-sum(pk*log(pk)))
    Adapted from this implimation: https://github.com/connorlee77/pytorch-mutual-information/blob/master/MutualInformation.py
    """

    def __init__(self, normalize=False, kernel_function=guassian_kernel) -> None:
        self.normalize = normalize
        self.kernel_function = kernel_function
        self.epsilon = 10 ** (-5)

    def joint_pdf_from_kernel(self, kernel_values1, kernel_values2):
        joint_kernel_values = torch.matmul(kernel_values1.T, kernel_values2)[0]
        normalization = torch.sum(joint_kernel_values, dim=-1)

        pdf = joint_kernel_values / normalization
        return pdf.detach().numpy()

    def marginal_pdf(self, distribution):
        kernel_marginalization = self.kernel_function(distribution)
        pdf = torch.mean(kernel_marginalization, dim=-1)
        normalization = torch.sum(pdf, dim=-1) + self.epsilon
        pdf = pdf / normalization

        return pdf, kernel_marginalization

    def __call__(self, x1, x2) -> Any:
        """Compute the mutual information

        Args:
            x1 (Tensor): distribution 1, with NxC
            x2 (Tensor): distribution 2, with NxC (same dim as x1)

        Returns:
            float: Mutual information (H_x1 + H_x2 - H_x1x2) for x1 and x2
        """

        pdf_x1, x1_marginal = self.marginal_pdf(x1)
        pdf_x2, x2_marginal = self.marginal_pdf(x2)
        pdf_x1x2 = self.joint_pdf_from_kernel(x1_marginal, x2_marginal)

        H_x1 = stats.entropy(pk=pdf_x1.detach().numpy(), axis=0)
        H_x2 = stats.entropy(pk=pdf_x2.detach().numpy(), axis=0)
        H_x1x2 = stats.entropy(pk=pdf_x1x2, axis=0)

        mutual_information = H_x1 + H_x2 - H_x1x2

        if self.normalize:
            mutual_information = 2 * mutual_information / (H_x1 + H_x2)

        return mutual_information


class DistributionEntropy:
    """
    The entropy difference in distributions, as defined as
    k_B (sum(p_1*ln(p_1) - sum(p_2*ln(p_2))))

    Assumes you can just sum over the total probablities for a graph
    """

    def __call__(self, distribution1, distribution2) -> Any:
        step1_entropy = (
            torch.distributions.categorical.Categorical(logits=distribution1)
            .entropy()
            .sum()
        )
        step2_entropy = (
            torch.distributions.categorical.Categorical(logits=distribution2)
            .entropy()
            .sum()
        )
        return (step1_entropy - step2_entropy).detach().numpy()
