from re import X
from data_generation import cnn_training_data
import torch
import torch.nn as nn


def contrastive_data_generation(Antigenic_dist, Virus_seq):
    Special_batch = []
    for i in Virus_seq['description'].values:
        match_list = Antigenic_dist[Antigenic_dist['Strain1'] == i]
        match_list2 = Antigenic_dist[Antigenic_dist['Strain2'] == i]
        match_list2[['Strain1', 'Strain2']
                    ] = match_list2[['Strain2', 'Strain1']]
        match_list = match_list.append(match_list2)
        if not match_list.empty:
            feature, label = cnn_training_data(match_list, Virus_seq)
            Special_batch.append([feature, label])
    print(f'Generation Complete! got {len(Special_batch)} groups')
    return Special_batch


def NTXent_data(X):
    return X, -1*X


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362

        :param temperature: int
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):
        """

        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device(
            "cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(
            projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered -
                      torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(
            1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (
            torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(
            log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(
            supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss
