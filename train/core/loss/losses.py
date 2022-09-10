import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Callable


class TripletMarginLossOfflineMining(nn.Module):
    def __init__(self, margin=1.0, p=2.0):
        super().__init__()
        self.margin = margin
        self.p = p

    def forward(self, anchor_embedding, positive_embedding, negative_embedding):
        distance_anchor_positive = (
            (anchor_embedding - positive_embedding).pow(self.p).sum(dim=1).sqrt()
        )
        distance_anchor_negative = (
            (anchor_embedding - negative_embedding).pow(self.p).sum(dim=1).sqrt()
        )
        triplet_loss = F.relu(
            distance_anchor_positive - distance_anchor_negative + self.margin
        )
        return triplet_loss.sum()


class TripletMarginLossOnlineMining(nn.Module):
    def __init__(self, margin=1.0, p=2.0, mining_type="online_hard_negative"):
        super().__init__()
        self.margin = margin
        self.p = p
        self.mining = mining_type

        if mining_type == "online":
            self.loss_function = _normal_triplet_loss
        elif mining_type == "online_hard_negative":
            self.loss_function = _batch_all_triplet_loss
        elif mining_type == "online_hardest_negative":
            self.loss_function = _batch_hard_triplet_loss

    def forward(self, embeddings, labels):
        return self.loss_function(labels, embeddings, self.margin, self.p)


def _get_triplet_masks(labels):
    # indices_equal is a square matrix, 1 in the diagonal and 0 everywhere else
    indices_equal = torch.eye(
        labels.size(0), dtype=torch.bool, device=labels.device
    )
    # indices_not_equal is inversed of indices_equal, 0 in the diagonal and 1 everywhere else
    indices_not_equal = ~indices_equal

    # convention: i (anchor index), j (positive index), k (negative index)
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)
    # Check that anchor, positive, negative are distince to each other
    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    # i_equal_j: indices of anchor-positive pairs
    # ~i_equal_k: indices of anchor-negative pairs
    # indices of valid triplets
    indices_triplets = i_equal_j & (~i_equal_k)
    # Make sure that anchor, positive, negative are distince to each other
    indices_triplets = indices_triplets & distinct_indices
    return indices_triplets


def _normal_triplet_loss(labels, embeddings, margin, p):
    pairwise_distance = torch.cdist(embeddings, embeddings, p=p)

    anchor_positive_distance = pairwise_distance.unsqueeze(2)
    anchor_negative_distance = pairwise_distance.unsqueeze(1)

    # Indexes of all triplets
    mask = _get_triplet_masks(labels)
    # Calucalate triplet loss
    triplet_loss = mask.float() * (
        anchor_positive_distance - anchor_negative_distance + margin
    )

    # Total triplets (including positive and negative triplet)
    n_triplets = mask.sum()

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (n_triplets + 1e-16)
    return triplet_loss, -1


def _batch_all_triplet_loss(labels, embeddings, margin, p):
    pairwise_distance = torch.cdist(embeddings, embeddings, p=p)

    anchor_positive_distance = pairwise_distance.unsqueeze(2)
    anchor_negative_distance = pairwise_distance.unsqueeze(1)

    # Indexes of all triplets
    mask = _get_triplet_masks(labels)
    # Calucalate triplet loss
    triplet_loss = mask.float() * (
        anchor_positive_distance - anchor_negative_distance + margin
    )

    # Remove negative loss (easy triplets)
    triplet_loss[triplet_loss < 0] = 0

    # Count number of positive triplets (where triplet_loss > 0)
    positive_triplet = triplet_loss[triplet_loss > 1e-16]
    n_positive_triplet = positive_triplet.size(0)

    # Total triplets (including positive and negative triplet)
    n_triplets = mask.sum()
    # Fraction of postive triplets in total
    fraction_positive_triplets = n_positive_triplet / (n_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (n_positive_triplet + 1e-16)
    return triplet_loss, fraction_positive_triplets


def _get_anchor_positive_mask(labels):
    # Check that i and j are distince
    indices_equal = torch.eye(
        labels.size(0), dtype=torch.bool, device=labels.device
    )
    indices_not_equal = ~indices_equal

    # Check anchor and negative
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    return labels_equal & indices_not_equal


def _get_anchor_negative_mask(labels):
    return labels.unsqueeze(0) != labels.unsqueeze(1)


def _batch_hard_triplet_loss(labels, embeddings, margin, p):
    pairwise_distance = torch.cdist(embeddings, embeddings, p=p)
    # Indexes
    mask_anchor_positive = _get_anchor_positive_mask(labels).float()
    # Distance between anchors and positives
    anchor_positive_distance = mask_anchor_positive * pairwise_distance

    # Hardest postive for every anchor
    hardest_positive_distance, _ = anchor_positive_distance.max(1, keepdim=True)

    mask_anchor_negative = _get_anchor_negative_mask(labels).float()
    # Add max value in each row to invalid negatives
    max_anchor_negative_distance, _ = pairwise_distance.max(dim=1, keepdim=True)
    anchor_negative_distance = pairwise_distance + max_anchor_negative_distance * (
        1.0 - mask_anchor_negative
    )

    # Hardest negative for every anchor
    hardest_negative_distance, _ = anchor_negative_distance.min(dim=1, keepdim=True)

    triplet_loss = (
        hardest_positive_distance - hardest_negative_distance + margin
    )
    triplet_loss[triplet_loss < 0] = 0
    triplet_loss = triplet_loss.mean()
    return triplet_loss, -1
