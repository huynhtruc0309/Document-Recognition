import numpy as np


class TripletUtils:
    @staticmethod
    def random_triplets(targets):
        """Random triplet [anchor, positive, negative].
        Return list index tupple with the same size as targets list
        Args:
            targets (List[int]): list of targets from dataset
        """
        targets_set = set(targets)
        targets = np.asarray(targets)
        tar2idxs = {target: np.where(targets == target)[0] for target in targets_set}
        random_state = np.random.RandomState(29)
        triplets = [
            (
                i,
                random_state.choice(tar2idxs[targets[i]]),
                random_state.choice(
                    tar2idxs[np.random.choice(list(targets_set - set([targets[i]])))]
                ),
            )
            for i in range(len(targets))
        ]
        return triplets

    @staticmethod
    def all_triplets(targets):
        """[summary]

        Args:
            targets (List[int]): list of targets from dataset
        """
        anchor_positive_pairs = []
        for idx_anchor, _ in enumerate(targets):
            for idx_positive in range(idx_anchor + 1, len(targets)):
                if targets[idx_anchor] == targets[idx_positive]:
                    anchor_positive_pairs.append((idx_anchor, idx_positive))

        triplets = []
        for idx_anchor, idx_positive in anchor_positive_pairs:
            for idx_negative, _ in enumerate(targets):
                if targets[idx_anchor] != targets[idx_negative]:
                    triplets.append((idx_anchor, idx_positive, idx_negative))

        return triplets

    # def _get_triplet_mask_pytorch(labels):
    #     """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    #     A triplet (i, j, k) is valid if:
    #         - i, j, k are distinct
    #         - labeddls[i] == labels[j] and labels[i] != labels[k]
    #     Args:
    #         labels: tf.int32 `Tensor` with shape [batch_size]
    #     """
    #     # Check that i, j and k are distinct
    #     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #     labels = torch.from_numpy(np.asarray(labels)).to(device)
    #     indices_equal = torch.eye(labels.shape[0]).type(torch.bool).to(device)
    #     indices_not_equal = torch.bitwise_not(indices_equal)

    #     i_not_equal_j = indices_not_equal.unsqueeze(2)
    #     i_not_equal_k = indices_not_equal.unsqueeze(1)
    #     j_not_equal_k = indices_not_equal.unsqueeze(0)

    #     distinct_indices = i_not_equal_j.__and__(i_not_equal_k).__and__(j_not_equal_k)

    #     # Check if labels[i] == labels[j] and labels[i] != labels[k]

    #     label_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))

    #     i_equal_j = label_equal.unsqueeze(2)
    #     i_equal_k = label_equal.unsqueeze(1)

    #     valid_labels = i_equal_j.__and__(torch.bitwise_not(i_equal_k))

    #     # Combine the two masks
    #     mask = distinct_indices.__and__(valid_labels)

    #     return mask.nonzero().numpy()

    # def _get_all_triplets(targets: List[int]) -> List[Tuple[int, int, int]]:
    #     """Return all possible triplets
    #         A triplet (i, j, k) is valid if:
    #             - i, j, k are distinct
    #             - labeddls[i] == labels[j] and labels[i] != labels[k]
    #         Args:
    #             labels: tf.int32 `Tensor` with shape [batch_size]
    #         """
    #     # Check that i, j and k are distinct
    #     targets = np.asarray(targets)
    #     indices_equal = np.eye(targets.shape[0], dtype=bool)
    #     indices_not_equal = ~indices_equal

    #     i_not_equal_j = np.expand_dims(indices_not_equal, 2)
    #     i_not_equal_k = np.expand_dims(indices_not_equal, 1)
    #     j_not_equal_k = np.expand_dims(indices_not_equal, 0)

    #     distinct_indices = i_not_equal_j.__and__(i_not_equal_k).__and__(j_not_equal_k)
    #     # Check if labels[i] == labels[j] and labels[i] != labels[k]
    #     target_equal = np.equal(np.expand_dims(targets, 0), np.expand_dims(targets, 1))
    #     i_equal_j = np.expand_dims(target_equal, 2)
    #     i_not_equal_k = ~np.expand_dims(target_equal, 1)
    #     valid_targets = i_equal_j.__and__(i_not_equal_k)

    #     # Combine the two masks
    #     mask = distinct_indices.__and__(valid_targets)

    #     # a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    #     a, p, n = np.where(mask)
    #     return list(zip(a, p, n))
