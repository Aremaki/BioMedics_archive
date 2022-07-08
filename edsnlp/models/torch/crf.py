import torch

IMPOSSIBLE = -10000000


def multi_dim_triu(x, diagonal=0):
    return x.masked_fill(
        ~torch.ones(x.shape[-2], x.shape[-1], dtype=torch.bool, device=x.device).triu(
            diagonal=diagonal
        ),
        0,
    )


def masked_flip(x, mask, dim_x=-2):
    flipped_x = torch.zeros_like(x)
    flipped_x[mask] = x.flip(dim_x)[mask.flip(-1)]
    return flipped_x


@torch.jit.script
def logsumexp_reduce(log_A, log_B):
    # log_A: 2 * N * M
    # log_B: 2 *     M * O
    # out: 2 * N * O
    return (log_A.unsqueeze(-1) + log_B.unsqueeze(-3)).logsumexp(-2)


@torch.jit.script
def max_reduce(log_A, log_B):
    # log_A: 2 * N * M
    # log_B: 2 *     M * O
    # out: 2 * N * O
    return (log_A.unsqueeze(-1) + log_B.unsqueeze(-3)).max(-2)


# noinspection PyTypeChecker
class LinearChainCRF(torch.nn.Module):
    def __init__(
        self,
        forbidden_transitions,
        start_forbidden_transitions=None,
        end_forbidden_transitions=None,
        learnable_transitions=True,
        with_start_end_transitions=True,
    ):
        """
        A linear chain CRF in Pytorch

        Parameters
        ----------
        forbidden_transitions: torch.BoolTensor
            Shape: n_tags * n_tags
            Impossible transitions (1 means impossible) from position n to position n+1
        start_forbidden_transitions: Optional[torch.BoolTensor]
            Shape: n_tags
            Impossible transitions at the start of a sequence
        end_forbidden_transitions Optional[torch.BoolTensor]
            Shape: n_tags
            Impossible transitions at the end of a sequence
        learnable_transitions: bool
            Should we learn transition scores to complete the
            constraints ?
        with_start_end_transitions:
            Should we apply start-end transitions.
            If learnable_transitions is True, learn start/end transition scores
        """
        super().__init__()

        num_tags = forbidden_transitions.shape[0]

        self.register_buffer("forbidden_transitions", forbidden_transitions.bool())
        if start_forbidden_transitions is not None:
            self.register_buffer(
                "start_forbidden_transitions", start_forbidden_transitions.bool()
            )
        else:
            self.register_buffer(
                "start_forbidden_transitions", torch.zeros(num_tags, dtype=torch.bool)
            )
        if end_forbidden_transitions is not None:
            self.register_buffer(
                "end_forbidden_transitions", end_forbidden_transitions.bool()
            )
        else:
            self.register_buffer(
                "end_forbidden_transitions", torch.zeros(num_tags, dtype=torch.bool)
            )

        if learnable_transitions:
            self.transitions = torch.nn.Parameter(
                torch.zeros_like(forbidden_transitions, dtype=torch.float)
            )
        else:
            self.register_buffer(
                "transitions",
                torch.zeros_like(forbidden_transitions, dtype=torch.float),
            )

        if learnable_transitions and with_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(
                torch.zeros(num_tags, dtype=torch.float)
            )
        else:
            self.register_buffer(
                "start_transitions", torch.zeros(num_tags, dtype=torch.float)
            )

        if learnable_transitions and with_start_end_transitions:
            self.end_transitions = torch.nn.Parameter(
                torch.zeros(num_tags, dtype=torch.float)
            )
        else:
            self.register_buffer(
                "end_transitions", torch.zeros(num_tags, dtype=torch.float)
            )

    def decode(self, emissions, mask):
        """
        Decodes a sequence of tag scores using the Viterbi algorithm

        Parameters
        ----------
        emissions: torch.FloatTensor
            Shape: ... * n_tokens * n_tags
        mask: torch.BoolTensor
            Shape: ... * n_tokens

        Returns
        -------
        torch.LongTensor
            Backtrack indices (= argmax), ie best tag sequence
        """
        transitions = self.transitions.masked_fill(
            self.forbidden_transitions, IMPOSSIBLE
        )
        start_transitions = self.start_transitions.masked_fill(
            self.start_forbidden_transitions, IMPOSSIBLE
        )
        end_transitions = self.end_transitions.masked_fill(
            self.end_forbidden_transitions, IMPOSSIBLE
        )
        n_samples, n_tokens = mask.shape

        emissions[..., 1:][~mask] = IMPOSSIBLE
        emissions = emissions.transpose(0, 1)

        # emissions: n_tokens * n_samples * n_tags
        out = [emissions[0] + start_transitions]
        backtrack = []

        for k in range(1, len(emissions)):
            res, indices = max_reduce(out[-1], transitions)
            backtrack.append(indices)
            out.append(res + emissions[k])

        res, indices = max_reduce(out[-1], end_transitions.unsqueeze(-1))
        path = torch.zeros(n_samples, n_tokens, dtype=torch.long)
        path[:, -1] = indices.squeeze(-1)

        path_range = torch.arange(n_samples, device=path.device)
        if len(backtrack) > 1:
            # Backward max path following
            for k, b in enumerate(backtrack[::-1]):
                path[:, -k - 2] = b[path_range, path[:, -k - 1]]

        return path

    def propagate(
        self,
        emissions,
        mask,
        tags=None,
        ring_op_name="logsumexp",
        use_constraints=True,
        way="forward",
    ):
        """
        Propagate scores through the CRF to perform either
        - Viterbi decoding (ring_op_name = max)
        - Posterior inference for training (ring_op_name = posterior)
        - Marginalization to get smoothed log probs (ring_op_name = logsumexp)

        Parameters
        ----------
        emissions: torch.FloatTensor
            Shape: ... * n_tokens * n_tags
        mask: torch.BoolTensor
            Shape: ... * n_tokens
        tags: torch.BoolTensor
            Shape: ... * n_tokens * n_tags
        ring_op_name: str
            Which ring operation should we carry when reducing the
            emissions sequences
        use_constraints: bool
            Whether to enforce constraints (default = True)
        way: str
            Whether to run the propagation forward or backward

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]
        - Reduced score = normalization constant or max viterbi score)
        - Log probabilities = intermediate scores reduction at each time step
        - Backtrack indices (= argmax) only non null for viterbi decoding
        """
        """
        Each alpha is the potential for the state given all previous observations
        and the current one
        """
        emissions = emissions.transpose(0, 1).double()
        mask = mask.transpose(0, 1)

        if tags is not None:
            if len(tags.shape) == 2:
                tags = tags.transpose(0, 1).unsqueeze(1)
            elif len(tags.shape) == 3:
                tags = tags.permute(2, 0, 1)

        backtrack = None

        if ring_op_name == "logsumexp":

            def ring_op(last_potential, trans, loc):
                return (
                    last_potential.unsqueeze(-1) + trans.unsqueeze(0).unsqueeze(0)
                ).logsumexp(2)

        elif ring_op_name == "posterior":

            def ring_op(last_potential, trans, loc):
                return trans[tags[loc]] + last_potential[
                    torch.arange(tags.shape[1]).unsqueeze(1),
                    torch.arange(tags.shape[2]).unsqueeze(0),
                    tags[loc],
                ].unsqueeze(-1)

        elif ring_op_name == "max":
            backtrack = []

            def ring_op(last_potential, trans, loc):
                res, indices = (
                    last_potential.unsqueeze(-1) + trans.unsqueeze(0).unsqueeze(0)
                ).max(2)
                backtrack.append(indices)
                return res

        else:
            raise NotImplementedError()

        if use_constraints:
            start_transitions = self.start_transitions.masked_fill(
                self.start_forbidden_transitions, IMPOSSIBLE
            )
            transitions = self.transitions.masked_fill(
                self.forbidden_transitions, IMPOSSIBLE
            )
            end_transitions = self.end_transitions.masked_fill(
                self.end_forbidden_transitions, IMPOSSIBLE
            )
        else:
            start_transitions = self.start_transitions
            transitions = self.transitions
            end_transitions = self.end_transitions

        if way == "backward":
            assert ring_op_name != "max", "Unsupported"
            start_transitions, end_transitions = end_transitions, start_transitions
            transitions = transitions.t()
            emissions = masked_flip(
                emissions.transpose(0, 1), mask.transpose(0, 1), -2
            ).transpose(0, 1)

        log_probs = [
            (start_transitions + emissions[0])
            .unsqueeze(0)
            .repeat_interleave(tags.shape[1] if tags is not None else 1, dim=0)
        ]

        for k in range(1, len(emissions)):
            res = ring_op(log_probs[-1], transitions, k - 1)  # - max_transitions
            # log_probs.append(res + emissions[k] + max_transitions)
            log_probs.append(
                torch.where(mask[k].unsqueeze(-1), res + emissions[k], log_probs[-1])
            )

        if ring_op_name == "logsumexp":
            z = ring_op(log_probs[-1], end_transitions.unsqueeze(1), 0)
        else:
            z = ring_op(
                log_probs[-1],
                end_transitions.unsqueeze(1),
                (
                    (mask.sum(0) - 1).unsqueeze(0),
                    torch.arange(log_probs[-1].shape[0]).unsqueeze(1),
                    torch.arange(mask.shape[1]).unsqueeze(0),
                ),
            ).squeeze(-1)

        log_probs = torch.cat(log_probs, dim=0)

        if way == "backward":
            log_probs = masked_flip(
                log_probs.transpose(0, 1),
                mask.transpose(0, 1),
                dim_x=-2,
            ).transpose(0, 1)

        return z, log_probs, backtrack

    def marginal(self, emissions, mask):
        """
        Compute the marginal log-probabilities of the tags
        given the emissions and the transition probabilities and
        constraints of the CRF

        We could use the `propagate` method but this implementation
        is faster.

        Parameters
        ----------
        emissions: torch.FloatTensor
            Shape: ... * n_tokens * n_tags
        mask: torch.BoolTensor
            Shape: ... * n_tokens

        Returns
        -------
        torch.FloatTensor
            Shape: ... * n_tokens * n_tags
        """
        device = emissions.device

        transitions = self.transitions.masked_fill(
            self.forbidden_transitions, IMPOSSIBLE
        )
        start_transitions = self.start_transitions.masked_fill(
            self.start_forbidden_transitions, IMPOSSIBLE
        )
        end_transitions = self.end_transitions.masked_fill(
            self.end_forbidden_transitions, IMPOSSIBLE
        )

        bi_transitions = torch.stack([transitions, transitions.t()], dim=0)

        # add start transitions (ie cannot start with ...)
        emissions[:, 0] = emissions[:, 0] + start_transitions

        # add end transitions (ie cannot end with ...): flip the emissions along the
        # token axis, and add the end transitions
        # emissions = masked_flip(emissions, mask, dim_x=1)
        emissions[
            torch.arange(mask.shape[0], device=device), mask.long().sum(1) - 1
        ] = (
            emissions[
                torch.arange(mask.shape[0], device=device),
                mask.long().sum(1) - 1,
            ]
            + end_transitions
        )

        # stack start -> end emissions (needs to flip the previously flipped emissions),
        # and end -> start emissions
        bi_emissions = torch.stack(
            [emissions, masked_flip(emissions, mask, dim_x=1)], 1
        )
        bi_emissions = bi_emissions.transpose(0, 2)

        out = [bi_emissions[0]]
        for k in range(1, len(bi_emissions)):
            res = logsumexp_reduce(out[-1], bi_transitions)
            out.append(res + bi_emissions[k])
        out = torch.stack(out, dim=0).transpose(0, 2)

        forward = out[:, 0]
        backward = masked_flip(out[:, 1], mask, dim_x=1)
        backward_z = backward[:, 0].logsumexp(-1)

        return forward + backward - emissions - backward_z[:, None, None]

    def forward(self, emissions, mask, target):
        """
        Compute the posterior reduced log-probabilities of the tags
        given the emissions and the transition probabilities and
        constraints of the CRF, ie the loss.


        We could use the `propagate` method but this implementation
        is faster.

        Parameters
        ----------
        emissions: torch.FloatTensor
            Shape: ... * n_tokens * n_tags
        mask: torch.BoolTensor
            Shape: ... * n_tokens
        target: torch.BoolTensor
            Shape: ... * n_tokens * n_tags
            The target tags represented with 1-hot encoding
            We use 1-hot instead of long format to handle
            cases when multiple tags at a given position are
            allowed during training.

        Returns
        -------
        torch.FloatTensor
            Shape: ...
            The loss
        """
        transitions = self.transitions.masked_fill(
            self.forbidden_transitions, IMPOSSIBLE
        )
        start_transitions = self.start_transitions.masked_fill(
            self.start_forbidden_transitions, IMPOSSIBLE
        )
        end_transitions = self.end_transitions.masked_fill(
            self.end_forbidden_transitions, IMPOSSIBLE
        )

        bi_emissions = torch.stack(
            [emissions.masked_fill(~target, IMPOSSIBLE), emissions], 1
        ).transpose(0, 2)

        # emissions: n_samples * n_tokens * n_tags
        # bi_emissions: n_tokens * 2 * n_samples * n_tags
        out = [bi_emissions[0] + start_transitions]

        for k in range(1, len(bi_emissions)):
            res = logsumexp_reduce(out[-1], transitions)
            out.append(res + bi_emissions[k])
        out = torch.stack(out, dim=0).transpose(0, 2)
        # n_samples * 2 * n_tokens * n_tags
        z = (
            masked_flip(out, mask.unsqueeze(1).repeat(1, 2, 1), dim_x=2)[:, :, 0]
            + end_transitions
        )
        supervised_z = z[:, 0].logsumexp(-1)
        unsupervised_z = z[:, 1].logsumexp(-1)
        return -(supervised_z - unsupervised_z)


# noinspection PyTypeChecker
class MultiLabelBIOULDecoder(LinearChainCRF):
    def __init__(
        self,
        num_labels,
        with_start_end_transitions=True,
        learnable_transitions=True,
    ):
        """
        Create a linear chain CRF with hard constraints to enforce the BIOUL tagging
        scheme

        Parameters
        ----------
        num_labels: int
        with_start_end_transitions: bool
        learnable_transitions: bool
        """
        O, I, B, L, U = 0, 1, 2, 3, 4

        num_tags = 1 + num_labels * 4
        self.num_tags = num_tags
        forbidden_transitions = torch.ones(num_tags, num_tags, dtype=torch.bool)
        forbidden_transitions[O, O] = 0  # O to O
        for i in range(num_labels):
            STRIDE = 4 * i
            for j in range(num_labels):
                STRIDE_J = j * 4
                forbidden_transitions[L + STRIDE, B + STRIDE_J] = 0  # L-i to B-j
                forbidden_transitions[L + STRIDE, U + STRIDE_J] = 0  # L-i to U-j
                forbidden_transitions[U + STRIDE, B + STRIDE_J] = 0  # U-i to B-j
                forbidden_transitions[U + STRIDE, U + STRIDE_J] = 0  # U-i to U-j

            forbidden_transitions[O, B + STRIDE] = 0  # O to B-i
            forbidden_transitions[B + STRIDE, I + STRIDE] = 0  # B-i to I-i
            forbidden_transitions[I + STRIDE, I + STRIDE] = 0  # I-i to I-i
            forbidden_transitions[I + STRIDE, L + STRIDE] = 0  # I-i to L-i
            forbidden_transitions[B + STRIDE, L + STRIDE] = 0  # B-i to L-i

            forbidden_transitions[L + STRIDE, O] = 0  # L-i to O
            forbidden_transitions[O, U + STRIDE] = 0  # O to U-i
            forbidden_transitions[U + STRIDE, O] = 0  # U-i to O

        start_forbidden_transitions = torch.zeros(num_tags, dtype=torch.bool)
        if with_start_end_transitions:
            for i in range(num_labels):
                STRIDE = 4 * i
                start_forbidden_transitions[I + STRIDE] = 1  # forbidden to start by I-i
                start_forbidden_transitions[L + STRIDE] = 1  # forbidden to start by L-i

        end_forbidden_transitions = torch.zeros(num_tags, dtype=torch.bool)
        if with_start_end_transitions:
            for i in range(num_labels):
                STRIDE = 4 * i
                end_forbidden_transitions[I + STRIDE] = 1  # forbidden to end by I-i
                end_forbidden_transitions[B + STRIDE] = 1  # forbidden to end by B-i

        super().__init__(
            forbidden_transitions,
            start_forbidden_transitions,
            end_forbidden_transitions,
            with_start_end_transitions=with_start_end_transitions,
            learnable_transitions=learnable_transitions,
        )

    @staticmethod
    def spans_to_tags(
        spans: torch.Tensor, n_samples: int, n_labels: int, n_tokens: int
    ):
        """
        Convert a tensor of spans of shape n_spans * (doc_idx, label, begin, end)
        to a matrix of BIOUL tags of shape n_samples * n_labels * n_tokens

        Parameters
        ----------
        spans: torch.Tensor
        n_samples: int
        n_labels: int
        n_tokens: int

        Returns
        -------
        torch.Tensor
        """
        device = spans.device
        cpu = torch.device("cpu")
        if not len(spans):
            return torch.zeros(
                n_samples, n_labels, n_tokens, dtype=torch.long, device=device
            )
        doc_indices, label_indices, begins, ends = spans.cpu().unbind(-1)
        ends = ends - 1

        pos = torch.arange(n_tokens, device=cpu)
        b_tags, l_tags, u_tags, i_tags = torch.zeros(
            4, n_samples, n_labels, n_tokens, dtype=torch.bool, device=cpu
        ).unbind(0)
        tags = torch.zeros(n_samples, n_labels, n_tokens, dtype=torch.long, device=cpu)
        where_u = begins == ends
        u_tags[doc_indices[where_u], label_indices[where_u], begins[where_u]] = True
        b_tags[doc_indices[~where_u], label_indices[~where_u], begins[~where_u]] = True
        l_tags[doc_indices[~where_u], label_indices[~where_u], ends[~where_u]] = True
        i_tags.view(-1, n_tokens).index_add_(
            0,
            doc_indices * n_labels + label_indices,
            (begins.unsqueeze(-1) < pos) & (pos < ends.unsqueeze(-1)),
        )
        tags[u_tags] = 4
        tags[b_tags] = 2
        tags[l_tags] = 3
        tags[i_tags.bool()] = 1
        return tags.to(device)

    @staticmethod
    def tags_to_spans(tags):
        """
        Convert a sequence of multiple label BIOUL tags to a sequence of spans

        Parameters
        ----------
        tags: torch.LongTensor
            Shape: n_samples * n_labels * n_tokens
        mask: torch.BoolTensor
            Shape: n_samples * n_labels * n_tokens

        Returns
        -------
        torch.LongTensor
            Shape: n_spans *  4
            (doc_idx, label_idx, begin, end)
        """
        return torch.cat(
            [
                torch.nonzero((tags == 4) | (tags == 2)),
                torch.nonzero((tags == 4) | (tags == 3))[..., [-1]] + 1,
            ],
            dim=-1,
        )