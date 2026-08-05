"""Module 2 (v3): Similarity-Aware Neighbor Selector with RL (Dou et al.
CIKM 2020, Section 3.3, Eq. 5-7, Algorithm 1 lines 15-19).

2a. Top-p sampling: unchanged mechanism from v2 (rank a center node's
neighbours under relation r by similarity score descending, keep the first
p_r^(l) * |neighbors| of them). This part of v2 was already correct against
the paper and needed no change.

2b. RL threshold update: rewritten entirely, replacing v2's Q-learning-style
direction/step-decay mechanism. The paper's actual rule uses a FIXED step
size tau (no decay, no persisted direction), a direct reward-sign action
(p += reward * tau), a random +-tau action on the first epoch (no reference
distance to compare against yet), and a terminal condition (Eq. 7) that
freezes p permanently once the RECENT reward signal goes flat/balanced.

Eq. 7 correction (the paper excerpt was OCR-garbled the first time this was
implemented): |sum_{e-10}^{e} f(p_r^(l), a_r^(l))^(e)| <= 2, for e >= 10.
Two things the original implementation got wrong:
  1. The comparison is against the ABSOLUTE VALUE of the summed rewards, not
     the raw (possibly negative) sum. Termination fires on a BALANCED or
     OSCILLATING reward signal (sum near zero -- the policy has stabilized
     and keeps flip-flopping between raise/lower), not on a sustained
     LOSING STREAK (sum strongly negative -- the policy is still reliably
     getting worse and should keep adjusting, not freeze). Those are
     opposite situations; comparing the raw sum to <= 2 would incorrectly
     freeze on a losing streak (sum e.g. -10 <= 2 is True) while correctly
     also freezing on balance (sum e.g. -1 <= 2 is True) -- it could not
     tell the two apart. abs(sum) <= 2 fixes this: a losing streak has
     abs(sum) large (close to the window size), balance has abs(sum) small.
  2. The window e-10..e is INCLUSIVE, i.e. 11 terms, not 10. `terminal_window`
     keeps its name and value (10, matching the paper's "e-10" lookback) --
     what changed is that the code now correctly requires
     `terminal_window + 1` accumulated rewards before checking, not
     `terminal_window`.
"""

import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class RLSelectorState:
    p: float
    reward_history: list = field(default_factory=list)  # last window_size rewards (+-1), for Eq. 7
    prev_distance: Optional[float] = None
    terminated: bool = False  # once True, p is frozen for the rest of training
    full_history: list = field(default_factory=list)  # (epoch, p, distance, reward) for logging


class SimilarityAwareSelector(nn.Module):
    """
    Top-p neighbour filter with the paper's actual RL update rule: fixed step
    size tau, direct reward-sign action (no persisted direction, no decay),
    and a terminal condition that freezes p once the recent reward trend
    plateaus (Eq. 7).
    """

    def __init__(
        self,
        num_relations: int,
        init_p: float = 0.5,
        tau: float = 0.05,               # paper: "a fixed small value tau"
        terminal_window: int = 10,       # Eq. 7: the "e-10" lookback -- window_size = terminal_window + 1
        terminal_threshold: float = 2.0, # Eq. 7: |sum of window_size rewards| <= 2 -> freeze
        seed: int = 42,
    ):
        super().__init__()
        self.num_relations = num_relations
        self.tau = tau
        self.terminal_window = terminal_window
        self.window_size = terminal_window + 1  # e-10 through e, inclusive, per Eq. 7
        self.terminal_threshold = terminal_threshold

        self.register_buffer("p", torch.full((num_relations,), float(init_p)))
        self._states = [RLSelectorState(p=init_p) for _ in range(num_relations)]
        self._rng = random.Random(seed)

    def forward(
        self,
        similarity_scores: torch.Tensor,   # [N, max_neighbors]
        relation_idx: int,
        neighbor_mask: torch.Tensor,       # [N, max_neighbors] bool
    ) -> torch.Tensor:
        """Top-p filtering, unchanged from v2 -- this part was already correct."""
        p = float(self.p[relation_idx].item())
        N, K = similarity_scores.shape

        scores = similarity_scores.masked_fill(~neighbor_mask, float("-inf"))
        valid_counts = neighbor_mask.sum(dim=1)
        keep_counts = torch.clamp((valid_counts.float() * p).ceil().long(), min=1)

        order = torch.argsort(scores, dim=1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(
            1, order, torch.arange(K, device=scores.device).unsqueeze(0).expand(N, K)
        )
        return (rank < keep_counts.unsqueeze(1)) & neighbor_mask

    @torch.no_grad()
    def rl_step_relation(self, relation_idx: int, distance: float, epoch: int) -> float:
        """
        Algorithm 1, lines 15-19 + Eq. 5-7. Call once per epoch, per relation,
        per layer (this class represents ONE layer's selector -- care_gnn.py
        instantiates one SimilarityAwareSelector per layer, same pattern as
        the aggregator and similarity modules).

        distance: G(D_r^(l))^(e) -- average Eq.2 distance over selected
        neighbours of FRAUD-labelled training centers only (see
        label_aware_distance below for the required center-node filtering).
        """
        state = self._states[relation_idx]

        if state.terminated:
            return state.p  # frozen, per Eq. 7's terminal condition

        if epoch == 1 or state.prev_distance is None:
            # "we assign random actions for the first epoch since it has no reference"
            action = self.tau if self._rng.random() < 0.5 else -self.tau
            reward = None  # no reward defined yet on the first epoch
        else:
            improved = (state.prev_distance - distance) >= 0  # Eq. 6, no epsilon
            reward = 1.0 if improved else -1.0
            action = reward * self.tau  # Algorithm 1 line 19: p += f * tau, directly

        new_p = min(max(state.p + action, 0.0), 1.0)  # p in [0, 1], closed interval
        state.p = new_p
        state.prev_distance = distance

        if reward is not None:
            state.reward_history.append(reward)
            if len(state.reward_history) > self.window_size:
                state.reward_history.pop(0)
            # len(...) == window_size is itself sufficient to know the
            # e-10..e window is fully populated -- no separate epoch>=N
            # guard needed (reward-bearing epochs start at 2, so the
            # window can only first fill once enough epochs have passed).
            if len(state.reward_history) == self.window_size:
                if abs(sum(state.reward_history)) <= self.terminal_threshold:
                    state.terminated = True  # Eq. 7 satisfied -> freeze permanently

        state.full_history.append((epoch, state.p, distance, reward))
        self.p[relation_idx] = new_p
        return new_p

    def rl_step(self, distances: dict, epoch: int) -> dict:
        return {r: self.rl_step_relation(r, d, epoch) for r, d in distances.items()}

    def get_state(self, relation_idx: int) -> RLSelectorState:
        return self._states[relation_idx]


def label_aware_distance(
    layer1_pred: torch.Tensor,          # [num_nodes] this epoch's layer-1 MLP output (tanh scalar)
    fraud_center_idx: torch.Tensor,     # [B] TRAIN-SPLIT, FRAUD-LABELLED (y=1) center node indices only
    selected_neighbors: torch.Tensor,   # [B, max_k] bool, this relation's current selection
    neighbor_idx: torch.Tensor,         # [B, max_k] long
) -> float:
    """
    G(D_r^(l))^(e), Eq. 5. Averaged over fraud-labelled training centers only
    (paper Discussion, Section 3.3.2): "we compute the filtering thresholds by
    only considering positive center nodes (i.e., fraudsters) and apply them
    for all node classes." fraud_center_idx must already be filtered to
    train-split AND y=1 before calling this function.
    """
    center_pred = layer1_pred[fraud_center_idx].unsqueeze(1)
    neighbor_pred = layer1_pred[neighbor_idx]
    diff = (center_pred - neighbor_pred).abs()  # Eq. 2, per-pair distance

    mask = selected_neighbors
    denom = mask.sum().item()
    if denom == 0:
        return 0.0
    return (diff * mask).sum().item() / denom
