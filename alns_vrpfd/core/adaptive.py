"""Adaptive operator selection for the ALNS loop."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from alns_vrpfd.core.operators import DestroyOperator, RepairOperator

RewardLabel = str

_DEFAULT_SIGMA: Mapping[RewardLabel, float] = {
    "global": 40.0,          # σ₁: 找到新的全局最优解 (increased)
    "better": 18.0,          # σ₂: 显著改善当前解 (increased)
    "slight_better": 12.0,   # σ₃: 略有改善或保持稳定 (increased)
    "accepted_worse": 2.0,   # σ₄: 接受较差解但有助于跳出局部最优 (increased)
}


@dataclass
class OperatorStats:
    weight: float = 1.0
    uses: int = 0
    total_time: float = 0.0
    total_improvement: float = 0.0
    recent_rewards: Deque[float] = field(default_factory=deque)

    @property
    def avg_time(self) -> float:
        return self.total_time / self.uses if self.uses else 0.0

    @property
    def avg_improvement(self) -> float:
        return self.total_improvement / self.uses if self.uses else 0.0


@dataclass
class AdaptivePool:
    operators: Sequence[object]
    eta: float
    sigma: Mapping[RewardLabel, float]
    rho: float
    w_bounds: Tuple[float, float]
    p_floor: float
    rng: random.Random
    history: int = 0
    decay: float = 0.0
    stats: Dict[object, OperatorStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.history = max(0, int(self.history))
        self.decay = max(0.0, min(1.0, float(self.decay)))
        for operator in self.operators:
            if operator not in self.stats:
                initial_weight = getattr(operator, "initial_weight", 1.0)
                weight = max(0.1, float(initial_weight))
                self.stats[operator] = OperatorStats(weight=weight)

    def select(self) -> object:
        weights = [self.stats[op].weight for op in self.operators]
        total = sum(weights)
        if total <= 0.0:
            weights = [1.0 for _ in self.operators]
            total = float(len(weights))
        probabilities = [max(w / total, self.p_floor / len(weights))
                         for w in weights]
        normaliser = sum(probabilities)
        r = self.rng.random() * normaliser
        cumulative = 0.0
        for operator, prob in zip(self.operators, probabilities):
            cumulative += prob
            if r <= cumulative:
                return operator
        return self.operators[-1]

    def update(
        self,
        operator: object,
        reward: RewardLabel,
        delta_improvement: float,
        elapsed: float,
        credit: float,
        time_normaliser: float,
    ) -> None:
        stats = self.stats[operator]
        stats.uses += 1
        stats.total_time += elapsed
        stats.total_improvement += max(delta_improvement, 0.0)

        base_reward = self.sigma.get(reward, 0.0)
        # 对于接受的解(非rejected),根据改进量调整奖励
        if reward in ["global", "better", "slight_better"]:
            shaped = base_reward * \
                (1.0 + self.rho * max(delta_improvement, 0.0))
        else:  # accepted_worse 不需要改进量加成
            shaped = base_reward
        if time_normaliser > 0:
            shaped /= time_normaliser

        shaped *= credit
        target = shaped
        if self.history > 0:
            stats.recent_rewards.append(shaped)
            if len(stats.recent_rewards) > self.history:
                stats.recent_rewards.popleft()
            if stats.recent_rewards:
                target = sum(stats.recent_rewards) / len(stats.recent_rewards)

        new_weight = (1.0 - self.eta) * stats.weight + self.eta * target
        stats.weight = float(
            max(self.w_bounds[0], min(self.w_bounds[1], new_weight)))
        self._apply_decay(exclude=operator)

    def _apply_decay(self, exclude: object) -> None:
        if self.decay <= 0.0:
            return
        for op, stats in self.stats.items():
            if op is exclude:
                continue
            relaxed = stats.weight * (1.0 - self.decay) + self.decay
            stats.weight = float(
                max(self.w_bounds[0], min(self.w_bounds[1], relaxed)))

    def set_probability_floor(self, value: float) -> None:
        self.p_floor = max(0.0, min(value, 1.0))

    def snapshot(self) -> List[dict[str, float]]:
        data: List[dict[str, float]] = []
        for operator in self.operators:
            stats = self.stats[operator]
            data.append(
                {
                    "name": getattr(operator, "name", operator.__class__.__name__),
                    "weight": stats.weight,
                    "uses": stats.uses,
                    "avg_time": stats.avg_time,
                    "avg_improvement": stats.avg_improvement,
                }
            )
        return data


class AdaptiveOperatorManager:
    """Manage adaptive selection and weight updates for ALNS operators."""

    def __init__(
        self,
        destroy_ops: Iterable[DestroyOperator],
        repair_ops: Iterable[RepairOperator],
        *,
        eta: float = 0.2,
        reward_scale: Mapping[RewardLabel, float] | None = None,
        rho: float = 0.3,
        w_bounds: Tuple[float, float] = (0.1, 100.0),
        p_floor: float = 0.02,
        alpha: float = 0.6,
        rng: Optional[random.Random] = None,
        history: int = 0,
        decay: float = 0.0,
    ) -> None:
        if eta <= 0 or eta >= 1:
            raise ValueError("eta must be in (0,1).")
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be in [0,1].")

        self._alpha = alpha
        self._rng = rng or random.Random(random.getrandbits(32))
        sigma = reward_scale or _DEFAULT_SIGMA
        self._destroy_pool = AdaptivePool(
            tuple(destroy_ops),
            eta=eta,
            sigma=sigma,
            rho=rho,
            w_bounds=w_bounds,
            p_floor=p_floor,
            rng=self._rng,
            history=history,
            decay=decay,
        )
        self._repair_pool = AdaptivePool(
            tuple(repair_ops),
            eta=eta,
            sigma=sigma,
            rho=rho,
            w_bounds=w_bounds,
            p_floor=p_floor,
            rng=self._rng,
            history=history,
            decay=decay,
        )

    def select_destroy(self) -> DestroyOperator:
        return self._destroy_pool.select()

    def select_repair(self) -> RepairOperator:
        return self._repair_pool.select()

    def probability_floor(self) -> float:
        return self._destroy_pool.p_floor

    def set_probability_floor(self, value: float) -> None:
        value = max(0.0, min(value, 1.0))
        self._destroy_pool.set_probability_floor(value)
        self._repair_pool.set_probability_floor(value)

    def update(
        self,
        destroy_op: DestroyOperator,
        repair_op: RepairOperator,
        reward: RewardLabel,
        delta_improvement: float,
        destroy_time: float,
        repair_time: float,
    ) -> None:
        destroy_norm = self._time_normaliser(
            destroy_op, destroy_time, self._destroy_pool)
        repair_norm = self._time_normaliser(
            repair_op, repair_time, self._repair_pool)
        self._destroy_pool.update(
            destroy_op,
            reward,
            delta_improvement,
            destroy_time,
            credit=1.0 - self._alpha,
            time_normaliser=destroy_norm,
        )
        self._repair_pool.update(
            repair_op,
            reward,
            delta_improvement,
            repair_time,
            credit=self._alpha,
            time_normaliser=repair_norm,
        )

    def snapshot(self) -> dict[str, List[dict[str, float]]]:
        return {
            "destroy": self._destroy_pool.snapshot(),
            "repair": self._repair_pool.snapshot(),
        }

    def _time_normaliser(
        self,
        operator: object,
        elapsed: float,
        pool: AdaptivePool,
        tau_bounds: Tuple[float, float] = (0.5, 2.0),
    ) -> float:
        stats = pool.stats[operator]
        if elapsed <= 0 or stats.uses <= 1:
            return 1.0
        median_time = stats.total_time / stats.uses
        if median_time <= 0:
            return 1.0
        ratio = max(tau_bounds[0], min(tau_bounds[1], elapsed / median_time))
        return ratio
