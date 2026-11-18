"""
Meta IA: otimizador dos parâmetros de recompensa + petelecos aleatórios.

- A cada `interval_s` segundos (padrão 300 s = 5 min, tempo real) a Meta IA mede o desempenho da
  política atual numa avaliação gulosa (sem exploração) em 64 ambientes na GPU.
- A métrica é FIXA (fração de tempo estável - penalidade de fim de curso), e não a soma bruta dos
  pontos: como os próprios pesos de recompensa mudam, maximizar os pontos brutos permitiria
  "trapacear" aumentando os pesos positivos.
- Algoritmo (1+1)-ES: se a rodada melhorou, os pesos usados viram os "melhores"; senão volta aos
  melhores. Em seguida propõe uma nova perturbação a partir dos melhores pesos.
- Passo de mutação (sigma) diminui conforme a estabilidade sobe e fica 0 (congelado) ao atingir a
  estabilidade perfeita em avaliações consecutivas.
- Enquanto ligada, aplica "petelecos" (impulso angular aleatório, lado e força aleatórios) em
  intervalos de tempo simulado aleatórios para desequilibrar o pêndulo novamente.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import random
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from physics import REWARD_TABLE_SPECS, BatchedCartPoleEnv, PhysicsParams


PERFECT_STABILITY = 0.985
PERFECT_ROUNDS_TO_FREEZE = 2
SIGMA_MAX = 0.35
SIGMA_MIN = 0.01
GPU_FLICK_EVERY_CYCLES = 25


@torch.no_grad()
def evaluate_policy(
    policy_net,
    action_values: torch.Tensor,
    params: PhysicsParams,
    device: torch.device,
    n_envs: int = 64,
    steps: int = 500,
    tail: int = 250,
    flick_force_max: float = 2.5,
    flicks_enabled: bool = True,
) -> Tuple[float, float, float, float]:
    """Avaliação simultânea: (1) Estabilidade c/ petelecos e (2) Subida de baixo + 2s estável."""
    env_stab = BatchedCartPoleEnv(
        num_envs=n_envs, params=params, device=device, max_episode_steps=10 ** 9, reset_mode="stability"
    )
    env_swing = BatchedCartPoleEnv(
        num_envs=n_envs, params=params, device=device, max_episode_steps=10 ** 9, reset_mode="swingup_2s"
    )
    env_stab.reset_all()
    env_swing.reset_all()

    lim = max(params.rail_limit, 0.1)
    stable_acc = torch.zeros(n_envs, device=device)
    end_acc = torch.zeros(n_envs, device=device)
    x_acc = torch.zeros(n_envs, device=device)
    center_acc = torch.zeros(n_envs, device=device)
    swing_hold = torch.zeros(n_envs, device=device)
    swing_ok = torch.zeros(n_envs, device=device, dtype=torch.bool)
    swing_end_acc = torch.zeros(n_envs, device=device)
    tail = min(tail, steps)
    zone = max(0.05, float(getattr(params, "rew_center_x_zone", 0.35)))

    for t in range(steps):
        if flicks_enabled and t > 0 and (t % 150 == 0):
            env_stab.apply_random_flick(0.5, flick_force_max)

        if hasattr(policy_net, "act_deterministic"):
            u_s = policy_net.act_deterministic(env_stab.get_states()).squeeze(-1)
            u_w = policy_net.act_deterministic(env_swing.get_states()).squeeze(-1)
        else:
            u_s = action_values[torch.argmax(policy_net(env_stab.get_states()), dim=1)]
            u_w = action_values[torch.argmax(policy_net(env_swing.get_states()), dim=1)]

        env_stab.step(u_s)
        env_swing.step(u_w)

        w_stable = (
            (torch.abs(env_swing.theta) < 0.22)
            & (torch.abs(env_swing.x) < 0.7 * lim)
            & (torch.abs(env_swing.theta_dot) < 1.8)
        )
        swing_hold = torch.where(w_stable, swing_hold + params.dt, torch.zeros_like(swing_hold))
        goal_sec = float(max(0.2, min(30.0, getattr(params, "swing_hold_target", 2.0))))
        swing_ok = swing_ok | (swing_hold >= goal_sec)
        swing_end_acc += (torch.abs(env_swing.x) >= 0.98 * lim).float()

        if t >= steps - tail:
            s_stable = (
                (torch.abs(env_stab.theta) < 0.22)
                & (torch.abs(env_stab.x) < 0.7 * lim)
                & (torch.abs(env_stab.theta_dot) < 1.8)
            )
            stable_acc += s_stable.float()
            center_acc += torch.exp(-((env_stab.x / zone) ** 2))
            end_acc += (torch.abs(env_stab.x) >= 0.98 * lim).float()
            x_acc += torch.abs(env_stab.x) / lim

    stab_frac = float(stable_acc.mean().item() / tail)
    end_rate = max(float(end_acc.mean().item() / tail), float(swing_end_acc.mean().item() / steps))
    mean_abs_x = float(x_acc.mean().item() / tail)
    swing_rate = float(swing_ok.float().mean().item())
    center_score = float(center_acc.mean().item() / tail)
    return stab_frac, end_rate, mean_abs_x, swing_rate, center_score


class MetaAI:
    """Controlador da Meta IA (estado + lógica de otimização + petelecos)."""

    def __init__(self) -> None:
        self.enabled = False
        self.flicks_enabled = True
        self.interval_s = 300.0
        self.flick_min_s = 4.0
        self.flick_max_s = 12.0
        self.flick_force_max = 2.5

        self.period_start = 0.0
        self.flick_countdown = 8.0
        self.train_cycles = 0

        self.rounds = 0
        self.best_score: Optional[float] = None
        self.best_stab = 0.0
        self.best_swingup = 0.0
        self.best_weights: Optional[Dict[str, float]] = None
        self.last_score = 0.0
        self.last_stab = 0.0
        self.last_swingup = 0.0
        self.last_center = 0.0
        self.last_end_rate = 0.0
        self.sigma = SIGMA_MAX
        self.perfect_streak = 0
        self.frozen = False
        self.last_msg = "Meta IA desligada."
        self.last_flick_msg = "—"
        self.flick_count = 0
        self.flick_falls = 0
        self._awaiting_flick_check = False
        self._pending: Optional[Tuple[float, float]] = None
        self._running = False
        self.history: Deque[Tuple[int, float, float, float]] = deque(maxlen=50)
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ controle
    def set_enabled(self, flag: bool, now: float) -> None:
        flag = bool(flag)
        if flag and not self.enabled:
            self.period_start = now
            self.best_score = None
            self.best_weights = None
            self.best_stab = 0.0
            self.best_swingup = 0.0
            self.rounds = 0
            self.perfect_streak = 0
            self.frozen = False
            self.sigma = SIGMA_MAX
            self.history.clear()
            self._schedule_flick()
            self.last_msg = "Meta IA ligada: avaliando Estabilidade + Subida (2s) ao fim do período."
        elif not flag and self.enabled:
            self.last_msg = "Meta IA desligada."
            self._pending = None
        self.enabled = flag

    def set_flicks_enabled(self, flag: bool) -> None:
        self.flicks_enabled = bool(flag)
        if self.flicks_enabled:
            self._schedule_flick()
        else:
            self._awaiting_flick_check = False

    def set_param(self, key: str, value: float) -> None:
        v = float(value)
        if key == "meta_interval_s":
            self.interval_s = max(5.0, min(7200.0, v))
        elif key == "meta_flick_min_s":
            self.flick_min_s = max(0.5, min(300.0, v))
            self.flick_max_s = max(self.flick_max_s, self.flick_min_s)
        elif key == "meta_flick_max_s":
            self.flick_max_s = max(0.5, min(600.0, v))
            self.flick_min_s = min(self.flick_min_s, self.flick_max_s)
        elif key == "meta_flick_force_max":
            self.flick_force_max = max(0.5, min(25.0, v))

    def time_left(self, now: float) -> float:
        if not self.enabled:
            return 0.0
        return max(0.0, self.period_start + self.interval_s - now)

    def due(self, now: float) -> bool:
        return self.enabled and now >= self.period_start + self.interval_s

    # ------------------------------------------------------------------ petelecos
    def _schedule_flick(self) -> None:
        lo = min(self.flick_min_s, self.flick_max_s)
        hi = max(self.flick_min_s, self.flick_max_s)
        self.flick_countdown = random.uniform(lo, hi)

    def on_sim_steps(self, n_steps: int, dt: float, sim_env, now: Optional[float] = None) -> None:
        """Conta tempo simulado e aplica petelecos leves (sem resetar para baixo)."""
        if not self.flicks_enabled or n_steps <= 0:
            return
        if self._awaiting_flick_check and abs(sim_env.theta) > 1.2:
            self.flick_falls += 1
            self._awaiting_flick_check = False
        self.flick_countdown -= n_steps * dt
        if self.flick_countdown <= 0.0:
            direction, mag = sim_env.apply_random_flick(self.flick_force_max)
            self.flick_count += 1
            self._awaiting_flick_check = True
            side = "→ direita" if direction > 0 else "← esquerda"
            self.last_flick_msg = f"#{self.flick_count} {side} ({mag:.1f} rad/s)"
            self._schedule_flick()

    def on_train_cycle(self, gpu_env) -> None:
        """Aplica petelecos aleatórios em parte dos ambientes de estabilidade para aprender a recuperar."""
        if not self.flicks_enabled:
            return
        self.train_cycles += 1
        if self.train_cycles % GPU_FLICK_EVERY_CYCLES == 0:
            gpu_env.apply_random_flick(0.25, self.flick_force_max)

    # ------------------------------------------------------------------ otimização
    @staticmethod
    def _get_weights(params: PhysicsParams) -> Dict[str, float]:
        return {w_key: float(getattr(params, w_key)) for w_key, _, _, _ in REWARD_TABLE_SPECS}

    @staticmethod
    def _set_weights(params: PhysicsParams, weights: Dict[str, float]) -> None:
        for k, v in weights.items():
            setattr(params, k, float(v))

    def _compute_sigma(self) -> float:
        if self.frozen:
            return 0.0
        gap = max(0.0, min(1.0, 1.0 - self.best_stab))
        return SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * (gap ** 1.5)

    def maybe_optimize(
        self,
        now: float,
        agent,
        params: PhysicsParams,
        device: torch.device,
        evaluator: Optional[Callable[[], Tuple[float, float]]] = None,
    ) -> bool:
        """Não bloqueante: aplica o resultado pronto (se houver) e dispara a próxima avaliação em thread.

        Retorna True quando um resultado foi aplicado (os pesos de recompensa podem ter mudado).
        """
        applied = False
        if self._pending is not None:
            stab, end_rate, mean_abs_x, swing_rate, center_score = self._pending
            self._pending = None
            self._running = False
            self._apply_result(params, stab, end_rate, mean_abs_x, swing_rate, center_score)
            applied = True

        if self.due(now) and not self._running:
            self._running = True
            self.period_start = now
            self.last_msg = "Avaliando Estabilidade + Subida (2s) em segundo plano..."
            net = copy.deepcopy(agent.policy_net).eval()
            action_values = agent.action_values.clone()
            params_copy = dataclasses.replace(params)
            flicks_on = bool(self.flicks_enabled)
            flick_f = float(self.flick_force_max)
            threading.Thread(
                target=self._worker,
                args=(net, action_values, params_copy, device, flicks_on, flick_f, evaluator),
                daemon=True,
            ).start()
        return applied

    def _worker(self, net, action_values, params_copy, device, flicks_on, flick_f, evaluator) -> None:
        try:
            if evaluator is not None:
                result = evaluator()
            else:
                result = evaluate_policy(
                    net, action_values, params_copy, device,
                    flick_force_max=flick_f, flicks_enabled=flicks_on,
                )
            self._pending = (
                float(result[0]),
                float(result[1]),
                float(result[2]) if len(result) > 2 else 0.0,
                float(result[3]) if len(result) > 3 else float(result[0]),
                float(result[4]) if len(result) > 4 else max(0.0, 1.0 - (float(result[2]) if len(result) > 2 else 0.0)),
            )
        except Exception as exc:  # noqa: BLE001
            self._running = False
            self.last_msg = f"Erro na avaliação da Meta IA: {exc}"

    def optimize(
        self,
        agent,
        params: PhysicsParams,
        device: torch.device,
        evaluator: Optional[Callable[[], Tuple[float, ...]]] = None,
    ) -> str:
        """Versão síncrona (avalia e aplica na hora)."""
        if evaluator is not None:
            res = evaluator()
            stab, end_rate = res[0], res[1]
            mean_abs_x = res[2] if len(res) > 2 else 0.0
            swing_rate = res[3] if len(res) > 3 else stab
            center_score = res[4] if len(res) > 4 else max(0.0, 1.0 - mean_abs_x)
        else:
            stab, end_rate, mean_abs_x, swing_rate, center_score = evaluate_policy(
                agent.policy_net, agent.action_values, params, device,
                flick_force_max=self.flick_force_max, flicks_enabled=self.flicks_enabled,
            )
        return self._apply_result(params, stab, end_rate, mean_abs_x, swing_rate, center_score)

    def _apply_result(
        self,
        params: PhysicsParams,
        stab: float,
        end_rate: float,
        mean_abs_x: float = 0.0,
        swing_rate: float = 0.0,
        center_score: float = 0.0,
    ) -> str:
        combined = 0.45 * stab + 0.35 * swing_rate + 0.20 * center_score
        score = combined - 0.30 * end_rate - 0.35 * mean_abs_x
        self.rounds += 1
        self.last_stab, self.last_swingup, self.last_center, self.last_end_rate, self.last_score = (
            stab, swing_rate, center_score, end_rate, score
        )

        current = self._get_weights(params)
        if self.best_weights is None or self.best_score is None or score > self.best_score:
            self.best_score, self.best_stab, self.best_swingup, self.best_weights = score, stab, swing_rate, dict(current)
            verdict = "melhorou" if self.rounds > 1 else "baseline"
        else:
            self._set_weights(params, self.best_weights)
            verdict = "piorou -> voltou aos melhores pesos"

        if combined >= PERFECT_STABILITY:
            self.perfect_streak += 1
        else:
            self.perfect_streak = 0
        self.frozen = self.perfect_streak >= PERFECT_ROUNDS_TO_FREEZE
        self.sigma = self._compute_sigma()

        if self.frozen:
            action = "estabilidade perfeita: pesos congelados"
        else:
            self._propose(params)
            action = f"nova perturbação (σ={self.sigma:.3f})"

        self.history.append((self.rounds, score, combined, self.sigma))
        self.last_msg = (
            f"Rodada {self.rounds}: estab. {stab * 100:.1f}%, subida {swing_rate * 100:.1f}%, "
            f"centro {center_score * 100:.1f}%, fim {end_rate * 100:.1f}% ({verdict}); {action}."
        )
        return self.last_msg

    def _propose(self, params: PhysicsParams) -> None:
        assert self.best_weights is not None
        self._set_weights(params, self.best_weights)
        params.rew_center_x_active = True
        params.rew_x_dist_active = True
        for w_key, a_key, _, _ in REWARD_TABLE_SPECS:
            if not bool(getattr(params, a_key)):
                continue
            w = self.best_weights[w_key]
            scale = max(abs(w), 0.5)
            nw = w + float(self._rng.normal(0.0, self.sigma * scale))
            if w != 0.0 and math.copysign(1.0, nw) != math.copysign(1.0, w):
                nw = math.copysign(max(abs(w) * 0.05, 0.01), w)
            if w_key == "rew_center_x":
                nw = max(1.5, nw)
            elif w_key == "rew_x_dist":
                nw = min(-0.30, nw)
            setattr(params, w_key, float(max(-500.0, min(500.0, nw))))

    # ------------------------------------------------------------------ telemetria
    def status(self, now: float) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "flicks_enabled": self.flicks_enabled,
            "time_left": self.time_left(now),
            "interval_s": self.interval_s,
            "flick_min_s": self.flick_min_s,
            "flick_max_s": self.flick_max_s,
            "flick_force_max": self.flick_force_max,
            "rounds": self.rounds,
            "sigma": self.sigma,
            "frozen": self.frozen,
            "best_score": self.best_score if self.best_score is not None else 0.0,
            "best_stab": self.best_stab,
            "best_swingup": self.best_swingup,
            "last_score": self.last_score,
            "last_stab": self.last_stab,
            "last_swingup": self.last_swingup,
            "last_center": self.last_center,
            "last_end_rate": self.last_end_rate,
            "last_msg": self.last_msg,
            "last_flick": self.last_flick_msg,
            "flick_count": self.flick_count,
            "flick_falls": self.flick_falls,
            "history": [list(h) for h in self.history],
        }
