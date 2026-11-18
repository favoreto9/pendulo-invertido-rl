"""
Motor de física realista para o Simulador de Pêndulo Invertido com:
- Gravidade fixa em 9.81 m/s².
- Base móvel inclinável continuamente entre -45° e +45° (configurável por range).
- Velocidade do vento mutável (+ direita, - esquerda).
- Sensor de Ângulo Quantizado no formato 123,4° (1 casa decimal / resolução 0,1°):
  deriva velocidade e aceleração angular exclusivamente de leituras consecutivas em graus.
- Vetor de estado 10D para a IA: leituras quantizadas do ângulo (0,1°) + inputs conhecidos
  de peso do pêndulo (m_p), tamanho do pêndulo (L) e peso do carrinho (m_c).
  Sem vento, sem posição X e sem inclinação da base na observação da IA.
- Bônus de proximidade do centro do eixo X e penalidade severa silenciosa ao encostar no fim de curso.
- Mutação automática seletiva com ranges configuráveis sincronizada entre CPU e GPU (256 ambientes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


FIXED_GRAVITY = 9.81
MAX_TABLE_ANGLE_RAD = math.radians(45.0)
MAX_WIND_SPEED = 15.0
ENDSTOP_PENALTY = 25.0

REWARD_TABLE_SPECS = (
    ("rew_center_x", "rew_center_x_active", "Próximo do x=0", 2.0),
    ("rew_top", "rew_top_active", "Pêndulo no topo", 2.0),
    ("rew_stability", "rew_stability_active", "Estabilidade", 1.5),
    ("rew_x_dist", "rew_x_dist_active", "Penalidade por distância do x=0", -0.40),
    ("rew_endstop", "rew_endstop_active", "Fim de curso", -25.0),
    ("rew_up_time_acc", "rew_up_time_acc_active", "Acumulador tempo p/ cima", 1.0),
    ("rew_fall", "rew_fall_active", "Queda do pêndulo", -2.0),
    ("rew_rise", "rew_rise_active", "Subida do pêndulo", 1.0),
    ("rew_bottom", "rew_bottom_active", "Pêndulo para baixo", -0.50),
    ("rew_down_time_acc", "rew_down_time_acc_active", "Acumulador tempo p/ baixo", -1.0),
    ("rew_swingup", "rew_swingup_active", "Swing up", 0.40),
    ("rew_smooth", "rew_smooth_active", "Movimento brusco", -0.10),
    ("rew_full_rotation", "rew_full_rotation_active", "Rotação completa (360°)", -35.0),
)


@dataclass
class PhysicsParams:
    """Parâmetros físicos, velocidade da simulação e regras de auto mutação."""

    pendulum_length: float = 0.75
    pendulum_mass: float = 0.25
    cart_mass: float = 1.20
    gravity: float = FIXED_GRAVITY
    rail_limit: float = 1.80
    max_force: float = 30.0
    sim_speed: float = 1.0  # Segundos simulados por segundo real
    swing_hold_target: float = 2.0  # Tempo estável pós-subida antes de resetar (s)

    # Vento e inclinação da base
    wind_speed: float = 0.0
    table_angle: float = 0.0

    # Configuração da aba Auto Mutação
    auto_mutate: bool = True
    mutation_mode: str = "fixed"  # "constant" (oscilação contínua) ou "fixed" (fica parada por X s)
    mutation_interval: float = 4.0

    mutate_table: bool = True
    table_min_deg: float = -45.0
    table_max_deg: float = 45.0

    mutate_wind: bool = True
    wind_min: float = -15.0
    wind_max: float = 15.0

    mutate_pend_mass: bool = True
    pend_mass_min: float = 0.08
    pend_mass_max: float = 1.50

    mutate_cart_mass: bool = True
    cart_mass_min: float = 0.40
    cart_mass_max: float = 4.00

    mutate_length: bool = False
    length_min: float = 0.35
    length_max: float = 1.30

    # Tabela de recompensas configurável (valores positivos ou negativos + ativação por item)
    rew_center_x_zone: float = 0.35      # Raio (m) da zona de pontuação em torno de x = 0
    rew_center_x: float = 2.0
    rew_center_x_active: bool = True
    rew_top: float = 2.0
    rew_top_active: bool = True
    rew_stability: float = 1.5
    rew_stability_active: bool = True
    rew_x_dist: float = -0.40
    rew_x_dist_active: bool = True
    rew_endstop: float = -25.0
    rew_endstop_active: bool = True
    rew_up_time_acc: float = 1.0
    rew_up_time_acc_active: bool = True
    rew_fall: float = -2.0
    rew_fall_active: bool = True
    rew_rise: float = 1.0
    rew_rise_active: bool = True
    rew_bottom: float = -0.50
    rew_bottom_active: bool = True
    rew_down_time_acc: float = -1.0
    rew_down_time_acc_active: bool = True
    rew_swingup: float = 0.40
    rew_swingup_active: bool = True
    rew_smooth: float = -0.10
    rew_smooth_active: bool = True
    rew_full_rotation: float = -35.0
    rew_full_rotation_active: bool = True

    # Mecânica motor de passo + polia/correia
    pulley_radius: float = 0.04
    gear_ratio: float = 2.5
    rotor_inertia: float = 1.2e-4
    motor_tau: float = 0.015
    v_max: float = 5.5

    # Atritos, aerodinâmica do vento e batentes
    cart_viscous_friction: float = 0.25
    cart_coulomb_friction: float = 0.08
    pendulum_friction: float = 0.006
    wind_quad_coeff: float = 0.024
    wind_lin_coeff: float = 0.12
    bumper_restitution: float = 0.35

    # Integração numérica
    dt: float = 0.016
    substeps: int = 4

    @property
    def l_cm(self) -> float:
        return 0.5 * self.pendulum_length

    @property
    def effective_cart_mass(self) -> float:
        reflected = self.rotor_inertia * (self.gear_ratio ** 2) / (self.pulley_radius ** 2)
        return self.cart_mass + reflected

    @property
    def tau_max(self) -> float:
        return self.max_force * self.pulley_radius / self.gear_ratio


def wrap_angle_scalar(theta: float) -> float:
    """Normaliza o ângulo para (-pi, pi], onde 0 = topo vertical e pi = baixo vertical."""
    wrapped = ((theta + math.pi) % (2.0 * math.pi)) - math.pi
    if wrapped <= -math.pi:
        wrapped += 2.0 * math.pi
    return wrapped


def wrap_angle_torch(theta: torch.Tensor) -> torch.Tensor:
    """Normaliza ângulos em PyTorch para (-pi, pi]."""
    two_pi = 2.0 * math.pi
    wrapped = torch.remainder(theta + math.pi, two_pi) - math.pi
    return torch.where(wrapped <= -math.pi, wrapped + two_pi, wrapped)


def quantize_angle_deg_scalar(theta_rad: float) -> float:
    """Quantiza o ângulo em graus com 1 casa decimal (resolução 0,1°) em (-180.0, 180.0]."""
    deg = round(math.degrees(wrap_angle_scalar(theta_rad)), 1)
    if deg <= -180.0:
        deg += 360.0
    elif deg > 180.0:
        deg -= 360.0
    if abs(deg) < 0.05:
        deg = 0.0
    return deg


def wrap_delta_deg_scalar(delta_deg: float) -> float:
    """Calcula a menor diferença angular em graus dentro de (-180.0, 180.0]."""
    d = ((delta_deg + 180.0) % 360.0) - 180.0
    if d <= -180.0:
        d += 360.0
    return d


def quantize_angle_deg_torch(theta_rad: torch.Tensor) -> torch.Tensor:
    """Quantiza tensor de ângulos para graus com 1 casa decimal (0,1°) em (-180.0, 180.0]."""
    wrapped = wrap_angle_torch(theta_rad)
    deg = torch.round(torch.rad2deg(wrapped) * 10.0) / 10.0
    deg = torch.where(deg <= -180.0, deg + 360.0, deg)
    deg = torch.where(deg > 180.0, deg - 360.0, deg)
    return torch.where(torch.abs(deg) < 0.05, torch.zeros_like(deg), deg)


def wrap_delta_deg_torch(delta_deg: torch.Tensor) -> torch.Tensor:
    """Calcula a menor diferença angular em graus no tensor dentro de (-180.0, 180.0]."""
    d = torch.remainder(delta_deg + 180.0, 360.0) - 180.0
    return torch.where(d <= -180.0, d + 360.0, d)


def format_angle_deg_br(deg: float) -> str:
    """Formata o ângulo com 1 casa decimal e vírgula brasileira (ex: 123,4°)."""
    val = 0.0 if abs(deg) < 0.05 else float(deg)
    return f"{val:.1f}".replace(".", ",") + "°"


def _ordered_range(lo: float, hi: float, hard_min: float, hard_max: float) -> Tuple[float, float]:
    a = max(hard_min, min(hard_max, float(lo)))
    b = max(hard_min, min(hard_max, float(hi)))
    return (a, b) if a <= b else (b, a)


class CartPolePhysics:
    """Ambiente físico unitário com sensor de ângulo quantizado em 0,1° (formato 123,4°) e posição X."""

    STATE_DIM = 13

    def __init__(self, params: PhysicsParams | None = None) -> None:
        self.params = params if params is not None else PhysicsParams()
        self.params.gravity = FIXED_GRAVITY

        self.x = 0.0
        self.prev_x = 0.0
        self.x_dot = 0.0
        self.theta = math.pi
        self.theta_dot = 0.0
        self.tau_m = 0.0
        self.prev_u = 0.0
        self.pulley_angle = 0.0

        # Estado do sensor de ângulo quantizado (resolução 0,1° no formato 123,4°)
        self.angle_deg = 180.0
        self.prev_angle_deg = 180.0
        self.theta_q = math.pi
        self.omega_q = 0.0
        self.alpha_q = 0.0

        # Dinâmica contínua da inclinação da mesa alpha(t)
        self.table_angle = 0.0
        self.table_angle_vel = 0.0
        self.table_angle_acc = 0.0
        self.table_target = 0.0
        self.table_phase = float(np.random.uniform(0.0, 2.0 * math.pi))
        self.table_timer = self.params.mutation_interval

        # Mutação automática de vento, massas e comprimento
        self.wind_target = 0.0
        self.wind_timer = self.params.mutation_interval
        self.pend_mass_timer = self.params.mutation_interval
        self.cart_mass_timer = self.params.mutation_interval
        self.length_timer = self.params.mutation_interval

        self.force_motor = 0.0
        self.x_ddot = 0.0
        self.theta_ddot = 0.0
        self.bumper_hit = 0.0
        self.bumper_side = 0
        self.stable_time = 0.0
        self.best_stable_time = 0.0
        self.up_time = 0.0
        self.down_time = 0.0
        self.rot_accum = 0.0
        self.prev_theta = math.pi
        self.step_count = 0
        self.episode_reward = 0.0
        self.last_reward_breakdown: Dict[str, float] = {
            **{w_key: 0.0 for w_key, _, _, _ in REWARD_TABLE_SPECS},
            "total_pts": 0.0,
        }

        self.reset(mode="down")

    def reset(self, mode: str = "down") -> np.ndarray:
        """Reinicia o pêndulo e o sensor de ângulo quantizado (0,1°)."""
        self.x = 0.0
        self.prev_x = 0.0
        self.x_dot = 0.0
        self.tau_m = 0.0
        self.prev_u = 0.0
        self.pulley_angle = 0.0
        self.force_motor = 0.0
        self.x_ddot = 0.0
        self.theta_ddot = 0.0
        self.bumper_hit = 0.0
        self.bumper_side = 0
        self.stable_time = 0.0
        self.best_stable_time = 0.0
        self.up_time = 0.0
        self.down_time = 0.0
        self.rot_accum = 0.0
        self.step_count = 0
        self.episode_reward = 0.0

        if mode == "down":
            self.theta = math.pi
            self.theta_dot = 0.0
        elif mode == "up":
            self.theta = 0.0
            self.theta_dot = 0.0
        else:
            self.theta = wrap_angle_scalar(float(np.random.uniform(-math.pi, math.pi)))
            self.theta_dot = float(np.random.uniform(-0.5, 0.5))

        self.prev_theta = self.theta
        self.angle_deg = quantize_angle_deg_scalar(self.theta)
        self.prev_angle_deg = self.angle_deg
        self.theta_q = math.radians(self.angle_deg)
        self.omega_q = 0.0
        self.alpha_q = 0.0
        self._compute_reward(0.0, 0.0, False)
        return self.get_state()

    def randomize_pose(self) -> None:
        """Reposiciona carrinho (x) e pêndulo (theta) aleatoriamente para treinar subida."""
        x_lim = max(self.params.rail_limit * 0.65, 0.1)
        self.x = float(np.random.uniform(-x_lim, x_lim))
        self.prev_x = self.x
        self.x_dot = 0.0
        self.theta = wrap_angle_scalar(float(np.random.uniform(-math.pi, math.pi)))
        self.prev_theta = self.theta
        self.theta_dot = 0.0
        self.tau_m = 0.0
        self.prev_u = 0.0
        self.up_time = 0.0
        self.down_time = 0.0
        self.rot_accum = 0.0
        self.stable_time = 0.0
        self.angle_deg = quantize_angle_deg_scalar(self.theta)
        self.prev_angle_deg = self.angle_deg
        self.theta_q = math.radians(self.angle_deg)
        self.omega_q = 0.0
        self.alpha_q = 0.0
        self._compute_reward(0.0, 0.0, False)

    def apply_perturbation(self, angular_impulse: float = 4.0, cart_impulse: float = 0.8) -> None:
        sign = 1.0 if (0.0 <= self.theta < math.pi) else -1.0
        if abs(self.theta) < 0.05 or abs(abs(self.theta) - math.pi) < 0.05:
            sign = float(np.random.choice([-1.0, 1.0]))
        self.theta_dot += sign * angular_impulse
        self.x_dot += -sign * cart_impulse
        self.stable_time = 0.0

    def apply_mouse_nudge(self, d_theta_dot: float, d_x_dot: float = 0.0) -> None:
        self.theta_dot = float(np.clip(self.theta_dot + d_theta_dot, -30.0, 30.0))
        self.x_dot = float(np.clip(self.x_dot + d_x_dot, -12.0, 12.0))

    def apply_random_flick(self, force_max: float) -> Tuple[float, float]:
        """Peteleco: impulso angular com lado e força aleatórios. Retorna (direção ±1, força rad/s)."""
        direction = float(np.random.choice([-1.0, 1.0]))
        magnitude = float(np.random.uniform(0.25, 1.0) * force_max)
        self.theta_dot = float(np.clip(self.theta_dot + direction * magnitude, -30.0, 30.0))
        self.x_dot = float(np.clip(self.x_dot - direction * 0.08 * magnitude, -12.0, 12.0))
        self.stable_time = 0.0
        return direction, magnitude

    def set_manual_table_angle_deg(self, deg: float) -> None:
        rad = max(-MAX_TABLE_ANGLE_RAD, min(MAX_TABLE_ANGLE_RAD, math.radians(deg)))
        self.table_angle = rad
        self.table_target = rad
        self.table_angle_vel = 0.0
        self.table_angle_acc = 0.0
        self.params.table_angle = rad
        self.table_timer = max(0.2, self.params.mutation_interval)

    def set_manual_wind(self, wind_speed: float) -> None:
        w = max(-MAX_WIND_SPEED, min(MAX_WIND_SPEED, wind_speed))
        self.params.wind_speed = w
        self.wind_target = w
        self.wind_timer = max(0.2, self.params.mutation_interval)

    def set_mutation_interval(self, seconds: float) -> None:
        interval = max(0.2, min(300.0, float(seconds)))
        self.params.mutation_interval = interval
        self.table_timer = min(self.table_timer, interval)
        self.wind_timer = min(self.wind_timer, interval)
        self.pend_mass_timer = min(self.pend_mass_timer, interval)
        self.cart_mass_timer = min(self.cart_mass_timer, interval)
        self.length_timer = min(self.length_timer, interval)

    def _update_environment_mutations(self, dt: float) -> None:
        p = self.params
        p.gravity = FIXED_GRAVITY
        base_dur = max(0.2, min(300.0, float(p.mutation_interval)))

        # 1) Base inclinável
        is_fixed = getattr(p, "mutation_mode", "fixed") == "fixed"
        if p.auto_mutate and p.mutate_table:
            t_lo_deg, t_hi_deg = _ordered_range(p.table_min_deg, p.table_max_deg, -45.0, 45.0)
            t_lo = math.radians(t_lo_deg)
            t_hi = math.radians(t_hi_deg)
            self.table_timer -= dt
            if self.table_timer <= 0.0:
                self.table_target = float(np.random.uniform(t_lo, t_hi))
                self.table_timer = base_dur if is_fixed else base_dur * float(np.random.uniform(0.85, 1.15))

            if is_fixed:
                desired_alpha = max(t_lo, min(t_hi, self.table_target))
                omega_n = 5.5
                zeta = 1.0
            else:
                self.table_phase += dt * 1.85
                osc_amp = min(0.14, 0.25 * (t_hi - t_lo))
                desired_alpha = max(t_lo, min(t_hi, self.table_target + osc_amp * math.sin(self.table_phase)))
                omega_n = 2.3
                zeta = 0.82

            alpha_acc = (omega_n ** 2) * (desired_alpha - self.table_angle) - 2.0 * zeta * omega_n * self.table_angle_vel
            self.table_angle_vel += alpha_acc * dt
            self.table_angle += self.table_angle_vel * dt
            if is_fixed and abs(desired_alpha - self.table_angle) < 1e-3 and abs(self.table_angle_vel) < 5e-3:
                self.table_angle = desired_alpha
                self.table_angle_vel = 0.0
                alpha_acc = 0.0

            if self.table_angle > t_hi:
                self.table_angle = t_hi
                self.table_angle_vel = min(0.0, self.table_angle_vel)
                alpha_acc = 0.0
            elif self.table_angle < t_lo:
                self.table_angle = t_lo
                self.table_angle_vel = max(0.0, self.table_angle_vel)
                alpha_acc = 0.0

            self.table_angle_acc = alpha_acc
            p.table_angle = self.table_angle
        else:
            self.table_angle = max(-MAX_TABLE_ANGLE_RAD, min(MAX_TABLE_ANGLE_RAD, p.table_angle))
            self.table_angle_vel = 0.0
            self.table_angle_acc = 0.0

        # 2) Vento mutável
        if p.auto_mutate and p.mutate_wind:
            w_lo, w_hi = _ordered_range(p.wind_min, p.wind_max, -MAX_WIND_SPEED, MAX_WIND_SPEED)
            self.wind_timer -= dt
            if self.wind_timer <= 0.0:
                self.wind_target = float(np.random.uniform(w_lo, w_hi))
                self.wind_timer = base_dur * float(np.random.uniform(0.85, 1.15))
            self.wind_target = max(w_lo, min(w_hi, self.wind_target))
            p.wind_speed = max(w_lo, min(w_hi, p.wind_speed + (self.wind_target - p.wind_speed) * min(1.0, dt * 1.8)))

        # 3) Peso do pêndulo
        if p.auto_mutate and p.mutate_pend_mass:
            pm_lo, pm_hi = _ordered_range(p.pend_mass_min, p.pend_mass_max, 0.05, 3.0)
            self.pend_mass_timer -= dt
            if self.pend_mass_timer <= 0.0:
                p.pendulum_mass = float(np.random.uniform(pm_lo, pm_hi))
                self.pend_mass_timer = base_dur * float(np.random.uniform(0.85, 1.15))

        # 4) Peso do carrinho
        if p.auto_mutate and p.mutate_cart_mass:
            cm_lo, cm_hi = _ordered_range(p.cart_mass_min, p.cart_mass_max, 0.20, 8.0)
            self.cart_mass_timer -= dt
            if self.cart_mass_timer <= 0.0:
                p.cart_mass = float(np.random.uniform(cm_lo, cm_hi))
                self.cart_mass_timer = base_dur * float(np.random.uniform(0.85, 1.15))

        # 5) Comprimento L do pêndulo
        if p.auto_mutate and p.mutate_length:
            l_lo, l_hi = _ordered_range(p.length_min, p.length_max, 0.20, 2.0)
            self.length_timer -= dt
            if self.length_timer <= 0.0:
                p.pendulum_length = float(np.random.uniform(l_lo, l_hi))
                self.length_timer = base_dur * float(np.random.uniform(0.85, 1.15))

    def _derivatives(
        self,
        x: float,
        xd: float,
        th: float,
        thd: float,
        tm: float,
        u_cmd: float,
        ext_cart_force: float,
        ext_pendulum_torque: float,
    ) -> Tuple[float, float, float, float, float]:
        p = self.params
        l_cm = p.l_cm
        m_p = p.pendulum_mass
        g = FIXED_GRAVITY

        alpha = self.table_angle
        alpha_d = self.table_angle_vel
        alpha_dd = self.table_angle_acc

        m_eff = p.effective_cart_mass
        m_total = m_eff + m_p
        i_total = (4.0 / 3.0) * m_p * (l_cm ** 2)

        speed_ratio = abs(xd) / max(p.v_max, 0.5)
        tau_lim = p.tau_max * min(1.0, max(0.15, 1.0 - speed_ratio))
        tau_cmd = max(-tau_lim, min(tau_lim, u_cmd * p.tau_max))
        tau_m_dot = (tau_cmd - tm) / max(p.motor_tau, 1e-3)

        f_motor = (p.gear_ratio * tm) / p.pulley_radius

        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        v_rod_x = xd * cos_a + l_cm * thd * math.cos(th) - x * alpha_d * sin_a
        v_rel = p.wind_speed - v_rod_x
        f_wind = p.wind_quad_coeff * v_rel * abs(v_rel) + p.wind_lin_coeff * v_rel
        tau_wind = f_wind * l_cm * math.cos(th)

        f_net = (
            f_motor
            + ext_cart_force
            + f_wind * cos_a
            - p.cart_viscous_friction * xd
            - p.cart_coulomb_friction * math.tanh(10.0 * xd)
        )

        sum_ta = th + alpha
        sin_ta = math.sin(sum_ta)
        cos_ta = math.cos(sum_ta)
        ml = m_p * l_cm

        rhs_x = (
            f_net
            + ml * sin_ta * (thd ** 2)
            - m_total * g * sin_a
            + m_total * x * (alpha_d ** 2)
        )
        rhs_t = (
            ml * g * math.sin(th)
            - p.pendulum_friction * thd
            + tau_wind
            + ext_pendulum_torque
            + ml * (
                2.0 * xd * alpha_d * sin_ta
                + x * alpha_dd * sin_ta
                + x * (alpha_d ** 2) * cos_ta
            )
        )

        det = max(m_total * i_total - (ml * cos_ta) ** 2, 1e-8)
        x_ddot = (i_total * rhs_x - ml * cos_ta * rhs_t) / det
        theta_ddot = (m_total * rhs_t - ml * cos_ta * rhs_x) / det
        return xd, x_ddot, thd, theta_ddot, tau_m_dot

    def step(
        self,
        action_u: float,
        external_cart_force: float = 0.0,
        external_pendulum_torque: float = 0.0,
    ) -> Tuple[np.ndarray, float, Dict[str, float]]:
        """Avança um passo físico RK4 e atualiza o sensor de ângulo quantizado (0,1°)."""
        p = self.params
        self.prev_theta = self.theta
        self.prev_x = self.x

        self._update_environment_mutations(p.dt)

        u_cmd = float(max(-1.0, min(1.0, action_u)))
        h = p.dt / max(1, p.substeps)

        x = self.x
        xd = self.x_dot
        th = self.theta
        thd = self.theta_dot
        tm = self.tau_m

        bumper_hit = 0.0
        bumper_side = 0
        x_lim = p.rail_limit

        for _ in range(p.substeps):
            k1 = self._derivatives(x, xd, th, thd, tm, u_cmd, external_cart_force, external_pendulum_torque)
            k2 = self._derivatives(
                x + 0.5 * h * k1[0], xd + 0.5 * h * k1[1], th + 0.5 * h * k1[2],
                thd + 0.5 * h * k1[3], tm + 0.5 * h * k1[4],
                u_cmd, external_cart_force, external_pendulum_torque,
            )
            k3 = self._derivatives(
                x + 0.5 * h * k2[0], xd + 0.5 * h * k2[1], th + 0.5 * h * k2[2],
                thd + 0.5 * h * k2[3], tm + 0.5 * h * k2[4],
                u_cmd, external_cart_force, external_pendulum_torque,
            )
            k4 = self._derivatives(
                x + h * k3[0], xd + h * k3[1], th + h * k3[2],
                thd + h * k3[3], tm + h * k3[4],
                u_cmd, external_cart_force, external_pendulum_torque,
            )

            x += (h / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            xd += (h / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            th += (h / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
            thd += (h / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
            tm += (h / 6.0) * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])

            if x > x_lim:
                x = x_lim
                if xd > 0.0:
                    delta_xd = -(1.0 + p.bumper_restitution) * xd
                    xd = -p.bumper_restitution * xd
                    thd -= (delta_xd * math.cos(th + self.table_angle)) / ((4.0 / 3.0) * p.l_cm)
                    bumper_hit = max(bumper_hit, min(1.0, abs(delta_xd) / 2.5 + 0.2))
                    bumper_side = 1
            elif x < -x_lim:
                x = -x_lim
                if xd < 0.0:
                    delta_xd = -(1.0 + p.bumper_restitution) * xd
                    xd = -p.bumper_restitution * xd
                    thd -= (delta_xd * math.cos(th + self.table_angle)) / ((4.0 / 3.0) * p.l_cm)
                    bumper_hit = max(bumper_hit, min(1.0, abs(delta_xd) / 2.5 + 0.2))
                    bumper_side = -1

            th = wrap_angle_scalar(th)

        self.x = x
        self.x_dot = xd
        self.theta = th
        self.theta_dot = thd
        self.tau_m = tm
        self.force_motor = (p.gear_ratio * self.tau_m) / p.pulley_radius
        self.pulley_angle = (x / p.pulley_radius) * p.gear_ratio

        endstop_contact = (bumper_hit > 0.0) or (abs(x) >= 0.98 * x_lim)
        if endstop_contact and bumper_side == 0:
            bumper_side = 1 if x >= 0.0 else -1
        self.bumper_hit = max(bumper_hit, 0.5 if endstop_contact else 0.0)
        self.bumper_side = bumper_side

        # Sensor de ângulo quantizado em 1 casa decimal (formato 123,4°)
        self.prev_angle_deg = self.angle_deg
        new_angle_deg = quantize_angle_deg_scalar(th)
        delta_deg = wrap_delta_deg_scalar(new_angle_deg - self.prev_angle_deg)
        self.angle_deg = new_angle_deg
        self.theta_q = math.radians(new_angle_deg)

        dt_safe = max(p.dt, 1e-5)
        new_omega_q = math.radians(delta_deg) / dt_safe
        self.alpha_q = (new_omega_q - self.omega_q) / dt_safe
        self.omega_q = new_omega_q

        self.rot_accum += wrap_angle_scalar(th - self.prev_theta)
        full_rot = 0.0
        if abs(self.rot_accum) >= 2.0 * math.pi:
            full_rot = 1.0
            self.rot_accum -= math.copysign(2.0 * math.pi, self.rot_accum)

        if abs(self.theta) < 0.35 and not endstop_contact and bumper_hit <= 0.0:
            self.up_time += p.dt
        else:
            self.up_time = 0.0

        if math.cos(self.theta) < -0.3:
            self.down_time += p.dt
        else:
            self.down_time = 0.0

        reward = self._compute_reward(u_cmd, bumper_hit, endstop_contact, full_rot=full_rot)
        self.prev_u = u_cmd
        self.step_count += 1
        self.episode_reward += reward

        if abs(self.theta) < 0.22 and abs(self.x) < 0.75 * x_lim and abs(self.theta_dot) < 2.0:
            self.stable_time += p.dt
            if self.stable_time > self.best_stable_time:
                self.best_stable_time = self.stable_time
        else:
            self.stable_time = 0.0

        info = {
            "bumper_hit": bumper_hit,
            "stable_time": self.stable_time,
            "force_motor": self.force_motor,
        }
        return self.get_state(), reward, info

    def _compute_reward(
        self, u_cmd: float, bumper_hit: float, endstop_contact: bool, full_rot: float = 0.0
    ) -> float:
        p = self.params
        cos_t = math.cos(self.theta)
        cos_prev = math.cos(self.prev_theta)
        x_lim = max(p.rail_limit, 0.1)
        x_norm = self.x / x_lim
        dt_safe = max(p.dt, 1e-5)

        wall_gate = 0.0 if (endstop_contact or bumper_hit > 0.0) else max(0.0, min(1.0, (0.98 - abs(x_norm)) / 0.18))

        zone = max(0.05, float(p.rew_center_x_zone))
        f_center_x = math.exp(-((self.x / zone) ** 2)) * wall_gate
        f_top = (0.25 * (cos_t + 1.0) + 0.5 * math.exp(-4.0 * (self.theta ** 2))) * wall_gate
        stable_cond = abs(self.theta) < 0.22 and abs(self.x) < 0.7 * p.rail_limit and abs(self.theta_dot) < 1.8
        f_stability = (1.0 if stable_cond else 0.0) * wall_gate
        f_x_dist = 0.6 * abs(x_norm) + 0.4 * (x_norm ** 2)

        wall_prox = max(0.0, abs(x_norm) - 0.82) / 0.18
        f_endstop = (1.0 + 0.5 * bumper_hit if endstop_contact else 0.0) + 0.65 * (wall_prox ** 2)
        f_up_time_acc = min(self.up_time, 10.0) * wall_gate

        fell_out = 1.0 if (cos_prev >= 0.75 and cos_t < 0.75) else 0.0
        drop_rate = min(2.0, max(0.0, (cos_prev - cos_t) / (6.0 * dt_safe)))
        f_fall = fell_out + drop_rate

        rose_in = 1.0 if (cos_prev < 0.75 and cos_t >= 0.75) else 0.0
        rise_rate = min(2.0, max(0.0, (cos_t - cos_prev) / (6.0 * dt_safe)))
        f_rise = (rose_in + rise_rate) * wall_gate

        f_bottom = 0.5 * (1.0 - cos_t) * math.exp(-2.5 * (1.0 + cos_t))

        f_swingup = 0.0
        if cos_t < 0.85:
            l_cm = p.l_cm
            i_total = (4.0 / 3.0) * p.pendulum_mass * (l_cm ** 2)
            denom_e = max(p.pendulum_mass * FIXED_GRAVITY * l_cm, 1e-4)
            e_norm = (0.5 * i_total * (self.theta_dot ** 2)) / denom_e + cos_t
            f_swingup = math.exp(-0.6 * ((e_norm - 1.0) ** 2)) * wall_gate

        factors = {
            "rew_center_x": f_center_x,
            "rew_top": f_top,
            "rew_stability": f_stability,
            "rew_x_dist": f_x_dist,
            "rew_endstop": f_endstop,
            "rew_up_time_acc": f_up_time_acc,
            "rew_fall": f_fall,
            "rew_rise": f_rise,
            "rew_bottom": f_bottom,
            "rew_down_time_acc": min(self.down_time, 10.0),
            "rew_swingup": f_swingup,
            "rew_smooth": (u_cmd - self.prev_u) ** 2,
            "rew_full_rotation": full_rot,
        }
        breakdown: Dict[str, float] = {}
        total_pts = 0.0
        for w_key, a_key, _, _ in REWARD_TABLE_SPECS:
            pts = float(getattr(p, w_key)) * factors[w_key] if bool(getattr(p, a_key)) else 0.0
            breakdown[w_key] = pts
            total_pts += pts
        breakdown["total_pts"] = total_pts
        self.last_reward_breakdown = breakdown
        return total_pts

    def get_state(self) -> np.ndarray:
        """
        Retorna vetor 13D para a IA:
        - Sensor de ângulo quantizado 0,1° (angle_deg/180, sin, cos, omega_q, alpha_q, prev_angle_deg/180)
        - Ação anterior u_{t-1}
        - Inputs conhecidos: peso do pêndulo (m_p), tamanho do pêndulo (L) e peso do carrinho (m_c)
        - Posição normalizada no eixo X (x, prev_x) e inclinação da mesa (0,3°)
        """
        p = self.params
        x_lim = max(p.rail_limit, 0.1)
        table_deg_q = round(math.degrees(self.table_angle) / 0.3) * 0.3
        return np.array(
            [
                self.angle_deg / 180.0,
                math.sin(self.theta_q),
                math.cos(self.theta_q),
                np.clip(self.omega_q / 12.0, -3.0, 3.0),
                math.tanh(self.alpha_q / 40.0),
                self.prev_angle_deg / 180.0,
                self.prev_u,
                np.clip(p.pendulum_mass / 1.5, 0.0, 2.5),
                np.clip(p.pendulum_length / 1.4, 0.0, 2.0),
                np.clip(p.cart_mass / 4.0, 0.0, 2.5),
                np.clip(self.x / x_lim, -1.5, 1.5),
                np.clip(self.prev_x / x_lim, -1.5, 1.5),
                np.clip(table_deg_q / 45.0, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def get_pivot_world_position(self) -> Tuple[float, float]:
        return (
            self.x * math.cos(self.table_angle),
            self.x * math.sin(self.table_angle),
        )

    def get_tip_position(self) -> Tuple[float, float]:
        px, py = self.get_pivot_world_position()
        tip_x = px + self.params.pendulum_length * math.sin(self.theta)
        tip_y = py + self.params.pendulum_length * math.cos(self.theta)
        return tip_x, tip_y


class BatchedCartPoleEnv:
    """
    Simulador paralelo massivamente vetorizado em PyTorch (GPU/CUDA).

    Respeita as flags e ranges da aba Auto Mutação e utiliza o sensor de ângulo
    quantizado em 0,1° (formato 123,4°) + inputs conhecidos (m_p, L, m_c) em lote.
    """

    STATE_DIM = 13

    def __init__(
        self,
        num_envs: int = 256,
        params: PhysicsParams | None = None,
        device: torch.device | None = None,
        max_episode_steps: int = 450,
        reset_mode: str = "mixed",
        swingup_goal_sec: float = 2.0,
    ) -> None:
        self.num_envs = num_envs
        self.params = params if params is not None else PhysicsParams()
        self.params.gravity = FIXED_GRAVITY
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_episode_steps = max_episode_steps
        self.reset_mode = reset_mode
        self.swingup_goal_sec = float(max(0.2, min(30.0, getattr(self.params, "swing_hold_target", swingup_goal_sec))))
        self.swingup_hold_time = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.swingup_success_rate = 0.0

        self.x = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.prev_x = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.x_dot = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.theta = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.theta_dot = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.tau_m = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.prev_u = torch.zeros(num_envs, device=self.device, dtype=torch.float32)

        # Estado do sensor de ângulo quantizado em lote na GPU (resolução 0,1°)
        self.angle_deg = torch.full((num_envs,), 180.0, device=self.device, dtype=torch.float32)
        self.prev_angle_deg = torch.full((num_envs,), 180.0, device=self.device, dtype=torch.float32)
        self.theta_q = torch.full((num_envs,), math.pi, device=self.device, dtype=torch.float32)
        self.omega_q = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.alpha_q = torch.zeros(num_envs, device=self.device, dtype=torch.float32)

        # Parâmetros por ambiente na GPU
        self.m_p = torch.full((num_envs,), self.params.pendulum_mass, device=self.device, dtype=torch.float32)
        self.m_c = torch.full((num_envs,), self.params.cart_mass, device=self.device, dtype=torch.float32)
        self.l_full = torch.full((num_envs,), self.params.pendulum_length, device=self.device, dtype=torch.float32)
        self.wind_speed = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.wind_target = torch.zeros(num_envs, device=self.device, dtype=torch.float32)

        self.alpha = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.alpha_vel = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.alpha_acc = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.alpha_target = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.alpha_phase = torch.empty(num_envs, device=self.device, dtype=torch.float32).uniform_(0.0, 2.0 * math.pi)

        self.table_timer = torch.randint(40, 220, (num_envs,), device=self.device, dtype=torch.int32)
        self.wind_timer = torch.randint(50, 260, (num_envs,), device=self.device, dtype=torch.int32)
        self.pmass_timer = torch.randint(80, 320, (num_envs,), device=self.device, dtype=torch.int32)
        self.cmass_timer = torch.randint(100, 360, (num_envs,), device=self.device, dtype=torch.int32)
        self.length_timer = torch.randint(100, 360, (num_envs,), device=self.device, dtype=torch.int32)

        self.step_count = torch.zeros(num_envs, device=self.device, dtype=torch.int32)
        self.episode_reward = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.up_time = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.down_time = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.rot_accum = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.prev_theta = torch.full((num_envs,), math.pi, device=self.device, dtype=torch.float32)

        self.reset_all()

    def sync_params(self, params: PhysicsParams) -> None:
        self.params = params
        self.params.gravity = FIXED_GRAVITY
        self.swingup_goal_sec = float(max(0.2, min(30.0, getattr(params, "swing_hold_target", 2.0))))

    def reset_all(self) -> torch.Tensor:
        mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_mask(mask)
        return self.get_states()

    def apply_random_flick(self, fraction: float, force_max: float) -> None:
        """Peteleco em uma fração dos ambientes: lado e força aleatórios por ambiente."""
        n = self.num_envs
        pick = (torch.rand(n, device=self.device) < fraction).float()
        sign = torch.where(
            torch.rand(n, device=self.device) < 0.5,
            -torch.ones(n, device=self.device),
            torch.ones(n, device=self.device),
        )
        mag = torch.empty(n, device=self.device).uniform_(0.25, 1.0) * force_max
        self.theta_dot = torch.clamp(self.theta_dot + pick * sign * mag, -30.0, 30.0)

    def reset_mask(self, mask: torch.Tensor) -> None:
        n_reset = int(mask.sum().item())
        if n_reset == 0:
            return

        p = self.params
        x_lim = max(p.rail_limit * 0.35, 0.05)

        if self.reset_mode == "swingup_2s":
            self.x[mask] = 0.0
            self.prev_x[mask] = 0.0
            self.x_dot[mask] = 0.0
            self.theta_dot[mask] = 0.0
            new_th = torch.full((n_reset,), math.pi, device=self.device)
        elif self.reset_mode == "stability":
            init_x = torch.empty(n_reset, device=self.device).uniform_(-x_lim, x_lim)
            self.x[mask] = init_x
            self.prev_x[mask] = init_x
            self.x_dot[mask] = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
            self.theta_dot[mask] = torch.empty(n_reset, device=self.device).uniform_(-0.5, 0.5)
            new_th = wrap_angle_torch(torch.empty(n_reset, device=self.device).uniform_(-0.35, 0.35))
        else:
            init_x = torch.empty(n_reset, device=self.device).uniform_(-x_lim, x_lim)
            self.x[mask] = init_x
            self.prev_x[mask] = init_x
            self.x_dot[mask] = torch.empty(n_reset, device=self.device).uniform_(-0.4, 0.4)
            self.theta_dot[mask] = torch.empty(n_reset, device=self.device).uniform_(-0.7, 0.7)

            mode_roll = torch.rand(n_reset, device=self.device)
            theta_down = torch.full((n_reset,), math.pi, device=self.device)
            theta_rand = torch.empty(n_reset, device=self.device).uniform_(-math.pi, math.pi)
            theta_up = torch.empty(n_reset, device=self.device).uniform_(-0.35, 0.35)
            new_th = wrap_angle_torch(
                torch.where(mode_roll < 0.55, theta_down, torch.where(mode_roll < 0.80, theta_rand, theta_up))
            )

        self.theta[mask] = new_th
        self.prev_theta[mask] = new_th
        self.up_time[mask] = 0.0
        self.down_time[mask] = 0.0
        self.rot_accum[mask] = 0.0
        self.swingup_hold_time[mask] = 0.0
        self.tau_m[mask] = 0.0
        self.prev_u[mask] = 0.0
        self.step_count[mask] = 0
        self.episode_reward[mask] = 0.0

        # Inicializa sensor de ângulo quantizado em 0,1° na GPU
        q_deg = quantize_angle_deg_torch(new_th)
        self.angle_deg[mask] = q_deg
        self.prev_angle_deg[mask] = q_deg
        self.theta_q[mask] = torch.deg2rad(q_deg)
        self.omega_q[mask] = 0.0
        self.alpha_q[mask] = 0.0

        if p.auto_mutate and p.mutate_pend_mass:
            pm_lo, pm_hi = _ordered_range(p.pend_mass_min, p.pend_mass_max, 0.05, 3.0)
            self.m_p[mask] = torch.empty(n_reset, device=self.device).uniform_(pm_lo, pm_hi)
        else:
            self.m_p[mask] = p.pendulum_mass

        if p.auto_mutate and p.mutate_cart_mass:
            cm_lo, cm_hi = _ordered_range(p.cart_mass_min, p.cart_mass_max, 0.20, 8.0)
            self.m_c[mask] = torch.empty(n_reset, device=self.device).uniform_(cm_lo, cm_hi)
        else:
            self.m_c[mask] = p.cart_mass

        if p.auto_mutate and p.mutate_length:
            l_lo, l_hi = _ordered_range(p.length_min, p.length_max, 0.20, 2.0)
            self.l_full[mask] = torch.empty(n_reset, device=self.device).uniform_(l_lo, l_hi)
        else:
            self.l_full[mask] = p.pendulum_length

        if p.auto_mutate and p.mutate_wind:
            w_lo, w_hi = _ordered_range(p.wind_min, p.wind_max, -MAX_WIND_SPEED, MAX_WIND_SPEED)
            self.wind_speed[mask] = torch.empty(n_reset, device=self.device).uniform_(w_lo, w_hi)
            self.wind_target[mask] = torch.empty(n_reset, device=self.device).uniform_(w_lo, w_hi)
        else:
            self.wind_speed[mask] = p.wind_speed
            self.wind_target[mask] = p.wind_speed

        if p.auto_mutate and p.mutate_table:
            t_lo_deg, t_hi_deg = _ordered_range(p.table_min_deg, p.table_max_deg, -45.0, 45.0)
            t_lo, t_hi = math.radians(t_lo_deg), math.radians(t_hi_deg)
            self.alpha[mask] = torch.empty(n_reset, device=self.device).uniform_(t_lo, t_hi)
            self.alpha_target[mask] = torch.empty(n_reset, device=self.device).uniform_(t_lo, t_hi)
            self.alpha_vel[mask] = torch.empty(n_reset, device=self.device).uniform_(-0.35, 0.35)
        else:
            self.alpha[mask] = p.table_angle
            self.alpha_target[mask] = p.table_angle
            self.alpha_vel[mask] = 0.0
        self.alpha_acc[mask] = 0.0

    def _mutate_environments_gpu(self) -> None:
        p = self.params
        dt = p.dt
        base_dur = max(0.2, min(300.0, float(p.mutation_interval)))
        base_steps = max(5, int(round(base_dur / max(dt, 1e-4))))
        lo_steps = max(4, int(round(base_steps * 0.85)))
        hi_steps = max(lo_steps + 2, int(round(base_steps * 1.15)))

        self.table_timer = torch.clamp(self.table_timer, max=hi_steps)
        self.wind_timer = torch.clamp(self.wind_timer, max=hi_steps)
        self.pmass_timer = torch.clamp(self.pmass_timer, max=hi_steps)
        self.cmass_timer = torch.clamp(self.cmass_timer, max=hi_steps)
        self.length_timer = torch.clamp(self.length_timer, max=hi_steps)

        is_fixed = getattr(p, "mutation_mode", "fixed") == "fixed"
        if p.auto_mutate and p.mutate_table:
            t_lo_deg, t_hi_deg = _ordered_range(p.table_min_deg, p.table_max_deg, -45.0, 45.0)
            t_lo, t_hi = math.radians(t_lo_deg), math.radians(t_hi_deg)
            self.table_timer -= 1
            t_trig = self.table_timer <= 0
            if t_trig.any():
                n_t = int(t_trig.sum().item())
                self.alpha_target[t_trig] = torch.empty(n_t, device=self.device).uniform_(t_lo, t_hi)
                if is_fixed:
                    self.table_timer[t_trig] = base_steps
                else:
                    self.table_timer[t_trig] = torch.randint(lo_steps, hi_steps, (n_t,), device=self.device, dtype=torch.int32)

            if is_fixed:
                desired_a = torch.clamp(self.alpha_target, t_lo, t_hi)
                omega_n = 5.5
                zeta = 1.0
            else:
                self.alpha_phase += dt * 1.85
                osc_amp = min(0.14, 0.25 * (t_hi - t_lo))
                desired_a = torch.clamp(self.alpha_target + osc_amp * torch.sin(self.alpha_phase), t_lo, t_hi)
                omega_n = 2.3
                zeta = 0.82

            a_acc = (omega_n ** 2) * (desired_a - self.alpha) - 2.0 * zeta * omega_n * self.alpha_vel
            self.alpha_vel = self.alpha_vel + a_acc * dt
            self.alpha = self.alpha + self.alpha_vel * dt

            hit_hi = self.alpha > t_hi
            hit_lo = self.alpha < t_lo
            self.alpha = torch.clamp(self.alpha, t_lo, t_hi)
            self.alpha_vel = torch.where(hit_hi, torch.clamp(self.alpha_vel, max=0.0), self.alpha_vel)
            self.alpha_vel = torch.where(hit_lo, torch.clamp(self.alpha_vel, min=0.0), self.alpha_vel)
            self.alpha_acc = torch.where(hit_hi | hit_lo, torch.zeros_like(a_acc), a_acc)
        else:
            self.alpha.fill_(p.table_angle)
            self.alpha_vel.zero_()
            self.alpha_acc.zero_()

        if p.auto_mutate and p.mutate_wind:
            w_lo, w_hi = _ordered_range(p.wind_min, p.wind_max, -MAX_WIND_SPEED, MAX_WIND_SPEED)
            self.wind_timer -= 1
            w_trig = self.wind_timer <= 0
            if w_trig.any():
                n_w = int(w_trig.sum().item())
                self.wind_target[w_trig] = torch.empty(n_w, device=self.device).uniform_(w_lo, w_hi)
                self.wind_timer[w_trig] = torch.randint(lo_steps, hi_steps, (n_w,), device=self.device, dtype=torch.int32)
            self.wind_target = torch.clamp(self.wind_target, w_lo, w_hi)
            self.wind_speed = torch.clamp(
                self.wind_speed + (self.wind_target - self.wind_speed) * min(1.0, dt * 1.8), w_lo, w_hi
            )
        else:
            self.wind_speed.fill_(p.wind_speed)

        if p.auto_mutate and p.mutate_pend_mass:
            pm_lo, pm_hi = _ordered_range(p.pend_mass_min, p.pend_mass_max, 0.05, 3.0)
            self.pmass_timer -= 1
            pm_trig = self.pmass_timer <= 0
            if pm_trig.any():
                n_pm = int(pm_trig.sum().item())
                self.m_p[pm_trig] = torch.empty(n_pm, device=self.device).uniform_(pm_lo, pm_hi)
                self.pmass_timer[pm_trig] = torch.randint(lo_steps, hi_steps, (n_pm,), device=self.device, dtype=torch.int32)
        else:
            self.m_p.fill_(p.pendulum_mass)

        if p.auto_mutate and p.mutate_cart_mass:
            cm_lo, cm_hi = _ordered_range(p.cart_mass_min, p.cart_mass_max, 0.20, 8.0)
            self.cmass_timer -= 1
            cm_trig = self.cmass_timer <= 0
            if cm_trig.any():
                n_cm = int(cm_trig.sum().item())
                self.m_c[cm_trig] = torch.empty(n_cm, device=self.device).uniform_(cm_lo, cm_hi)
                self.cmass_timer[cm_trig] = torch.randint(lo_steps, hi_steps, (n_cm,), device=self.device, dtype=torch.int32)
        else:
            self.m_c.fill_(p.cart_mass)

        if p.auto_mutate and p.mutate_length:
            l_lo, l_hi = _ordered_range(p.length_min, p.length_max, 0.20, 2.0)
            self.length_timer -= 1
            l_trig = self.length_timer <= 0
            if l_trig.any():
                n_l = int(l_trig.sum().item())
                self.l_full[l_trig] = torch.empty(n_l, device=self.device).uniform_(l_lo, l_hi)
                self.length_timer[l_trig] = torch.randint(lo_steps, hi_steps, (n_l,), device=self.device, dtype=torch.int32)
        else:
            self.l_full.fill_(p.pendulum_length)

    def _derivatives(
        self,
        x: torch.Tensor,
        xd: torch.Tensor,
        th: torch.Tensor,
        thd: torch.Tensor,
        tm: torch.Tensor,
        u_cmd: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.params
        l_cm = 0.5 * self.l_full
        m_p = self.m_p
        reflected = p.rotor_inertia * (p.gear_ratio ** 2) / (p.pulley_radius ** 2)
        m_eff = self.m_c + reflected
        m_total = m_eff + m_p
        i_total = (4.0 / 3.0) * m_p * (l_cm ** 2)
        ml = m_p * l_cm
        g = FIXED_GRAVITY

        speed_ratio = torch.abs(xd) / max(p.v_max, 0.5)
        tau_lim = p.tau_max * torch.clamp(1.0 - speed_ratio, min=0.15, max=1.0)
        tau_cmd = torch.clamp(u_cmd * p.tau_max, -tau_lim, tau_lim)
        tau_m_dot = (tau_cmd - tm) / max(p.motor_tau, 1e-3)

        f_motor = (p.gear_ratio * tm) / p.pulley_radius

        cos_a = torch.cos(self.alpha)
        sin_a = torch.sin(self.alpha)
        v_rod_x = xd * cos_a + l_cm * thd * torch.cos(th) - x * self.alpha_vel * sin_a
        v_rel = self.wind_speed - v_rod_x
        f_wind = p.wind_quad_coeff * v_rel * torch.abs(v_rel) + p.wind_lin_coeff * v_rel
        tau_wind = f_wind * l_cm * torch.cos(th)

        f_net = (
            f_motor
            + f_wind * cos_a
            - p.cart_viscous_friction * xd
            - p.cart_coulomb_friction * torch.tanh(10.0 * xd)
        )

        sum_ta = th + self.alpha
        sin_ta = torch.sin(sum_ta)
        cos_ta = torch.cos(sum_ta)

        rhs_x = (
            f_net
            + ml * sin_ta * (thd ** 2)
            - m_total * g * sin_a
            + m_total * x * (self.alpha_vel ** 2)
        )
        rhs_t = (
            ml * g * torch.sin(th)
            - p.pendulum_friction * thd
            + tau_wind
            + ml * (
                2.0 * xd * self.alpha_vel * sin_ta
                + x * self.alpha_acc * sin_ta
                + x * (self.alpha_vel ** 2) * cos_ta
            )
        )

        det = torch.clamp(m_total * i_total - (ml * cos_ta) ** 2, min=1e-8)
        x_ddot = (i_total * rhs_x - ml * cos_ta * rhs_t) / det
        theta_ddot = (m_total * rhs_t - ml * cos_ta * rhs_x) / det
        return xd, x_ddot, thd, theta_ddot, tau_m_dot

    @torch.no_grad()
    def step(
        self, action_u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.params
        self.prev_theta = self.theta.clone()
        self.prev_x = self.x.clone()

        self._mutate_environments_gpu()

        u_cmd = torch.clamp(action_u, -1.0, 1.0)
        h = p.dt / max(1, p.substeps)
        x_lim = p.rail_limit
        restitution = p.bumper_restitution
        inv_effective_len = 1.0 / ((4.0 / 3.0) * torch.clamp(0.5 * self.l_full, min=1e-4))

        x = self.x
        xd = self.x_dot
        th = self.theta
        thd = self.theta_dot
        tm = self.tau_m
        bumper_hit = torch.zeros_like(x)

        for _ in range(p.substeps):
            k1 = self._derivatives(x, xd, th, thd, tm, u_cmd)
            k2 = self._derivatives(
                x + 0.5 * h * k1[0], xd + 0.5 * h * k1[1], th + 0.5 * h * k1[2],
                thd + 0.5 * h * k1[3], tm + 0.5 * h * k1[4], u_cmd,
            )
            k3 = self._derivatives(
                x + 0.5 * h * k2[0], xd + 0.5 * h * k2[1], th + 0.5 * h * k2[2],
                thd + 0.5 * h * k2[3], tm + 0.5 * h * k2[4], u_cmd,
            )
            k4 = self._derivatives(
                x + h * k3[0], xd + h * k3[1], th + h * k3[2],
                thd + h * k3[3], tm + h * k3[4], u_cmd,
            )

            x = x + (h / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            xd = xd + (h / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            th = th + (h / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
            thd = thd + (h / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
            tm = tm + (h / 6.0) * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])

            hit_right = (x > x_lim) & (xd > 0.0)
            hit_left = (x < -x_lim) & (xd < 0.0)
            hit_any = hit_right | hit_left
            x = torch.clamp(x, -x_lim, x_lim)
            if hit_any.any():
                delta_xd = -(1.0 + restitution) * xd
                xd = torch.where(hit_any, -restitution * xd, xd)
                thd = torch.where(
                    hit_any,
                    thd - (delta_xd * torch.cos(th + self.alpha)) * inv_effective_len,
                    thd,
                )
                bumper_hit = torch.where(hit_any, torch.ones_like(bumper_hit), bumper_hit)

            th = wrap_angle_torch(th)

        self.x = x
        self.x_dot = xd
        self.theta = th
        self.theta_dot = thd
        self.tau_m = tm

        # Atualiza sensor de ângulo quantizado em 0,1° na GPU
        self.prev_angle_deg = self.angle_deg
        new_angle_deg = quantize_angle_deg_torch(th)
        delta_deg = wrap_delta_deg_torch(new_angle_deg - self.prev_angle_deg)
        self.angle_deg = new_angle_deg
        self.theta_q = torch.deg2rad(new_angle_deg)

        dt_safe = max(p.dt, 1e-5)
        new_omega_q = torch.deg2rad(delta_deg) / dt_safe
        self.alpha_q = (new_omega_q - self.omega_q) / dt_safe
        self.omega_q = new_omega_q

        two_pi = 2.0 * math.pi
        self.rot_accum = self.rot_accum + wrap_angle_torch(th - self.prev_theta)
        rot_hit = torch.abs(self.rot_accum) >= two_pi
        full_rot = rot_hit.float()
        self.rot_accum = torch.where(
            rot_hit,
            self.rot_accum - torch.sign(self.rot_accum) * two_pi,
            self.rot_accum,
        )

        endstop_hit = (bumper_hit > 0.0) | (torch.abs(self.x) >= 0.98 * max(x_lim, 0.1))
        self.up_time = torch.where(
            (torch.abs(self.theta) < 0.35) & (~endstop_hit),
            self.up_time + p.dt,
            torch.zeros_like(self.up_time),
        )
        self.down_time = torch.where(
            torch.cos(self.theta) < -0.3,
            self.down_time + p.dt,
            torch.zeros_like(self.down_time),
        )

        rewards = self._compute_rewards(u_cmd, bumper_hit, full_rot)

        self.prev_u = u_cmd
        self.step_count += 1
        self.episode_reward += rewards

        next_states = self.get_states()
        if self.reset_mode == "swingup_2s":
            stable_now = (
                (torch.abs(self.theta) < 0.22)
                & (torch.abs(self.x) < 0.7 * max(x_lim, 0.1))
                & (torch.abs(self.theta_dot) < 1.8)
            )
            self.swingup_hold_time = torch.where(
                stable_now,
                self.swingup_hold_time + p.dt,
                torch.zeros_like(self.swingup_hold_time),
            )
            succeeded = self.swingup_hold_time >= self.swingup_goal_sec
            truncated = (self.step_count >= self.max_episode_steps) | succeeded
            if truncated.any():
                batch_rate = float(succeeded[truncated].float().mean().item())
                self.swingup_success_rate = 0.85 * self.swingup_success_rate + 0.15 * batch_rate
        else:
            truncated = self.step_count >= self.max_episode_steps

        bootstrap_dones = torch.zeros_like(truncated)

        completed_rewards = self.episode_reward[truncated].clone()
        if truncated.any():
            self.reset_mask(truncated)

        return next_states, rewards, bootstrap_dones, completed_rewards

    def _compute_rewards(
        self, u_cmd: torch.Tensor, bumper_hit: torch.Tensor, full_rot: torch.Tensor
    ) -> torch.Tensor:
        p = self.params
        cos_t = torch.cos(self.theta)
        cos_prev = torch.cos(self.prev_theta)
        x_lim = max(p.rail_limit, 0.1)
        x_norm = self.x / x_lim
        dt_safe = max(p.dt, 1e-5)

        wall_prox = torch.clamp((torch.abs(x_norm) - 0.82) / 0.18, min=0.0)
        endstop_hit = (bumper_hit > 0.0) | (torch.abs(self.x) >= 0.98 * p.rail_limit)
        wall_gate = torch.where(
            endstop_hit,
            torch.zeros_like(x_norm),
            torch.clamp((0.98 - torch.abs(x_norm)) / 0.18, 0.0, 1.0),
        )

        zone = max(0.05, float(p.rew_center_x_zone))
        f_center_x = torch.exp(-((self.x / zone) ** 2)) * wall_gate
        f_top = (0.25 * (cos_t + 1.0) + 0.5 * torch.exp(-4.0 * (self.theta ** 2))) * wall_gate
        stable_mask = (
            (torch.abs(self.theta) < 0.22)
            & (torch.abs(self.x) < 0.7 * p.rail_limit)
            & (torch.abs(self.theta_dot) < 1.8)
        )
        f_stability = stable_mask.float() * wall_gate
        f_x_dist = 0.6 * abs(x_norm) + 0.4 * (x_norm ** 2)

        endstop_factor = torch.where(
            endstop_hit,
            1.0 + 0.5 * bumper_hit,
            torch.zeros_like(x_norm),
        )
        f_endstop = endstop_factor + 0.65 * (wall_prox ** 2)
        f_up_time_acc = torch.clamp(self.up_time, max=10.0) * wall_gate

        fell_out = ((cos_prev >= 0.75) & (cos_t < 0.75)).float()
        drop_rate = torch.clamp((cos_prev - cos_t) / (6.0 * dt_safe), 0.0, 2.0)
        f_fall = fell_out + drop_rate

        rose_in = ((cos_prev < 0.75) & (cos_t >= 0.75)).float()
        rise_rate = torch.clamp((cos_t - cos_prev) / (6.0 * dt_safe), 0.0, 2.0)
        f_rise = (rose_in + rise_rate) * wall_gate

        f_bottom = 0.5 * (1.0 - cos_t) * torch.exp(-2.5 * (1.0 + cos_t))

        l_cm = 0.5 * self.l_full
        i_total = (4.0 / 3.0) * self.m_p * (l_cm ** 2)
        denom_e = torch.clamp(self.m_p * FIXED_GRAVITY * l_cm, min=1e-4)
        e_norm = (0.5 * i_total * (self.theta_dot ** 2)) / denom_e + cos_t
        f_swingup = torch.where(
            cos_t < 0.85,
            torch.exp(-0.6 * ((e_norm - 1.0) ** 2)),
            torch.zeros_like(cos_t),
        ) * wall_gate

        factors = {
            "rew_center_x": f_center_x,
            "rew_top": f_top,
            "rew_stability": f_stability,
            "rew_x_dist": f_x_dist,
            "rew_endstop": f_endstop,
            "rew_up_time_acc": f_up_time_acc,
            "rew_fall": f_fall,
            "rew_rise": f_rise,
            "rew_bottom": f_bottom,
            "rew_down_time_acc": torch.clamp(self.down_time, max=10.0),
            "rew_swingup": f_swingup,
            "rew_smooth": (u_cmd - self.prev_u) ** 2,
            "rew_full_rotation": full_rot,
        }
        total = torch.zeros_like(cos_t)
        for w_key, a_key, _, _ in REWARD_TABLE_SPECS:
            if bool(getattr(p, a_key)):
                total = total + float(getattr(p, w_key)) * factors[w_key]
        return total

    def get_states(self) -> torch.Tensor:
        """Retorna matriz (N, 13) com ângulo (0,1°), m_p/L/m_c, posição X e inclinação da mesa (0,3°)."""
        x_lim = max(self.params.rail_limit, 0.1)
        table_deg_q = torch.round(torch.rad2deg(self.alpha) / 0.3) * 0.3
        return torch.stack(
            [
                self.angle_deg / 180.0,
                torch.sin(self.theta_q),
                torch.cos(self.theta_q),
                torch.clamp(self.omega_q / 12.0, -3.0, 3.0),
                torch.tanh(self.alpha_q / 40.0),
                self.prev_angle_deg / 180.0,
                self.prev_u,
                torch.clamp(self.m_p / 1.5, 0.0, 2.5),
                torch.clamp(self.l_full / 1.4, 0.0, 2.0),
                torch.clamp(self.m_c / 4.0, 0.0, 2.5),
                torch.clamp(self.x / x_lim, -1.5, 1.5),
                torch.clamp(self.prev_x / x_lim, -1.5, 1.5),
                torch.clamp(table_deg_q / 45.0, -1.0, 1.0),
            ],
            dim=1,
        )
