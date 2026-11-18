"""
Interface Gráfica Desktop em Pygame (60+ FPS) para o Simulador de Pêndulo Invertido:
- Aba 'Controle': Sliders apenas para Inclinação Base (α) e Vento; campos digitáveis (sem barra)
  para Velocidade da Simulação (s/s), Peso Pêndulo, Peso Carrinho, Comprimento L, Tempo de Mutação,
  Força Motor e Taxa de Aprendizado.
- Aba 'Auto Mutação': Seleção individual do que randomizar (Base, Vento, Peso Pêndulo, Peso Carrinho,
  Comprimento L) com ranges [Mín, Máx] digitáveis e duração controlável.
- Gráficos: Sensor Ângulo (123,4° & ω_q), Eixo X (centro = 0) e Recompensa Atual (janela 30s).
- Sem alerta visual de perda de pontos no fim de curso (penalidade aplicada silenciosamente no RL).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pygame

from physics import CartPolePhysics, FIXED_GRAVITY, PhysicsParams, REWARD_TABLE_SPECS, format_angle_deg_br
from rl_agent import RLAgent


BG_DARK = (13, 17, 23)
PANEL_BG = (22, 27, 34)
PANEL_ALT = (28, 35, 45)
BORDER_COLOR = (46, 57, 73)
GRID_COLOR = (25, 33, 44)
TEXT_PRIMARY = (230, 237, 243)
TEXT_SECONDARY = (139, 148, 158)
ACCENT_CYAN = (56, 189, 248)
ACCENT_GREEN = (16, 185, 129)
ACCENT_AMBER = (245, 158, 11)
ACCENT_ROSE = (244, 63, 94)
ACCENT_VIOLET = (167, 139, 250)
RAIL_METAL = (95, 108, 126)
CART_BODY = (52, 65, 84)
CART_TRIM = (88, 166, 255)


@dataclass
class TelemetryData:
    """Buffers e métricas em tempo real exibidas na interface e nos gráficos."""

    angles_deg: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    ang_vels_q: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    positions_x: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    reward_30s: Deque[Tuple[float, float]] = field(default_factory=deque)

    def prune_reward_30s(self, now_t: Optional[float] = None, window_sec: float = 30.0) -> None:
        """Remove amostras com mais de 30 segundos da janela deslizante."""
        ref = time.perf_counter() if now_t is None else float(now_t)
        while self.reward_30s and (ref - self.reward_30s[0][0]) > window_sec:
            self.reward_30s.popleft()

    def record_reward(self, value: float, now_t: Optional[float] = None, window_sec: float = 30.0) -> None:
        """Registra a recompensa atual com timestamp e descarta registros antigos."""
        ref = time.perf_counter() if now_t is None else float(now_t)
        self.reward_30s.append((ref, float(value)))
        self.prune_reward_30s(now_t=ref, window_sec=window_sec)

    fps: float = 60.0
    steps_per_sec: float = 0.0
    total_env_steps: int = 0
    total_grad_steps: int = 0
    buffer_size: int = 0
    ai_active: bool = False
    training_active: bool = True
    auto_mutate: bool = True
    epsilon_auto: float = 1.0
    current_u: float = 0.0
    last_reward: float = 0.0
    rail_limit: float = 1.80
    reward_breakdown: Dict[str, float] = field(default_factory=dict)
    meta_status: Dict[str, object] = field(default_factory=dict)
    device_info: Dict[str, object] = field(default_factory=dict)
    swing_successes: int = 0
    swing_attempts: int = 0
    swing_hold_time: float = 0.0
    swing_hold_target: float = 2.0


class Checkbox:
    """Widget de checkbox estilizado."""

    def __init__(
        self,
        key: str,
        label: str,
        checked: bool = False,
        accent: Tuple[int, int, int] = ACCENT_CYAN,
    ) -> None:
        self.key = key
        self.label = label
        self.checked = checked
        self.accent = accent
        self.rect = pygame.Rect(0, 0, 200, 22)
        self.hovered = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.checked = not self.checked
                return True
        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        box_size = 16
        box_rect = pygame.Rect(
            self.rect.x,
            self.rect.y + (self.rect.height - box_size) // 2,
            box_size,
            box_size,
        )
        bg = self.accent if self.checked else PANEL_ALT
        border = self.accent if (self.checked or self.hovered) else BORDER_COLOR
        pygame.draw.rect(surface, bg, box_rect, border_radius=4)
        pygame.draw.rect(surface, border, box_rect, width=1, border_radius=4)

        if self.checked:
            p1 = (box_rect.x + 3, box_rect.y + 8)
            p2 = (box_rect.x + 7, box_rect.y + 12)
            p3 = (box_rect.x + 13, box_rect.y + 4)
            pygame.draw.lines(surface, BG_DARK, False, [p1, p2, p3], width=2)

        text_surf = font.render(self.label, True, TEXT_PRIMARY)
        surface.blit(text_surf, (box_rect.right + 8, self.rect.y + (self.rect.height - text_surf.get_height()) // 2))


class Button:
    """Botão interativo com realce de hover e clique."""

    def __init__(
        self,
        key: str,
        label: str,
        color: Tuple[int, int, int] = PANEL_ALT,
        border_accent: Tuple[int, int, int] = BORDER_COLOR,
    ) -> None:
        self.key = key
        self.label = label
        self.color = color
        self.border_accent = border_accent
        self.rect = pygame.Rect(0, 0, 150, 26)
        self.hovered = False
        self.pressed = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
                return False
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_pressed = self.pressed
            self.pressed = False
            if was_pressed and self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, active: bool = False) -> None:
        base = self.color
        if active:
            bg = tuple(min(255, c + 24) for c in base)
        elif self.pressed:
            bg = tuple(max(0, c - 15) for c in base)
        elif self.hovered:
            bg = tuple(min(255, c + 16) for c in base)
        else:
            bg = base

        border = self.border_accent if (self.hovered or active) else BORDER_COLOR
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, border, self.rect, width=1, border_radius=6)

        text_surf = font.render(self.label, True, TEXT_PRIMARY)
        tx = self.rect.x + (self.rect.width - text_surf.get_width()) // 2
        ty = self.rect.y + (self.rect.height - text_surf.get_height()) // 2
        surface.blit(text_surf, (tx, ty))


class NumberField:
    """Campo numérico puramente digitável (sem barra deslizante)."""

    def __init__(
        self,
        key: str,
        label: str,
        min_val: float,
        max_val: float,
        value: float,
        fmt: str = "{:.2f}",
        unit: str = "",
        accent: Tuple[int, int, int] = ACCENT_CYAN,
        compact_inline: bool = False,
    ) -> None:
        self.key = key
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.fmt = fmt
        self.unit = unit
        self.accent = accent
        self.compact_inline = compact_inline

        self.rect = pygame.Rect(0, 0, 160, 24)
        self.input_rect = pygame.Rect(0, 0, 84, 21)
        self.hovered = False
        self.editing = False
        self.edit_text = ""

    def set_value(self, value: float) -> None:
        if not self.editing:
            self.value = float(max(self.min_val, min(self.max_val, value)))

    def update_layout(self, x: int, y: int, width: int, height: int = 24, box_w: int = 88) -> None:
        self.rect = pygame.Rect(x, y, width, height)
        if self.compact_inline:
            self.input_rect = pygame.Rect(x, y, width, height)
        else:
            self.input_rect = pygame.Rect(x + width - box_w, y + 1, box_w, height - 2)

    def _commit_edit(self) -> bool:
        self.editing = False
        txt = self.edit_text.strip().replace(",", ".")
        if not txt or txt in {"-", "+", "."}:
            return False
        try:
            parsed = float(txt)
        except ValueError:
            return False
        if not math.isfinite(parsed):
            return False
        self.value = float(max(self.min_val, min(self.max_val, parsed)))
        return True

    def handle_event(self, event: pygame.event.Event) -> bool:
        if self.editing:
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return self._commit_edit()
                if event.key == pygame.K_ESCAPE:
                    self.editing = False
                    return False
                if event.key == pygame.K_BACKSPACE:
                    self.edit_text = self.edit_text[:-1]
                    return False
                if event.unicode and event.unicode in "0123456789.-+,eE" and len(self.edit_text) < 12:
                    self.edit_text += event.unicode
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not self.input_rect.collidepoint(event.pos):
                    return self._commit_edit()
                return False

        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.input_rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.input_rect.collidepoint(event.pos):
                self.editing = True
                self.edit_text = self.fmt.format(self.value).lstrip("+")
                return False
        return False

    def draw(self, surface: pygame.Surface, font_label: pygame.font.Font, font_val: pygame.font.Font) -> None:
        if not self.compact_inline and self.label:
            lbl = font_label.render(self.label, True, TEXT_SECONDARY)
            surface.blit(lbl, (self.rect.x, self.rect.y + (self.rect.height - lbl.get_height()) // 2))

        bg = (16, 22, 30) if self.editing else PANEL_ALT
        border = self.accent if self.editing else (TEXT_SECONDARY if self.hovered else BORDER_COLOR)
        pygame.draw.rect(surface, bg, self.input_rect, border_radius=4)
        pygame.draw.rect(surface, border, self.input_rect, width=1, border_radius=4)

        if self.editing:
            cursor = "|" if (pygame.time.get_ticks() // 450) % 2 == 0 else ""
            val_str = self.edit_text + cursor
        else:
            val_str = self.fmt.format(self.value) + (f" {self.unit}" if (self.unit and not self.compact_inline) else "")

        val_surf = font_val.render(val_str, True, TEXT_PRIMARY)
        vx = max(self.input_rect.x + 4, self.input_rect.right - val_surf.get_width() - 5)
        vy = self.input_rect.y + (self.input_rect.height - val_surf.get_height()) // 2
        surface.blit(val_surf, (vx, vy))


class Slider:
    """Slider contínuo com barra e caixa numérica digitável (para Base e Vento)."""

    def __init__(
        self,
        key: str,
        label: str,
        min_val: float,
        max_val: float,
        value: float,
        fmt: str = "{:.1f}",
        unit: str = "",
        accent: Tuple[int, int, int] = ACCENT_CYAN,
    ) -> None:
        self.key = key
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.fmt = fmt
        self.unit = unit
        self.accent = accent

        self.rect = pygame.Rect(0, 0, 300, 42)
        self.track_rect = pygame.Rect(0, 0, 300, 6)
        self.input_rect = pygame.Rect(0, 0, 84, 19)
        self.dragging = False
        self.hovered = False
        self.input_hovered = False
        self.editing = False
        self.edit_text = ""

    def _val_to_ratio(self, val: float) -> float:
        val = max(self.min_val, min(self.max_val, val))
        return (val - self.min_val) / max(self.max_val - self.min_val, 1e-9)

    def _ratio_to_val(self, ratio: float) -> float:
        ratio = max(0.0, min(1.0, ratio))
        return self.min_val + ratio * (self.max_val - self.min_val)

    def set_value(self, value: float) -> None:
        if not self.dragging and not self.editing:
            self.value = float(max(self.min_val, min(self.max_val, value)))

    def update_layout(self, x: int, y: int, width: int) -> None:
        self.rect = pygame.Rect(x, y, width, 42)
        self.track_rect = pygame.Rect(x, y + 26, width, 6)
        box_w = min(92, max(76, width // 4))
        self.input_rect = pygame.Rect(x + width - box_w, y, box_w, 19)

    def _commit_edit(self) -> bool:
        self.editing = False
        txt = self.edit_text.strip().replace(",", ".")
        if not txt or txt in {"-", "+", "."}:
            return False
        try:
            parsed = float(txt)
        except ValueError:
            return False
        if not math.isfinite(parsed):
            return False
        self.value = float(max(self.min_val, min(self.max_val, parsed)))
        return True

    def handle_event(self, event: pygame.event.Event) -> bool:
        hit_rect = self.track_rect.inflate(14, 18)

        if self.editing:
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return self._commit_edit()
                if event.key == pygame.K_ESCAPE:
                    self.editing = False
                    return False
                if event.key == pygame.K_BACKSPACE:
                    self.edit_text = self.edit_text[:-1]
                    return False
                if event.unicode and event.unicode in "0123456789.-+" and len(self.edit_text) < 12:
                    self.edit_text += event.unicode
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not self.input_rect.collidepoint(event.pos):
                    committed = self._commit_edit()
                    if hit_rect.collidepoint(event.pos):
                        self.dragging = True
                        ratio = (event.pos[0] - self.track_rect.x) / max(1, self.track_rect.width)
                        self.value = self._ratio_to_val(ratio)
                        return True
                    return committed
                return False

        if event.type == pygame.MOUSEMOTION:
            self.hovered = hit_rect.collidepoint(event.pos)
            self.input_hovered = self.input_rect.collidepoint(event.pos)
            if self.dragging:
                ratio = (event.pos[0] - self.track_rect.x) / max(1, self.track_rect.width)
                self.value = self._ratio_to_val(ratio)
                return True
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.input_rect.collidepoint(event.pos):
                self.editing = True
                self.edit_text = self.fmt.format(self.value).lstrip("+")
                return False
            if hit_rect.collidepoint(event.pos):
                self.dragging = True
                ratio = (event.pos[0] - self.track_rect.x) / max(1, self.track_rect.width)
                self.value = self._ratio_to_val(ratio)
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True
        return False

    def draw(self, surface: pygame.Surface, font_label: pygame.font.Font, font_val: pygame.font.Font) -> None:
        label_surf = font_label.render(self.label, True, TEXT_SECONDARY)
        surface.blit(label_surf, (self.rect.x, self.rect.y + 2))

        box_bg = (16, 22, 30) if self.editing else PANEL_ALT
        box_border = self.accent if self.editing else (TEXT_SECONDARY if self.input_hovered else BORDER_COLOR)
        pygame.draw.rect(surface, box_bg, self.input_rect, border_radius=4)
        pygame.draw.rect(surface, box_border, self.input_rect, width=1, border_radius=4)

        if self.editing:
            cursor = "|" if (pygame.time.get_ticks() // 450) % 2 == 0 else ""
            val_str = self.edit_text + cursor
        else:
            val_str = self.fmt.format(self.value) + (f" {self.unit}" if self.unit else "")
        val_surf = font_val.render(val_str, True, TEXT_PRIMARY)
        vx = max(self.input_rect.x + 4, self.input_rect.right - val_surf.get_width() - 5)
        vy = self.input_rect.y + (self.input_rect.height - val_surf.get_height()) // 2
        surface.blit(val_surf, (vx, vy))

        pygame.draw.rect(surface, PANEL_ALT, self.track_rect, border_radius=3)
        ratio = self._val_to_ratio(self.value)
        fill_w = max(0, int(ratio * self.track_rect.width))
        if fill_w > 0:
            fill_rect = pygame.Rect(self.track_rect.x, self.track_rect.y, fill_w, self.track_rect.height)
            pygame.draw.rect(surface, self.accent, fill_rect, border_radius=3)

        knob_x = self.track_rect.x + fill_w
        knob_y = self.track_rect.centery
        radius = 8 if (self.dragging or self.hovered) else 6
        pygame.draw.circle(surface, TEXT_PRIMARY, (knob_x, knob_y), radius)
        pygame.draw.circle(surface, self.accent, (knob_x, knob_y), radius, width=2)


class SimulatorUI:
    """Gerenciador completo de layout, abas, gráficos e eventos em Pygame."""

    def __init__(self, params: PhysicsParams, width: int = 1440, height: int = 860) -> None:
        pygame.init()
        pygame.display.set_caption("Simulador Pêndulo Invertido RL — Sensor Ângulo 0,1° & Auto Mutação (CUDA)")

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.font_title = self._make_font(15, bold=True)
        self.font_med = self._make_font(13, bold=True)
        self.font_small = self._make_font(12, bold=False)
        self.font_mono = self._make_font(12, bold=True, mono=True)

        self.tip_trail: Deque[Tuple[float, float]] = deque(maxlen=50)
        self.swing_trail: Deque[Tuple[float, float]] = deque(maxlen=50)
        self.toast_text = ""
        self.toast_color = ACCENT_CYAN
        self.toast_timer = 0.0

        rng = np.random.default_rng(42)
        self.wind_particles = [
            [float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.12, 0.90)), float(rng.uniform(0.6, 1.4))]
            for _ in range(38)
        ]

        self.mouse_dragging_cart = False
        self.mouse_drag_pane = "stab"
        self.mouse_target_rail_s: Optional[float] = None
        self.last_mouse_pos: Optional[Tuple[int, int]] = None

        self.active_tab = "control"  # "control", "auto_mutate", "reward"
        self.tab_buttons: Dict[str, Button] = {
            "tab_control": Button("tab_control", "Controle", PANEL_ALT, ACCENT_CYAN),
            "tab_mutate": Button("tab_mutate", "Auto Mut.", PANEL_ALT, ACCENT_AMBER),
            "tab_reward": Button("tab_reward", "Recomp.", PANEL_ALT, ACCENT_GREEN),
            "tab_meta": Button("tab_meta", "Meta IA", PANEL_ALT, ACCENT_VIOLET),
        }

        # Widgets da Aba Controle
        self.checkboxes: Dict[str, Checkbox] = {
            "ai_control": Checkbox("ai_control", "Ativar Controle por IA", checked=False, accent=ACCENT_GREEN),
            "gpu_training": Checkbox("gpu_training", "Treinar IA na GPU (Turbo CUDA)", checked=True, accent=ACCENT_CYAN),
            "auto_mutate": Checkbox("auto_mutate", "Mutação Automática Ativa", checked=params.auto_mutate, accent=ACCENT_AMBER),
        }

        self.buttons: Dict[str, Button] = {
            "set_down": Button("set_down", "Colocar Pêndulo para Baixo", PANEL_ALT, ACCENT_AMBER),
            "set_up": Button("set_up", "Colocar no Topo (0)", PANEL_ALT, ACCENT_CYAN),
            "perturb": Button("perturb", "Perturbar (Impulso)", PANEL_ALT, ACCENT_ROSE),
            "save_model": Button("save_model", "Salvar Modelo", (23, 42, 58), ACCENT_CYAN),
            "load_model": Button("load_model", "Carregar Modelo", (23, 42, 58), ACCENT_GREEN),
            "reset_training": Button("reset_training", "Resetar / Limpar Treinamento", (58, 25, 34), ACCENT_ROSE),
        }

        # Sliders apenas para Base e Vento
        self.sliders: Dict[str, Slider] = {
            "table_angle_deg": Slider("table_angle_deg", "Inclinação Base (α)", -45.0, 45.0, 0.0, "{:+.1f}", "°", accent=ACCENT_AMBER),
            "wind_speed": Slider("wind_speed", "Velocidade Vento", -15.0, 15.0, params.wind_speed, "{:+.1f}", "m/s", accent=ACCENT_CYAN),
        }

        # Campos puramente digitáveis (sem barra) na aba Controle
        self.number_fields: Dict[str, NumberField] = {
            "sim_speed": NumberField("sim_speed", "Vel. Simulação (s/s)", 0.05, 50.0, params.sim_speed, "{:.2f}", "x", ACCENT_GREEN),
            "swing_hold_target": NumberField("swing_hold_target", "Estab. Subida (s)", 0.2, 30.0, params.swing_hold_target, "{:.1f}", "s", ACCENT_CYAN),
            "pendulum_mass": NumberField("pendulum_mass", "Peso Pêndulo (m_p)", 0.05, 3.00, params.pendulum_mass, "{:.2f}", "kg", ACCENT_CYAN),
            "cart_mass": NumberField("cart_mass", "Peso Carrinho (m_c)", 0.20, 8.00, params.cart_mass, "{:.2f}", "kg", ACCENT_CYAN),
            "pendulum_length": NumberField("pendulum_length", "Comprimento L (m)", 0.20, 2.00, params.pendulum_length, "{:.2f}", "m", ACCENT_GREEN),
            "mutation_interval": NumberField("mutation_interval", "Tempo Mutação (s)", 0.2, 300.0, params.mutation_interval, "{:.1f}", "s", ACCENT_AMBER),
            "max_force": NumberField("max_force", "Força Motor (N)", 5.0, 100.0, params.max_force, "{:.1f}", "N", ACCENT_GREEN),
            "learning_rate": NumberField("learning_rate", "Taxa Aprend. (lr)", 1e-5, 3e-3, 3e-4, "{:.1e}", "", ACCENT_VIOLET),
        }

        # Widgets da Aba Auto Mutação
        self.mutate_checkboxes: Dict[str, Checkbox] = {
            "auto_mutate_tab": Checkbox("auto_mutate", "Ativar Auto Mutação", checked=params.auto_mutate, accent=ACCENT_AMBER),
            "mutate_table": Checkbox("mutate_table", "Inclinação da Base (°)", checked=params.mutate_table, accent=ACCENT_AMBER),
            "mutate_wind": Checkbox("mutate_wind", "Velocidade do Vento (m/s)", checked=params.mutate_wind, accent=ACCENT_CYAN),
            "mutate_pend_mass": Checkbox("mutate_pend_mass", "Peso do Pêndulo (kg)", checked=params.mutate_pend_mass, accent=ACCENT_CYAN),
            "mutate_cart_mass": Checkbox("mutate_cart_mass", "Peso do Carrinho (kg)", checked=params.mutate_cart_mass, accent=ACCENT_CYAN),
            "mutate_length": Checkbox("mutate_length", "Comprimento L (m)", checked=params.mutate_length, accent=ACCENT_GREEN),
        }
        self.mutation_mode = getattr(params, "mutation_mode", "fixed")
        self.mutation_mode_buttons: Dict[str, Button] = {
            "constant": Button("mode_constant", "Mutação Constante", PANEL_ALT, ACCENT_AMBER),
            "fixed": Button("mode_fixed", "Mutação Fixa", PANEL_ALT, ACCENT_CYAN),
        }

        self.mutate_fields: Dict[str, NumberField] = {
            "mutation_interval_tab": NumberField("mutation_interval", "Tempo Mutação (s)", 0.2, 300.0, params.mutation_interval, "{:.1f}", "s", ACCENT_AMBER),
            "table_min_deg": NumberField("table_min_deg", "Mín", -45.0, 45.0, params.table_min_deg, "{:.1f}", "", ACCENT_AMBER, compact_inline=True),
            "table_max_deg": NumberField("table_max_deg", "Máx", -45.0, 45.0, params.table_max_deg, "{:.1f}", "", ACCENT_AMBER, compact_inline=True),
            "wind_min": NumberField("wind_min", "Mín", -15.0, 15.0, params.wind_min, "{:.1f}", "", ACCENT_CYAN, compact_inline=True),
            "wind_max": NumberField("wind_max", "Máx", -15.0, 15.0, params.wind_max, "{:.1f}", "", ACCENT_CYAN, compact_inline=True),
            "pend_mass_min": NumberField("pend_mass_min", "Mín", 0.05, 3.00, params.pend_mass_min, "{:.2f}", "", ACCENT_CYAN, compact_inline=True),
            "pend_mass_max": NumberField("pend_mass_max", "Máx", 0.05, 3.00, params.pend_mass_max, "{:.2f}", "", ACCENT_CYAN, compact_inline=True),
            "cart_mass_min": NumberField("cart_mass_min", "Mín", 0.20, 8.00, params.cart_mass_min, "{:.2f}", "", ACCENT_CYAN, compact_inline=True),
            "cart_mass_max": NumberField("cart_mass_max", "Máx", 0.20, 8.00, params.cart_mass_max, "{:.2f}", "", ACCENT_CYAN, compact_inline=True),
            "length_min": NumberField("length_min", "Mín", 0.20, 2.00, params.length_min, "{:.2f}", "", ACCENT_GREEN, compact_inline=True),
            "length_max": NumberField("length_max", "Máx", 0.20, 2.00, params.length_max, "{:.2f}", "", ACCENT_GREEN, compact_inline=True),
        }

        self.mutate_rows = [
            ("mutate_table", "table_min_deg", "table_max_deg"),
            ("mutate_wind", "wind_min", "wind_max"),
            ("mutate_pend_mass", "pend_mass_min", "pend_mass_max"),
            ("mutate_cart_mass", "cart_mass_min", "cart_mass_max"),
            ("mutate_length", "length_min", "length_max"),
        ]

        # Widgets da Aba Recompensa (Tabela de 10 atividades +/- com ativação individual)
        short_reward_labels = {
            "rew_center_x": "Próximo do x=0",
            "rew_top": "Pêndulo no topo",
            "rew_stability": "Estabilidade",
            "rew_x_dist": "Penal. dist. x=0",
            "rew_endstop": "Fim de curso",
            "rew_up_time_acc": "Acum. tempo p/ cima",
            "rew_fall": "Queda do pêndulo",
            "rew_rise": "Subida do pêndulo",
            "rew_bottom": "Pêndulo para baixo",
            "rew_down_time_acc": "Acum. tempo p/ baixo",
            "rew_swingup": "Swing up",
            "rew_smooth": "Movimento brusco",
            "rew_full_rotation": "Rotação completa",
        }
        self.reward_rows: List[Tuple[str, str, str]] = [
            (w_key, a_key, short_reward_labels.get(w_key, label))
            for w_key, a_key, label, _ in REWARD_TABLE_SPECS
        ]
        self.reward_checkboxes: Dict[str, Checkbox] = {
            a_key: Checkbox(
                a_key,
                short_lbl,
                checked=bool(getattr(params, a_key)),
                accent=ACCENT_GREEN if float(getattr(params, w_key)) >= 0 else ACCENT_ROSE,
            )
            for w_key, a_key, short_lbl in self.reward_rows
        }
        self.reward_fields: Dict[str, NumberField] = {
            w_key: NumberField(
                w_key,
                short_lbl,
                -500.0,
                500.0,
                float(getattr(params, w_key)),
                "{:+.2f}",
                "pts",
                ACCENT_GREEN if float(getattr(params, w_key)) >= 0 else ACCENT_ROSE,
                compact_inline=True,
            )
            for w_key, _, short_lbl in self.reward_rows
        }

        self.sim_rect = pygame.Rect(0, 0, 100, 100)
        self.charts_rect = pygame.Rect(0, 0, 100, 100)
        self.sidebar_rect = pygame.Rect(0, 0, 100, 100)
        self.chart_rects: List[pygame.Rect] = []
        self.status_box_rect = pygame.Rect(0, 0, 100, 42)
        self.reward_live_box_rect = pygame.Rect(0, 0, 100, 44)
        self.reward_header_y = 0
        self.reward_pts_positions: Dict[str, Tuple[int, int]] = {}

        # Widgets da Aba Meta IA
        self.meta_checkbox = Checkbox("meta_enabled", "Ativar Meta IA (otimiza 2 tasks)", checked=False, accent=ACCENT_VIOLET)
        self.flicks_checkbox = Checkbox("flicks_enabled", "Ativar Petelecos (perturbações)", checked=True, accent=ACCENT_ROSE)
        self.meta_fields: Dict[str, NumberField] = {
            "meta_interval_s": NumberField("meta_interval_s", "Intervalo otimização (s)", 5.0, 7200.0, 300.0, "{:.0f}", "s", ACCENT_VIOLET),
            "meta_flick_min_s": NumberField("meta_flick_min_s", "Peteleco: tempo mín (s)", 0.5, 300.0, 4.0, "{:.1f}", "s", ACCENT_ROSE),
            "meta_flick_max_s": NumberField("meta_flick_max_s", "Peteleco: tempo máx (s)", 0.5, 600.0, 12.0, "{:.1f}", "s", ACCENT_ROSE),
            "meta_flick_force_max": NumberField("meta_flick_force_max", "Peteleco: força máx", 0.5, 25.0, 2.5, "{:.1f}", "rad/s", ACCENT_ROSE),
        }
        self.meta_status_rect = pygame.Rect(0, 0, 100, 150)
        self.mutate_header_y = 0
        self._compute_layout()

    @staticmethod
    def _make_font(size: int, bold: bool = False, mono: bool = False) -> pygame.font.Font:
        names = "consolas,cascadiamono,dejavusansmono" if mono else "segoeui,arial,dejavusans"
        try:
            return pygame.font.SysFont(names, size, bold=bold)
        except Exception:
            return pygame.font.Font(None, size)

    def is_editing_text(self) -> bool:
        return (
            any(sl.editing for sl in self.sliders.values())
            or any(nf.editing for nf in self.number_fields.values())
            or any(mf.editing for mf in self.mutate_fields.values())
            or any(rf.editing for rf in self.reward_fields.values())
            or any(mf.editing for mf in self.meta_fields.values())
        )

    def show_toast(self, message: str, color: Tuple[int, int, int] = ACCENT_CYAN, duration: float = 2.6) -> None:
        self.toast_text = message
        self.toast_color = color
        self.toast_timer = duration

    def clear_trail(self) -> None:
        self.tip_trail.clear()
        self.swing_trail.clear()

    def _compute_layout(self) -> None:
        pad = 12
        sidebar_w = min(380, max(335, int(self.width * 0.27)))
        main_w = self.width - sidebar_w - 3 * pad

        sim_h = max(320, int((self.height - 3 * pad) * 0.59))
        charts_h = self.height - sim_h - 3 * pad

        self.sim_rect = pygame.Rect(pad, pad, main_w, sim_h)
        self.sim_top_bar = pygame.Rect(pad, pad, main_w, 34)
        pane_y = pad + 40
        pane_h = max(260, sim_h - 40)
        half_sim_w = (main_w - 10) // 2
        self.sim_rect_stab = pygame.Rect(pad, pane_y, half_sim_w, pane_h)
        self.sim_rect_swing = pygame.Rect(pad + half_sim_w + 10, pane_y, main_w - half_sim_w - 10, pane_h)
        self.charts_rect = pygame.Rect(pad, pad * 2 + sim_h, main_w, charts_h)
        self.sidebar_rect = pygame.Rect(pad * 2 + main_w, pad, sidebar_w, self.height - 2 * pad)

        cw = (self.charts_rect.width - pad) // 2
        ch = (self.charts_rect.height - pad) // 2
        self.chart_rects = [
            pygame.Rect(self.charts_rect.x, self.charts_rect.y, cw, ch),
            pygame.Rect(self.charts_rect.x + cw + pad, self.charts_rect.y, self.charts_rect.width - cw - pad, ch),
            pygame.Rect(self.charts_rect.x, self.charts_rect.y + ch + pad, self.charts_rect.width, self.charts_rect.height - ch - pad),
        ]

        sx = self.sidebar_rect.x + 14
        sw = self.sidebar_rect.width - 28
        half_w = (sw - 8) // 2

        # Abas no topo da barra lateral (3 abas)
        tab_y = self.sidebar_rect.y + 38
        third_w = (sw - 18) // 4
        self.tab_buttons["tab_control"].rect = pygame.Rect(sx, tab_y, third_w, 28)
        self.tab_buttons["tab_mutate"].rect = pygame.Rect(sx + third_w + 6, tab_y, third_w, 28)
        self.tab_buttons["tab_reward"].rect = pygame.Rect(sx + (third_w + 6) * 2, tab_y, third_w, 28)
        self.tab_buttons["tab_meta"].rect = pygame.Rect(sx + (third_w + 6) * 3, tab_y, sw - 3 * (third_w + 6), 28)

        # Layout Aba Meta IA
        my2 = tab_y + 38
        self.meta_checkbox.rect = pygame.Rect(sx, my2, sw, 24)
        my2 += 26
        self.flicks_checkbox.rect = pygame.Rect(sx, my2, sw, 24)
        my2 += 28
        self.meta_status_rect = pygame.Rect(sx, my2, sw, 174)
        my2 += 182
        for mf in self.meta_fields.values():
            mf.update_layout(sx, my2, sw, height=26, box_w=92)
            my2 += 30

        # Layout Aba Controle
        cy = tab_y + 36
        for key in ("ai_control", "gpu_training", "auto_mutate"):
            self.checkboxes[key].rect = pygame.Rect(sx, cy, sw, 22)
            cy += 24
        cy += 4

        self.buttons["set_down"].rect = pygame.Rect(sx, cy, sw, 28)
        cy += 32
        self.buttons["set_up"].rect = pygame.Rect(sx, cy, half_w, 26)
        self.buttons["perturb"].rect = pygame.Rect(sx + half_w + 8, cy, half_w, 26)
        cy += 31
        self.buttons["save_model"].rect = pygame.Rect(sx, cy, half_w, 26)
        self.buttons["load_model"].rect = pygame.Rect(sx + half_w + 8, cy, half_w, 26)
        cy += 31
        self.buttons["reset_training"].rect = pygame.Rect(sx, cy, sw, 26)
        cy += 32

        self.status_box_rect = pygame.Rect(sx, cy, sw, 40)
        cy += 46

        for slider in self.sliders.values():
            slider.update_layout(sx, cy, sw)
            cy += 48

        cy += 4
        avail_nf = max(175, self.sidebar_rect.bottom - cy - 6)
        nf_step = min(28, avail_nf // len(self.number_fields))
        for nf in self.number_fields.values():
            nf.update_layout(sx, cy, sw, height=max(22, nf_step - 3), box_w=96)
            cy += nf_step

        # Layout Aba Auto Mutação
        my = tab_y + 38
        self.mutate_checkboxes["auto_mutate_tab"].rect = pygame.Rect(sx, my, sw, 24)
        my += 26
        self.mutation_mode_buttons["constant"].rect = pygame.Rect(sx, my, half_w, 24)
        self.mutation_mode_buttons["fixed"].rect = pygame.Rect(sx + half_w + 8, my, half_w, 24)
        my += 29
        self.mutate_fields["mutation_interval_tab"].update_layout(sx, my, sw, height=25, box_w=96)
        my += 32
        self.mutate_header_y = my
        my += 22

        col_w = (sw - 12) // 2
        for cb_key, min_key, max_key in self.mutate_rows:
            self.mutate_checkboxes[cb_key].rect = pygame.Rect(sx, my, sw, 22)
            my += 24
            self.mutate_fields[min_key].update_layout(sx + 18, my, col_w - 18, height=23)
            self.mutate_fields[max_key].update_layout(sx + col_w + 12, my, col_w - 4, height=23)
            my += 32

        # Layout Aba Recompensa (10 linhas compactas)
        ry = tab_y + 38
        self.reward_live_box_rect = pygame.Rect(sx, ry, sw, 44)
        self.reward_header_y = ry + 50
        row_y = self.reward_header_y + 18
        avail_r = max(280, self.sidebar_rect.bottom - row_y - 8)
        row_h = min(44, avail_r // len(self.reward_rows))
        self.reward_pts_positions = {}
        in_w = 78
        pts_w = 54
        cb_w = max(120, sw - in_w - pts_w - 10)
        for w_key, a_key, _ in self.reward_rows:
            self.reward_checkboxes[a_key].rect = pygame.Rect(sx, row_y + 2, cb_w, max(20, row_h - 6))
            self.reward_pts_positions[w_key] = (sx + cb_w + 4, row_y + (row_h - 14) // 2)
            self.reward_fields[w_key].update_layout(sx + sw - in_w, row_y + 2, in_w, height=min(26, max(22, row_h - 6)))
            row_y += row_h

    def _sim_geometry(self, params: PhysicsParams, rect: Optional[pygame.Rect] = None) -> Tuple[int, int, float]:
        r = rect if rect is not None else self.sim_rect_stab
        cx = r.centerx
        hud_top = 34
        usable_h = max(130, r.height - hud_top - 36)
        rail_y = r.y + hud_top + int(usable_h * 0.54)

        px_x = (r.width - 70) / max(2.0 * params.rail_limit, 0.5)
        max_vert_reach = params.rail_limit * 0.71 + params.pendulum_length
        px_y = (usable_h * 0.45) / max(max_vert_reach * 0.72, 0.85)
        px_per_m = max(26.0, min(px_x, px_y, 130.0))
        return cx, rail_y, px_per_m

    @staticmethod
    def _rail_to_screen(cx: int, rail_y: int, px_per_m: float, alpha: float, s: float, n: float = 0.0) -> Tuple[int, int]:
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        wx = s * cos_a - n * sin_a
        wy = s * sin_a + n * cos_a
        return int(cx + wx * px_per_m), int(rail_y - wy * px_per_m)

    def screen_to_rail_s(
        self, sx: int, sy: int, physics: CartPolePhysics, rect: Optional[pygame.Rect] = None
    ) -> float:
        cx, rail_y, px_per_m = self._sim_geometry(physics.params, rect=rect)
        wx = (sx - cx) / px_per_m
        wy = (rail_y - sy) / px_per_m
        return wx * math.cos(physics.table_angle) + wy * math.sin(physics.table_angle)

    def handle_event(
        self,
        event: pygame.event.Event,
        physics: CartPolePhysics,
        swing_physics: Optional[CartPolePhysics] = None,
    ) -> Dict[str, object]:
        actions: Dict[str, object] = {}

        if event.type == pygame.QUIT:
            actions["quit"] = True
            return actions

        if event.type == pygame.VIDEORESIZE:
            self.width = max(1180, event.w)
            self.height = max(740, event.h)
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
            self._compute_layout()
            return actions

        # Processa inputs numéricos / sliders visíveis na aba ativa
        changed_params: List[Tuple[str, float]] = []
        captured = False

        if self.active_tab == "control":
            active_sliders = list(self.sliders.values())
            active_fields = list(self.number_fields.values())
        elif self.active_tab == "auto_mutate":
            active_sliders = []
            active_fields = list(self.mutate_fields.values())
        elif self.active_tab == "meta":
            active_sliders = []
            active_fields = list(self.meta_fields.values())
        else:
            active_sliders = []
            active_fields = list(self.reward_fields.values())

        for w in active_sliders + active_fields:
            was_ed = w.editing
            if w.handle_event(event):
                changed_params.append((w.key, w.value))
            if getattr(w, "dragging", False) or (w.editing and not was_ed):
                captured = True

        if changed_params:
            actions["params"] = changed_params
        if captured or self.is_editing_text():
            return actions

        # Troca de abas
        if self.tab_buttons["tab_control"].handle_event(event):
            self.active_tab = "control"
            return actions
        if self.tab_buttons["tab_mutate"].handle_event(event):
            self.active_tab = "auto_mutate"
            return actions
        if self.tab_buttons["tab_reward"].handle_event(event):
            self.active_tab = "reward"
            return actions
        if self.tab_buttons["tab_meta"].handle_event(event):
            self.active_tab = "meta"
            return actions

        if self.active_tab == "control":
            for key, cb in self.checkboxes.items():
                if cb.handle_event(event):
                    actions["checkbox"] = (key, cb.checked)
                    return actions
            for key, btn in self.buttons.items():
                if btn.handle_event(event):
                    actions["button"] = key
                    return actions
        elif self.active_tab == "auto_mutate":
            for m_key, m_btn in self.mutation_mode_buttons.items():
                if m_btn.handle_event(event):
                    self.mutation_mode = m_key
                    actions["mutation_mode"] = m_key
                    return actions
            for key, cb in self.mutate_checkboxes.items():
                if cb.handle_event(event):
                    actions["checkbox"] = (cb.key, cb.checked)
                    return actions
        elif self.active_tab == "meta":
            if self.meta_checkbox.handle_event(event):
                actions["checkbox"] = (self.meta_checkbox.key, self.meta_checkbox.checked)
                return actions
            if self.flicks_checkbox.handle_event(event):
                actions["checkbox"] = (self.flicks_checkbox.key, self.flicks_checkbox.checked)
                return actions
        elif self.active_tab == "reward":
            for key, cb in self.reward_checkboxes.items():
                if cb.handle_event(event):
                    actions["checkbox"] = (cb.key, cb.checked)
                    return actions

        # Interação direta na simulação (ambos os painéis: Esquerdo e Direito)
        if event.type == pygame.MOUSEBUTTONDOWN and (
            self.sim_rect_stab.collidepoint(event.pos) or self.sim_rect_swing.collidepoint(event.pos)
        ):
            is_swing = self.sim_rect_swing.collidepoint(event.pos)
            pane_rect = self.sim_rect_swing if is_swing else self.sim_rect_stab
            target_ph = swing_physics if (is_swing and swing_physics is not None) else physics
            self.mouse_drag_pane = "swing" if is_swing else "stab"

            cx, rail_y, px_per_m = self._sim_geometry(target_ph.params, pane_rect)
            tip_wx, tip_wy = target_ph.get_tip_position()
            tip_sx = cx + tip_wx * px_per_m
            tip_sy = rail_y - tip_wy * px_per_m

            if event.button == 1:
                dist_tip = math.hypot(event.pos[0] - tip_sx, event.pos[1] - tip_sy)
                if dist_tip < 32:
                    self.last_mouse_pos = event.pos
                else:
                    self.mouse_dragging_cart = True
                    self.mouse_target_rail_s = self.screen_to_rail_s(
                        event.pos[0], event.pos[1], target_ph, rect=pane_rect
                    )
            elif event.button == 3:
                push_dir = 1.0 if event.pos[0] >= tip_sx else -1.0
                nudge_key = "nudge_swing" if is_swing else "nudge"
                actions[nudge_key] = (push_dir * 3.8, -push_dir * 0.4)

        elif event.type == pygame.MOUSEMOTION:
            is_swing = self.mouse_drag_pane == "swing"
            pane_rect = self.sim_rect_swing if is_swing else self.sim_rect_stab
            target_ph = swing_physics if (is_swing and swing_physics is not None) else physics
            if self.mouse_dragging_cart:
                self.mouse_target_rail_s = self.screen_to_rail_s(
                    event.pos[0], event.pos[1], target_ph, rect=pane_rect
                )
            elif self.last_mouse_pos is not None:
                dx = event.pos[0] - self.last_mouse_pos[0]
                self.last_mouse_pos = event.pos
                nudge_key = "nudge_swing" if is_swing else "nudge"
                actions[nudge_key] = (dx * 0.14, 0.0)

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.mouse_dragging_cart = False
            self.mouse_target_rail_s = None
            self.last_mouse_pos = None

        return actions

    def render(
        self,
        physics: CartPolePhysics,
        agent: RLAgent,
        telemetry: TelemetryData,
        dt_frame: float,
        swing_physics: Optional[CartPolePhysics] = None,
    ) -> None:
        self.screen.fill(BG_DARK)
        self._draw_simulation_canvas(physics, swing_physics, telemetry, dt_frame)
        self._draw_telemetry_charts(physics, agent, telemetry)
        self._draw_sidebar(telemetry)
        pygame.display.flip()

    def _draw_wind_field(self, r: pygame.Rect, wind_speed: float, dt_frame: float) -> None:
        abs_w = abs(wind_speed)
        if abs_w < 0.25:
            return
        alpha_norm = min(1.0, abs_w / 15.0)
        col = (int(40 + 30 * alpha_norm), int(120 + 80 * alpha_norm), int(180 + 70 * alpha_norm))
        usable_w = max(1, r.width - 20)
        usable_h = max(1, r.height - 60)

        for pt in self.wind_particles:
            pt[0] = (pt[0] + (wind_speed / 24.0) * dt_frame * pt[2]) % 1.0
            px = r.x + 10 + int(pt[0] * usable_w)
            py = r.y + 48 + int(pt[1] * usable_h)
            streak = int((8 + 28 * alpha_norm) * pt[2])
            dx = streak if wind_speed > 0 else -streak
            pygame.draw.line(self.screen, col, (px, py), (px + dx, py), 1)

    def _draw_simulation_canvas(
        self,
        physics: CartPolePhysics,
        swing_physics: Optional[CartPolePhysics],
        telemetry: TelemetryData,
        dt_frame: float,
    ) -> None:
        # Barra superior unificada (GPU, FPS, Velocidade de Treino)
        tb = self.sim_top_bar
        pygame.draw.rect(self.screen, PANEL_ALT, tb, border_radius=8)
        pygame.draw.rect(self.screen, BORDER_COLOR, tb, width=1, border_radius=8)

        dev = telemetry.device_info
        is_cuda = bool(dev.get("is_cuda", False))
        dev_color = ACCENT_GREEN if is_cuda else ACCENT_AMBER
        if is_cuda:
            dev_txt = f"CUDA: {dev.get('name', 'GPU')} | VRAM: {dev.get('alloc_mb', 0.0):.0f}/{dev.get('total_mb', 0.0):.0f} MB"
        else:
            dev_txt = f"MODO: {dev.get('name', 'CPU')}"

        pygame.draw.circle(self.screen, dev_color, (tb.x + 16, tb.centery), 5)
        self.screen.blit(self.font_med.render(dev_txt, True, dev_color), (tb.x + 28, tb.y + 8))

        p = physics.params
        wind_dir = "→ DIR" if p.wind_speed > 0.2 else ("← ESQ" if p.wind_speed < -0.2 else "CALMO")
        perf_txt = (
            f"FPS: {telemetry.fps:4.0f} | Sim: {p.sim_speed:.2f}x | "
            f"Treino GPU: {telemetry.steps_per_sec:,.0f} steps/s | "
            f"Vento: {p.wind_speed:+.1f} m/s ({wind_dir})"
        )
        perf_surf = self.font_mono.render(perf_txt, True, TEXT_PRIMARY)
        self.screen.blit(perf_surf, (tb.right - perf_surf.get_width() - 12, tb.y + 9))

        self._draw_pendulum_pane(
            r=self.sim_rect_stab,
            physics=physics,
            telemetry=telemetry,
            dt_frame=dt_frame,
            trail=self.tip_trail,
            title_badge="TREINO 1: ESTABILIDADE (CIMA)",
            title_color=ACCENT_GREEN if telemetry.ai_active else ACCENT_AMBER,
            right_badge=f"θ: {format_angle_deg_br(physics.angle_deg)} | Estável: {physics.stable_time:4.1f}s (Rec: {physics.best_stable_time:4.1f}s)",
            footer_hint="Clique Esq: Arrastar | Dir: Peteleco | Espaço: IA",
            is_swing_pane=False,
        )

        if swing_physics is not None:
            swing_ok_rate = (
                (telemetry.swing_successes / telemetry.swing_attempts * 100.0)
                if telemetry.swing_attempts > 0
                else 0.0
            )
            goal_t = max(0.2, float(telemetry.swing_hold_target))
            self._draw_pendulum_pane(
                r=self.sim_rect_swing,
                physics=swing_physics,
                telemetry=telemetry,
                dt_frame=dt_frame,
                trail=self.swing_trail,
                title_badge=f"TREINO 2: SUBIDA ({goal_t:3.1f}s)",
                title_color=ACCENT_CYAN,
                right_badge=f"Hold: {telemetry.swing_hold_time:3.1f}/{goal_t:3.1f}s | Subidas: {telemetry.swing_successes}/{telemetry.swing_attempts} ({swing_ok_rate:2.0f}%)",
                footer_hint=f"Inicia de baixo (θ=180°) -> Sobe e segura {goal_t:3.1f}s",
                is_swing_pane=True,
            )

        if self.toast_timer > 0.0:
            self.toast_timer = max(0.0, self.toast_timer - dt_frame)
            toast_surf = self.font_med.render(self.toast_text, True, TEXT_PRIMARY)
            tw = toast_surf.get_width() + 28
            toast_rect = pygame.Rect(self.sim_rect.centerx - tw // 2, self.sim_top_bar.bottom + 8, tw, 28)
            pygame.draw.rect(self.screen, (20, 28, 38), toast_rect, border_radius=6)
            pygame.draw.rect(self.screen, self.toast_color, toast_rect, width=1, border_radius=6)
            self.screen.blit(toast_surf, (toast_rect.x + 14, toast_rect.y + 5))

    def _draw_pendulum_pane(
        self,
        r: pygame.Rect,
        physics: CartPolePhysics,
        telemetry: TelemetryData,
        dt_frame: float,
        trail: Deque[Tuple[float, float]],
        title_badge: str,
        title_color: Tuple[int, int, int],
        right_badge: str,
        footer_hint: str,
        is_swing_pane: bool,
    ) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, r, border_radius=10)
        border_col = ACCENT_CYAN if is_swing_pane else BORDER_COLOR
        pygame.draw.rect(self.screen, border_col, r, width=1, border_radius=10)

        old_clip = self.screen.get_clip()
        self.screen.set_clip(r.inflate(-2, -2))

        p = physics.params
        cx, rail_y, px_per_m = self._sim_geometry(p, rect=r)
        alpha = physics.table_angle

        grid_step = max(25, int(0.5 * px_per_m))
        gx = cx % grid_step
        while gx < r.right:
            if gx > r.x:
                pygame.draw.line(self.screen, GRID_COLOR, (gx, r.y + 34), (gx, r.bottom), 1)
            gx += grid_step
        gy = rail_y % grid_step
        while gy < r.bottom:
            if gy > r.y + 34:
                pygame.draw.line(self.screen, GRID_COLOR, (r.x, gy), (r.right, gy), 1)
            gy += grid_step

        self._draw_wind_field(r, p.wind_speed, dt_frame)

        hud_rect = pygame.Rect(r.x, r.y, r.width, 32)
        pygame.draw.rect(self.screen, PANEL_ALT, hud_rect)
        pygame.draw.line(self.screen, BORDER_COLOR, (r.x, hud_rect.bottom), (r.right, hud_rect.bottom), 1)

        self.screen.blit(self.font_med.render(title_badge, True, title_color), (r.x + 10, r.y + 7))
        rb_surf = self.font_mono.render(right_badge, True, TEXT_PRIMARY)
        self.screen.blit(rb_surf, (r.right - rb_surf.get_width() - 10, r.y + 8))

        pygame.draw.circle(self.screen, (62, 74, 92), (cx, rail_y), 8)
        pygame.draw.circle(self.screen, ACCENT_AMBER, (cx, rail_y), 3)

        s_left = -p.rail_limit - 0.24
        s_right = p.rail_limit + 0.24
        rail_poly = [
            self._rail_to_screen(cx, rail_y, px_per_m, alpha, s_left, -0.07),
            self._rail_to_screen(cx, rail_y, px_per_m, alpha, s_right, -0.07),
            self._rail_to_screen(cx, rail_y, px_per_m, alpha, s_right, -0.15),
            self._rail_to_screen(cx, rail_y, px_per_m, alpha, s_left, -0.15),
        ]
        pygame.draw.polygon(self.screen, (40, 49, 63), rail_poly)
        p_rl = self._rail_to_screen(cx, rail_y, px_per_m, alpha, s_left, -0.07)
        p_rr = self._rail_to_screen(cx, rail_y, px_per_m, alpha, s_right, -0.07)
        pygame.draw.line(self.screen, RAIL_METAL, p_rl, p_rr, 2)

        m_tick = -math.floor(p.rail_limit * 2.0) * 0.5
        while m_tick <= p.rail_limit + 1e-6:
            t1 = self._rail_to_screen(cx, rail_y, px_per_m, alpha, m_tick, -0.15)
            t2 = self._rail_to_screen(cx, rail_y, px_per_m, alpha, m_tick, -0.23 if abs(m_tick) < 1e-3 else -0.20)
            tick_col = ACCENT_CYAN if abs(m_tick) < 1e-3 else TEXT_SECONDARY
            pygame.draw.line(self.screen, tick_col, t1, t2, 2 if abs(m_tick) < 1e-3 else 1)
            m_tick += 0.5

        pul_l_s = -p.rail_limit - 0.30
        pul_r_s = p.rail_limit + 0.30
        pul_n = -0.12
        pul_r_px = 9
        l_pul_px = self._rail_to_screen(cx, rail_y, px_per_m, alpha, pul_l_s, pul_n)
        r_pul_px = self._rail_to_screen(cx, rail_y, px_per_m, alpha, pul_r_s, pul_n)

        belt_top_l = self._rail_to_screen(cx, rail_y, px_per_m, alpha, pul_l_s, pul_n + 0.08)
        belt_top_r = self._rail_to_screen(cx, rail_y, px_per_m, alpha, pul_r_s, pul_n + 0.08)
        belt_bot_l = self._rail_to_screen(cx, rail_y, px_per_m, alpha, pul_l_s, pul_n - 0.08)
        belt_bot_r = self._rail_to_screen(cx, rail_y, px_per_m, alpha, pul_r_s, pul_n - 0.08)
        pygame.draw.line(self.screen, (72, 84, 102), belt_top_l, belt_top_r, 2)
        pygame.draw.line(self.screen, (72, 84, 102), belt_bot_l, belt_bot_r, 2)

        for px_center in (l_pul_px, r_pul_px):
            pygame.draw.circle(self.screen, (52, 62, 78), px_center, pul_r_px)
            pygame.draw.circle(self.screen, RAIL_METAL, px_center, pul_r_px, width=2)
            for spoke in range(3):
                ang = physics.pulley_angle + alpha + spoke * (2.0 * math.pi / 3.0)
                sx2 = px_center[0] + int((pul_r_px - 3) * math.cos(ang))
                sy2 = px_center[1] + int((pul_r_px - 3) * math.sin(ang))
                pygame.draw.line(self.screen, ACCENT_CYAN, px_center, (sx2, sy2), 2)

        for side, s_edge in ((-1, -p.rail_limit), (1, p.rail_limit)):
            is_hit = physics.bumper_hit > 0.05 and physics.bumper_side == side
            b_col = ACCENT_ROSE if is_hit else (180, 95, 65)
            s_a = s_edge + (0.02 if side > 0 else -0.11)
            s_b = s_edge + (0.11 if side > 0 else -0.02)
            b_poly = [
                self._rail_to_screen(cx, rail_y, px_per_m, alpha, s_a, 0.14),
                self._rail_to_screen(cx, rail_y, px_per_m, alpha, s_b, 0.14),
                self._rail_to_screen(cx, rail_y, px_per_m, alpha, s_b, -0.07),
                self._rail_to_screen(cx, rail_y, px_per_m, alpha, s_a, -0.07),
            ]
            pygame.draw.polygon(self.screen, b_col, b_poly)

        cart_half_s = 0.24
        cart_poly = [
            self._rail_to_screen(cx, rail_y, px_per_m, alpha, physics.x - cart_half_s, 0.14),
            self._rail_to_screen(cx, rail_y, px_per_m, alpha, physics.x + cart_half_s, 0.14),
            self._rail_to_screen(cx, rail_y, px_per_m, alpha, physics.x + cart_half_s, -0.06),
            self._rail_to_screen(cx, rail_y, px_per_m, alpha, physics.x - cart_half_s, -0.06),
        ]
        pygame.draw.polygon(self.screen, CART_BODY, cart_poly)
        cart_border = ACCENT_CYAN if is_swing_pane else (ACCENT_GREEN if telemetry.ai_active else CART_TRIM)
        pygame.draw.polygon(self.screen, cart_border, cart_poly, width=2)

        for w_off in (-0.14, 0.14):
            w_pos = self._rail_to_screen(cx, rail_y, px_per_m, alpha, physics.x + w_off, -0.07)
            pygame.draw.circle(self.screen, (140, 155, 175), w_pos, 4)

        pivot_pos = self._rail_to_screen(cx, rail_y, px_per_m, alpha, physics.x, 0.04)
        if abs(physics.force_motor) > 0.4:
            f_len_s = (physics.force_motor / max(p.max_force, 1.0)) * 0.55
            f_end = self._rail_to_screen(cx, rail_y, px_per_m, alpha, physics.x + f_len_s, 0.04)
            f_col = ACCENT_CYAN if is_swing_pane else (ACCENT_GREEN if telemetry.ai_active else ACCENT_AMBER)
            pygame.draw.line(self.screen, f_col, pivot_pos, f_end, 3)

        ghost_len = int(p.pendulum_length * px_per_m)
        for dy in range(12, ghost_len, 14):
            pygame.draw.line(
                self.screen, (52, 66, 84),
                (pivot_pos[0], pivot_pos[1] - dy),
                (pivot_pos[0], pivot_pos[1] - min(ghost_len, dy + 6)), 1,
            )

        tip_wx, tip_wy = physics.get_tip_position()
        trail.append((tip_wx, tip_wy))
        if len(trail) >= 2:
            trail_pts = list(trail)
            for i in range(1, len(trail_pts)):
                a_norm = i / len(trail_pts)
                if is_swing_pane:
                    col = (int(15 + 30 * a_norm), int(45 + 160 * a_norm), int(80 + 170 * a_norm))
                else:
                    col = (int(20 + 36 * a_norm), int(45 + 144 * a_norm), int(70 + 178 * a_norm))
                p1 = (int(cx + trail_pts[i - 1][0] * px_per_m), int(rail_y - trail_pts[i - 1][1] * px_per_m))
                p2 = (int(cx + trail_pts[i][0] * px_per_m), int(rail_y - trail_pts[i][1] * px_per_m))
                pygame.draw.line(self.screen, col, p1, p2, 2)

        tip_pos = (int(cx + tip_wx * px_per_m), int(rail_y - tip_wy * px_per_m))
        rod_color = ACCENT_GREEN if abs(physics.theta) < 0.22 else TEXT_PRIMARY
        pygame.draw.line(self.screen, rod_color, pivot_pos, tip_pos, 4)
        pygame.draw.circle(self.screen, ACCENT_CYAN, pivot_pos, 5)

        bob_radius = int(max(7, min(16, 6 + p.pendulum_mass * 6.0)))
        pygame.draw.circle(self.screen, ACCENT_CYAN if is_swing_pane else ACCENT_AMBER, tip_pos, bob_radius)
        pygame.draw.circle(self.screen, TEXT_PRIMARY, tip_pos, bob_radius, width=2)

        drag_this_pane = (is_swing_pane and self.mouse_drag_pane == "swing") or (
            not is_swing_pane and self.mouse_drag_pane == "stab"
        )
        if drag_this_pane and self.mouse_dragging_cart and self.mouse_target_rail_s is not None:
            s_clamped = max(-p.rail_limit, min(p.rail_limit, self.mouse_target_rail_s))
            m_pos = self._rail_to_screen(cx, rail_y, px_per_m, alpha, s_clamped, 0.04)
            pygame.draw.circle(self.screen, ACCENT_AMBER, m_pos, 8, width=2)
            pygame.draw.line(self.screen, ACCENT_AMBER, pivot_pos, m_pos, 1)

        if is_swing_pane:
            prog = max(0.0, min(1.0, telemetry.swing_hold_time / max(0.2, float(telemetry.swing_hold_target))))
            bar_w = 140
            bar_rect = pygame.Rect(r.x + 12, r.y + 40, bar_w, 12)
            pygame.draw.rect(self.screen, (28, 36, 48), bar_rect, border_radius=4)
            if prog > 0.01:
                fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, max(4, int(bar_w * prog)), bar_rect.height)
                pygame.draw.rect(self.screen, ACCENT_GREEN if prog >= 0.99 else ACCENT_CYAN, fill_rect, border_radius=4)
            pygame.draw.rect(self.screen, BORDER_COLOR, bar_rect, width=1, border_radius=4)

        stats_line = (
            f"θ: {format_angle_deg_br(physics.angle_deg)} | "
            f"x: {physics.x:+.2f}m | F: {physics.force_motor:+4.1f}N"
        )
        self.screen.blit(self.font_mono.render(stats_line, True, TEXT_PRIMARY), (r.x + 10, r.bottom - 24))
        hint_surf = self.font_small.render(footer_hint, True, TEXT_SECONDARY)
        if hint_surf.get_width() + 260 < r.width:
            self.screen.blit(hint_surf, (r.right - hint_surf.get_width() - 10, r.bottom - 23))

        self.screen.set_clip(old_clip)

    def _draw_telemetry_charts(self, physics: CartPolePhysics, agent: RLAgent, telemetry: TelemetryData) -> None:
        self._draw_dual_chart(
            rect=self.chart_rects[0],
            title=f"Sensor Ângulo ({format_angle_deg_br(physics.angle_deg)}) & ω_q",
            series1=list(telemetry.angles_deg),
            color1=ACCENT_CYAN,
            label1="θ(°)",
            range1=(-180.0, 180.0),
            series2=list(telemetry.ang_vels_q),
            color2=ACCENT_AMBER,
            label2="ω_q",
            range2=(-12.0, 12.0),
        )
        lim = max(0.8, telemetry.rail_limit)
        self._draw_dual_chart(
            rect=self.chart_rects[1],
            title="Posição Eixo X (m) — Alvo Centro (0.0)",
            series1=list(telemetry.positions_x),
            color1=ACCENT_GREEN,
            label1="x(m)",
            range1=(-lim, lim),
            series2=[0.0, 0.0],
            color2=ACCENT_CYAN,
            label2="centro",
            range2=(-lim, lim),
        )
        now_t = time.perf_counter()
        telemetry.prune_reward_30s(now_t=now_t, window_sec=30.0)
        self._draw_reward_30s_chart(
            rect=self.chart_rects[2],
            title="Recompensa Atual — Janela Deslizante 30s (pts/passo)",
            samples=list(telemetry.reward_30s),
            now_t=now_t,
            window_sec=30.0,
        )

    def _draw_dual_chart(
        self,
        rect: pygame.Rect,
        title: str,
        series1: List[float],
        color1: Tuple[int, int, int],
        label1: str,
        range1: Tuple[float, float],
        series2: List[float],
        color2: Tuple[int, int, int],
        label2: str,
        range2: Tuple[float, float],
    ) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(self.screen, BORDER_COLOR, rect, width=1, border_radius=8)

        self.screen.blit(self.font_med.render(title, True, TEXT_PRIMARY), (rect.x + 10, rect.y + 7))

        v1 = f"{series1[-1]:+.2f}" if series1 else "0.00"
        v2 = f"{series2[-1]:+.2f}" if series2 else "0.00"
        leg2 = self.font_small.render(f"{label2}:{v2}", True, color2)
        leg1 = self.font_small.render(f"{label1}:{v1}", True, color1)
        self.screen.blit(leg2, (rect.right - leg2.get_width() - 10, rect.y + 8))
        self.screen.blit(leg1, (rect.right - leg2.get_width() - leg1.get_width() - 16, rect.y + 8))

        plot = pygame.Rect(rect.x + 10, rect.y + 30, rect.width - 20, rect.height - 40)
        pygame.draw.rect(self.screen, BG_DARK, plot, border_radius=5)

        zero_y = plot.y + plot.height // 2
        pygame.draw.line(self.screen, GRID_COLOR, (plot.x, zero_y), (plot.right, zero_y), 1)

        self._plot_line(plot, series1, range1[0], range1[1], color1, width=2)
        self._plot_line(plot, series2, range2[0], range2[1], color2, width=1)

    def _draw_reward_30s_chart(
        self,
        rect: pygame.Rect,
        title: str,
        samples: List[Tuple[float, float]],
        now_t: float,
        window_sec: float = 30.0,
    ) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(self.screen, BORDER_COLOR, rect, width=1, border_radius=8)

        recent = [(t, v) for (t, v) in samples if 0.0 <= (now_t - t) <= window_sec]
        self.screen.blit(self.font_med.render(title, True, TEXT_PRIMARY), (rect.x + 10, rect.y + 7))

        cur_val = recent[-1][1] if recent else 0.0
        cur_txt = f"{cur_val:+.2f} pts" if recent else "--"
        cur_col = ACCENT_GREEN if cur_val >= 0.0 else ACCENT_ROSE
        cur_surf = self.font_mono.render(cur_txt, True, cur_col)
        self.screen.blit(cur_surf, (rect.right - cur_surf.get_width() - 10, rect.y + 8))

        plot = pygame.Rect(rect.x + 10, rect.y + 30, rect.width - 20, rect.height - 40)
        pygame.draw.rect(self.screen, BG_DARK, plot, border_radius=5)

        for mark in (10.0, 20.0):
            gx = plot.x + 2 + int((mark / window_sec) * max(1, plot.width - 4))
            pygame.draw.line(self.screen, GRID_COLOR, (gx, plot.y), (gx, plot.bottom), 1)

        if len(recent) < 2:
            wait_surf = self.font_small.render("Coletando recompensa na janela de 30s...", True, TEXT_SECONDARY)
            self.screen.blit(wait_surf, (plot.centerx - wait_surf.get_width() // 2, plot.centery - 7))
            return

        vals = [v for _, v in recent]
        vmin = min(min(vals), 0.0)
        vmax = max(max(vals), 0.0)
        span = max(vmax - vmin, 0.5)
        vmin -= 0.08 * span
        vmax += 0.08 * span
        full_span = max(vmax - vmin, 1e-6)

        y_zero_norm = max(0.0, min(1.0, (0.0 - vmin) / full_span))
        y_zero = int(plot.bottom - 3 - y_zero_norm * max(1, plot.height - 6))
        pygame.draw.line(self.screen, BORDER_COLOR, (plot.x, y_zero), (plot.right, y_zero), 1)

        if len(recent) > 450:
            step = max(1, len(recent) // 450)
            recent = recent[::step] + ([recent[-1]] if len(recent) % step != 0 else [])

        pts = []
        t_start = now_t - window_sec
        usable_w = max(1, plot.width - 4)
        usable_h = max(1, plot.height - 6)
        for t, v in recent:
            ratio_x = max(0.0, min(1.0, (t - t_start) / window_sec))
            px = plot.x + 2 + ratio_x * usable_w
            norm_y = max(0.0, min(1.0, (v - vmin) / full_span))
            py = plot.bottom - 3 - norm_y * usable_h
            pts.append((px, py))

        pygame.draw.aalines(self.screen, ACCENT_CYAN, False, pts)

        if len(pts) >= 5:
            win = 15
            ma_pts = []
            acc = 0.0
            for i, (px, _) in enumerate(pts):
                v = recent[i][1]
                acc += v
                if i >= win:
                    acc -= recent[i - win][1]
                mv = acc / min(i + 1, win)
                norm_y = max(0.0, min(1.0, (mv - vmin) / full_span))
                py = plot.bottom - 3 - norm_y * usable_h
                ma_pts.append((px, py))
            pygame.draw.lines(self.screen, ACCENT_GREEN, False, ma_pts, width=2)

        lbl_l = self.font_small.render("-30s", True, TEXT_SECONDARY)
        lbl_m = self.font_small.render("-15s", True, TEXT_SECONDARY)
        lbl_r = self.font_small.render("0s", True, TEXT_SECONDARY)
        self.screen.blit(lbl_l, (plot.x + 4, plot.bottom - 16))
        self.screen.blit(lbl_m, (plot.centerx - lbl_m.get_width() // 2, plot.bottom - 16))
        self.screen.blit(lbl_r, (plot.right - lbl_r.get_width() - 4, plot.bottom - 16))

    def _plot_line(
        self,
        plot: pygame.Rect,
        series: List[float],
        vmin: float,
        vmax: float,
        color: Tuple[int, int, int],
        width: int = 2,
    ) -> None:
        n = len(series)
        if n < 2:
            return
        span = max(vmax - vmin, 1e-6)
        pts = []
        for i, val in enumerate(series):
            px = plot.x + 2 + (i / (n - 1)) * max(1, plot.width - 4)
            norm = max(0.0, min(1.0, (val - vmin) / span))
            py = plot.bottom - 3 - norm * max(1, plot.height - 6)
            pts.append((px, py))

        if width == 1:
            pygame.draw.aalines(self.screen, color, False, pts)
        else:
            pygame.draw.lines(self.screen, color, False, pts, width=width)

    def sync_from_params(self, params: PhysicsParams, table_deg: float, lr: float, meta: object) -> None:
        """Sincroniza todos os widgets com as configurações restauradas do disco."""
        self.checkboxes["auto_mutate"].checked = params.auto_mutate
        self.mutate_checkboxes["auto_mutate_tab"].checked = params.auto_mutate
        self.mutation_mode = getattr(params, "mutation_mode", "fixed")
        for cb_key, _, _ in self.mutate_rows:
            self.mutate_checkboxes[cb_key].checked = bool(getattr(params, cb_key))
        for k, nf in self.mutate_fields.items():
            if k == "mutation_interval_tab":
                nf.set_value(params.mutation_interval)
            elif hasattr(params, k):
                nf.set_value(float(getattr(params, k)))
        self.sliders["table_angle_deg"].set_value(table_deg)
        self.sliders["wind_speed"].set_value(params.wind_speed)
        for k, nf in self.number_fields.items():
            if k == "learning_rate":
                nf.set_value(lr)
            elif hasattr(params, k):
                nf.set_value(float(getattr(params, k)))
        for w_key, a_key, _ in self.reward_rows:
            self.reward_checkboxes[a_key].checked = bool(getattr(params, a_key))
            self.reward_fields[w_key].set_value(float(getattr(params, w_key)))
        self.meta_checkbox.checked = bool(getattr(meta, "enabled", False))
        self.flicks_checkbox.checked = bool(getattr(meta, "flicks_enabled", True))
        for k, nf in self.meta_fields.items():
            attr = k.replace("meta_", "", 1)
            if hasattr(meta, attr):
                nf.set_value(float(getattr(meta, attr)))

    def _draw_meta_tab(self, telemetry: TelemetryData) -> None:
        m = telemetry.meta_status or {}
        self.meta_checkbox.checked = bool(m.get("enabled", self.meta_checkbox.checked))
        self.meta_checkbox.draw(self.screen, self.font_med)
        self.flicks_checkbox.checked = bool(m.get("flicks_enabled", self.flicks_checkbox.checked))
        self.flicks_checkbox.draw(self.screen, self.font_med)

        box = self.meta_status_rect
        pygame.draw.rect(self.screen, PANEL_ALT, box, border_radius=6)
        pygame.draw.rect(self.screen, BORDER_COLOR, box, width=1, border_radius=6)

        enabled = bool(m.get("enabled", False))
        frozen = bool(m.get("frozen", False))
        if not enabled:
            state_txt, state_col = "META IA: DESLIGADA", TEXT_SECONDARY
        elif frozen:
            state_txt, state_col = "META IA: ESTABILIDADE PERFEITA", ACCENT_GREEN
        else:
            state_txt, state_col = "META IA: OTIMIZANDO (2 TASKS)", ACCENT_AMBER

        tl = int(max(0.0, float(m.get("time_left", 0.0))))
        time_txt = f"Próxima otimização: {tl // 60:02d}:{tl % 60:02d}" if enabled else "Próxima otimização: --:--"
        lines = [
            (state_txt, state_col),
            (time_txt, TEXT_PRIMARY),
            (f"Rodadas: {int(m.get('rounds', 0))} | σ mutação: {float(m.get('sigma', 0.0)):.3f}", TEXT_PRIMARY),
            (
                f"Estab: {float(m.get('best_stab', 0.0)) * 100:5.1f}% | Subida(2s): {float(m.get('best_swingup', 0.0)) * 100:5.1f}%",
                ACCENT_GREEN,
            ),
            (
                f"Última: est {float(m.get('last_stab', 0.0)) * 100:4.1f}% | sub {float(m.get('last_swingup', 0.0)) * 100:4.1f}%",
                ACCENT_CYAN,
            ),
            (f"Último peteleco: {m.get('last_flick', '—')}", ACCENT_ROSE),
        ]
        ly = box.y + 6
        for txt, col in lines:
            self.screen.blit(self.font_mono.render(txt[:46], True, col), (box.x + 10, ly))
            ly += 17

        msg = str(m.get("last_msg", ""))
        words, cur, wrapped = msg.split(" "), "", []
        for w in words:
            if len(cur) + len(w) + 1 > 44:
                wrapped.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            wrapped.append(cur)
        for row in wrapped[:3]:
            self.screen.blit(self.font_small.render(row, True, TEXT_SECONDARY), (box.x + 10, ly))
            ly += 14

        for mf in self.meta_fields.values():
            mf.draw(self.screen, self.font_small, self.font_mono)

    def _draw_sidebar(self, telemetry: TelemetryData) -> None:
        r = self.sidebar_rect
        pygame.draw.rect(self.screen, PANEL_BG, r, border_radius=10)
        pygame.draw.rect(self.screen, BORDER_COLOR, r, width=1, border_radius=10)

        header = self.font_title.render("PAINEL DE CONTROLE & SIMULAÇÃO", True, TEXT_PRIMARY)
        self.screen.blit(header, (r.x + 14, r.y + 11))

        self.tab_buttons["tab_control"].draw(self.screen, self.font_med, active=(self.active_tab == "control"))
        self.tab_buttons["tab_mutate"].draw(self.screen, self.font_med, active=(self.active_tab == "auto_mutate"))
        self.tab_buttons["tab_reward"].draw(self.screen, self.font_med, active=(self.active_tab == "reward"))
        self.tab_buttons["tab_meta"].draw(self.screen, self.font_med, active=(self.active_tab == "meta"))

        if self.active_tab == "control":
            for cb in self.checkboxes.values():
                cb.draw(self.screen, self.font_med)
            for btn in self.buttons.values():
                btn.draw(self.screen, self.font_small)

            pygame.draw.rect(self.screen, PANEL_ALT, self.status_box_rect, border_radius=6)
            pygame.draw.rect(self.screen, BORDER_COLOR, self.status_box_rect, width=1, border_radius=6)
            g_line = self.font_mono.render(f"Gravidade Fixa: {FIXED_GRAVITY:.2f} m/s² (Constante)", True, ACCENT_AMBER)
            e_line = self.font_mono.render(
                f"Exploração Auto (Varredura): {telemetry.epsilon_auto * 100.0:4.1f}%",
                True,
                ACCENT_VIOLET,
            )
            self.screen.blit(g_line, (self.status_box_rect.x + 10, self.status_box_rect.y + 5))
            self.screen.blit(e_line, (self.status_box_rect.x + 10, self.status_box_rect.y + 21))

            for slider in self.sliders.values():
                slider.draw(self.screen, self.font_small, self.font_mono)

            for nf in self.number_fields.values():
                nf.draw(self.screen, self.font_small, self.font_mono)
        elif self.active_tab == "auto_mutate":
            self.mutate_checkboxes["auto_mutate_tab"].draw(self.screen, self.font_med)
            for m_key, m_btn in self.mutation_mode_buttons.items():
                active = self.mutation_mode == m_key
                m_btn.base_color = (28, 52, 68) if active else PANEL_ALT
                m_btn.draw(self.screen, self.font_small, active=active)
            self.mutate_fields["mutation_interval_tab"].draw(self.screen, self.font_small, self.font_mono)

            sx = r.x + 14
            sw = r.width - 28
            col_w = (sw - 12) // 2
            pygame.draw.line(self.screen, BORDER_COLOR, (sx, self.mutate_header_y - 4), (sx + sw, self.mutate_header_y - 4), 1)
            self.screen.blit(self.font_small.render("Parâmetro", True, TEXT_SECONDARY), (sx, self.mutate_header_y))
            self.screen.blit(self.font_small.render("Mín", True, TEXT_SECONDARY), (sx + 22, self.mutate_header_y))
            self.screen.blit(self.font_small.render("Máx", True, TEXT_SECONDARY), (sx + col_w + 16, self.mutate_header_y))

            for cb_key, min_key, max_key in self.mutate_rows:
                self.mutate_checkboxes[cb_key].draw(self.screen, self.font_med)
                self.mutate_fields[min_key].draw(self.screen, self.font_small, self.font_mono)
                self.mutate_fields[max_key].draw(self.screen, self.font_small, self.font_mono)
        elif self.active_tab == "meta":
            self._draw_meta_tab(telemetry)
        else:
            lb = self.reward_live_box_rect
            pygame.draw.rect(self.screen, PANEL_ALT, lb, border_radius=6)
            pygame.draw.rect(self.screen, BORDER_COLOR, lb, width=1, border_radius=6)
            bd = telemetry.reward_breakdown or {}
            total_pts = float(bd.get("total_pts", 0.0))
            tot_col = ACCENT_GREEN if total_pts >= 0.0 else ACCENT_ROSE
            self.screen.blit(
                self.font_small.render("TABELA DE RECOMPENSAS (ATIVAR / +/-)", True, TEXT_PRIMARY),
                (lb.x + 10, lb.y + 6),
            )
            self.screen.blit(
                self.font_mono.render(f"Total Passo: {total_pts:+8.3f} pts", True, tot_col),
                (lb.x + 10, lb.y + 23),
            )

            sx = r.x + 14
            sw = r.width - 28
            hy = self.reward_header_y
            self.screen.blit(self.font_small.render("Atividade", True, TEXT_SECONDARY), (sx + 2, hy))
            self.screen.blit(self.font_small.render("Pts", True, TEXT_SECONDARY), (sx + sw - 134, hy))
            self.screen.blit(self.font_small.render("Peso (+/-)", True, TEXT_SECONDARY), (sx + sw - 74, hy))
            pygame.draw.line(self.screen, BORDER_COLOR, (sx, hy + 15), (sx + sw, hy + 15), 1)

            for w_key, a_key, _ in self.reward_rows:
                cb = self.reward_checkboxes[a_key]
                nf = self.reward_fields[w_key]
                cb.draw(self.screen, self.font_small)
                nf.draw(self.screen, self.font_small, self.font_mono)
                px, py = self.reward_pts_positions[w_key]
                val_pts = float(bd.get(w_key, 0.0))
                if not cb.checked:
                    pts_col = TEXT_SECONDARY
                    pts_str = "  off "
                else:
                    pts_col = ACCENT_GREEN if val_pts > 1e-6 else (ACCENT_ROSE if val_pts < -1e-6 else TEXT_SECONDARY)
                    pts_str = f"{val_pts:+6.2f}"
                self.screen.blit(self.font_mono.render(pts_str, True, pts_col), (px, py))
