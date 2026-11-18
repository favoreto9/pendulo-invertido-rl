"""
Ponto de entrada principal do Simulador de Pêndulo Invertido em Carrinho com RL (PyTorch/CUDA).

Recursos integrados:
- Velocidade da simulação configurável (segundos simulados por segundo real).
- Aba Controle (sliders apenas para Base e Vento; campos digitáveis para pesos, comprimento, tempo, lr e força).
- Aba Auto Mutação (seleção individual do que randomizar e ranges [Mín, Máx] digitáveis).
- IA recebendo ângulo quantizado (0,1° no formato 123,4°) + inputs de m_p, L e m_c.
- Bônus de centro no eixo X e penalidade severa silenciosa no fim de curso.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Optional

import numpy as np
import pygame
import torch

from meta_ai import MetaAI
from physics import BatchedCartPoleEnv, CartPolePhysics, FIXED_GRAVITY, PhysicsParams
from rl_agent import RLAgent
from storage import latest_memory_path, load_app_config, next_memory_path, save_app_config
from ui import ACCENT_AMBER, ACCENT_CYAN, ACCENT_GREEN, ACCENT_ROSE, SimulatorUI, TelemetryData


APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(APP_DIR, "modelo_pendulo_gpu.pth")


def ask_save_path(default_path: str) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            title="Salvar Modelo RL (.pth)",
            initialdir=os.path.dirname(default_path),
            initialfile=os.path.basename(default_path),
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")],
        )
        root.destroy()
        return path if path else None
    except Exception:
        return default_path


def ask_load_path(default_path: str) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Carregar Modelo RL (.pth)",
            initialdir=os.path.dirname(default_path),
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")],
        )
        root.destroy()
        return path if path else None
    except Exception:
        return default_path if os.path.exists(default_path) else None


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = PhysicsParams(gravity=FIXED_GRAVITY)
    sim_env = CartPolePhysics(params=params)
    swing_sim_env = CartPolePhysics(params=params)
    agent = RLAgent(state_dim=CartPolePhysics.STATE_DIM, device=device)
    meta = MetaAI()
    load_app_config(APP_DIR, params, agent, meta, sim_env, now_t=time.perf_counter())
    swing_sim_env.reset("down")

    batch_envs = 2048 if device.type == "cuda" else 128
    gpu_env = BatchedCartPoleEnv(num_envs=batch_envs, params=params, device=device, max_episode_steps=450, reset_mode="stability")
    gpu_swing_env = BatchedCartPoleEnv(num_envs=batch_envs, params=params, device=device, max_episode_steps=500, reset_mode="swingup_2s")

    train_lock = threading.RLock()

    def sync_all_envs() -> None:
        with train_lock:
            gpu_env.sync_params(params)
            gpu_swing_env.sync_params(params)

    ui = SimulatorUI(params=params)
    ui.sync_from_params(params, math.degrees(sim_env.table_angle), agent.lr, meta)
    telemetry = TelemetryData()
    telemetry.device_info = agent.get_device_info()

    ai_active = ui.checkboxes["ai_control"].checked
    training_active = ui.checkboxes["gpu_training"].checked

    if device.type == "cuda":
        ui.show_toast(f"GPU CUDA pronta: {telemetry.device_info['name']}", ACCENT_GREEN, duration=3.5)
    else:
        ui.show_toast("CUDA indisponível — rodando em CPU Fallback", ACCENT_AMBER, duration=3.5)

    perf_window_start = time.perf_counter()
    perf_window_steps = 0
    steps_per_sec = 0.0
    frame_counter = 0
    sim_time_acc = 0.0
    last_reward = 0.0
    chosen_u = 0.0
    swing_hold_time = 0.0
    swing_elapsed = 0.0
    swing_successes = 0
    swing_attempts = 0
    running = True

    train_state = {
        "running": True,
        "training_active": training_active,
        "steps_window": 0,
        "window_start": time.perf_counter(),
        "steps_per_sec": 0.0,
        "pending_toast": None,
    }

    def train_worker() -> None:
        while train_state["running"]:
            if not train_state["training_active"]:
                time.sleep(0.02)
                continue
            with train_lock:
                for env_task in (gpu_env, gpu_swing_env):
                    states_b = env_task.get_states()
                    act_idx_b, act_u_b = agent.select_actions_batch(states_b)
                    next_states_b, rewards_b, dones_b, completed_ep_rewards = env_task.step(act_u_b)

                    agent.replay_buffer.add_batch(states_b, act_idx_b, rewards_b, next_states_b, dones_b)
                    agent.advance_epsilon(env_task.num_envs)
                    train_state["steps_window"] += env_task.num_envs

                    if completed_ep_rewards.numel() > 0:
                        agent.record_episode_rewards(completed_ep_rewards.tolist())

                meta.on_train_cycle(gpu_env)
                for _ in range(6):
                    agent.train_step()

                now_w = time.perf_counter()
                el_w = now_w - train_state["window_start"]
                if el_w >= 0.35:
                    inst = train_state["steps_window"] / el_w
                    prev = train_state["steps_per_sec"]
                    train_state["steps_per_sec"] = inst if prev == 0.0 else (0.65 * prev + 0.35 * inst)
                    train_state["window_start"] = now_w
                    train_state["steps_window"] = 0

                if meta.maybe_optimize(now_w, agent, params, device):
                    gpu_env.sync_params(params)
                    gpu_swing_env.sync_params(params)
                    train_state["pending_toast"] = meta.last_msg[:90]

            time.sleep(0.0005)

    train_thread = threading.Thread(target=train_worker, daemon=True)
    train_thread.start()

    while running:
        dt_frame = min(0.05, max(1e-4, ui.clock.tick(65) / 1000.0))
        fps = ui.clock.get_fps()
        frame_counter += 1

        # 1. Eventos da interface e teclado
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and not ui.is_editing_text():
                if event.key == pygame.K_SPACE:
                    ai_active = not ai_active
                    ui.checkboxes["ai_control"].checked = ai_active
                    ui.show_toast(
                        "Piloto Automático IA ATIVADO" if ai_active else "Controle Manual ATIVADO",
                        ACCENT_GREEN if ai_active else ACCENT_AMBER,
                    )
                elif event.key == pygame.K_r:
                    sim_env.reset("down")
                    ui.clear_trail()
                    ui.show_toast("Pêndulo posicionado para baixo (θ = 180,0°)", ACCENT_AMBER)

            act = ui.handle_event(event, sim_env, swing_sim_env)
            if act.get("quit"):
                running = False
                break

            if "params" in act:
                for p_key, p_val in act["params"]:
                    val = float(p_val)
                    if p_key == "learning_rate":
                        with train_lock:
                            agent.set_learning_rate(val)
                    elif p_key == "sim_speed":
                        params.sim_speed = max(0.05, min(50.0, val))
                    elif p_key == "swing_hold_target":
                        params.swing_hold_target = max(0.2, min(30.0, val))
                        sync_all_envs()
                    elif p_key == "table_angle_deg":
                        sim_env.set_manual_table_angle_deg(val)
                        sync_all_envs()
                    elif p_key == "wind_speed":
                        sim_env.set_manual_wind(val)
                        sync_all_envs()
                    elif p_key == "mutation_interval":
                        sim_env.set_mutation_interval(val)
                        ui.number_fields["mutation_interval"].set_value(params.mutation_interval)
                        ui.mutate_fields["mutation_interval_tab"].set_value(params.mutation_interval)
                        sync_all_envs()
                    elif p_key == "pendulum_mass":
                        params.pendulum_mass = val
                        sim_env.pend_mass_timer = max(0.2, params.mutation_interval)
                        sync_all_envs()
                    elif p_key == "cart_mass":
                        params.cart_mass = val
                        sim_env.cart_mass_timer = max(0.2, params.mutation_interval)
                        sync_all_envs()
                    elif p_key == "pendulum_length":
                        params.pendulum_length = val
                        sim_env.length_timer = max(0.2, params.mutation_interval)
                        sync_all_envs()
                    elif p_key.startswith("meta_"):
                        meta.set_param(p_key, val)
                    elif hasattr(params, p_key):
                        setattr(params, p_key, val)
                        sync_all_envs()

            if "checkbox" in act:
                key, checked = act["checkbox"]
                if key == "ai_control":
                    ai_active = bool(checked)
                    ui.show_toast(
                        "Piloto Automático IA ATIVADO" if ai_active else "Controle Manual ATIVADO",
                        ACCENT_GREEN if ai_active else ACCENT_AMBER,
                    )
                elif key == "meta_enabled":
                    meta.set_enabled(bool(checked), time.perf_counter())
                    ui.show_toast(
                        "Meta IA LIGADA (otimiza 2 tasks)" if checked else "Meta IA DESLIGADA",
                        ACCENT_CYAN if checked else ACCENT_AMBER,
                    )
                elif key == "flicks_enabled":
                    meta.set_flicks_enabled(bool(checked))
                    ui.show_toast(
                        "Petelecos ATIVADOS" if checked else "Petelecos DESATIVADOS",
                        ACCENT_ROSE if checked else ACCENT_AMBER,
                    )
                elif key == "gpu_training":
                    training_active = bool(checked)
                    train_state["training_active"] = training_active
                    ui.show_toast(
                        "Treino paralelo na GPU ATIVADO" if training_active else "Treino em pausa",
                        ACCENT_CYAN if training_active else ACCENT_AMBER,
                    )
                elif key == "auto_mutate":
                    params.auto_mutate = bool(checked)
                    ui.checkboxes["auto_mutate"].checked = params.auto_mutate
                    ui.mutate_checkboxes["auto_mutate_tab"].checked = params.auto_mutate
                    sync_all_envs()
                    ui.show_toast(
                        "Mutação automática ATIVADA" if params.auto_mutate else "Mutação automática PAUSADA",
                        ACCENT_AMBER,
                    )
                elif hasattr(params, key):
                    setattr(params, key, bool(checked))
                    sync_all_envs()

            elif "mutation_mode" in act:
                params.mutation_mode = str(act["mutation_mode"])
                ui.mutation_mode = params.mutation_mode
                sync_all_envs()
                ui.show_toast(
                    "Mutação Fixa (parada pelo tempo selecionado)"
                    if params.mutation_mode == "fixed"
                    else "Mutação Constante (oscilando)",
                    ACCENT_CYAN if params.mutation_mode == "fixed" else ACCENT_AMBER,
                )

            elif "button" in act:
                btn_key = act["button"]
                if btn_key == "set_down":
                    sim_env.reset("down")
                    ui.clear_trail()
                    ui.show_toast("Pêndulo resetado para baixo (θ = 180,0°)", ACCENT_AMBER)
                elif btn_key == "set_up":
                    sim_env.reset("up")
                    ui.clear_trail()
                    ui.show_toast("Pêndulo posicionado no topo (θ = 0)", ACCENT_CYAN)
                elif btn_key == "perturb":
                    sim_env.apply_perturbation(angular_impulse=4.2, cart_impulse=0.9)
                    ui.show_toast("Impulso de perturbação aplicado!", ACCENT_ROSE)
                elif btn_key == "save_model":
                    save_path = ask_save_path(next_memory_path(APP_DIR))
                    if save_path:
                        with train_lock:
                            agent.save_model(save_path)
                        ui.show_toast(f"Modelo salvo: {os.path.basename(save_path)}", ACCENT_GREEN)
                elif btn_key == "load_model":
                    load_path = ask_load_path(latest_memory_path(APP_DIR) or DEFAULT_MODEL_PATH)
                    if load_path and os.path.exists(load_path):
                        try:
                            with train_lock:
                                agent.load_model(load_path)
                            ui.number_fields["learning_rate"].set_value(agent.lr)
                            ui.show_toast(f"Modelo carregado: {os.path.basename(load_path)}", ACCENT_GREEN)
                        except Exception as exc:
                            ui.show_toast(f"Erro ao carregar modelo: {exc}", ACCENT_ROSE, duration=4.0)
                    elif load_path:
                        ui.show_toast("Arquivo de modelo não encontrado.", ACCENT_ROSE)
                elif btn_key == "reset_training":
                    with train_lock:
                        agent.reset()
                        gpu_env.reset_all()
                        gpu_swing_env.reset_all()
                    sim_env.reset("down")
                    swing_sim_env.reset("down")
                    swing_hold_time = 0.0
                    swing_elapsed = 0.0
                    telemetry.reward_30s.clear()
                    ui.clear_trail()
                    ui.show_toast("Rede neural, replay buffer e varredura reiniciados!", ACCENT_ROSE)

            elif "nudge" in act:
                d_thd, d_xd = act["nudge"]
                sim_env.apply_mouse_nudge(float(d_thd), float(d_xd))
            elif "nudge_swing" in act:
                d_thd, d_xd = act["nudge_swing"]
                swing_sim_env.apply_mouse_nudge(float(d_thd), float(d_xd))

        if not running:
            break

        # 2. Teclado / Mouse
        manual_u = 0.0
        if not ui.is_editing_text():
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                manual_u -= 1.0
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                manual_u += 1.0

        ext_cart_force = 0.0
        ext_swing_force = 0.0
        manual_swing_u = manual_u
        if ui.mouse_dragging_cart and ui.mouse_target_rail_s is not None:
            target_s = float(np.clip(ui.mouse_target_rail_s, -params.rail_limit, params.rail_limit))
            if ui.mouse_drag_pane == "swing":
                if not ai_active and manual_swing_u == 0.0:
                    manual_swing_u = float(np.clip((target_s - swing_sim_env.x) * 3.0 - swing_sim_env.x_dot * 0.45, -1.0, 1.0))
                elif ai_active:
                    ext_swing_force = float(np.clip((target_s - swing_sim_env.x) * 35.0, -40.0, 40.0))
            else:
                if not ai_active and manual_u == 0.0:
                    manual_u = float(np.clip((target_s - sim_env.x) * 3.0 - sim_env.x_dot * 0.45, -1.0, 1.0))
                elif ai_active:
                    ext_cart_force = float(np.clip((target_s - sim_env.x) * 35.0, -40.0, 40.0))

        # 3. Passo simultâneo das duas simulações visuais (Estabilidade + Subida 2s)
        sim_time_acc += dt_frame * params.sim_speed
        sim_steps_run = 0
        while sim_time_acc >= params.dt and sim_steps_run < 16:
            sim_time_acc -= params.dt
            sim_steps_run += 1

            if ai_active:
                _, chosen_u = agent.select_action_single(sim_env.get_state(), deterministic=True)
                _, swing_u = agent.select_action_single(swing_sim_env.get_state(), deterministic=True)
            else:
                chosen_u = manual_u
                swing_u = manual_swing_u

            _, last_reward, _ = sim_env.step(chosen_u, external_cart_force=ext_cart_force)

            swing_sim_env.table_angle = sim_env.table_angle
            swing_sim_env.step(swing_u, external_cart_force=ext_swing_force)
            swing_elapsed += params.dt
            goal_t = max(0.2, float(params.swing_hold_target))
            if abs(swing_sim_env.theta) <= 0.25 and abs(swing_sim_env.theta_dot) <= 1.2:
                swing_hold_time += params.dt
                if swing_hold_time >= goal_t:
                    swing_successes += 1
                    swing_attempts += 1
                    swing_hold_time = 0.0
                    swing_elapsed = 0.0
                    swing_sim_env.reset("down")
                    ui.swing_trail.clear()
            else:
                swing_hold_time = 0.0
                if swing_elapsed >= max(15.0, goal_t + 10.0):
                    swing_attempts += 1
                    swing_elapsed = 0.0
                    swing_sim_env.reset("down")
                    ui.swing_trail.clear()

            now_rew_t = time.perf_counter()
            telemetry.angles_deg.append(sim_env.angle_deg)
            telemetry.ang_vels_q.append(sim_env.omega_q)
            telemetry.positions_x.append(sim_env.x)
            telemetry.record_reward(last_reward, now_t=now_rew_t)

        telemetry.prune_reward_30s(time.perf_counter())
        meta.on_sim_steps(sim_steps_run, params.dt, sim_env)

        if sim_time_acc > params.dt * 2.0:
            sim_time_acc = 0.0

        # Sincroniza campos visuais com os valores mutados
        ui.sliders["table_angle_deg"].set_value(math.degrees(sim_env.table_angle))
        ui.sliders["wind_speed"].set_value(params.wind_speed)
        ui.number_fields["pendulum_mass"].set_value(params.pendulum_mass)
        ui.number_fields["cart_mass"].set_value(params.cart_mass)
        ui.number_fields["pendulum_length"].set_value(params.pendulum_length)
        for rw_key, rw_field in ui.reward_fields.items():
            rw_field.set_value(float(getattr(params, rw_key)))
        for mf_key, mf_field in ui.meta_fields.items():
            mf_field.set_value(float(getattr(meta, mf_key.replace("meta_", "", 1))))

        # 4. Treinamento paralelo ocorre em background thread (train_worker)
        if train_state["pending_toast"]:
            ui.show_toast(str(train_state["pending_toast"]), ACCENT_CYAN, duration=5.0)
            train_state["pending_toast"] = None

        # 5. Telemetria
        now = time.perf_counter()
        if (now - perf_window_start) >= 0.35:
            perf_window_start = now
            telemetry.device_info = agent.get_device_info()

        telemetry.fps = fps if fps > 0 else (1.0 / dt_frame)
        telemetry.steps_per_sec = train_state["steps_per_sec"] if training_active else 0.0
        telemetry.total_env_steps = agent.total_env_steps
        telemetry.total_grad_steps = agent.total_grad_steps
        telemetry.buffer_size = agent.replay_buffer.size
        telemetry.ai_active = ai_active
        telemetry.training_active = training_active
        telemetry.auto_mutate = params.auto_mutate
        telemetry.epsilon_auto = agent.epsilon
        telemetry.current_u = chosen_u
        telemetry.last_reward = last_reward
        telemetry.rail_limit = params.rail_limit
        telemetry.reward_breakdown = sim_env.last_reward_breakdown
        telemetry.meta_status = meta.status(now)
        telemetry.swing_successes = swing_successes
        telemetry.swing_attempts = swing_attempts
        telemetry.swing_hold_time = swing_hold_time
        telemetry.swing_hold_target = params.swing_hold_target

        if frame_counter % 30 == 0 and device.type == "cuda":
            telemetry.device_info = agent.get_device_info()

        ui.render(sim_env, agent, telemetry, dt_frame, swing_physics=swing_sim_env)

    train_state["running"] = False
    train_thread.join(timeout=1.5)
    save_app_config(APP_DIR, params, agent, meta, sim_env)
    pygame.quit()


if __name__ == "__main__":
    main()
