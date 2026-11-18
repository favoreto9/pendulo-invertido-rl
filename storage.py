"""
Persistência de configurações do simulador e salvamento sequencial na pasta Memoria/.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict
from typing import Any, Dict

from physics import FIXED_GRAVITY, PhysicsParams


CONFIG_FILENAME = "config_simulador.json"
MEMORY_DIR_NAME = "Memoria"
MEMORY_FILE_RE = re.compile(r"^modelo_pendulo_gpu_(\d+)\.pth$", re.IGNORECASE)


def next_memory_path(app_dir: str) -> str:
    """Retorna o próximo caminho disponível em Memoria/modelo_pendulo_gpu_{N}.pth sem sobrescrever."""
    mem_dir = os.path.join(app_dir, MEMORY_DIR_NAME)
    os.makedirs(mem_dir, exist_ok=True)
    max_idx = 0
    for name in os.listdir(mem_dir):
        m = MEMORY_FILE_RE.match(name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return os.path.join(mem_dir, f"modelo_pendulo_gpu_{max_idx + 1}.pth")


def latest_memory_path(app_dir: str) -> str | None:
    """Retorna o checkpoint mais recente na pasta Memoria/ ou modelo_pendulo_gpu.pth na raiz."""
    mem_dir = os.path.join(app_dir, MEMORY_DIR_NAME)
    best_idx = 0
    best_path: str | None = None
    if os.path.isdir(mem_dir):
        for name in os.listdir(mem_dir):
            m = MEMORY_FILE_RE.match(name)
            if m and int(m.group(1)) > best_idx:
                best_idx = int(m.group(1))
                best_path = os.path.join(mem_dir, name)
    if best_path is not None:
        return best_path
    root_path = os.path.join(app_dir, "modelo_pendulo_gpu.pth")
    return root_path if os.path.exists(root_path) else None


def save_app_config(
    app_dir: str,
    params: PhysicsParams,
    agent: Any,
    meta: Any,
    sim_env: Any | None = None,
) -> str:
    """Salva todas as configurações (mutação, pesos, parâmetros físicos, lr e Meta IA) em JSON."""
    p_dict = asdict(params)
    if sim_env is not None:
        p_dict["table_angle"] = float(sim_env.table_angle)
    payload: Dict[str, Any] = {
        "params": p_dict,
        "learning_rate": float(agent.lr),
        "meta": {
            "enabled": bool(meta.enabled),
            "flicks_enabled": bool(meta.flicks_enabled),
            "interval_s": float(meta.interval_s),
            "flick_min_s": float(meta.flick_min_s),
            "flick_max_s": float(meta.flick_max_s),
            "flick_force_max": float(meta.flick_force_max),
        },
    }
    cfg_path = os.path.join(app_dir, CONFIG_FILENAME)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return cfg_path


def load_app_config(
    app_dir: str,
    params: PhysicsParams,
    agent: Any,
    meta: Any,
    sim_env: Any | None = None,
    now_t: float = 0.0,
) -> bool:
    """Carrega configurações salvas anteriormente se config_simulador.json existir."""
    cfg_path = os.path.join(app_dir, CONFIG_FILENAME)
    if not os.path.exists(cfg_path):
        return False
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False

    p_data = payload.get("params", {})
    for key, val in p_data.items():
        if key == "gravity":
            continue
        if hasattr(params, key):
            cur = getattr(params, key)
            if isinstance(cur, bool):
                setattr(params, key, bool(val))
            elif isinstance(cur, int):
                setattr(params, key, int(val))
            elif isinstance(cur, float):
                setattr(params, key, float(val))
            elif isinstance(cur, str):
                setattr(params, key, str(val))
    params.gravity = FIXED_GRAVITY

    if sim_env is not None and "table_angle" in p_data:
        sim_env.set_manual_table_angle_deg(math.degrees(float(p_data["table_angle"])))
    if sim_env is not None and "wind_speed" in p_data:
        sim_env.set_manual_wind(float(p_data["wind_speed"]))

    if "learning_rate" in payload:
        agent.set_learning_rate(float(payload["learning_rate"]))

    m_data = payload.get("meta", {})
    for m_key, p_name in (
        ("interval_s", "meta_interval_s"),
        ("flick_min_s", "meta_flick_min_s"),
        ("flick_max_s", "meta_flick_max_s"),
        ("flick_force_max", "meta_flick_force_max"),
    ):
        if m_key in m_data:
            meta.set_param(p_name, float(m_data[m_key]))
    if "flicks_enabled" in m_data:
        meta.set_flicks_enabled(bool(m_data["flicks_enabled"]))
    if "enabled" in m_data:
        meta.set_enabled(bool(m_data["enabled"]), now_t)
    return True
