"""
Servidor Web Local + Motor de Física e Treinamento RL em GPU (PyTorch/CUDA).

Recursos:
- Velocidade da simulação configurável (segundos simulados por segundo real).
- Aba Controle (sliders só para Base e Vento; campos numéricos puros para pesos, L, tempo, força e lr).
- Aba Auto Mutação com checkboxes individuais e ranges [Mín, Máx] editáveis.
- Sensor de Ângulo Quantizado em 0,1° (formato 123,4°) + inputs de m_p, L e m_c na IA.
- Gráfico de Posição X (centro = 0) e sem gráfico de Loss nem texto de fim de curso.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Deque, Dict, Optional

import numpy as np
import torch

from meta_ai import MetaAI
from physics import BatchedCartPoleEnv, CartPolePhysics, FIXED_GRAVITY, PhysicsParams, REWARD_TABLE_SPECS, format_angle_deg_br
from rl_agent import RLAgent
from storage import latest_memory_path, load_app_config, next_memory_path, save_app_config


APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(APP_DIR, "modelo_pendulo_gpu.pth")
REWARD_WINDOW_SEC = 30.0


WEB_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Pêndulo Invertido RL — Sensor Ângulo 0,1° &amp; Auto Mutação (CUDA)</title>
<style>
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --panel2: #1c232d;
    --border: #2c3646;
    --text: #e6edf3;
    --muted: #8b949e;
    --cyan: #38bdf8;
    --green: #10b981;
    --amber: #f59e0b;
    --rose: #f43f5e;
    --violet: #a78bfa;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    font-size: 13px;
  }
  .layout {
    display: grid;
    grid-template-columns: 1fr 370px;
    gap: 12px;
    padding: 12px;
    min-height: 100vh;
  }
  .left-col {
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-width: 0;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  .hud {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    padding: 9px 14px;
    background: var(--panel2);
    border-bottom: 1px solid var(--border);
    font-family: Consolas, "Cascadia Mono", monospace;
    font-size: 12px;
  }
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-weight: 600;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); }
  .sim-wrap { position: relative; height: 410px; }
  canvas { display: block; width: 100%; height: 100%; }
  .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .chart-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
  }
  .chart-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    font-size: 12px;
    font-weight: 600;
  }
  .chart-canvas-wrap {
    height: 120px;
    background: var(--bg);
    border-radius: 6px;
    overflow: hidden;
  }
  .sidebar {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 13px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .tabs {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 5px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  .tab-btn.active {
    background: #233246;
    border-color: var(--cyan);
  }
  .tab-panel { display: flex; flex-direction: column; gap: 9px; }
  .hidden { display: none !important; }
  .check-row {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    user-select: none;
    font-weight: 600;
  }
  .check-row input { width: 15px; height: 15px; accent-color: var(--green); cursor: pointer; }
  .btn-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .btn-full { grid-column: 1 / -1; }
  button {
    background: var(--panel2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
  }
  button:hover { background: #263140; border-color: var(--cyan); }
  button.amber { border-color: rgba(245, 158, 11, 0.55); }
  button.green { background: rgba(16, 185, 129, 0.15); border-color: rgba(16, 185, 129, 0.55); }
  button.rose { background: rgba(244, 63, 94, 0.15); border-color: rgba(244, 63, 94, 0.55); }
  .fixed-box {
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 7px 10px;
    font-family: Consolas, monospace;
    font-size: 11.5px;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .slider-item label, .num-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    color: var(--muted);
    gap: 6px;
  }
  .num-wrap { display: inline-flex; align-items: center; gap: 4px; }
  input.num-input {
    width: 82px;
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 3px 6px;
    font-family: Consolas, monospace;
    font-size: 12px;
    font-weight: 600;
    text-align: right;
  }
  input.num-input:focus { outline: none; border-color: var(--cyan); }
  span.unit-lbl { min-width: 24px; font-size: 11px; color: var(--muted); }
  .slider-item input[type=range] { width: 100%; accent-color: var(--cyan); cursor: pointer; }
  .fields-box {
    display: flex;
    flex-direction: column;
    gap: 6px;
    border-top: 1px solid var(--border);
    padding-top: 8px;
  }
  .mut-row {
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 7px 9px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .mut-ranges {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .mut-ranges label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 4px;
    font-size: 11px;
    color: var(--muted);
  }
  .rew-table {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .rew-head, .rew-row {
    display: grid;
    grid-template-columns: 1fr 56px 76px;
    align-items: center;
    gap: 6px;
  }
  .rew-head {
    font-size: 11px;
    color: var(--muted);
    padding: 2px 4px;
    border-bottom: 1px solid var(--border);
  }
  .rew-row {
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 4px 6px;
    font-size: 11.5px;
  }
  .rew-row label {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    overflow: hidden;
    white-space: nowrap;
  }
  .rew-row label span { overflow: hidden; text-overflow: ellipsis; }
  .rew-pts {
    font-family: Consolas, monospace;
    font-size: 11px;
    text-align: right;
  }
  .rew-row input.num-input { width: 74px; padding: 2px 5px; font-size: 11.5px; }
  .toast {
    position: absolute;
    top: 12px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(18, 26, 36, 0.94);
    border: 1px solid var(--cyan);
    padding: 6px 14px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 12px;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.2s;
  }
  .toast.show { opacity: 1; }
  .hint {
    font-size: 11px;
    color: var(--muted);
    padding: 6px 12px;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
  }
  @media (max-width: 1040px) { .layout { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="layout">
  <div class="left-col">
    <div class="card">
      <div class="hud">
        <span class="badge"><span id="gpuDot" class="dot"></span><span id="gpuLabel">Conectando à GPU...</span></span>
        <span id="perfLabel">Treino: 0 steps/s | Ângulo: 180,0° | Sim: 1.00x</span>
      </div>
      <div class="sim-wrap">
        <canvas id="simCanvas"></canvas>
        <div id="toast" class="toast"></div>
      </div>
      <div class="hint">
        <span id="liveStats">θ: 180,0° | x: +0.00 m | Base α: +0.0° | Vento: +0.0 m/s</span>
        <span>←/→ ou A/D: Motor | Arrastar | Dir.: Empurrar | Espaço: IA | R: Baixo</span>
      </div>
    </div>

    <div class="charts-grid">
      <div class="chart-card">
        <div class="chart-head"><span id="titleAngle">Sensor Ângulo (0,1°)</span><span id="lblAngle" style="color:var(--cyan)">180,0°</span></div>
        <div class="chart-canvas-wrap"><canvas id="chartAngle"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-head"><span>Posição Eixo X (m) — Alvo Centro (0.0)</span><span id="lblPosX" style="color:var(--green)">0.00 m</span></div>
        <div class="chart-canvas-wrap"><canvas id="chartPosX"></canvas></div>
      </div>
      <div class="chart-card" style="grid-column: 1 / -1;">
        <div class="chart-head"><span>Recompensa Atual — Janela Deslizante 30s (pts/passo)</span><span id="lblRew30s" style="color:var(--green)">--</span></div>
        <div class="chart-canvas-wrap" style="height:105px;"><canvas id="chartRew30s"></canvas></div>
      </div>
    </div>
  </div>

  <aside class="sidebar">
    <div class="tabs">
      <button id="btnTabControl" class="tab-btn active" onclick="switchTab('control')">Controle</button>
      <button id="btnTabMutate" class="tab-btn" onclick="switchTab('auto_mutate')">Auto Mut.</button>
      <button id="btnTabReward" class="tab-btn" onclick="switchTab('reward')">Recompensa</button>
      <button id="btnTabMeta" class="tab-btn" onclick="switchTab('meta')">Meta IA</button>
    </div>

    <!-- ABA CONTROLE -->
    <div id="tabControl" class="tab-panel">
      <label class="check-row"><input id="chkAI" type="checkbox" /> Ativar Controle por IA</label>
      <label class="check-row"><input id="chkTrain" type="checkbox" checked /> Treinar IA na GPU (Turbo CUDA)</label>
      <label class="check-row"><input id="chkMutate" type="checkbox" checked /> Mutação Automática Ativa</label>

      <div class="btn-grid">
        <button class="btn-full amber" onclick="sendAction('set_down')">Colocar Pêndulo para Baixo</button>
        <button onclick="sendAction('set_up')">Colocar no Topo (0)</button>
        <button class="rose" onclick="sendAction('perturb')">Perturbar (Impulso)</button>
        <button class="green" onclick="sendAction('save_model')">Salvar Modelo</button>
        <button onclick="sendAction('load_model')">Carregar Modelo</button>
        <button class="btn-full rose" onclick="sendAction('reset_training')">Resetar / Limpar Treinamento</button>
      </div>

      <div class="fixed-box">
        <span style="color:var(--amber)">Gravidade Fixa: 9.81 m/s² (Constante)</span>
        <span id="epsAutoLabel" style="color:var(--violet)">Exploração Auto (Varredura): 100.0%</span>
      </div>

      <div id="slidersContainer"></div>
      <div id="fieldsContainer" class="fields-box"></div>
    </div>

    <!-- ABA AUTO MUTAÇÃO -->
    <div id="tabMutate" class="tab-panel hidden">
      <label class="check-row"><input id="chkMutateTab" type="checkbox" checked /> Ativar Auto Mutação</label>
      <div class="btn-grid">
        <button class="btn" id="btnModeConst" onclick="setMutMode('constant')">Mutação Constante</button>
        <button class="btn" id="btnModeFixed" onclick="setMutMode('fixed')">Mutação Fixa</button>
      </div>
      <div class="num-row">
        <span>Tempo de Mutação (s)</span>
        <span class="num-wrap">
          <input class="num-input" id="mutTimeTab" type="text" inputmode="decimal" value="4.0" />
          <span class="unit-lbl">s</span>
        </span>
      </div>
      <div id="mutRowsContainer" style="display:flex;flex-direction:column;gap:7px;"></div>
    </div>

    <!-- ABA RECOMPENSA -->
    <div id="tabReward" class="tab-panel hidden">
      <div class="fixed-box" style="flex-direction:row;justify-content:space-between;align-items:center;">
        <span style="color:var(--text);font-weight:700;">TABELA DE RECOMPENSAS (+/-)</span>
        <span id="rwTotal" style="color:var(--green);font-weight:700;">Total: +0.000</span>
      </div>
      <div class="rew-head">
        <span>Atividade</span>
        <span style="text-align:right;">Pts</span>
        <span style="text-align:right;">Peso</span>
      </div>
      <div id="rewTableContainer" class="rew-table"></div>
    </div>

    <!-- ABA META IA -->
    <div id="tabMeta" class="tab-panel hidden">
      <label class="check-row"><input id="chkMeta" type="checkbox" /> Ativar Meta IA (otimiza 2 tasks)</label>
      <label class="check-row"><input id="chkFlicks" type="checkbox" checked /> Ativar Petelecos (perturbações)</label>
      <div class="fixed-box">
        <span id="metaState" style="font-weight:700;color:var(--amber)">META IA: DESLIGADA</span>
        <span id="metaTime">Próxima otimização em: --:--</span>
        <span id="metaRounds">Rodadas: 0 | σ (passo de mutação): 0.350</span>
        <span id="metaStab" style="color:var(--green)">Estab: 0.0% | Subida(2s): 0.0%</span>
        <span id="metaScore">Última: est 0.0% | sub 0.0%</span>
        <span id="metaFlick" style="color:var(--rose)">Último peteleco: —</span>
        <span id="metaMsg" style="color:var(--muted);white-space:normal;">—</span>
      </div>
      <div id="metaFieldsContainer" class="fields-box"></div>
    </div>
  </aside>
</div>

<script>
const SLIDER_SPECS = [
  { key: "table_angle_deg", label: "Inclinação Base (α)", min: -45.0, max: 45.0, step: 0.5, val: 0.0, unit: "°", inMin: -45.0, inMax: 45.0, fmt: v => v.toFixed(1) },
  { key: "wind_speed", label: "Velocidade Vento", min: -15.0, max: 15.0, step: 0.2, val: 0.0, unit: "m/s", inMin: -15.0, inMax: 15.0, fmt: v => v.toFixed(1) },
];

const FIELD_SPECS = [
  { key: "sim_speed", label: "Vel. Simulação (s/s)", min: 0.05, max: 50.0, val: 1.0, unit: "x", fmt: v => v.toFixed(2) },
  { key: "swing_hold_target", label: "Estab. Subida (s)", min: 0.2, max: 30.0, val: 2.0, unit: "s", fmt: v => v.toFixed(1) },
  { key: "pendulum_mass", label: "Peso Pêndulo (m_p)", min: 0.05, max: 3.00, val: 0.25, unit: "kg", fmt: v => v.toFixed(2) },
  { key: "cart_mass", label: "Peso Carrinho (m_c)", min: 0.20, max: 8.00, val: 1.20, unit: "kg", fmt: v => v.toFixed(2) },
  { key: "pendulum_length", label: "Comprimento L (m)", min: 0.20, max: 2.00, val: 0.75, unit: "m", fmt: v => v.toFixed(2) },
  { key: "mutation_interval", label: "Tempo Mutação (s)", min: 0.2, max: 300.0, val: 4.0, unit: "s", fmt: v => v.toFixed(1) },
  { key: "max_force", label: "Força Motor (N)", min: 5.0, max: 100.0, val: 30.0, unit: "N", fmt: v => v.toFixed(1) },
  { key: "learning_rate", label: "Taxa Aprend. (lr)", min: 1e-5, max: 3e-3, val: 3e-4, unit: "", fmt: v => v.toExponential(1) },
];

const MUTATE_SPECS = [
  { flag: "mutate_table", label: "Inclinação da Base (°)", minKey: "table_min_deg", maxKey: "table_max_deg", hMin: -45, hMax: 45, defMin: -45, defMax: 45, fmt: v => v.toFixed(1) },
  { flag: "mutate_wind", label: "Velocidade Vento (m/s)", minKey: "wind_min", maxKey: "wind_max", hMin: -15, hMax: 15, defMin: -15, defMax: 15, fmt: v => v.toFixed(1) },
  { flag: "mutate_pend_mass", label: "Peso Pêndulo (kg)", minKey: "pend_mass_min", maxKey: "pend_mass_max", hMin: 0.05, hMax: 3.0, defMin: 0.08, defMax: 1.50, fmt: v => v.toFixed(2) },
  { flag: "mutate_cart_mass", label: "Peso Carrinho (kg)", minKey: "cart_mass_min", maxKey: "cart_mass_max", hMin: 0.20, hMax: 8.0, defMin: 0.40, defMax: 4.00, fmt: v => v.toFixed(2) },
  { flag: "mutate_length", label: "Comprimento L (m)", minKey: "length_min", maxKey: "length_max", hMin: 0.20, hMax: 2.0, defMin: 0.35, defMax: 1.30, fmt: v => v.toFixed(2) },
];

const REWARD_SPECS = [
  { key: "rew_center_x", activeKey: "rew_center_x_active", label: "Próximo do x=0", val: 2.0 },
  { key: "rew_top", activeKey: "rew_top_active", label: "Pêndulo no topo", val: 2.0 },
  { key: "rew_stability", activeKey: "rew_stability_active", label: "Estabilidade", val: 1.5 },
  { key: "rew_x_dist", activeKey: "rew_x_dist_active", label: "Penal. distância do x=0", val: -0.40 },
  { key: "rew_endstop", activeKey: "rew_endstop_active", label: "Fim de curso", val: -25.0 },
  { key: "rew_up_time_acc", activeKey: "rew_up_time_acc_active", label: "Acum. tempo p/ cima", val: 1.0 },
  { key: "rew_fall", activeKey: "rew_fall_active", label: "Queda do pêndulo", val: -2.0 },
  { key: "rew_rise", activeKey: "rew_rise_active", label: "Subida do pêndulo", val: 1.0 },
  { key: "rew_bottom", activeKey: "rew_bottom_active", label: "Pêndulo para baixo", val: -0.50 },
  { key: "rew_down_time_acc", activeKey: "rew_down_time_acc_active", label: "Acum. tempo p/ baixo", val: -1.0 },
  { key: "rew_swingup", activeKey: "rew_swingup_active", label: "Swing up", val: 0.40 },
  { key: "rew_smooth", activeKey: "rew_smooth_active", label: "Movimento brusco", val: -0.10 },
  { key: "rew_full_rotation", activeKey: "rew_full_rotation_active", label: "Rotação completa", val: -35.0 },
];
const fmtSigned = v => (Number(v) >= 0 ? "+" : "") + Number(v).toFixed(2);

function switchTab(tab) {
  document.getElementById("tabControl").classList.toggle("hidden", tab !== "control");
  document.getElementById("tabMutate").classList.toggle("hidden", tab !== "auto_mutate");
  document.getElementById("tabReward").classList.toggle("hidden", tab !== "reward");
  document.getElementById("tabMeta").classList.toggle("hidden", tab !== "meta");
  document.getElementById("btnTabMeta").classList.toggle("active", tab === "meta");
  document.getElementById("btnTabControl").classList.toggle("active", tab === "control");
  document.getElementById("btnTabMutate").classList.toggle("active", tab === "auto_mutate");
  document.getElementById("btnTabReward").classList.toggle("active", tab === "reward");
}

function postParams(obj) {
  fetch("/api/params", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(obj) });
}

const sliderEls = {};
const sCont = document.getElementById("slidersContainer");
SLIDER_SPECS.forEach(sp => {
  const div = document.createElement("div");
  div.className = "slider-item";
  div.innerHTML = `<label><span>${sp.label}</span><span class="num-wrap"><input class="num-input" id="num_${sp.key}" type="text" inputmode="decimal" value="${sp.fmt(sp.val)}" /><span class="unit-lbl">${sp.unit}</span></span></label>
    <input type="range" id="sl_${sp.key}" min="${sp.min}" max="${sp.max}" step="${sp.step}" value="${sp.val}" />`;
  sCont.appendChild(div);
  const input = div.querySelector("input[type=range]");
  const numInput = div.querySelector("input.num-input");
  sliderEls[sp.key] = { input, numInput, spec: sp, dragging: false };

  input.addEventListener("pointerdown", () => sliderEls[sp.key].dragging = true);
  window.addEventListener("pointerup", () => sliderEls[sp.key].dragging = false);
  input.addEventListener("input", () => {
    const val = parseFloat(input.value);
    numInput.value = sp.fmt(val);
    postParams({ [sp.key]: val });
  });
  const commit = () => {
    const p = parseFloat(numInput.value.replace(",", "."));
    if (!Number.isFinite(p)) return;
    const c = Math.max(sp.inMin, Math.min(sp.inMax, p));
    numInput.value = sp.fmt(c); input.value = c;
    postParams({ [sp.key]: c });
  };
  numInput.addEventListener("change", commit);
  numInput.addEventListener("keydown", e => { if (e.key === "Enter") { commit(); numInput.blur(); } });
});

const fieldEls = {};
const fCont = document.getElementById("fieldsContainer");
FIELD_SPECS.forEach(sp => {
  const row = document.createElement("div");
  row.className = "num-row";
  row.innerHTML = `<span>${sp.label}</span><span class="num-wrap"><input class="num-input" id="nf_${sp.key}" type="text" inputmode="decimal" value="${sp.fmt(sp.val)}" /><span class="unit-lbl">${sp.unit}</span></span>`;
  fCont.appendChild(row);
  const numInput = row.querySelector("input");
  fieldEls[sp.key] = { numInput, spec: sp };
  const commit = () => {
    const p = parseFloat(numInput.value.replace(",", "."));
    if (!Number.isFinite(p)) return;
    const c = Math.max(sp.min, Math.min(sp.max, p));
    numInput.value = sp.fmt(c);
    postParams({ [sp.key]: c });
  };
  numInput.addEventListener("change", commit);
  numInput.addEventListener("keydown", e => { if (e.key === "Enter") { commit(); numInput.blur(); } });
});

const rewFieldEls = {};
const rwCont = document.getElementById("rewTableContainer");
REWARD_SPECS.forEach(sp => {
  const row = document.createElement("div");
  row.className = "rew-row";
  row.innerHTML = `
    <label title="${sp.label}"><input type="checkbox" id="rc_${sp.activeKey}" checked /><span>${sp.label}</span></label>
    <span class="rew-pts" id="rp_${sp.key}">+0.00</span>
    <input class="num-input" id="rf_${sp.key}" type="text" inputmode="decimal" value="${fmtSigned(sp.val)}" />`;
  rwCont.appendChild(row);
  const chk = row.querySelector("input[type=checkbox]");
  const ptsEl = document.getElementById(`rp_${sp.key}`);
  const numInput = document.getElementById(`rf_${sp.key}`);
  rewFieldEls[sp.key] = { chk, ptsEl, numInput, spec: sp };
  chk.addEventListener("change", () => sendControl({ [sp.activeKey]: chk.checked }));
  const commit = () => {
    const p = parseFloat(numInput.value.replace(",", "."));
    if (!Number.isFinite(p)) return;
    const c = Math.max(-500.0, Math.min(500.0, p));
    numInput.value = fmtSigned(c);
    postParams({ [sp.key]: c });
  };
  numInput.addEventListener("change", commit);
  numInput.addEventListener("keydown", e => { if (e.key === "Enter") { commit(); numInput.blur(); } });
});

const META_FIELD_SPECS = [
  { key: "meta_interval_s", stKey: "interval_s", label: "Intervalo de otimização (s)", min: 5, max: 7200, val: 300, unit: "s", fmt: v => v.toFixed(0) },
  { key: "meta_flick_min_s", stKey: "flick_min_s", label: "Peteleco: tempo mínimo (s)", min: 0.5, max: 300, val: 4, unit: "s", fmt: v => v.toFixed(1) },
  { key: "meta_flick_max_s", stKey: "flick_max_s", label: "Peteleco: tempo máximo (s)", min: 0.5, max: 600, val: 12, unit: "s", fmt: v => v.toFixed(1) },
  { key: "meta_flick_force_max", stKey: "flick_force_max", label: "Peteleco: força máxima", min: 0.5, max: 25, val: 2.5, unit: "rad/s", fmt: v => v.toFixed(1) },
];
const metaFieldEls = {};
const metaCont = document.getElementById("metaFieldsContainer");
META_FIELD_SPECS.forEach(sp => {
  const row = document.createElement("div");
  row.className = "num-row";
  row.innerHTML = `<span>${sp.label}</span><span class="num-wrap"><input class="num-input" id="mf_${sp.key}" type="text" inputmode="decimal" value="${sp.fmt(sp.val)}" /><span class="unit-lbl">${sp.unit}</span></span>`;
  metaCont.appendChild(row);
  const numInput = row.querySelector("input");
  metaFieldEls[sp.key] = { numInput, spec: sp };
  const commit = () => {
    const p = parseFloat(numInput.value.replace(",", "."));
    if (!Number.isFinite(p)) return;
    const c = Math.max(sp.min, Math.min(sp.max, p));
    numInput.value = sp.fmt(c);
    postParams({ [sp.key]: c });
  };
  numInput.addEventListener("change", commit);
  numInput.addEventListener("keydown", e => { if (e.key === "Enter") { commit(); numInput.blur(); } });
});
document.getElementById("chkMeta").addEventListener("change", e => {
  sendControl({ meta_enabled: e.target.checked });
  showToast(e.target.checked ? "Meta IA LIGADA" : "Meta IA DESLIGADA");
});
document.getElementById("chkFlicks").addEventListener("change", e => {
  sendControl({ flicks_enabled: e.target.checked });
  showToast(e.target.checked ? "Petelecos ATIVADOS" : "Petelecos DESATIVADOS");
});

function updateMetaPanel(m) {
  if (!m) return;
  document.getElementById("chkMeta").checked = !!m.enabled;
  document.getElementById("chkFlicks").checked = m.flicks_enabled !== false;
  const stEl = document.getElementById("metaState");
  stEl.textContent = !m.enabled ? "META IA: DESLIGADA" : (m.frozen ? "META IA: ESTABILIDADE PERFEITA (pesos congelados)" : "META IA: OTIMIZANDO (2 TASKS)");
  stEl.style.color = !m.enabled ? "var(--muted)" : (m.frozen ? "var(--green)" : "var(--amber)");
  const tl = Math.max(0, Math.round(m.time_left || 0));
  document.getElementById("metaTime").textContent = m.enabled
    ? `Próxima otimização em: ${String(Math.floor(tl / 60)).padStart(2, "0")}:${String(tl % 60).padStart(2, "0")}`
    : "Próxima otimização em: --:--";
  document.getElementById("metaRounds").textContent = `Rodadas: ${m.rounds} | σ (passo de mutação): ${m.sigma.toFixed(3)}`;
  document.getElementById("metaStab").textContent = `Estab: ${(m.best_stab * 100).toFixed(1)}% | Subida(2s): ${((m.best_swingup || 0) * 100).toFixed(1)}%`;
  document.getElementById("metaScore").textContent = `Última: est ${(m.last_stab * 100).toFixed(1)}% | sub ${((m.last_swingup || 0) * 100).toFixed(1)}%`;
  document.getElementById("metaFlick").textContent = `Último peteleco: ${m.last_flick}`;
  document.getElementById("metaMsg").textContent = m.last_msg;
  for (const k of Object.keys(metaFieldEls)) {
    const it = metaFieldEls[k];
    if (document.activeElement !== it.numInput && m[it.spec.stKey] !== undefined) it.numInput.value = it.spec.fmt(m[it.spec.stKey]);
  }
}

const mutTimeTab = document.getElementById("mutTimeTab");
const commitMutTimeTab = () => {
  const p = parseFloat(mutTimeTab.value.replace(",", "."));
  if (!Number.isFinite(p)) return;
  const c = Math.max(0.2, Math.min(300.0, p));
  mutTimeTab.value = c.toFixed(1);
  postParams({ mutation_interval: c });
};
mutTimeTab.addEventListener("change", commitMutTimeTab);
mutTimeTab.addEventListener("keydown", e => { if (e.key === "Enter") { commitMutTimeTab(); mutTimeTab.blur(); } });

const mutEls = {};
const mutCont = document.getElementById("mutRowsContainer");
MUTATE_SPECS.forEach(ms => {
  const card = document.createElement("div");
  card.className = "mut-row";
  card.innerHTML = `
    <label class="check-row"><input type="checkbox" id="mc_${ms.flag}" checked /> ${ms.label}</label>
    <div class="mut-ranges">
      <label>Mín <input class="num-input" style="width:68px;" id="mn_${ms.minKey}" type="text" inputmode="decimal" value="${ms.fmt(ms.defMin)}" /></label>
      <label>Máx <input class="num-input" style="width:68px;" id="mx_${ms.maxKey}" type="text" inputmode="decimal" value="${ms.fmt(ms.defMax)}" /></label>
    </div>`;
  mutCont.appendChild(card);
  const chk = card.querySelector("input[type=checkbox]");
  const minIn = document.getElementById(`mn_${ms.minKey}`);
  const maxIn = document.getElementById(`mx_${ms.maxKey}`);
  mutEls[ms.flag] = { chk, minIn, maxIn, spec: ms };

  chk.addEventListener("change", () => sendControl({ [ms.flag]: chk.checked }));
  const bindNum = (el, key) => {
    const commit = () => {
      const p = parseFloat(el.value.replace(",", "."));
      if (!Number.isFinite(p)) return;
      const c = Math.max(ms.hMin, Math.min(ms.hMax, p));
      el.value = ms.fmt(c);
      postParams({ [key]: c });
    };
    el.addEventListener("change", commit);
    el.addEventListener("keydown", e => { if (e.key === "Enter") { commit(); el.blur(); } });
  };
  bindNum(minIn, ms.minKey);
  bindNum(maxIn, ms.maxKey);
});

let stateData = null;
const trail = [];
const swingTrail = [];
const windParticles = Array.from({ length: 38 }, () => ({ x: Math.random(), y: 0.1 + 0.8 * Math.random(), l: 0.6 + 0.8 * Math.random() }));
const keysDown = new Set();
let mouseDragRailS = null;

function showToast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg; el.classList.add("show");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => el.classList.remove("show"), 2400);
}

async function sendAction(action, extra = {}) {
  const res = await fetch("/api/action", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action, ...extra })
  });
  const data = await res.json();
  if (action === "set_down" || action === "set_up" || action === "reset_training") { trail.length = 0; swingTrail.length = 0; }
  if (data.message) showToast(data.message);
}

function sendControl(patch) {
  fetch("/api/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(patch) });
}

function setMutMode(mode) {
  sendControl({ mutation_mode: mode });
  showToast(mode === "fixed" ? "Mutação Fixa (parada X s)" : "Mutação Constante (oscilando)");
}
document.getElementById("chkAI").addEventListener("change", e => { sendControl({ ai_active: e.target.checked }); showToast(e.target.checked ? "Piloto IA ATIVADO" : "Manual ATIVADO"); });
document.getElementById("chkTrain").addEventListener("change", e => sendControl({ training_active: e.target.checked }));
document.getElementById("chkMutate").addEventListener("change", e => {
  document.getElementById("chkMutateTab").checked = e.target.checked;
  sendControl({ auto_mutate: e.target.checked });
});
document.getElementById("chkMutateTab").addEventListener("change", e => {
  document.getElementById("chkMutate").checked = e.target.checked;
  sendControl({ auto_mutate: e.target.checked });
});

window.addEventListener("keydown", e => {
  if (document.activeElement && document.activeElement.classList.contains("num-input")) return;
  if (["ArrowLeft", "ArrowRight", " "].includes(e.key)) e.preventDefault();
  if (e.key === " ") {
    const chk = document.getElementById("chkAI");
    chk.checked = !chk.checked; sendControl({ ai_active: chk.checked }); return;
  }
  if (e.key.toLowerCase() === "r") { sendAction("set_down"); return; }
  keysDown.add(e.key); updateManualInput();
});
window.addEventListener("keyup", e => { keysDown.delete(e.key); updateManualInput(); });

function updateManualInput() {
  let u = 0;
  if (keysDown.has("ArrowLeft") || keysDown.has("a") || keysDown.has("A")) u -= 1;
  if (keysDown.has("ArrowRight") || keysDown.has("d") || keysDown.has("D")) u += 1;
  sendControl({ manual_u: u, mouse_target_rail_s: mouseDragRailS });
}

const simCanvas = document.getElementById("simCanvas");
function getSimGeo(p, x0 = 0, pw = null) {
  const w = pw ?? Math.max(120, (simCanvas.width - 10) * 0.5);
  const h = simCanvas.height;
  const cx = x0 + w * 0.5, railY = h * 0.54;
  const maxVert = p.rail_limit * 0.71 + p.pendulum_length;
  const pxPerM = Math.max(22, Math.min((w - 60) / (2 * p.rail_limit), (h * 0.44) / Math.max(maxVert * 0.72, 0.85), 125));
  return { cx, railY, pxPerM };
}
function railToScreen(cx, railY, pxPerM, alpha, s, n = 0) {
  const ca = Math.cos(alpha), sa = Math.sin(alpha);
  return [cx + (s * ca - n * sa) * pxPerM, railY - (s * sa + n * ca) * pxPerM];
}

simCanvas.addEventListener("pointerdown", e => {
  if (!stateData) return;
  const rect = simCanvas.getBoundingClientRect();
  const sx = (e.clientX - rect.left) * (simCanvas.width / rect.width);
  const sy = (e.clientY - rect.top) * (simCanvas.height / rect.height);
  const { cx, railY, pxPerM } = getSimGeo(stateData.params);
  const ph = stateData.physics, p = stateData.params;
  if (e.button === 2) {
    const tipX = cx + (ph.x * Math.cos(ph.table_angle) + p.pendulum_length * Math.sin(ph.theta)) * pxPerM;
    const dir = sx >= tipX ? 1 : -1;
    sendAction("nudge", { d_theta_dot: dir * 3.8, d_x_dot: -dir * 0.4 });
    return;
  }
  simCanvas.setPointerCapture(e.pointerId);
  const wx = (sx - cx) / pxPerM, wy = (railY - sy) / pxPerM;
  mouseDragRailS = wx * Math.cos(ph.table_angle) + wy * Math.sin(ph.table_angle);
  updateManualInput();
});
simCanvas.addEventListener("pointermove", e => {
  if (mouseDragRailS === null || !stateData) return;
  const rect = simCanvas.getBoundingClientRect();
  const sx = (e.clientX - rect.left) * (simCanvas.width / rect.width);
  const sy = (e.clientY - rect.top) * (simCanvas.height / rect.height);
  const { cx, railY, pxPerM } = getSimGeo(stateData.params);
  const wx = (sx - cx) / pxPerM, wy = (railY - sy) / pxPerM;
  mouseDragRailS = wx * Math.cos(stateData.physics.table_angle) + wy * Math.sin(stateData.physics.table_angle);
  updateManualInput();
});
simCanvas.addEventListener("pointerup", () => { if (mouseDragRailS !== null) { mouseDragRailS = null; updateManualInput(); } });
simCanvas.addEventListener("contextmenu", e => e.preventDefault());

function resizeCanvas(c) {
  const r = c.parentElement.getBoundingClientRect();
  if (c.width !== Math.floor(r.width) || c.height !== Math.floor(r.height)) {
    c.width = Math.max(100, Math.floor(r.width)); c.height = Math.max(80, Math.floor(r.height));
  }
}

function drawPendulumPane(ctx, x0, y0, pw, phBox, ph, p, st, trailArr, opts) {
  ctx.save();
  ctx.beginPath(); ctx.rect(x0, y0, pw, phBox); ctx.clip();

  ctx.fillStyle = "#0d1117"; ctx.fillRect(x0, y0, pw, phBox);
  ctx.strokeStyle = opts.isSwing ? "#38bdf8" : "#273244";
  ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y0 + 0.5, pw - 1, phBox - 1);

  const alpha = ph.table_angle;
  const { cx, railY, pxPerM } = getSimGeo(p, x0, pw);

  ctx.strokeStyle = "#17202b"; ctx.lineWidth = 1;
  const step = Math.max(24, 0.5 * pxPerM);
  for (let gx = x0 + ((cx - x0) % step); gx < x0 + pw; gx += step) { ctx.beginPath(); ctx.moveTo(gx, y0); ctx.lineTo(gx, y0 + phBox); ctx.stroke(); }
  for (let gy = railY % step; gy < y0 + phBox; gy += step) { ctx.beginPath(); ctx.moveTo(x0, gy); ctx.lineTo(x0 + pw, gy); ctx.stroke(); }

  const ws = p.wind_speed;
  if (Math.abs(ws) > 0.25) {
    const normW = Math.min(1, Math.abs(ws) / 15);
    ctx.strokeStyle = `rgba(56, 189, 248, ${0.22 + 0.55 * normW})`; ctx.lineWidth = 1.4;
    for (const pt of windParticles) {
      if (!opts.isSwing) pt.x = ((pt.x + (ws / 24) * 0.024 * pt.l * Math.min(4, p.sim_speed)) % 1 + 1) % 1;
      const px = x0 + pt.x * pw, py = y0 + pt.y * phBox;
      const len = (8 + 24 * normW) * pt.l * (ws > 0 ? 1 : -1);
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px + len, py); ctx.stroke();
    }
  }

  ctx.fillStyle = "#3c485c"; ctx.beginPath(); ctx.arc(cx, railY, 8, 0, Math.PI * 2); ctx.fill();

  const sL = -p.rail_limit - 0.24, sR = p.rail_limit + 0.24;
  const r1 = railToScreen(cx, railY, pxPerM, alpha, sL, -0.07);
  const r2 = railToScreen(cx, railY, pxPerM, alpha, sR, -0.07);
  const r3 = railToScreen(cx, railY, pxPerM, alpha, sR, -0.15);
  const r4 = railToScreen(cx, railY, pxPerM, alpha, sL, -0.15);
  ctx.fillStyle = "#28313f";
  ctx.beginPath(); ctx.moveTo(r1[0], r1[1]); ctx.lineTo(r2[0], r2[1]); ctx.lineTo(r3[0], r3[1]); ctx.lineTo(r4[0], r4[1]); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = "#5f6c7e"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(r1[0], r1[1]); ctx.lineTo(r2[0], r2[1]); ctx.stroke();

  const lPul = railToScreen(cx, railY, pxPerM, alpha, -p.rail_limit - 0.30, -0.12);
  const rPul = railToScreen(cx, railY, pxPerM, alpha, p.rail_limit + 0.30, -0.12);
  for (const pp of [lPul, rPul]) {
    ctx.fillStyle = "#343e4e"; ctx.beginPath(); ctx.arc(pp[0], pp[1], 9, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#5f6c7e"; ctx.stroke();
    for (let k = 0; k < 3; k++) {
      const a = ph.pulley_angle + alpha + k * (Math.PI * 2 / 3);
      ctx.strokeStyle = "#38bdf8"; ctx.beginPath(); ctx.moveTo(pp[0], pp[1]); ctx.lineTo(pp[0] + 6 * Math.cos(a), pp[1] + 6 * Math.sin(a)); ctx.stroke();
    }
  }

  for (const side of [-1, 1]) {
    const sEdge = side * p.rail_limit;
    const sa = sEdge + (side > 0 ? 0.02 : -0.11), sb = sEdge + (side > 0 ? 0.11 : -0.02);
    const bp = [
      railToScreen(cx, railY, pxPerM, alpha, sa, 0.14), railToScreen(cx, railY, pxPerM, alpha, sb, 0.14),
      railToScreen(cx, railY, pxPerM, alpha, sb, -0.07), railToScreen(cx, railY, pxPerM, alpha, sa, -0.07),
    ];
    ctx.fillStyle = (ph.bumper_hit > 0.05 && ph.bumper_side === side) ? "#f43f5e" : "#b45f41";
    ctx.beginPath(); ctx.moveTo(bp[0][0], bp[0][1]); bp.slice(1).forEach(pt => ctx.lineTo(pt[0], pt[1])); ctx.closePath(); ctx.fill();
  }

  const cp = [
    railToScreen(cx, railY, pxPerM, alpha, ph.x - 0.24, 0.14), railToScreen(cx, railY, pxPerM, alpha, ph.x + 0.24, 0.14),
    railToScreen(cx, railY, pxPerM, alpha, ph.x + 0.24, -0.06), railToScreen(cx, railY, pxPerM, alpha, ph.x - 0.24, -0.06),
  ];
  ctx.fillStyle = "#344154"; ctx.strokeStyle = opts.isSwing ? "#38bdf8" : (st.ai_active ? "#10b981" : "#58a6ff"); ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(cp[0][0], cp[0][1]); cp.slice(1).forEach(pt => ctx.lineTo(pt[0], pt[1])); ctx.closePath(); ctx.fill(); ctx.stroke();

  const piv = railToScreen(cx, railY, pxPerM, alpha, ph.x, 0.04);
  if (Math.abs(ph.force_motor) > 0.4) {
    const fEnd = railToScreen(cx, railY, pxPerM, alpha, ph.x + (ph.force_motor / Math.max(p.max_force, 1)) * 0.55, 0.04);
    ctx.strokeStyle = opts.isSwing ? "#38bdf8" : (st.ai_active ? "#10b981" : "#f59e0b"); ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(piv[0], piv[1]); ctx.lineTo(fEnd[0], fEnd[1]); ctx.stroke();
  }

  const pivWx = ph.x * Math.cos(alpha), pivWy = ph.x * Math.sin(alpha);
  const tipWx = pivWx + p.pendulum_length * Math.sin(ph.theta);
  const tipWy = pivWy + p.pendulum_length * Math.cos(ph.theta);
  trailArr.push([tipWx, tipWy]); if (trailArr.length > 44) trailArr.shift();
  for (let i = 1; i < trailArr.length; i++) {
    ctx.strokeStyle = `rgba(56, 189, 248, ${(i / trailArr.length) * 0.65})`; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cx + trailArr[i - 1][0] * pxPerM, railY - trailArr[i - 1][1] * pxPerM);
    ctx.lineTo(cx + trailArr[i][0] * pxPerM, railY - trailArr[i][1] * pxPerM); ctx.stroke();
  }

  const tipSx = cx + tipWx * pxPerM, tipSy = railY - tipWy * pxPerM;
  ctx.strokeStyle = Math.abs(ph.theta) < 0.22 ? "#10b981" : "#e6edf3"; ctx.lineWidth = 4;
  ctx.beginPath(); ctx.moveTo(piv[0], piv[1]); ctx.lineTo(tipSx, tipSy); ctx.stroke();

  const bobR = Math.max(7, Math.min(16, 6 + p.pendulum_mass * 6));
  ctx.fillStyle = opts.isSwing ? "#38bdf8" : "#f59e0b"; ctx.strokeStyle = "#e6edf3"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(tipSx, tipSy, bobR, 0, Math.PI * 2); ctx.fill(); ctx.stroke();

  ctx.fillStyle = opts.titleColor; ctx.font = "bold 12px Segoe UI";
  ctx.fillText(opts.title, x0 + 10, y0 + 20);
  ctx.fillStyle = "#e6edf3"; ctx.font = "bold 11px Consolas, monospace";
  ctx.fillText(opts.rightBadge, x0 + pw - ctx.measureText(opts.rightBadge).width - 10, y0 + 20);

  if (opts.isSwing && opts.swingStats) {
    const goalT = Math.max(0.2, p.swing_hold_target || 2.0);
    const prog = Math.max(0, Math.min(1, (opts.swingStats.hold_time || 0) / goalT));
    ctx.fillStyle = "#1b2432"; ctx.fillRect(x0 + 10, y0 + 28, 120, 9);
    if (prog > 0.01) {
      ctx.fillStyle = prog >= 0.99 ? "#10b981" : "#38bdf8";
      ctx.fillRect(x0 + 10, y0 + 28, Math.max(4, 120 * prog), 9);
    }
    ctx.strokeStyle = "#2b384c"; ctx.strokeRect(x0 + 10, y0 + 28, 120, 9);
  }
  ctx.restore();
}

function drawSim() {
  resizeCanvas(simCanvas);
  if (!stateData) return;
  const ctx = simCanvas.getContext("2d");
  const w = simCanvas.width, h = simCanvas.height;
  const ph = stateData.physics, sph = stateData.swing_physics || ph, p = stateData.params, st = stateData.status;
  const sw = stateData.swing_stats || { hold_time: 0, successes: 0, attempts: 0 };

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#080c12"; ctx.fillRect(0, 0, w, h);

  const halfW = Math.floor((w - 8) / 2);
  drawPendulumPane(ctx, 0, 0, halfW, h, ph, p, st, trail, {
    isSwing: false,
    title: "TREINO 1: ESTABILIDADE",
    titleColor: st.ai_active ? "#10b981" : "#f59e0b",
    rightBadge: `θ: ${fmtDegBr(ph.angle_deg)} | Estável: ${ph.stable_time.toFixed(1)}s`,
  });
  const goalT = Math.max(0.2, p.swing_hold_target || 2.0);
  const okRate = sw.attempts > 0 ? Math.round((sw.successes / sw.attempts) * 100) : 0;
  drawPendulumPane(ctx, halfW + 8, 0, w - halfW - 8, h, sph, p, st, swingTrail, {
    isSwing: true,
    swingStats: sw,
    title: `TREINO 2: SUBIDA (${goalT.toFixed(1)}s)`,
    titleColor: "#38bdf8",
    rightBadge: `Hold: ${sw.hold_time.toFixed(1)}/${goalT.toFixed(1)}s | Subidas: ${sw.successes}/${sw.attempts} (${okRate}%)`,
  });
}

function fmtDegBr(v) {
  const n = Math.abs(v) < 0.05 ? 0 : Number(v || 0);
  return n.toFixed(1).replace(".", ",") + "°";
}

function drawChart(canvasId, seriesList, fixedRanges = null) {
  const c = document.getElementById(canvasId); resizeCanvas(c);
  const ctx = c.getContext("2d"), w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#17202b"; ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();

  seriesList.forEach((s, idx) => {
    const arr = s.data; if (!arr || arr.length < 2) return;
    let vmin, vmax;
    if (fixedRanges && fixedRanges[idx]) [vmin, vmax] = fixedRanges[idx];
    else { vmin = Math.min(...arr); vmax = Math.max(...arr); const sp = Math.max(vmax - vmin, 1e-4); vmin -= 0.08 * sp; vmax += 0.08 * sp; }
    const span = Math.max(vmax - vmin, 1e-6);
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.8;
    ctx.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const px = 4 + (i / (arr.length - 1)) * (w - 8);
      const py = h - 5 - Math.max(0, Math.min(1, (arr[i] - vmin) / span)) * (h - 10);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
  });
}

function drawReward30sChart(canvasId, relTimes, values, windowSec = 30.0) {
  const c = document.getElementById(canvasId); resizeCanvas(c);
  const ctx = c.getContext("2d"), w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = "#17202b"; ctx.lineWidth = 1;
  [10, 20].forEach(sec => {
    const gx = 4 + (sec / windowSec) * (w - 8);
    ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
  });

  if (!relTimes || !values || values.length < 2) {
    ctx.fillStyle = "#8b949e"; ctx.font = "11px Consolas, monospace";
    ctx.fillText("Aguardando amostras nos últimos 30s...", 12, h / 2);
    return;
  }

  let vmin = Math.min(0, ...values), vmax = Math.max(0, ...values);
  const sp = Math.max(vmax - vmin, 0.5);
  vmin -= 0.08 * sp; vmax += 0.08 * sp;
  const fullSpan = Math.max(vmax - vmin, 1e-6);

  const yZero = h - 5 - Math.max(0, Math.min(1, (0 - vmin) / fullSpan)) * (h - 10);
  ctx.strokeStyle = "#2c3646"; ctx.beginPath(); ctx.moveTo(0, yZero); ctx.lineTo(w, yZero); ctx.stroke();

  const toXY = (rt, val) => [
    4 + Math.max(0, Math.min(1, (rt + windowSec) / windowSec)) * (w - 8),
    h - 5 - Math.max(0, Math.min(1, (val - vmin) / fullSpan)) * (h - 10)
  ];

  ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 1.4;
  ctx.beginPath();
  for (let i = 0; i < values.length; i++) {
    const [px, py] = toXY(relTimes[i], values[i]);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();

  if (values.length >= 5) {
    const win = 15;
    let acc = 0;
    ctx.strokeStyle = "#10b981"; ctx.lineWidth = 2.1;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
      acc += values[i];
      if (i >= win) acc -= values[i - win];
      const mv = acc / Math.min(i + 1, win);
      const [px, py] = toXY(relTimes[i], mv);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  ctx.fillStyle = "#8b949e"; ctx.font = "10px Consolas, monospace";
  ctx.fillText("-30s", 6, h - 4);
  ctx.fillText("-15s", w / 2 - 12, h - 4);
  ctx.fillText("0s", w - 18, h - 4);
}

async function poll() {
  try {
    const res = await fetch("/api/state");
    stateData = await res.json();
    const ph = stateData.physics, st = stateData.status, p = stateData.params, sr = stateData.series;
    const dev = st.device_info || {};
    document.getElementById("gpuDot").style.background = dev.is_cuda ? "#10b981" : "#f59e0b";
    document.getElementById("gpuLabel").textContent = dev.is_cuda
      ? `CUDA: ${dev.name} | VRAM: ${dev.alloc_mb.toFixed(0)}/${dev.total_mb.toFixed(0)} MB`
      : `MODO: ${dev.name || "CPU"}`;
    const angBr = ph.angle_deg_br || fmtDegBr(ph.angle_deg);
    document.getElementById("perfLabel").textContent =
      `Treino: ${Math.round(st.steps_per_sec).toLocaleString()} steps/s | Ângulo: ${angBr} | Recorde: ${(ph.best_stable_time || 0).toFixed(1)}s | Sim: ${p.sim_speed.toFixed(2)}x`;

    document.getElementById("liveStats").textContent =
      `θ: ${angBr} | x: ${ph.x.toFixed(2)} m | m_p: ${p.pendulum_mass.toFixed(2)} kg | L: ${p.pendulum_length.toFixed(2)} m | m_c: ${p.cart_mass.toFixed(2)} kg`;
    document.getElementById("epsAutoLabel").textContent = `Exploração Auto (Varredura): ${(st.epsilon_auto * 100).toFixed(1)}%`;

    document.getElementById("chkAI").checked = st.ai_active;
    document.getElementById("chkTrain").checked = st.training_active;
    document.getElementById("chkMutate").checked = p.auto_mutate;
    document.getElementById("chkMutateTab").checked = p.auto_mutate;
    document.getElementById("btnModeConst").style.borderColor = p.mutation_mode === "constant" ? "var(--amber)" : "var(--border)";
    document.getElementById("btnModeFixed").style.borderColor = p.mutation_mode !== "constant" ? "var(--cyan)" : "var(--border)";

    for (const k of Object.keys(sliderEls)) {
      const item = sliderEls[k];
      if (!item.dragging && document.activeElement !== item.numInput && p[k] !== undefined) {
        item.input.value = p[k]; item.numInput.value = item.spec.fmt(p[k]);
      }
    }
    for (const k of Object.keys(fieldEls)) {
      const item = fieldEls[k];
      if (document.activeElement !== item.numInput && p[k] !== undefined) {
        item.numInput.value = item.spec.fmt(p[k]);
      }
    }
    if (document.activeElement !== mutTimeTab && p.mutation_interval !== undefined) {
      mutTimeTab.value = p.mutation_interval.toFixed(1);
    }
    for (const flag of Object.keys(mutEls)) {
      const m = mutEls[flag];
      if (p[flag] !== undefined) m.chk.checked = !!p[flag];
      if (document.activeElement !== m.minIn && p[m.spec.minKey] !== undefined) m.minIn.value = m.spec.fmt(p[m.spec.minKey]);
      if (document.activeElement !== m.maxIn && p[m.spec.maxKey] !== undefined) m.maxIn.value = m.spec.fmt(p[m.spec.maxKey]);
    }
    const bd = st.reward_breakdown || {};
    for (const k of Object.keys(rewFieldEls)) {
      const item = rewFieldEls[k];
      if (p[item.spec.activeKey] !== undefined) item.chk.checked = !!p[item.spec.activeKey];
      if (document.activeElement !== item.numInput && p[k] !== undefined) item.numInput.value = fmtSigned(p[k]);
      const pts = Number(bd[k] || 0);
      if (!item.chk.checked) {
        item.ptsEl.textContent = "off"; item.ptsEl.style.color = "var(--muted)";
      } else {
        item.ptsEl.textContent = (pts >= 0 ? "+" : "") + pts.toFixed(2);
        item.ptsEl.style.color = pts > 1e-6 ? "var(--green)" : (pts < -1e-6 ? "var(--rose)" : "var(--muted)");
      }
    }
    const tot = Number(bd.total_pts || 0);
    const rwTotEl = document.getElementById("rwTotal");
    rwTotEl.textContent = `Total: ${tot >= 0 ? "+" : ""}${tot.toFixed(3)}`;
    rwTotEl.style.color = tot >= 0 ? "var(--green)" : "var(--rose)";
    updateMetaPanel(stateData.meta);

    drawSim();
    const lim = Math.max(0.8, p.rail_limit);
    drawChart("chartAngle", [
      { data: sr.angles_deg, color: "#38bdf8", width: 2 },
      { data: sr.ang_vels_q, color: "#f59e0b", width: 1.2 },
    ], [[-180, 180], [-12, 12]]);
    drawChart("chartPosX", [
      { data: sr.positions_x, color: "#10b981", width: 2 },
      { data: [0, 0], color: "#38bdf8", width: 1 },
    ], [[-lim, lim], [-lim, lim]]);
    drawReward30sChart("chartRew30s", sr.rewards_30s_rel_t, sr.rewards_30s, 30.0);

    document.getElementById("titleAngle").textContent = `Sensor Ângulo (${angBr}) & ω_q`;
    document.getElementById("lblAngle").textContent = angBr;
    document.getElementById("lblPosX").textContent = `${ph.x.toFixed(2)} m`;
    const rArr = sr.rewards_30s || [];
    const lastR = rArr.length ? rArr[rArr.length - 1] : null;
    const lblR = document.getElementById("lblRew30s");
    lblR.textContent = lastR !== null ? `${lastR >= 0 ? "+" : ""}${lastR.toFixed(2)} pts` : "--";
    lblR.style.color = (lastR === null || lastR >= 0) ? "var(--green)" : "var(--rose)";
  } catch {}
  setTimeout(poll, 22);
}

poll();
</script>
</body>
</html>
"""


class SimulationEngine:
    """Thread dedicada com velocidade de simulação ajustável e varredura GPU."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._train_lock = threading.RLock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = PhysicsParams(gravity=FIXED_GRAVITY)
        self.sim_env = CartPolePhysics(params=self.params)
        self.swing_sim_env = CartPolePhysics(params=self.params)
        self.agent = RLAgent(state_dim=CartPolePhysics.STATE_DIM, device=self.device)
        self.meta = MetaAI()
        load_app_config(APP_DIR, self.params, self.agent, self.meta, self.sim_env, now_t=time.perf_counter())
        self.swing_sim_env.reset("down")

        batch_envs = 2048 if self.device.type == "cuda" else 128
        self.gpu_env = BatchedCartPoleEnv(num_envs=batch_envs, params=self.params, device=self.device, max_episode_steps=450, reset_mode="stability")
        self.gpu_swing_env = BatchedCartPoleEnv(num_envs=batch_envs, params=self.params, device=self.device, max_episode_steps=500, reset_mode="swingup_2s")
        atexit.register(lambda: save_app_config(APP_DIR, self.params, self.agent, self.meta, self.sim_env))

        self.ai_active = False
        self.training_active = True
        self.manual_u = 0.0
        self.mouse_target_rail_s: Optional[float] = None
        self.current_u = 0.0
        self.swing_hold_time = 0.0
        self.swing_elapsed = 0.0
        self.swing_successes = 0
        self.swing_attempts = 0
        self.swing_recent: Deque[int] = deque(maxlen=20)
        self.current_memory_name = "Novo (zero)"
        self.recent_eval_window: Deque[Tuple[float, bool, float]] = deque()
        self.recent_top_durations: Deque[float] = deque(maxlen=30)
        self._prev_stable_time = 0.0

        self.angles_deg: Deque[float] = deque(maxlen=180)
        self.ang_vels_q: Deque[float] = deque(maxlen=180)
        self.positions_x: Deque[float] = deque(maxlen=180)
        self.rewards_30s: Deque[Tuple[float, float]] = deque()

        self.steps_per_sec = 0.0
        self.device_info = self.agent.get_device_info()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._train_thread = threading.Thread(target=self._train_loop, daemon=True)
        self._train_thread.start()

    def _sync_all_envs(self) -> None:
        with self._train_lock:
            self.gpu_env.sync_params(self.params)
            self.gpu_swing_env.sync_params(self.params)

    def _prune_rewards_30s(self, now_t: Optional[float] = None) -> None:
        ref = time.perf_counter() if now_t is None else float(now_t)
        while self.rewards_30s and (ref - self.rewards_30s[0][0]) > REWARD_WINDOW_SEC:
            self.rewards_30s.popleft()

    def _loop(self) -> None:
        perf_start = time.perf_counter()
        perf_steps = 0
        last_tick = time.perf_counter()
        sim_acc = 0.0

        while self._running:
            t0 = time.perf_counter()
            dt_real = min(0.05, max(1e-4, t0 - last_tick))
            last_tick = t0

            with self.lock:
                ext_force = 0.0
                manual_cmd = self.manual_u
                if self.mouse_target_rail_s is not None:
                    target_s = float(np.clip(self.mouse_target_rail_s, -self.params.rail_limit, self.params.rail_limit))
                    if not self.ai_active and manual_cmd == 0.0:
                        manual_cmd = float(np.clip((target_s - self.sim_env.x) * 3.0 - self.sim_env.x_dot * 0.45, -1.0, 1.0))
                    elif self.ai_active:
                        ext_force = float(np.clip((target_s - self.sim_env.x) * 35.0, -40.0, 40.0))

                sim_acc += dt_real * self.params.sim_speed
                steps_run = 0
                while sim_acc >= self.params.dt and steps_run < 16:
                    sim_acc -= self.params.dt
                    steps_run += 1

                    if self.ai_active:
                        _, chosen_u = self.agent.select_action_single(self.sim_env.get_state(), deterministic=True)
                    else:
                        chosen_u = manual_cmd

                    _, swing_u = self.agent.select_action_single(self.swing_sim_env.get_state(), deterministic=True)

                    self.current_u = chosen_u
                    self.sim_env.step(chosen_u, external_cart_force=ext_force)
                    if self.sim_env.stable_time > 0.0:
                        self._prev_stable_time = self.sim_env.stable_time
                    elif self._prev_stable_time > 0.1:
                        self.recent_top_durations.append(self._prev_stable_time)
                        self._prev_stable_time = 0.0

                    self.swing_sim_env.table_angle = self.sim_env.table_angle
                    self.swing_sim_env.step(swing_u)
                    self.swing_elapsed += self.params.dt
                    goal_t = max(0.2, float(self.params.swing_hold_target))
                    if abs(self.swing_sim_env.theta) <= 0.25 and abs(self.swing_sim_env.theta_dot) <= 1.2:
                        self.swing_hold_time += self.params.dt
                        if self.swing_hold_time >= goal_t:
                            self.swing_successes += 1
                            self.swing_attempts += 1
                            self.swing_recent.append(1)
                            self.swing_hold_time = 0.0
                            self.swing_elapsed = 0.0
                            self.swing_sim_env.reset("down")
                    else:
                        self.swing_hold_time = 0.0
                        if self.swing_elapsed >= max(15.0, goal_t + 10.0):
                            self.swing_attempts += 1
                            self.swing_recent.append(0)
                            self.swing_elapsed = 0.0
                            self.swing_sim_env.reset("down")

                    step_t = time.perf_counter()
                    lim = max(self.params.rail_limit, 0.2)
                    is_stab_now = (
                        abs(self.sim_env.theta) < 0.22
                        and abs(self.sim_env.x) < 0.7 * lim
                        and abs(self.sim_env.theta_dot) < 1.8
                    )
                    self.recent_eval_window.append((step_t, is_stab_now, abs(float(self.sim_env.x))))
                    while self.recent_eval_window and (step_t - self.recent_eval_window[0][0]) > 60.0:
                        self.recent_eval_window.popleft()
                    self.angles_deg.append(float(self.sim_env.angle_deg))
                    self.ang_vels_q.append(float(self.sim_env.omega_q))
                    self.positions_x.append(float(self.sim_env.x))
                    self.rewards_30s.append((step_t, float(self.sim_env.last_reward_breakdown.get("total_pts", 0.0))))

                self._prune_rewards_30s(time.perf_counter())
                self.meta.on_sim_steps(steps_run, self.params.dt, self.sim_env)

                if sim_acc > self.params.dt * 2.0:
                    sim_acc = 0.0

            dt_spent = time.perf_counter() - t0
            time.sleep(max(0.001, 0.016 - dt_spent))

    def _train_loop(self) -> None:
        perf_start = time.perf_counter()
        perf_steps = 0
        while self._running:
            if not self.training_active:
                time.sleep(0.02)
                continue
            with self._train_lock:
                for env_task in (self.gpu_env, self.gpu_swing_env):
                    states_b = env_task.get_states()
                    act_idx_b, act_u_b = self.agent.select_actions_batch(states_b)
                    next_states_b, rewards_b, dones_b, completed_ep_rewards = env_task.step(act_u_b)

                    self.agent.replay_buffer.add_batch(states_b, act_idx_b, rewards_b, next_states_b, dones_b)
                    self.agent.advance_epsilon(env_task.num_envs)
                    perf_steps += env_task.num_envs

                    if completed_ep_rewards.numel() > 0:
                        self.agent.record_episode_rewards(completed_ep_rewards.tolist())

                self.meta.on_train_cycle(self.gpu_env)
                for _ in range(6):
                    self.agent.train_step()

                now = time.perf_counter()
                if self.meta.maybe_optimize(now, self.agent, self.params, self.device):
                    self._sync_all_envs()

                elapsed = now - perf_start
                if elapsed >= 0.35:
                    inst_sps = perf_steps / elapsed
                    self.steps_per_sec = inst_sps if self.steps_per_sec == 0.0 else (0.65 * self.steps_per_sec + 0.35 * inst_sps)
                    perf_start = now
                    perf_steps = 0
                    self.device_info = self.agent.get_device_info()

            time.sleep(0.0005)

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            now_t = time.perf_counter()
            self._prune_rewards_30s(now_t)
            pts_30s = list(self.rewards_30s)
            if len(pts_30s) > 360:
                stride = max(1, len(pts_30s) // 360)
                sampled = pts_30s[::stride]
                if sampled[-1] != pts_30s[-1]:
                    sampled.append(pts_30s[-1])
                pts_30s = sampled
            rel_t_30s = [round(t - now_t, 3) for t, _ in pts_30s]
            vals_30s = [v for _, v in pts_30s]
            return {
                "physics": {
                    "x": self.sim_env.x,
                    "x_dot": self.sim_env.x_dot,
                    "theta": self.sim_env.theta,
                    "theta_dot": self.sim_env.theta_dot,
                    "angle_deg": self.sim_env.angle_deg,
                    "angle_deg_br": format_angle_deg_br(self.sim_env.angle_deg),
                    "theta_q": self.sim_env.theta_q,
                    "omega_q": self.sim_env.omega_q,
                    "table_angle": self.sim_env.table_angle,
                    "force_motor": self.sim_env.force_motor,
                    "pulley_angle": self.sim_env.pulley_angle,
                    "bumper_hit": self.sim_env.bumper_hit,
                    "bumper_side": self.sim_env.bumper_side,
                    "stable_time": self.sim_env.stable_time,
                    "best_stable_time": self.sim_env.best_stable_time,
                },
                "swing_physics": {
                    "x": self.swing_sim_env.x,
                    "x_dot": self.swing_sim_env.x_dot,
                    "theta": self.swing_sim_env.theta,
                    "theta_dot": self.swing_sim_env.theta_dot,
                    "angle_deg": self.swing_sim_env.angle_deg,
                    "angle_deg_br": format_angle_deg_br(self.swing_sim_env.angle_deg),
                    "table_angle": self.swing_sim_env.table_angle,
                    "force_motor": self.swing_sim_env.force_motor,
                    "pulley_angle": self.swing_sim_env.pulley_angle,
                    "bumper_hit": self.swing_sim_env.bumper_hit,
                    "bumper_side": self.swing_sim_env.bumper_side,
                },
                "swing_stats": {
                    "hold_time": self.swing_hold_time,
                    "successes": self.swing_successes,
                    "attempts": self.swing_attempts,
                    "recent_20_ok": sum(self.swing_recent),
                    "recent_20_n": len(self.swing_recent),
                },
                "recent_metrics": {
                    "stab_pct": (sum(1 for _, ok, _ in self.recent_eval_window if ok) / len(self.recent_eval_window) * 100.0) if self.recent_eval_window else 0.0,
                    "mean_abs_x": (sum(ax for _, _, ax in self.recent_eval_window) / len(self.recent_eval_window)) if self.recent_eval_window else 0.0,
                    "mean_top_s": (
                        (sum(self.recent_top_durations) + (self.sim_env.stable_time if self.sim_env.stable_time > 0.0 else 0.0))
                        / (len(self.recent_top_durations) + (1 if self.sim_env.stable_time > 0.0 else 0))
                    ) if (self.recent_top_durations or self.sim_env.stable_time > 0.0) else 0.0,
                    "current_memory": self.current_memory_name,
                },
                "params": {
                    "sim_speed": self.params.sim_speed,
                    "swing_hold_target": self.params.swing_hold_target,
                    "table_angle_deg": math.degrees(self.sim_env.table_angle),
                    "wind_speed": self.params.wind_speed,
                    "pendulum_mass": self.params.pendulum_mass,
                    "cart_mass": self.params.cart_mass,
                    "pendulum_length": self.params.pendulum_length,
                    "mutation_interval": self.params.mutation_interval,
                    "rail_limit": self.params.rail_limit,
                    "max_force": self.params.max_force,
                    "learning_rate": self.agent.lr,
                    "auto_mutate": self.params.auto_mutate,
                    "mutation_mode": self.params.mutation_mode,
                    "mutate_table": self.params.mutate_table,
                    "table_min_deg": self.params.table_min_deg,
                    "table_max_deg": self.params.table_max_deg,
                    "mutate_wind": self.params.mutate_wind,
                    "wind_min": self.params.wind_min,
                    "wind_max": self.params.wind_max,
                    "mutate_pend_mass": self.params.mutate_pend_mass,
                    "pend_mass_min": self.params.pend_mass_min,
                    "pend_mass_max": self.params.pend_mass_max,
                    "mutate_cart_mass": self.params.mutate_cart_mass,
                    "cart_mass_min": self.params.cart_mass_min,
                    "cart_mass_max": self.params.cart_mass_max,
                    "mutate_length": self.params.mutate_length,
                    "length_min": self.params.length_min,
                    "length_max": self.params.length_max,
                    "rew_center_x_zone": self.params.rew_center_x_zone,
                    **{w_key: getattr(self.params, w_key) for w_key, _, _, _ in REWARD_TABLE_SPECS},
                    **{a_key: bool(getattr(self.params, a_key)) for _, a_key, _, _ in REWARD_TABLE_SPECS},
                    "gravity": FIXED_GRAVITY,
                },
                "status": {
                    "ai_active": self.ai_active,
                    "training_active": self.training_active,
                    "epsilon_auto": self.agent.epsilon,
                    "current_u": self.current_u,
                    "steps_per_sec": self.steps_per_sec if self.training_active else 0.0,
                    "total_env_steps": self.agent.total_env_steps,
                    "total_grad_steps": self.agent.total_grad_steps,
                    "buffer_size": self.agent.replay_buffer.size,
                    "reward_breakdown": self.sim_env.last_reward_breakdown,
                    "device_info": self.device_info,
                },
                "meta": self.meta.status(now_t),
                "series": {
                    "angles_deg": list(self.angles_deg),
                    "ang_vels_q": list(self.ang_vels_q),
                    "positions_x": list(self.positions_x),
                    "rewards_30s_rel_t": rel_t_30s,
                    "rewards_30s": vals_30s,
                },
            }

    def update_control(self, data: Dict[str, object]) -> None:
        with self.lock:
            if "ai_active" in data:
                self.ai_active = bool(data["ai_active"])
            if "training_active" in data:
                self.training_active = bool(data["training_active"])
            if "meta_enabled" in data:
                self.meta.set_enabled(bool(data["meta_enabled"]), time.perf_counter())
            if "flicks_enabled" in data:
                self.meta.set_flicks_enabled(bool(data["flicks_enabled"]))
            if "manual_u" in data:
                self.manual_u = float(np.clip(float(data["manual_u"]), -1.0, 1.0))
            if "mouse_target_rail_s" in data:
                val = data["mouse_target_rail_s"]
                self.mouse_target_rail_s = None if val is None else float(val)

            if "mutation_mode" in data:
                mode_val = str(data["mutation_mode"])
                if mode_val in ("constant", "fixed"):
                    self.params.mutation_mode = mode_val
                    self._sync_all_envs()
            for flag in (
                "auto_mutate",
                "mutate_table",
                "mutate_wind",
                "mutate_pend_mass",
                "mutate_cart_mass",
                "mutate_length",
                *[a_key for _, a_key, _, _ in REWARD_TABLE_SPECS],
            ):
                if flag in data:
                    setattr(self.params, flag, bool(data[flag]))
                    self._sync_all_envs()
            save_app_config(APP_DIR, self.params, self.agent, self.meta, self.sim_env)

    def update_params(self, data: Dict[str, object]) -> None:
        reward_weight_keys = {w_key for w_key, _, _, _ in REWARD_TABLE_SPECS}
        reward_active_keys = {a_key for _, a_key, _, _ in REWARD_TABLE_SPECS}
        with self.lock:
            for key, raw_val in data.items():
                if key in reward_active_keys:
                    setattr(self.params, key, bool(raw_val))
                    self.gpu_env.sync_params(self.params)
                    continue
                val = float(raw_val)
                if key.startswith("meta_"):
                    self.meta.set_param(key, val)
                    continue
                if key in reward_weight_keys:
                    setattr(self.params, key, max(-500.0, min(500.0, val)))
                    self.gpu_env.sync_params(self.params)
                    continue
                if key == "learning_rate":
                    with self._train_lock:
                        self.agent.set_learning_rate(val)
                elif key == "sim_speed":
                    self.params.sim_speed = max(0.05, min(50.0, val))
                elif key == "swing_hold_target":
                    self.params.swing_hold_target = max(0.2, min(30.0, val))
                    self._sync_all_envs()
                elif key == "table_angle_deg":
                    self.sim_env.set_manual_table_angle_deg(val)
                    self.gpu_env.sync_params(self.params)
                elif key == "wind_speed":
                    self.sim_env.set_manual_wind(val)
                    self.gpu_env.sync_params(self.params)
                elif key == "mutation_interval":
                    self.sim_env.set_mutation_interval(val)
                    self.gpu_env.sync_params(self.params)
                elif key == "pendulum_mass":
                    self.params.pendulum_mass = val
                    self.sim_env.pend_mass_timer = max(0.2, self.params.mutation_interval)
                    self.gpu_env.sync_params(self.params)
                elif key == "cart_mass":
                    self.params.cart_mass = val
                    self.sim_env.cart_mass_timer = max(0.2, self.params.mutation_interval)
                    self.gpu_env.sync_params(self.params)
                elif key == "pendulum_length":
                    self.params.pendulum_length = val
                    self.sim_env.length_timer = max(0.2, self.params.mutation_interval)
                    self.gpu_env.sync_params(self.params)
                elif key != "gravity" and hasattr(self.params, key):
                    setattr(self.params, key, val)
                    self._sync_all_envs()
            self._sync_all_envs()
            save_app_config(APP_DIR, self.params, self.agent, self.meta, self.sim_env)

    def perform_action(self, data: Dict[str, object]) -> Dict[str, object]:
        action = str(data.get("action", ""))
        with self.lock:
            if action == "set_down":
                self.sim_env.reset("down")
                return {"ok": True, "message": "Pêndulo para baixo (θ = 180,0°)"}
            if action == "set_up":
                self.sim_env.reset("up")
                return {"ok": True, "message": "Pêndulo posicionado no topo (θ = 0)"}
            if action == "perturb":
                self.sim_env.apply_perturbation(angular_impulse=4.2, cart_impulse=0.9)
                return {"ok": True, "message": "Impulso de perturbação aplicado!"}
            if action == "nudge":
                self.sim_env.apply_mouse_nudge(
                    float(data.get("d_theta_dot", 0.0)),
                    float(data.get("d_x_dot", 0.0)),
                )
                return {"ok": True}
            if action == "reset_training":
                with self._train_lock:
                    self.agent.reset()
                    self.gpu_env.reset_all()
                    self.gpu_swing_env.reset_all()
                self.sim_env.reset("down")
                self.sim_env.best_stable_time = 0.0
                self.swing_sim_env.reset("down")
                self.swing_hold_time = 0.0
                self.swing_elapsed = 0.0
                self.swing_successes = 0
                self.swing_attempts = 0
                self.swing_recent.clear()
                self.recent_eval_window.clear()
                self.recent_top_durations.clear()
                self._prev_stable_time = 0.0
                self.current_memory_name = "Novo (zero)"
                self.rewards_30s.clear()
                if self.meta.enabled:
                    now_t = time.perf_counter()
                    self.meta.set_enabled(False, now_t)
                    self.meta.set_enabled(True, now_t)
                return {"ok": True, "message": "Rede neural, replay buffer e varredura reiniciados!"}
            if action == "save_model":
                custom_path = data.get("path")
                save_path = str(custom_path) if custom_path else next_memory_path(APP_DIR)
                with self._train_lock:
                    self.agent.save_model(save_path)
                if not custom_path:
                    self.current_memory_name = os.path.basename(save_path)
                    save_app_config(APP_DIR, self.params, self.agent, self.meta, self.sim_env)
                return {"ok": True, "message": f"Modelo salvo em {os.path.basename(save_path)}"}
            if action == "load_model":
                load_path = str(data.get("path")) if data.get("path") else latest_memory_path(APP_DIR)
                if not load_path or not os.path.exists(load_path):
                    return {"ok": False, "message": "Nenhum arquivo de modelo salvo ainda."}
                try:
                    with self._train_lock:
                        self.agent.load_model(load_path)
                    if data.get("display_name"):
                        self.current_memory_name = str(data["display_name"])
                    elif not data.get("path"):
                        self.current_memory_name = os.path.basename(load_path)
                    return {"ok": True, "message": f"Modelo carregado de {self.current_memory_name}"}
                except Exception as exc:
                    return {"ok": False, "message": str(exc)}
        return {"ok": False, "message": "Ação desconhecida"}


ENGINE: Optional[SimulationEngine] = None


class RequestHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        assert ENGINE is not None
        if self.path == "/" or self.path.startswith("/?"):
            body = WEB_DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            body = json.dumps(ENGINE.snapshot()).encode("utf-8")
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()

    def do_POST(self) -> None:
        assert ENGINE is not None
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {}

        if self.path == "/api/control":
            ENGINE.update_control(payload)
            resp = {"ok": True}
        elif self.path == "/api/params":
            ENGINE.update_params(payload)
            resp = {"ok": True}
        elif self.path == "/api/action":
            resp = ENGINE.perform_action(payload)
        else:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()
            return

        body = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(host: str = "127.0.0.1", port: int = 8765) -> None:
    global ENGINE
    ENGINE = SimulationEngine()
    server = ThreadingHTTPServer((host, port), RequestHandler)
    print(f"Servidor CUDA ativo em http://{host}:{port} | Device: {ENGINE.device_info}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ENGINE._running = False
        server.server_close()


if __name__ == "__main__":
    main()
