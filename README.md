# Simulador de Pêndulo Invertido em Carrinho com Deep RL (PyTorch + CUDA)

Simulador físico em tempo real de um **Pêndulo Invertido sobre Carrinho** com aceleração em **GPU local (PyTorch/CUDA)**, controle manual, piloto automático por **Dueling Double DQN**, **Meta IA** para otimização periódica de recompensas + petelecos aleatórios, e interfaces **Desktop (Pygame)** e **Web (`http://127.0.0.1:8765`)**.

---

## Estrutura do Projeto

| Arquivo | Descrição |
| :--- | :--- |
| `main.py` | Ponto de entrada da interface Desktop (Pygame 60+ FPS) com loop de treino na GPU. |
| `web_server.py` | Servidor HTTP local (`127.0.0.1:8765`) + API REST (`/api/state`, `/api/control`, `/api/params`, `/api/action`) e dashboard web. |
| `physics.py` | Motor físico RK4 (Lagrange/Euler-Newton) em CPU (`CartPolePhysics`) e em lote na GPU (`BatchedCartPoleEnv`, 256 ambientes paralelos). |
| `rl_agent.py` | Agente **Dueling Double DQN** (`DuelingQNetwork`, `GPUReplayBuffer`, exploração cíclica `3%–15%`, save/load `.pth`). |
| `meta_ai.py` | **Meta IA**: avalia a política em 128 ambientes na GPU, ajusta os pesos da tabela de recompensas, reduz o passo `σ` conforme a estabilidade sobe, aplica petelecos leves e reposiciona `(x, θ)` periodicamente. |
| `ui.py` | Interface gráfica Pygame com abas **Controle**, **Auto Mut.**, **Recompensa** e **Meta IA**, além dos gráficos em tempo real. |
| `requirements.txt` | Dependências Python (`torch`, `numpy`, `pygame`). |
| `Iniciar_Simulador.bat` | Atalho para iniciar o simulador no Windows usando o ambiente virtual `.venv`. |

---

## Física e Entradas da IA (`STATE_DIM = 4`)

- **Dinâmica realista:** integração RK4 com atrito viscoso/Coulomb, motor de passo com polia/correia, batentes elásticos nos fins de curso, base móvel inclinável (`-45°` a `+45°`), vento horizontal (`-15` a `+15 m/s`) e gravidade fixa em `9,81 m/s²`.
- **Vetor de observação da IA (4 entradas):**
  1. `angle_360 / 360.0` — ângulo do pêndulo de `0,0°` a `359,9°` (quantizado em `0,1°`);
  2. `x` — posição atual no eixo X (m);
  3. `rail_limit` — limite do trilho de cada lado (m);
  4. `table_deg_q / 45.0` — inclinação da mesa quantizada em `0,3°`.

---

## Tabela de Recompensas (11 itens configuráveis `+/-` + checkbox)

Cada item pode ser ativado/desativado individualmente e receber pesos positivos ou negativos (`[-500, +500]`):

1. `rew_center_x` — Próximo do `x=0`
2. `rew_top` — Pêndulo no topo
3. `rew_stability` — Estabilidade
4. `rew_x_dist` — Penalidade por distância do `x=0` (`0,6·|x_norm| + 0,4·x_norm²`)
5. `rew_endstop` — Fim de curso
6. `rew_up_time_acc` — Acumulador de tempo para cima
7. `rew_fall` — Queda do pêndulo
8. `rew_rise` — Subida do pêndulo
9. `rew_bottom` — Pêndulo para baixo
10. `rew_swingup` — Swing up
11. `rew_smooth` — Movimento brusco `(u - prev_u)²`

---

## Como Instalar e Executar Localmente (Windows + GPU NVIDIA)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

- **Interface Desktop (Pygame):**
  ```powershell
  .\.venv\Scripts\python.exe main.py
  ```
- **Interface Web (`http://127.0.0.1:8765`):**
  ```powershell
  .\.venv\Scripts\python.exe web_server.py
  ```
