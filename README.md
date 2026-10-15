# Q-Learning Grid World Simulator

**S5 Machine Learning Laboratory Project**  
**Course:** Machine Learning Lab (Semester 5)  
**Topic:** Model-Free Reinforcement Learning — Tabular Q-Learning for Grid World Navigation  
**Architecture:** Python (Flask Backend & Numerical RL Model) + HTML5 / CSS3 / Modern JavaScript (Interactive Frontend)

---

## 1. Project Overview

This project implements a complete, end-to-end **Reinforcement Learning (RL) system** based on the **Tabular Q-Learning algorithm**. An autonomous agent learns to navigate a discrete 5 × 5 Grid World from an initial starting coordinate `(0, 0)` to a designated goal state `(4, 4)` while avoiding static obstacle walls and boundaries.

The project features a **modular Python backend** serving a **clean, modern HTML/CSS/JavaScript web application**. As the agent moves across the grid, the web interface dynamically computes and highlights the underlying Bellman optimality calculations, state-action transitions, reward gains, and convergence curves in real-time.

---

## 2. Theoretical Background & Mathematical Formulation

### 2.1 Markov Decision Process (MDP) Formulation
The Grid World environment is modeled as a discrete Markov Decision Process:
- **State Space (`S`):** 25 grid coordinates `(r, c)` where `r, c ∈ {0, 1, 2, 3, 4}`.
  - Start State: `S_0 = (0, 0)`
  - Goal State: `G = (4, 4)`
  - Static Obstacles: `{(1, 1), (1, 3), (2, 3), (3, 1)}`
- **Action Space (`A`):** 4 discrete cardinal actions:
  - `0`: **UP** (Δr = -1, Δc = 0)
  - `1`: **DOWN** (Δr = +1, Δc = 0)
  - `2`: **LEFT** (Δr = 0, Δc = -1)
  - `3`: **RIGHT** (Δr = 0, Δc = +1)
- **Reward Function — `R(s, a, s')`:**
  - Goal transition (`s' = G`): **+20.0** (Terminal condition)
  - Obstacle / Wall collision: **-5.0** (Agent bounces back to state `s`)
  - Boundary violation: **-5.0** (Agent remains within the grid)
  - Normal step transition: **-1.0** (Step cost to incentivize the shortest path)

### 2.2 The Bellman Optimality Equation
The agent maintains a state-action lookup table `Q(s, a)` of shape `(5, 5, 4)`. All state-action values are initialized to zero: `Q_0(s, a) = 0`.

During training, whenever the agent performs action `a` in state `s`, receives reward `r`, and observes next state `s'`, the Q-value is updated using the Temporal Difference (TD) control rule:

```text
Q_new(s, a) = Q_old(s, a) + α * [ r + γ * max_a' Q(s', a') - Q_old(s, a) ]
```

Where:
- **TD Target:** `r + γ * max_a' Q(s', a')`
- **TD Error (δ):** `TD Target - Q_old(s, a)`
- **Value Gain:** `α * TD Error`
- **α (Learning Rate):** Determines how heavily new information overrides old estimates (default = `0.10`).
- **γ (Discount Factor):** Models the present value of future rewards (default = `0.90`).

### 2.3 Exploration vs. Exploitation (ε-Greedy Policy)
To balance discovery of new routes with exploitation of known high-value paths:
- With probability **ε (epsilon)**, the agent selects an action uniformly at random from `A` (exploration).
- With probability **1 - ε**, the agent greedily selects the action with the maximum estimated Q-value (exploitation):
  ```text
  a* = argmax_a Q(s, a)
  ```
- Epsilon decays multiplicatively after each episode:
  ```text
  ε_next = max(ε_min, ε_current * decay)
  ```
  *(Default parameters: initial ε = 1.0, decay = 0.992, ε_min = 0.01)*

---

## 3. Directory Structure

The project has been kept strictly minimal, modular, and professional:

```text
ML_Pro/
│
├── app.py                      # Flask backend API & web server
├── environment.py              # Discrete Grid World environment class
├── agent.py                    # Tabular Q-Learning agent with Bellman math tracking
├── main.py                     # CLI verification test runner
├── README.md                   # Project documentation
│
├── templates/
│   └── index.html              # Modern, responsive Web dashboard
│
└── static/
    ├── css/
    │   └── style.css           # Clean, responsive UI styling
    └── js/
        └── main.js             # Client-side dynamic state rendering, D-pad & Chart.js
```

---

## 4. How the System Works

```text
+-------------------------------------------------------------------------------+
|                       Browser (HTML5 / CSS3 / JavaScript)                     |
|  - 5x5 Grid Canvas with Agent Token                                           |
|  - Real-Time Dynamic Mathematical Calculations Table                          |
|  - Manual D-Pad & Keyboard Controls (Arrow Keys)                              |
|  - Live Performance Charts (Episode vs Reward, Steps to Goal via Chart.js)    |
+-------------------------------------------------------------------------------+
                                      ▲  │
                         HTTP / JSON  │  │  REST API Calls
                          Responses   │  ▼
+-------------------------------------------------------------------------------+
|                          Flask Server Backend (app.py)                        |
|  - GET  /api/state               : Current environment, agent pos, Q-table    |
|  - POST /api/train               : Runs Q-learning loop for N episodes        |
|  - POST /api/step                : Executes single step & returns math tuple  |
|  - POST /api/reset               : Resets agent to start state (0, 0)         |
|  - GET  /api/hyperparameter-test : Runs 6 parameter sensitivity benchmarks    |
+-------------------------------------------------------------------------------+
                                      ▲  │
                       Model Queries  │  │  State Updates
                                      │  ▼
+-------------------------------------------------------------------------------+
|                      Python RL Core (agent.py, environment.py)                |
|  - Maintains Q-Table: [5 x 5 x 4] NumPy float array                           |
|  - Evaluates TD Target: r + gamma * max Q(s')                                 |
|  - Computes Value Gain: alpha * TD_Error                                      |
+-------------------------------------------------------------------------------+
```

---

## 5. Quick Start & Execution Guide

### Prerequisites
Make sure Python 3.8+ is installed on your machine. Install standard backend dependencies:
```bash
pip install flask flask-cors numpy
```

### Option A: Launch the Web Simulator (Recommended)
1. Open PowerShell or Command Prompt inside `Desktop\ML_Pro`:
   ```bash
   python app.py
   # (or 'py app.py' on Windows)
   ```
2. Open your web browser and navigate to:
   ```text
   http://127.0.0.1:5000
   ```
3. Use the interface to:
   - Click **Train Agent** to train the model with custom α, γ, and episode inputs.
   - Click **Run Optimal Path** to watch the agent execute the shortest path with step-by-step math highlighting.
   - Use the **Manual D-Pad** (or Arrow Keys on your keyboard) to step manually and observe immediate Q-value updates.
   - Click **Run Hyperparameter Experiments** to generate comparative benchmarks across 6 learning configurations.

### Option B: Command-Line Pipeline Test
To quickly verify convergence in the terminal without opening a browser:
```bash
python main.py
# (or 'py main.py' on Windows)
```

---

## 6. Step-by-Step Value Flow Across the Learned Path

When following the learned greedy policy from start `(0, 0)` to goal `(4, 4)`, the agent executes the minimum theoretical route in **8 steps**:

```text
(0, 0) ──RIGHT──> (0, 1) ──RIGHT──> (0, 2) ──DOWN──> (1, 2)
                                                         │
                                                       DOWN
                                                         │
(4, 4) <──RIGHT── (4, 3) <──DOWN── (3, 3) <──RIGHT── (2, 2)
 [GOAL]
```

### Step-by-Step Mathematical Calculation Table:

| Step | State (s) | Action (a) | Next State (s') | Immediate Reward (r) | Best Future max Q(s') | Bellman Target: r + 0.90 * max Q(s') | Q-Value Q(s, a) | Cumulative Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `(0, 0)` | `RIGHT` | `(0, 1)` | -1.0 | 5.94 | -1.0 + (0.90 * 5.94) = **4.35** | **4.35** | -1.0 |
| **2** | `(0, 1)` | `RIGHT` | `(0, 2)` | -1.0 | 7.71 | -1.0 + (0.90 * 7.71) = **5.94** | **5.94** | -2.0 |
| **3** | `(0, 2)` | `DOWN`  | `(1, 2)` | -1.0 | 9.68 | -1.0 + (0.90 * 9.68) = **7.71** | **7.71** | -3.0 |
| **4** | `(1, 2)` | `DOWN`  | `(2, 2)` | -1.0 | 11.87 | -1.0 + (0.90 * 11.87) = **9.68** | **9.68** | -4.0 |
| **5** | `(2, 2)` | `DOWN`  | `(3, 2)` | -1.0 | 14.30 | -1.0 + (0.90 * 14.30) = **11.87** | **11.87** | -5.0 |
| **6** | `(3, 2)` | `RIGHT` | `(3, 3)` | -1.0 | 17.00 | -1.0 + (0.90 * 17.00) = **14.30** | **14.30** | -6.0 |
| **7** | `(3, 3)` | `DOWN`  | `(4, 3)` | -1.0 | 20.00 | -1.0 + (0.90 * 20.00) = **17.00** | **17.00** | -7.0 |
| **8** | `(4, 3)` | `RIGHT` | `(4, 4)` | **+20.0** | 0.00 (Goal) | +20.0 + (0.90 * 0.00) = **20.00** | **20.00** | **+13.0** |

- **Minimum Steps Taken:** 8 steps
- **Optimal Cumulative Reward:** +13.0 (7 transitions at -1.0 + 1 goal transition at +20.0)
- **Obstacle Collisions:** 0

---

## 7. Hyperparameter Analysis & Observations

The simulator provides a built-in benchmark evaluating 6 different hyperparameter configurations over 250 episodes:

| Configuration | Learning Rate (α) | Discount Factor (γ) | Epsilon Decay | Final Avg Reward | Avg Steps to Goal | Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **α = 0.01 (Slow)** | 0.01 | 0.90 | 0.995 | +6.88 | 12.4 | Converged slowly |
| **α = 0.10 (Standard)** | 0.10 | 0.90 | 0.995 | +10.44 | 9.5 | Stable convergence |
| **α = 0.50 (Aggressive)** | 0.50 | 0.90 | 0.995 | +11.00 | 9.2 | Rapid learning |
| **γ = 0.50 (Myopic)** | 0.10 | 0.50 | 0.995 | +10.20 | 10.1 | Short horizon bias |
| **γ = 0.95 (Far-sighted)**| 0.10 | 0.95 | 0.995 | +10.46 | 9.5 | Strong reward propagation |
| **Fast Decay (0.98)** | 0.10 | 0.90 | 0.980 | **+12.96** | **8.0** | **Optimal (Theoretical Minimum)** |

### Key Experimental Insights:
1. **Impact of Learning Rate (α):** A small learning rate (α = 0.01) updates Q-values too conservatively, requiring more training episodes to propagate goal rewards backward. α = 0.10 provides balanced convergence.
2. **Impact of Discount Factor (γ):** Higher values (γ = 0.90 - 0.95) are essential for 2D maze navigation because the +20.0 goal signal must propagate across at least 8 transitions to reach the starting state `(0, 0)`.
3. **Exploration Schedule (ε):** Faster decay schedules that transition smoothly from broad random search to strict exploitation yield optimal 8-step policies with the highest cumulative reward.

---

## 8. Team Members (Max 6 Members)
1. Member 1 (Roll No / ID)
2. Member 2 (Roll No / ID)
3. Member 3 (Roll No / ID)
4. Member 4 (Roll No / ID)
5. Member 5 (Roll No / ID)
6. Member 6 (Roll No / ID)

---

## 9. How to Clone and Run on Any Device (GitHub Setup Guide)

Follow these step-by-step terminal instructions to clone the repository, set up a dedicated working folder, install prerequisites, and launch the simulator on another computer (Windows, macOS, or Linux).

### Step 1: Open Terminal or Command Prompt
- **Windows:** Press `Win + R`, type `powershell` (or `cmd`), and press Enter.
- **macOS:** Press `Cmd + Space`, type `Terminal`, and press Enter.
- **Linux:** Press `Ctrl + Alt + T`.

### Step 2: Choose a Directory and Clone the Repository
Navigate to your desired directory (for example, your Desktop or Projects directory) and clone the repository:

```bash
# Navigate to Desktop (or your preferred workspace)
cd Desktop

# Clone the repository into a dedicated folder
git clone https://github.com/Vibhathcross/Q-learning-Game.git

# Enter the cloned project folder
cd Q-learning-Game
```

### Step 3: Set Up a Python Virtual Environment (Recommended)
Creating an isolated virtual environment prevents library conflicts across projects:

- **On Windows (PowerShell / CMD):**
  ```bash
  # Create a virtual environment named 'venv'
  python -m venv venv

  # Activate the virtual environment
  .\venv\Scripts\activate
  ```

- **On macOS / Linux:**
  ```bash
  # Create a virtual environment named 'venv'
  python3 -m venv venv

  # Activate the virtual environment
  source venv/bin/activate
  ```

### Step 4: Install Required Dependencies
Install the required packages using `pip`:
```bash
pip install flask flask-cors numpy
```

### Step 5: Launch the Web Simulator
Start the Flask application server:

- **Windows:**
  ```bash
  python app.py
  # (or 'py app.py' if using the Python Launcher)
  ```

- **macOS / Linux:**
  ```bash
  python3 app.py
  ```

Once running, you will see output similar to:
```text
=======================================================
  Q-Learning Grid World Server Starting...
  Open your browser at: http://127.0.0.1:5000
=======================================================
```

Open your browser and visit:
```text
http://127.0.0.1:5000
```

### Step 6: Command-Line Pipeline Verification (Optional)
To test model convergence directly in your terminal without a web browser:
```bash
python main.py
# (or 'py main.py' on Windows / 'python3 main.py' on macOS & Linux)
```
This runs the full 350-episode training loop and prints the learned 8-step optimal trajectory with cumulative rewards.
