# Q-Learning Grid World Simulator

**S5 Machine Learning Laboratory Project**  
**Course:** Machine Learning Lab (Semester 5)  
**Topic:** Model-Free Reinforcement Learning — Tabular Q-Learning for Grid World Navigation  
**Architecture:** Python (Flask Backend & Numerical RL Model) + HTML5 / CSS3 / Modern JavaScript (Interactive Frontend)  
**Repository:** [https://github.com/Vibhathcross/Q-learning-Game](https://github.com/Vibhathcross/Q-learning-Game)

---
---

# Part 1: Introduction, Problem Statement and Theoretical Background

---

## 1.1 Introduction

Reinforcement Learning (RL) is a paradigm of machine learning where an autonomous agent learns optimal decision-making by interacting with an environment through trial and error. Unlike supervised learning, the agent does not receive labeled input-output pairs. Instead, it discovers which actions yield the highest cumulative reward over time by exploring the environment, observing feedback (rewards or penalties), and gradually refining its behavior.

This project implements a **Tabular Q-Learning** agent — a model-free, off-policy RL algorithm — trained to navigate a 2D grid-based environment. The agent starts at a fixed position, must reach a goal position, and must learn to avoid obstacle walls, all without any prior knowledge of the environment layout. The entire learning process is driven by numerical reward signals.

## 1.2 Problem Statement

Design and implement a reinforcement learning system in which an intelligent agent learns to navigate through a **5 × 5 Grid World** from a designated start state to a goal state. The system must:

1. Define the states, possible actions, reward structure, and Q-table, and initialize the Q-values.
2. Implement the Q-Learning algorithm with suitable learning rate (α), discount factor (γ), and ε-greedy exploration.
3. Train the agent over multiple episodes, updating Q-values based on received rewards.
4. After training, extract the optimal policy/path from the Q-table.
5. Provide visualizations of the grid, learned path, training performance, and hyperparameter sensitivity.

## 1.3 Theoretical Background

### 1.3.1 Markov Decision Process (MDP)

The Grid World is formulated as a **Markov Decision Process (MDP)**, defined by the tuple `(S, A, R, T, γ)`:

- **State Space (`S`):** 25 discrete grid coordinates `(row, col)` where `row, col ∈ {0, 1, 2, 3, 4}`.

  ```text
  Grid Layout:
  
  (0,0) (0,1) (0,2) (0,3) (0,4)       S = Start (0,0)
  (1,0) [WALL] (1,2) [WALL] (1,4)     G = Goal  (4,4)
  (2,0) (2,1) (2,2) [WALL] (2,4)      ■ = Wall / Obstacle
  (3,0) [WALL] (3,2) (3,3) (3,4)
  (4,0) (4,1) (4,2) (4,3) (4,4)
  ```

  - **Start State:** `(0, 0)` — Top-left corner
  - **Goal State:** `(4, 4)` — Bottom-right corner
  - **Obstacle Cells:** `(1, 1)`, `(1, 3)`, `(2, 3)`, `(3, 1)` — Impassable wall blocks

- **Action Space (`A`):** 4 discrete cardinal movements:
  | Action ID | Name | Direction Vector |
  | :---: | :---: | :---: |
  | `0` | **UP** | (Δrow = -1, Δcol = 0) |
  | `1` | **DOWN** | (Δrow = +1, Δcol = 0) |
  | `2` | **LEFT** | (Δrow = 0, Δcol = -1) |
  | `3` | **RIGHT** | (Δrow = 0, Δcol = +1) |

- **Reward Function — `R(s, a, s')`:**
  | Transition Type | Reward | Effect |
  | :--- | :---: | :--- |
  | Agent reaches Goal `(4, 4)` | **+20.0** | Episode terminates (success) |
  | Agent collides with Obstacle | **-5.0** | Agent stays in current cell |
  | Agent hits grid boundary | **-5.0** | Agent stays in current cell |
  | Agent moves to empty cell | **-1.0** | Step cost (shortest path incentive) |

- **Transition Function (`T`):** Deterministic. The environment always moves the agent to the intended cell if it is valid and empty.

- **Discount Factor (`γ`):** Default = `0.90`. Controls how much future rewards are valued relative to immediate rewards.

### 1.3.2 Q-Learning Algorithm

Q-Learning is a **model-free**, **off-policy**, **Temporal Difference (TD)** control algorithm. The agent maintains a lookup table `Q(s, a)` mapping every state-action pair to an estimated value.

**Core Update Rule (Bellman Optimality Equation):**
```text
Q_new(s, a) = Q_old(s, a) + α * [ r + γ * max_a' Q(s', a') - Q_old(s, a) ]
```

The components of this update are:

| Term | Formula | Meaning |
| :--- | :--- | :--- |
| **Q_old(s, a)** | Current Q-table entry | The agent's current estimate for taking action `a` in state `s` |
| **r** | Immediate reward | The numerical feedback from the environment for this step |
| **max Q(s', a')** | Best future estimate | The highest Q-value available from the next state `s'` |
| **TD Target** | `r + γ * max Q(s', a')` | What the value *should* be according to the new experience |
| **TD Error (δ)** | `TD Target - Q_old` | The "surprise" — how far off the old estimate was |
| **Value Gain** | `α * TD Error` | The actual correction applied to the Q-table |
| **α (Learning Rate)** | Default = `0.10` | How aggressively new information overrides old estimates |
| **γ (Discount Factor)** | Default = `0.90` | How much weight is given to future rewards vs. immediate ones |

### 1.3.3 Exploration vs. Exploitation (ε-Greedy Policy)

During training, the agent must balance two competing needs:
- **Exploration:** Try random actions to discover new, potentially better routes.
- **Exploitation:** Use the current Q-table to take the best known action.

This is managed through the **ε-greedy strategy:**
- With probability **ε**: Choose a **random** action (explore).
- With probability **1 - ε**: Choose the **best known** action (exploit):
  ```text
  a* = argmax_a Q(s, a)
  ```

**Epsilon Decay Schedule:**
```text
ε_next = max(ε_min, ε_current * decay_factor)
```

| Phase | Episode Range | Epsilon (ε) | Behavior |
| :--- | :---: | :---: | :--- |
| Early Training | 1 — 80 | 1.00 → 0.50 | Mostly random exploration |
| Mid Training | 80 — 200 | 0.50 → 0.20 | Balanced exploration + exploitation |
| Late Training | 200 — 350 | 0.20 → 0.06 | Mostly greedy exploitation |

*(Default parameters: ε_initial = 1.0, decay = 0.992, ε_min = 0.01)*

---
---

# Part 2: System Design and Methodology

---

## 2.1 Project Directory Structure

```text
ML_Pro/
│
├── app.py                      # Flask REST API backend & web server
├── environment.py              # Discrete 5x5 Grid World MDP environment
├── agent.py                    # Tabular Q-Learning agent with Bellman math tracking
├── main.py                     # Command-line test runner & convergence validator
├── README.md                   # Project documentation (this file)
│
├── templates/
│   └── index.html              # Responsive single-page web application
│
└── static/
    ├── css/
    │   └── style.css           # Clean, modern UI stylesheet
    └── js/
        └── main.js             # Dynamic grid rendering, API calls, charts & D-pad
```

## 2.2 System Architecture

The system follows a **three-tier client-server architecture**:

```text
+-------------------------------------------------------------------------------+
|                 TIER 1: FRONTEND (Browser — HTML5 / CSS3 / JS)                |
|  - Interactive 5x5 Grid Canvas with animated agent token                       |
|  - Real-time mathematical calculations table (highlights active step)          |
|  - Manual D-pad buttons + keyboard arrow key controls                          |
|  - Live Chart.js graphs for training reward & steps convergence                |
|  - Hyperparameter configuration input fields                                   |
+-------------------------------------------------------------------------------+
                                      ▲  │
                         HTTP / JSON  │  │  REST API Calls
                          Responses   │  ▼
+-------------------------------------------------------------------------------+
|                 TIER 2: BACKEND (Flask — Python app.py)                        |
|  - GET  /api/state               : Current grid state, agent pos, Q-table      |
|  - POST /api/train               : Run Q-learning for N episodes               |
|  - POST /api/step                : Execute one action & return math breakdown  |
|  - POST /api/reset               : Reset agent to start (0, 0)                |
|  - GET  /api/hyperparameter-test : Run 6 parameter sensitivity experiments     |
+-------------------------------------------------------------------------------+
                                      ▲  │
                       Model Queries  │  │  State Transitions
                                      │  ▼
+-------------------------------------------------------------------------------+
|                 TIER 3: RL CORE (agent.py, environment.py)                     |
|  - NumPy Q-Table: [5 x 5 x 4] float array                                     |
|  - Bellman update: computes TD Target, TD Error, Value Gain per step           |
|  - ε-greedy action selection with multiplicative decay                          |
|  - Policy grid extraction for frontend visualization                            |
+-------------------------------------------------------------------------------+
```

## 2.3 Detailed Step-by-Step Algorithm

The Q-Learning pipeline executes through **five concrete stages**:

```text
+---------------------------------------------------------------------------------+
|                       STAGE 1: ENVIRONMENT INITIALIZATION                       |
|  - Create 5x5 coordinate matrix                                                |
|  - Place Agent at (0, 0), Goal at (4, 4)                                       |
|  - Define Walls at (1,1), (1,3), (2,3), (3,1)                                  |
|  - Initialize Q-Table: 5 x 5 x 4 array of all zeros                            |
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                       STAGE 2: EPISODE EXECUTION LOOP                           |
|  Repeat for each Episode (1 to 350):                                            |
|    a. Reset agent position to start state s = (0, 0)                            |
|    b. For each step within the episode (max 100 steps):                         |
|       - Sample random number p between 0.0 and 1.0                              |
|       - If p < ε: Choose random action a ∈ {UP, DOWN, LEFT, RIGHT}              |
|       - If p ≥ ε: Choose greedy action a* = argmax Q(s, a)                      |
|       - Execute action 'a' in environment                                       |
|       - Environment returns: next_state s', reward r, done flag                 |
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                       STAGE 3: BELLMAN VALUE UPDATE                              |
|    For each transition tuple (s, a, r, s'):                                     |
|      Step 1: Read current estimate:  Q_old = Q[s, a]                            |
|      Step 2: Find best downstream:   best_next = max Q[s', a'] (0 if done)      |
|      Step 3: Compute TD Target    =  r + γ * best_next                          |
|      Step 4: Compute TD Error     =  TD_Target - Q_old                          |
|      Step 5: Compute Value Gain   =  α * TD_Error                               |
|      Step 6: Update Q-Table       =  Q_old + Value_Gain                         |
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                       STAGE 4: EPSILON DECAY                                    |
|  - After each episode, multiply ε by decay factor (0.992)                       |
|  - Early episodes: Wide random exploration (ε > 0.50)                           |
|  - Mid episodes:   Path consolidation toward goal (ε ~ 0.20)                   |
|  - Late episodes:  Pure exploitation of shortest route (ε ≤ 0.06)              |
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                       STAGE 5: OPTIMAL POLICY EXTRACTION                        |
|  - Freeze Q-table (set ε = 0, no more exploration)                             |
|  - At each state, select optimal action: π*(s) = argmax_a Q(s, a)               |
|  - Trace greedy trajectory from (0, 0) to (4, 4)                               |
|  - Result: 8-step collision-free path with cumulative reward = +13.0            |
+---------------------------------------------------------------------------------+
```

### 2.4 Numerical Trace: How a Single Q-Value Updates

**Scenario:** Episode 1, Step 1. Agent at `(0, 0)`, Q-table is all zeros. Agent steps `RIGHT`:

| Calculation Step | Formula | Value |
| :--- | :--- | :--- |
| 1. Old Value | `Q_old((0,0), RIGHT)` | `0.00` |
| 2. Immediate Reward | `r` (step cost) | `-1.00` |
| 3. Best Future Estimate | `max Q((0,1), all actions)` | `0.00` |
| 4. TD Target | `r + γ * max Q = -1.0 + (0.90 * 0.00)` | `-1.00` |
| 5. TD Error (Surprise) | `Target - Q_old = -1.00 - 0.00` | `-1.00` |
| 6. Value Gain | `α * Error = 0.10 * (-1.00)` | `-0.10` |
| 7. New Q-Value | `Q_old + Gain = 0.00 + (-0.10)` | **`-0.10`** |

As training episodes progress and the agent discovers the goal `(4, 4)` yielding `+20.0`, that high reward propagates backward cell-by-cell through the Bellman equation, eventually creating a smooth value gradient from start to goal.

---
---

# Part 3: Application Interface and Functionality

---

## 3.1 Web Application Features

The web interface (served at `http://127.0.0.1:5000`) provides the following interactive components:

### 3.1.1 Interactive Grid World Arena
- **5 × 5 visual grid** displaying labeled cells: `START`, `GOAL (+20)`, `WALL`, and empty cells.
- **Animated agent token** (red circle) moves across the grid in real-time.
- **Policy arrows** (↑ ↓ ← →) displayed on each cell showing the learned best action.
- **Q-value labels** shown on each cell showing the maximum estimated value.
- **Red trail path** marks previously visited cells as the agent walks.

### 3.1.2 Hyperparameter Configuration Panel
- **Learning Rate (α):** Editable input field (default: 0.10).
- **Discount Factor (γ):** Editable input field (default: 0.90).
- **Training Episodes:** Editable input field (default: 350).
- Changes are sent to the Python backend on the next training run.

### 3.1.3 Control Buttons
| Button | Function |
| :--- | :--- |
| **Train Agent** | Sends hyperparameters to backend, runs full training loop, updates grid |
| **Run Optimal Path** | Animates the learned 8-step shortest path with synchronized math table |
| **Single Step** | Executes one greedy step from the agent's current position |
| **Toggle Arrows / Q-Values** | Shows or hides policy arrows and Q-value labels on the grid |
| **Reset Agent** | Moves agent back to `(0, 0)` without clearing the learned Q-table |
| **Run Hyperparameter Experiments** | Triggers 6 sensitivity benchmarks on the Python backend |

### 3.1.4 Manual Navigation D-Pad
- On-screen directional buttons (↑ ↓ ← →) for manually stepping the agent.
- **Keyboard arrow keys** also supported — allows testing any custom path.
- Each manual step triggers a backend API call and updates the calculation display.

### 3.1.5 Real-Time Calculation Display
- **Live calculation box** updates on every step showing:
  ```text
  Step 3: (0, 2) → (1, 2) via DOWN [Normal step]
  Target = -1.0 + (0.90 × 9.68) = 7.71 • TD Error = 0.00 • Q_new = 7.71
  ```

### 3.1.6 Dynamic Step-by-Step Calculations Table
A full table showing every step of the optimal path with columns:
- Step number, State `(s)`, Action taken, Next State `(s')`, Immediate Reward, Best Future Q-value, Bellman Target formula and result, Current Q-value, and Cumulative score.
- **Active step is highlighted in yellow** as the agent walks through the grid.

### 3.1.7 Performance Charts
- **Cumulative Reward per Episode** — Line chart showing how episode reward improves during training.
- **Steps to Goal per Episode** — Line chart showing how the number of steps decreases with training.
- Charts are rendered using **Chart.js** and update dynamically after each training run.

### 3.1.8 Hyperparameter Comparison Table
- Displays results from 6 distinct hyperparameter configurations.
- Columns: Configuration name, α, γ, Decay rate, Final Avg Reward, Avg Steps, and Outcome status.

## 3.2 REST API Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | GET | Serves the main web application (index.html) |
| `/api/state` | GET | Returns current grid metadata, agent position, epsilon, and policy grid |
| `/api/train` | POST | Trains agent with provided `{episodes, alpha, gamma}` and returns training history + optimal steps |
| `/api/step` | POST | Executes one step (greedy or manual action), returns full Bellman math breakdown |
| `/api/reset` | POST | Resets agent to `(0, 0)` without clearing learned Q-table |
| `/api/hyperparameter-test` | GET | Runs 6 benchmark configurations and returns comparative results |

---
---

# Part 4: Results and Performance Evaluation

---

## 4.1 Learned Optimal Path

After 350 episodes of training, the agent converges to the **minimum theoretical shortest path** of exactly **8 steps**:

```text
(0, 0) ──RIGHT──> (0, 1) ──RIGHT──> (0, 2) ──DOWN──> (1, 2)
                                                         │
                                                       DOWN
                                                         │
(4, 4) <──RIGHT── (4, 3) <──DOWN── (3, 3) <──RIGHT── (2, 2)
 [GOAL]
```

## 4.2 Step-by-Step Mathematical Calculation Table

| Step | State (s) | Action (a) | Next State (s') | Reward (r) | Best Future max Q(s') | Bellman Target: r + 0.90 * max Q(s') | Q-Value Q(s, a) | Cumulative Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `(0, 0)` | `RIGHT` | `(0, 1)` | -1.0 | 5.94 | -1.0 + (0.90 * 5.94) = **4.35** | **4.35** | -1.0 |
| **2** | `(0, 1)` | `RIGHT` | `(0, 2)` | -1.0 | 7.71 | -1.0 + (0.90 * 7.71) = **5.94** | **5.94** | -2.0 |
| **3** | `(0, 2)` | `DOWN`  | `(1, 2)` | -1.0 | 9.68 | -1.0 + (0.90 * 9.68) = **7.71** | **7.71** | -3.0 |
| **4** | `(1, 2)` | `DOWN`  | `(2, 2)` | -1.0 | 11.87 | -1.0 + (0.90 * 11.87) = **9.68** | **9.68** | -4.0 |
| **5** | `(2, 2)` | `DOWN`  | `(3, 2)` | -1.0 | 14.30 | -1.0 + (0.90 * 14.30) = **11.87** | **11.87** | -5.0 |
| **6** | `(3, 2)` | `RIGHT` | `(3, 3)` | -1.0 | 17.00 | -1.0 + (0.90 * 17.00) = **14.30** | **14.30** | -6.0 |
| **7** | `(3, 3)` | `DOWN`  | `(4, 3)` | -1.0 | 20.00 | -1.0 + (0.90 * 20.00) = **17.00** | **17.00** | -7.0 |
| **8** | `(4, 3)` | `RIGHT` | `(4, 4)` | **+20.0** | 0.00 (Goal) | +20.0 + (0.90 * 0.00) = **20.00** | **20.00** | **+13.0** |

## 4.3 Performance Metrics Summary

| Metric | Value |
| :--- | :--- |
| **Steps to Goal (Trained Agent)** | **8 steps** (Theoretical minimum) |
| **Optimal Cumulative Reward** | **+13.0** (7 × -1.0 + 1 × +20.0) |
| **Obstacle Collisions** | **0** (100% collision-free route) |
| **Success Rate (last 50 episodes)** | **100%** |
| **Convergence Point** | **~Episode 180** (ε < 0.20) |
| **Training Episodes** | 350 |
| **Final Exploration Rate (ε)** | 0.06 |

## 4.4 Value Propagation Observation

The Q-values form a smooth gradient from start to goal, showing how the +20.0 goal reward propagates backward:

```text
Cell:     (0,0) → (0,1) → (0,2) → (1,2) → (2,2) → (3,2) → (3,3) → (4,3) → (4,4)
Q-value:  4.35    5.94    7.71    9.68    11.87   14.30   17.00   20.00   [GOAL]
```

Each upstream cell holds exactly `90%` of its downstream neighbor's value minus the `-1.0` step penalty. This is the Bellman equation at equilibrium.

## 4.5 Training Convergence Behavior

| Training Phase | Episodes | Agent Behavior |
| :--- | :---: | :--- |
| **Random Exploration** | 1 — 80 | Agent moves randomly, 30-80 steps per episode, frequent wall collisions |
| **Path Discovery** | 80 — 150 | Agent finds the goal more consistently, steps drop below 15 |
| **Policy Consolidation** | 150 — 250 | Agent locks into near-optimal routes, reward stabilizes near +10 |
| **Full Convergence** | 250 — 350 | Agent consistently executes 8-step optimal path, reward = +13.0 |

---
---

# Part 5: Hyperparameter Analysis, Discussion and Conclusion

---

## 5.1 Hyperparameter Sensitivity Experiments

The built-in benchmark evaluates 6 hyperparameter configurations over 250 episodes each:

| Configuration | Learning Rate (α) | Discount Factor (γ) | Epsilon Decay | Final Avg Reward | Avg Steps to Goal | Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **α = 0.01 (Slow)** | 0.01 | 0.90 | 0.995 | +6.88 | 12.4 | Converged slowly |
| **α = 0.10 (Standard)** | 0.10 | 0.90 | 0.995 | +10.44 | 9.5 | Stable convergence |
| **α = 0.50 (Aggressive)** | 0.50 | 0.90 | 0.995 | +11.00 | 9.2 | Rapid learning |
| **γ = 0.50 (Myopic)** | 0.10 | 0.50 | 0.995 | +10.20 | 10.1 | Short horizon bias |
| **γ = 0.95 (Far-sighted)** | 0.10 | 0.95 | 0.995 | +10.46 | 9.5 | Strong propagation |
| **Fast Decay (0.98)** | 0.10 | 0.90 | 0.980 | **+12.96** | **8.0** | **Optimal** |

## 5.2 Discussion of Results

### Impact of Learning Rate (α)
- **α = 0.01 (Too Slow):** Q-values update by tiny increments each step. The +20.0 goal reward needs hundreds of episodes to propagate across 8 cells to `(0, 0)`. Final average steps remain at 12.4 — the agent hasn't fully converged.
- **α = 0.10 (Balanced):** Each step updates Q-values meaningfully but not excessively. Converges to near-optimal within 200 episodes.
- **α = 0.50 (Aggressive):** Learns very fast — fewer episodes needed. However, in more complex environments, such a high rate can cause instability and oscillation.

### Impact of Discount Factor (γ)
- **γ = 0.50 (Myopic):** The agent "sees" only 2-3 steps ahead. Since the goal is 8 steps away, the +20.0 signal arrives at `(0, 0)` heavily attenuated: `20 × 0.50^8 ≈ 0.08`. The start cell barely "knows" the goal exists.
- **γ = 0.90 (Standard):** Goal value at start: `20 × 0.90^8 ≈ 8.6`. The signal is strong enough to guide the agent across 8 transitions.
- **γ = 0.95 (Far-sighted):** Even stronger propagation: `20 × 0.95^8 ≈ 13.2`. Works well here but can slow convergence in very large grids due to overvaluation of distant states.

### Impact of Exploration Decay (ε)
- **Slow Decay (0.995):** The agent continues exploring randomly for longer, which can waste many episodes on suboptimal paths after good routes are already discovered.
- **Fast Decay (0.98):** The agent transitions quickly from exploration to exploitation. After discovering the goal early, it rapidly locks into the shortest path. This produces the best final performance (+12.96 avg reward, exactly 8 steps) — matching the theoretical minimum.

### Best Configuration
For this 5 × 5 Grid World with 4 obstacles:
```text
α = 0.10,  γ = 0.90,  ε_decay = 0.98
```
This achieves the **theoretical optimal** of 8 steps with a cumulative reward of +13.0.

## 5.3 Conclusion

The Tabular Q-Learning algorithm successfully solved the Grid World navigation problem. Key findings:

1. **The agent converged to the theoretical shortest path** of 8 steps, achieving the maximum possible cumulative reward of +13.0 with zero obstacle collisions.

2. **The Bellman equation creates a natural value gradient** across the grid — each cell's Q-value reflects its distance to the goal, forming a smooth slope that guides the agent optimally.

3. **Hyperparameter tuning is critical.** Learning rate controls convergence speed, discount factor determines how far the goal signal propagates, and exploration decay controls how quickly the agent transitions from random search to optimal exploitation.

4. **The interactive web interface** makes the entire learning process transparent — every mathematical calculation, transition, and value update is dynamically computed and displayed, making Q-Learning intuitive to understand and demonstrate.

---
---

# Appendix

---

## Team Members (Max 6 Members)
1. Member 1 (Roll No / ID)
2. Member 2 (Roll No / ID)
3. Member 3 (Roll No / ID)
4. Member 4 (Roll No / ID)
5. Member 5 (Roll No / ID)
6. Member 6 (Roll No / ID)

---

## How to Clone and Run on Any Device (GitHub Setup Guide)

Follow these step-by-step terminal instructions to clone the repository, set up a dedicated working folder, install prerequisites, and launch the simulator on another computer (Windows, macOS, or Linux).

### Step 1: Open Terminal or Command Prompt
- **Windows:** Press `Win + R`, type `powershell` (or `cmd`), and press Enter.
- **macOS:** Press `Cmd + Space`, type `Terminal`, and press Enter.
- **Linux:** Press `Ctrl + Alt + T`.

### Step 2: Choose a Directory and Clone the Repository

```bash
# Navigate to Desktop (or your preferred workspace)
cd Desktop

# Clone the repository into a dedicated folder
git clone https://github.com/Vibhathcross/Q-learning-Game.git

# Enter the cloned project folder
cd Q-learning-Game
```

### Step 3: Set Up a Python Virtual Environment (Recommended)

- **On Windows (PowerShell / CMD):**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```

- **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 4: Install Required Dependencies
```bash
pip install flask flask-cors numpy
```

### Step 5: Launch the Web Simulator

- **Windows:**
  ```bash
  python app.py
  # (or 'py app.py' if using the Python Launcher)
  ```

- **macOS / Linux:**
  ```bash
  python3 app.py
  ```

Once running, you will see:
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
```bash
python main.py
# (or 'py main.py' on Windows / 'python3 main.py' on macOS & Linux)
```
This runs the full 350-episode training loop and prints the learned 8-step optimal trajectory with cumulative rewards.
