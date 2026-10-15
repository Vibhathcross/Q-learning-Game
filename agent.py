"""
agent.py
--------
Tabular Q-Learning Agent implementing the Bellman Optimality update:
Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
"""

import random
import numpy as np
from environment import ACTION_NAMES


class QLearningAgent:
    def __init__(self, rows=5, cols=5, num_actions=4, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.992):
        self.rows = rows
        self.cols = cols
        self.num_actions = num_actions
        self.alpha = float(alpha)               # Learning rate
        self.gamma = float(gamma)               # Discount factor
        self.epsilon = float(epsilon)           # Exploration rate
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        # 3D Q-Table: [row, col, action]
        self.q_table = np.zeros((rows, cols, num_actions), dtype=float)

    def choose_action(self, state, explore=True):
        """Epsilon-greedy policy for action selection."""
        r, c = state
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            q_vals = self.q_table[r, c]
            max_val = np.max(q_vals)
            best_actions = np.where(q_vals == max_val)[0]
            return int(np.random.choice(best_actions))

    def update(self, state, action, reward, next_state, done):
        """
        Applies Bellman update and returns a detailed dictionary of calculations.
        """
        r, c = state
        nr, nc = next_state

        q_old = float(self.q_table[r, c, action])
        best_next_q = 0.0 if done else float(np.max(self.q_table[nr, nc]))
        td_target = float(reward + self.gamma * best_next_q)
        td_error = float(td_target - q_old)
        gain = float(self.alpha * td_error)
        q_new = float(q_old + gain)

        self.q_table[r, c, action] = q_new

        return {
            "state": list(state),
            "action": action,
            "action_name": ACTION_NAMES[action],
            "reward": float(reward),
            "next_state": list(next_state),
            "q_old": q_old,
            "best_next_q": best_next_q,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "td_target": td_target,
            "td_error": td_error,
            "gain": gain,
            "q_new": q_new,
            "done": bool(done)
        }

    def decay_epsilon(self):
        """Multiplicative epsilon decay."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_table_list(self):
        """Returns JSON-serializable representation of the 3D Q-table."""
        return np.round(self.q_table, 3).tolist()

    def get_policy_grid(self, env):
        """
        Computes the greedy policy arrow and best Q-value for each cell.
        """
        policy_grid = []
        for r in range(self.rows):
            row_data = []
            for c in range(self.cols):
                cell_type = "empty"
                if (r, c) == env.start:
                    cell_type = "start"
                elif (r, c) == env.goal:
                    cell_type = "goal"
                elif (r, c) in env.obstacles:
                    cell_type = "obstacle"

                q_vals = [float(v) for v in self.q_table[r, c]]
                best_act_idx = int(np.argmax(q_vals))
                best_q = float(np.max(q_vals))

                row_data.append({
                    "row": r,
                    "col": c,
                    "type": cell_type,
                    "q_values": q_vals,
                    "best_action": best_act_idx,
                    "best_action_name": ACTION_NAMES[best_act_idx],
                    "best_q": best_q
                })
            policy_grid.append(row_data)
        return policy_grid
