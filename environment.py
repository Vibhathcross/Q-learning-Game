"""
environment.py
--------------
Discrete 5x5 Grid World Environment for Reinforcement Learning.
Defines states, transitions, obstacles, boundaries, and rewards.
"""

ACTIONS = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1)    # RIGHT
}

ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_SYMBOLS = {0: '^', 1: 'v', 2: '<', 3: '>'}


class GridWorld:
    def __init__(self, rows=5, cols=5, start=(0, 0), goal=(4, 4), obstacles=None):
        self.rows = rows
        self.cols = cols
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.obstacles = [tuple(o) for o in (obstacles if obstacles is not None else [(1, 1), (1, 3), (2, 3), (3, 1)])]
        self.state = self.start

    def reset(self):
        """Resets agent back to start state."""
        self.state = self.start
        return self.state

    def step(self, action):
        """
        Executes an action from current state.
        Returns: (next_state, reward, done, info)
        """
        dr, dc = ACTIONS[action]
        nr = self.state[0] + dr
        nc = self.state[1] + dc

        # Check boundary collision
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
            next_state = self.state
            reward = -5.0
            done = False
            info = "Hit boundary wall"
        # Check obstacle collision
        elif (nr, nc) in self.obstacles:
            next_state = self.state
            reward = -5.0
            done = False
            info = "Hit obstacle wall"
        # Reached goal
        elif (nr, nc) == self.goal:
            next_state = (nr, nc)
            reward = 20.0
            done = True
            info = "Reached goal"
        # Normal step
        else:
            next_state = (nr, nc)
            reward = -1.0
            done = False
            info = "Normal step"

        self.state = next_state
        return next_state, reward, done, info

    def to_dict(self):
        """Returns JSON-serializable environment metadata."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "start": list(self.start),
            "goal": list(self.goal),
            "obstacles": [list(o) for o in self.obstacles],
            "actions": ACTION_NAMES
        }
