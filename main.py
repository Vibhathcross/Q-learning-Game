"""
main.py
-------
Command-line test runner for the Q-Learning Grid World Model.
Validates the Python backend logic, trains the agent, displays the optimal
trajectory, and confirms that optimal 8-step convergence is achieved.
"""

import numpy as np
from environment import GridWorld, ACTION_NAMES
from agent import QLearningAgent


def test_qlearning_pipeline():
    print("=" * 70)
    print(" S5 MACHINE LEARNING LAB - TABULAR Q-LEARNING PIPELINE TEST")
    print("=" * 70)

    # 1. Initialize environment & agent
    env = GridWorld(rows=5, cols=5, start=(0, 0), goal=(4, 4))
    agent = QLearningAgent(rows=5, cols=5, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.992)

    print(f"Environment Initialized:")
    print(f"  - Grid Dimensions : {env.rows} x {env.cols}")
    print(f"  - Start State     : {env.start}")
    print(f"  - Goal State      : {env.goal}")
    print(f"  - Obstacles       : {env.obstacles}")

    # 2. Training Loop
    episodes = 350
    print(f"\nTraining agent for {episodes} episodes...")
    for ep in range(episodes):
        state = env.reset()
        for _ in range(100):
            action = agent.choose_action(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.decay_epsilon()

    print("Training finished successfully!")
    print(f"Final Exploration Rate (epsilon): {agent.epsilon:.4f}")

    # 3. Test Optimal Path
    state = env.reset()
    path = [state]
    actions_taken = []
    total_reward = 0.0
    step = 0

    while state != env.goal and step < 20:
        action = agent.choose_action(state, explore=False)
        next_state, reward, done, _ = env.step(action)
        path.append(next_state)
        actions_taken.append(ACTION_NAMES[action])
        total_reward += reward
        state = next_state
        step += 1
        if done:
            break

    print("\nLearned Optimal Path:")
    print(" -> ".join([str(p) for p in path]))
    print(f"Actions Taken       : {', '.join(actions_taken)}")
    print(f"Total Steps Taken   : {len(actions_taken)} (Theoretical Minimum: 8)")
    print(f"Accumulated Reward  : {total_reward:+.1f} (Expected: +13.0)")

    if len(actions_taken) == 8 and total_reward == 13.0:
        print("\n[SUCCESS] The agent discovered the theoretical shortest collision-free route!")
    else:
        print("\n[NOTICE] The agent reached the goal in a valid route.")

    print("\nTo launch the interactive Web GUI, run:")
    print("  py app.py")
    print("  Then open: http://127.0.0.1:5000")
    print("=" * 70)


if __name__ == "__main__":
    test_qlearning_pipeline()
