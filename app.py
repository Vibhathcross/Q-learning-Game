"""
app.py
------
Flask backend for the Q-Learning Grid World game and simulator.
Serves the web UI and exposes RESTful API endpoints for training,
interactive stepping, manual play, and hyperparameter analysis.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np

from environment import GridWorld, ACTION_NAMES
from agent import QLearningAgent

app = Flask(__name__)
CORS(app)

# Global simulation state
env = GridWorld(rows=5, cols=5, start=(0, 0), goal=(4, 4))
agent = QLearningAgent(rows=5, cols=5, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.992)
current_episode = 0
total_score = 0.0
is_trained = False
training_history = {
    "rewards": [],
    "steps": [],
    "epsilons": []
}


@app.route("/")
def index():
    """Serves the main web application."""
    return render_template("index.html")


@app.route("/api/state", methods=["GET"])
def get_state():
    """Returns current environment state, agent position, and Q-table summary."""
    return jsonify({
        "env": env.to_dict(),
        "agent_pos": list(env.state),
        "is_trained": is_trained,
        "episodes_trained": current_episode,
        "epsilon": round(agent.epsilon, 4),
        "alpha": agent.alpha,
        "gamma": agent.gamma,
        "total_score": round(total_score, 2),
        "policy_grid": agent.get_policy_grid(env)
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    """Resets the agent position back to start state without clearing Q-table."""
    global total_score
    env.reset()
    total_score = 0.0
    return jsonify({
        "status": "success",
        "agent_pos": list(env.state),
        "total_score": total_score
    })


@app.route("/api/step", methods=["POST"])
def step():
    """
    Executes a single action (either provided manually or chosen greedily by the agent).
    Returns the action taken, transition, reward, and exact Bellman calculations.
    """
    global total_score
    data = request.get_json() or {}
    manual_action = data.get("action", None)

    state = env.state
    if manual_action is not None and 0 <= manual_action <= 3:
        action = int(manual_action)
    else:
        # Agent picks best action using learned Q-values
        action = agent.choose_action(state, explore=False)

    next_state, reward, done, info = env.step(action)
    total_score += reward

    # Compute transition math breakdown
    calc = agent.update(state, action, reward, next_state, done)
    calc["info"] = info
    calc["total_score"] = round(total_score, 2)
    calc["policy_grid"] = agent.get_policy_grid(env)

    return jsonify(calc)


@app.route("/api/train", methods=["POST"])
def train():
    """
    Trains the agent for N episodes with customizable hyperparameters.
    Tracks episode rewards, steps taken, and epsilon decay.
    """
    global agent, env, is_trained, current_episode, total_score, training_history
    data = request.get_json() or {}

    episodes = int(data.get("episodes", 300))
    alpha = float(data.get("alpha", 0.1))
    gamma = float(data.get("gamma", 0.9))
    epsilon_decay = float(data.get("epsilon_decay", 0.992))

    # Re-initialize agent with new parameters
    agent = QLearningAgent(
        rows=env.rows, cols=env.cols,
        alpha=alpha, gamma=gamma,
        epsilon=1.0, epsilon_decay=epsilon_decay
    )
    env.reset()
    total_score = 0.0

    rewards_list = []
    steps_list = []
    epsilons_list = []

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0.0
        steps = 0

        for st in range(100):
            action = agent.choose_action(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward
            steps += 1
            if done:
                break

        agent.decay_epsilon()
        rewards_list.append(round(ep_reward, 2))
        steps_list.append(steps)
        epsilons_list.append(round(agent.epsilon, 4))

    env.reset()
    is_trained = True
    current_episode = episodes
    training_history = {
        "rewards": rewards_list,
        "steps": steps_list,
        "epsilons": epsilons_list
    }

    # Pre-calculate optimal path steps and detailed calculations
    optimal_path_steps = []
    test_state = env.reset()
    sim_score = 0.0
    step_num = 1

    while test_state != env.goal and step_num <= 15:
        best_act = agent.choose_action(test_state, explore=False)
        r, c = test_state
        q_val = float(agent.q_table[r, c, best_act])
        next_s, rew, done, _ = env.step(best_act)
        best_future = 0.0 if done else float(np.max(agent.q_table[next_s[0], next_s[1]]))
        target = float(rew + agent.gamma * best_future)
        sim_score += rew

        optimal_path_steps.append({
            "step": step_num,
            "state": list(test_state),
            "action": best_act,
            "action_name": ACTION_NAMES[best_act],
            "next_state": list(next_s),
            "reward": float(rew),
            "q_val": q_val,
            "best_future_q": best_future,
            "target": target,
            "score": round(sim_score, 2)
        })

        test_state = next_s
        step_num += 1
        if done:
            break

    env.reset()

    return jsonify({
        "status": "success",
        "episodes": episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_final": round(agent.epsilon, 4),
        "history": training_history,
        "policy_grid": agent.get_policy_grid(env),
        "optimal_steps": optimal_path_steps
    })


@app.route("/api/hyperparameter-test", methods=["GET"])
def hyperparameter_test():
    """
    Runs sensitivity experiments across 6 configurations and returns
    structured data for dynamic frontend chart rendering.
    """
    configs = [
        {"name": "alpha=0.01 (Slow)", "alpha": 0.01, "gamma": 0.90, "decay": 0.995},
        {"name": "alpha=0.10 (Standard)", "alpha": 0.10, "gamma": 0.90, "decay": 0.995},
        {"name": "alpha=0.50 (Aggressive)", "alpha": 0.50, "gamma": 0.90, "decay": 0.995},
        {"name": "gamma=0.50 (Myopic)", "alpha": 0.10, "gamma": 0.50, "decay": 0.995},
        {"name": "gamma=0.95 (Far-sighted)", "alpha": 0.10, "gamma": 0.95, "decay": 0.995},
        {"name": "Fast Decay (0.98)", "alpha": 0.10, "gamma": 0.90, "decay": 0.980}
    ]

    results = []
    episodes = 250

    for cfg in configs:
        test_agent = QLearningAgent(
            rows=env.rows, cols=env.cols,
            alpha=cfg["alpha"], gamma=cfg["gamma"],
            epsilon=1.0, epsilon_decay=cfg["decay"]
        )
        test_rewards = []
        test_steps = []

        for ep in range(episodes):
            s = env.reset()
            ep_rew = 0.0
            st_count = 0
            for _ in range(100):
                a = test_agent.choose_action(s, explore=True)
                ns, r, done, _ = env.step(a)
                test_agent.update(s, a, r, ns, done)
                s = ns
                ep_rew += r
                st_count += 1
                if done:
                    break
            test_agent.decay_epsilon()
            test_rewards.append(round(ep_rew, 2))
            test_steps.append(st_count)

        # Average of last 40 episodes
        final_avg_reward = round(float(np.mean(test_rewards[-40:])), 2)
        final_avg_steps = round(float(np.mean(test_steps[-40:])), 1)

        # Moving average for chart
        window = 15
        ma_rewards = np.convolve(test_rewards, np.ones(window)/window, mode='valid').round(2).tolist()

        results.append({
            "name": cfg["name"],
            "alpha": cfg["alpha"],
            "gamma": cfg["gamma"],
            "decay": cfg["decay"],
            "final_avg_reward": final_avg_reward,
            "final_avg_steps": final_avg_steps,
            "moving_avg_rewards": ma_rewards
        })

    env.reset()
    return jsonify({"results": results})


if __name__ == "__main__":
    print("=======================================================")
    print("  Q-Learning Grid World Server Starting...")
    print("  Open your browser at: http://127.0.0.1:5000")
    print("=======================================================")
    app.run(host="127.0.0.1", port=5000, debug=False)
