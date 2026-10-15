/**
 * main.js
 * -------
 * Frontend application logic for Q-Learning Grid World Simulator.
 * Handles API calls, dynamic grid rendering, math calculation highlighting,
 * manual keyboard/D-pad controls, and Chart.js graphs.
 */

const ACTION_ARROWS = { 0: '↑', 1: '↓', 2: '←', 3: '→' };
const ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT'];

let currentEnv = null;
let currentAgentPos = [0, 0];
let currentPolicyGrid = null;
let showArrows = true;
let isMoving = false;
let stepCount = 0;
let visitedTrail = [];

// Chart references
let rewardChart = null;
let stepsChart = null;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
  initCharts();
  await loadState();
  // Auto-train once initially so the board opens ready-to-run
  trainAgent();
});

// Fetch current state from Python backend
async function loadState() {
  try {
    const res = await fetch('/api/state');
    const data = await res.json();
    currentEnv = data.env;
    currentAgentPos = data.agent_pos;
    currentPolicyGrid = data.policy_grid;

    document.getElementById('val-epsilon').innerText = data.epsilon.toFixed(3);
    document.getElementById('val-reward').innerText = data.total_score.toFixed(1);
    document.getElementById('pos-indicator').innerText = `Pos: (${currentAgentPos[0]}, ${currentAgentPos[1]})`;
    
    if (data.is_trained) {
      document.getElementById('val-status').innerText = 'Trained';
      document.getElementById('val-status').style.color = 'var(--success)';
    }

    renderGrid();
  } catch (err) {
    console.error('Failed to load state:', err);
  }
}

// Render the 5x5 Grid
function renderGrid() {
  const board = document.getElementById('grid-board');
  if (!currentEnv) return;
  board.innerHTML = '';

  const rows = currentEnv.rows;
  const cols = currentEnv.cols;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.id = `cell-${r}-${c}`;

      // Coordinate marker
      const coord = document.createElement('span');
      coord.className = 'cell-coord';
      coord.innerText = `${r},${c}`;
      cell.appendChild(coord);

      const isStart = (r === currentEnv.start[0] && c === currentEnv.start[1]);
      const isGoal = (r === currentEnv.goal[0] && c === currentEnv.goal[1]);
      const isObstacle = currentEnv.obstacles.some(o => o[0] === r && o[1] === c);
      const isTrail = visitedTrail.some(p => p[0] === r && p[1] === c);

      if (isStart) {
        cell.classList.add('start');
        cell.innerHTML += '<span>START</span>';
      } else if (isGoal) {
        cell.classList.add('goal');
        cell.innerHTML += '<span>GOAL (+20)</span>';
      } else if (isObstacle) {
        cell.classList.add('obstacle');
        cell.innerHTML += '<span>WALL</span>';
      } else if (isTrail) {
        cell.classList.add('path-visited');
      }

      // Display learned policy arrows and Q-values
      if (showArrows && currentPolicyGrid && !isObstacle && !isGoal) {
        const cellData = currentPolicyGrid[r][c];
        const arrow = ACTION_ARROWS[cellData.best_action];
        const bestQ = cellData.best_q.toFixed(1);

        const arrowSpan = document.createElement('span');
        arrowSpan.className = 'policy-arrow';
        arrowSpan.innerText = arrow;

        const qSpan = document.createElement('span');
        qSpan.className = 'policy-q';
        qSpan.innerText = `Q:${bestQ}`;

        cell.appendChild(arrowSpan);
        cell.appendChild(qSpan);
      }

      // Agent avatar
      if (r === currentAgentPos[0] && c === currentAgentPos[1]) {
        const agent = document.createElement('div');
        agent.className = 'agent-token';
        agent.innerHTML = '&#9679;';
        cell.appendChild(agent);
      }

      board.appendChild(cell);
    }
  }
}

// Toggle Arrow view
function toggleArrows() {
  showArrows = !showArrows;
  renderGrid();
}

// Reset Agent Position
async function resetAgent() {
  try {
    const res = await fetch('/api/reset', { method: 'POST' });
    const data = await res.json();
    currentAgentPos = data.agent_pos;
    stepCount = 0;
    visitedTrail = [currentAgentPos];

    document.getElementById('val-steps').innerText = '0';
    document.getElementById('val-reward').innerText = '0.0';
    document.getElementById('pos-indicator').innerText = `Pos: (${currentAgentPos[0]}, ${currentAgentPos[1]})`;
    
    // Clear active table highlights
    document.querySelectorAll('#calc-table-body tr').forEach(tr => tr.classList.remove('highlight-active'));

    document.getElementById('calc-title').innerText = 'Agent Reset to Start (0, 0)';
    document.getElementById('calc-math').innerText = 'Click "Run Optimal Path" or step manually to begin.';

    renderGrid();
  } catch (err) {
    console.error('Reset error:', err);
  }
}

// Train the Agent via Backend API
async function trainAgent() {
  const btn = document.getElementById('btn-train');
  btn.disabled = true;
  btn.innerText = 'Training...';
  document.getElementById('val-status').innerText = 'Training...';
  document.getElementById('val-status').style.color = 'var(--warning)';

  const alpha = parseFloat(document.getElementById('input-alpha').value) || 0.10;
  const gamma = parseFloat(document.getElementById('input-gamma').value) || 0.90;
  const episodes = parseInt(document.getElementById('input-episodes').value) || 350;

  try {
    const res = await fetch('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ alpha, gamma, episodes })
    });
    const data = await res.json();

    currentPolicyGrid = data.policy_grid;
    document.getElementById('val-status').innerText = 'Trained';
    document.getElementById('val-status').style.color = 'var(--success)';
    document.getElementById('val-epsilon').innerText = data.epsilon_final.toFixed(3);

    // Update charts
    updateCharts(data.history);

    // Populate calculations table
    renderCalculationsTable(data.optimal_steps);

    await resetAgent();
  } catch (err) {
    console.error('Training error:', err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '&#9889; Train Agent';
  }
}

// Populate the mathematical calculations table
function renderCalculationsTable(steps) {
  const tbody = document.getElementById('calc-table-body');
  tbody.innerHTML = '';

  steps.forEach(st => {
    const tr = document.createElement('tr');
    tr.id = `row-step-${st.step}`;

    const rClass = st.reward > 0 ? 'reward-pos' : 'reward-neg';
    const rSign = st.reward > 0 ? '+' : '';

    tr.innerHTML = `
      <td><b>${st.step}</b></td>
      <td>(${st.state[0]}, ${st.state[1]})</td>
      <td><b>${st.action_name}</b> (${ACTION_ARROWS[st.action]})</td>
      <td>(${st.next_state[0]}, ${st.next_state[1]})</td>
      <td class="${rClass}">${rSign}${st.reward.toFixed(1)}</td>
      <td>${st.best_future_q.toFixed(2)}</td>
      <td class="math-cell">${rSign}${st.reward.toFixed(1)} + (0.90 &times; ${st.best_future_q.toFixed(2)}) = <b>${st.target.toFixed(2)}</b></td>
      <td><b>${st.q_val.toFixed(2)}</b></td>
      <td><b>${st.score > 0 ? '+' : ''}${st.score.toFixed(1)}</b></td>
    `;
    tbody.appendChild(tr);
  });
}

// Step single action from Agent or Manual
async function stepAction(actionIndex = null) {
  try {
    const payload = actionIndex !== null ? { action: actionIndex } : {};
    const res = await fetch('/api/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    stepCount++;
    currentAgentPos = data.next_state;
    visitedTrail.push(currentAgentPos);
    currentPolicyGrid = data.policy_grid;

    document.getElementById('val-steps').innerText = `${stepCount}`;
    document.getElementById('val-reward').innerText = data.total_score.toFixed(1);
    document.getElementById('pos-indicator').innerText = `Pos: (${currentAgentPos[0]}, ${currentAgentPos[1]})`;

    // Highlight row in calculations table
    document.querySelectorAll('#calc-table-body tr').forEach(tr => tr.classList.remove('highlight-active'));
    const row = document.getElementById(`row-step-${stepCount}`);
    if (row) row.classList.add('highlight-active');

    // Update real-time calculation box
    const rSign = data.reward > 0 ? '+' : '';
    document.getElementById('calc-title').innerText =
      `Step ${stepCount}: (${data.state}) → (${data.next_state}) via ${data.action_name} [${data.info}]`;
    document.getElementById('calc-math').innerHTML =
      `Target = ${rSign}${data.reward.toFixed(1)} + (${data.gamma} &times; ${data.best_next_q.toFixed(2)}) = <b>${data.td_target.toFixed(2)}</b> &bull; TD Error = <b>${data.td_error.toFixed(2)}</b> &bull; Q_new = <b>${data.q_new.toFixed(2)}</b>`;

    renderGrid();
    return data.done;
  } catch (err) {
    console.error('Step error:', err);
    return true;
  }
}

// Manual D-Pad step
async function manualStep(act) {
  if (isMoving) return;
  await stepAction(act);
}

// Single greedy step
async function stepGreedy() {
  if (isMoving) return;
  await stepAction(null);
}

// Run the full optimal path with animation
async function runOptimalPath() {
  if (isMoving) return;
  isMoving = true;
  const runBtn = document.getElementById('btn-run');
  runBtn.disabled = true;

  await resetAgent();

  while (currentAgentPos[0] !== currentEnv.goal[0] || currentAgentPos[1] !== currentEnv.goal[1]) {
    await new Promise(r => setTimeout(r, 400));
    const done = await stepAction(null);
    if (done || stepCount >= 18) break;
  }

  isMoving = false;
  runBtn.disabled = false;
}

// Run Hyperparameter Sensitivity Analysis
async function fetchHyperparameterAnalysis() {
  const card = document.getElementById('hyperparam-card');
  const tbody = document.getElementById('hyperparam-table-body');
  card.style.display = 'block';
  tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:15px;">Running 6 hyperparameter experiments on Python backend...</td></tr>';

  try {
    const res = await fetch('/api/hyperparameter-test');
    const data = await res.json();
    tbody.innerHTML = '';

    data.results.forEach(item => {
      const tr = document.createElement('tr');
      const isBest = item.final_avg_steps <= 8.5;
      tr.innerHTML = `
        <td><b>${item.name}</b></td>
        <td>${item.alpha}</td>
        <td>${item.gamma}</td>
        <td>${item.decay}</td>
        <td><b>${item.final_avg_reward > 0 ? '+' : ''}${item.final_avg_reward.toFixed(2)}</b></td>
        <td><b>${item.final_avg_steps.toFixed(1)}</b></td>
        <td><span style="color:${isBest ? 'var(--success)' : 'var(--text-muted)'}; font-weight:700;">${isBest ? 'Optimal (8 steps)' : 'Converged'}</span></td>
      `;
      tbody.appendChild(tr);
    });

    card.scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    console.error('Hyperparameter error:', err);
    tbody.innerHTML = '<tr><td colspan="7" style="color:red; text-align:center;">Failed to run experiments.</td></tr>';
  }
}

// Chart.js initialization
function initCharts() {
  const ctxR = document.getElementById('rewardChart').getContext('2d');
  rewardChart = new Chart(ctxR, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Cumulative Reward',
        data: [],
        borderColor: '#2563eb',
        backgroundColor: 'rgba(37, 99, 235, 0.08)',
        borderWidth: 2,
        tension: 0.25,
        fill: true,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Episode' }, grid: { display: false } },
        y: { title: { display: true, text: 'Reward' } }
      }
    }
  });

  const ctxS = document.getElementById('stepsChart').getContext('2d');
  stepsChart = new Chart(ctxS, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Steps to Goal',
        data: [],
        borderColor: '#16a34a',
        backgroundColor: 'rgba(22, 163, 74, 0.08)',
        borderWidth: 2,
        tension: 0.25,
        fill: true,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Episode' }, grid: { display: false } },
        y: { title: { display: true, text: 'Steps' }, min: 0 }
      }
    }
  });
}

// Update charts with training history
function updateCharts(history) {
  if (!history || !rewardChart || !stepsChart) return;
  const labels = Array.from({ length: history.rewards.length }, (_, i) => i + 1);

  rewardChart.data.labels = labels;
  rewardChart.data.datasets[0].data = history.rewards;
  rewardChart.update();

  stepsChart.data.labels = labels;
  stepsChart.data.datasets[0].data = history.steps;
  stepsChart.update();
}

// Keyboard arrow navigation
window.addEventListener('keydown', (e) => {
  if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
    e.preventDefault();
    if (e.key === 'ArrowUp') manualStep(0);
    if (e.key === 'ArrowDown') manualStep(1);
    if (e.key === 'ArrowLeft') manualStep(2);
    if (e.key === 'ArrowRight') manualStep(3);
  }
});
