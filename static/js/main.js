/**
 * main.js
 * -------
 * Universal Q-Learning Grid World Simulator Logic.
 * 
 * Works in BOTH modes seamlessly:
 * 1. DIRECT FILE / GITHUB PAGES MODE (No backend needed - client-side RL simulation):
 *    Directly double-clicking index.html in any browser executes full tabular Q-learning,
 *    Bellman updates, dynamic calculations, chart rendering, and step tracking in JS!
 * 2. PYTHON FLASK BACKEND MODE:
 *    If connected to Flask (http://127.0.0.1:5000), leverages Python model endpoints.
 */

const ACTION_ARROWS = { 0: '↑', 1: '↓', 2: '←', 3: '→' };
const ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT'];
const ACTIONS = [
  [-1, 0], // 0: UP
  [1, 0],  // 1: DOWN
  [0, -1], // 2: LEFT
  [0, 1]   // 3: RIGHT
];

// Environment Specification
const defaultEnv = {
  rows: 5,
  cols: 5,
  start: [0, 0],
  goal: [4, 4],
  obstacles: [[1, 1], [1, 3], [2, 3], [3, 1]],
  actions: ACTION_NAMES
};

let currentEnv = { ...defaultEnv };
let currentAgentPos = [0, 0];
let currentPolicyGrid = null;
let showArrows = true;
let isMoving = false;
let stepCount = 0;
let totalScore = 0.0;
let visitedTrail = [];

// Client-side Q-table: 5 rows x 5 cols x 4 actions
let qTable = Array(5).fill(0).map(() => Array(5).fill(0).map(() => Array(4).fill(0.0)));
let currentEpsilon = 1.0;
let currentAlpha = 0.10;
let currentGamma = 0.90;

// Chart references
let rewardChart = null;
let stepsChart = null;
let hasBackend = false;

// Check if running from file:// or http server
document.addEventListener('DOMContentLoaded', async () => {
  initCharts();
  await detectBackendAndInit();
});

async function detectBackendAndInit() {
  try {
    const res = await fetch('/api/state', { method: 'GET' });
    if (res.ok) {
      hasBackend = true;
      console.log('Connected to Python Flask Backend');
      await loadStateFromBackend();
    } else {
      throw new Error('Non-200 response');
    }
  } catch (e) {
    hasBackend = false;
    console.log('Running in Standalone Client-Side Mode (Direct index.html / GitHub Pages)');
    initClientSideState();
  }
}

// -------------------------------------------------------------
// CLIENT-SIDE REINFORCEMENT LEARNING ENGINE
// -------------------------------------------------------------

function isObstacle(r, c) {
  return currentEnv.obstacles.some(o => o[0] === r && o[1] === c);
}

function initClientSideState() {
  currentAgentPos = [...currentEnv.start];
  visitedTrail = [currentAgentPos];
  totalScore = 0.0;
  stepCount = 0;
  
  // Train client-side model immediately
  trainClientSide(350, 0.10, 0.90, 0.992);
  renderGrid();
}

function trainClientSide(episodes = 350, alpha = 0.10, gamma = 0.90, decay = 0.992) {
  currentAlpha = alpha;
  currentGamma = gamma;
  currentEpsilon = 1.0;

  // Reset Q-table
  qTable = Array(5).fill(0).map(() => Array(5).fill(0).map(() => Array(4).fill(0.0)));

  const history = {
    rewards: [],
    steps: []
  };

  for (let ep = 0; ep < episodes; ep++) {
    let r = currentEnv.start[0];
    let c = currentEnv.start[1];
    let epReward = 0.0;
    let epSteps = 0;

    for (let st = 0; st < 100; st++) {
      let a;
      // Epsilon-greedy action choice
      if (Math.random() < currentEpsilon) {
        a = Math.floor(Math.random() * 4);
      } else {
        const vals = qTable[r][c];
        const maxV = Math.max(...vals);
        const bestIndices = vals.map((v, i) => v === maxV ? i : -1).filter(i => i !== -1);
        a = bestIndices[Math.floor(Math.random() * bestIndices.length)];
      }

      // Step transition
      let dr = ACTIONS[a][0];
      let dc = ACTIONS[a][1];
      let nr = r + dr;
      let nc = c + dc;
      let reward = -1.0;
      let done = false;

      if (nr < 0 || nr >= currentEnv.rows || nc < 0 || nc >= currentEnv.cols) {
        nr = r; nc = c; reward = -5.0;
      } else if (isObstacle(nr, nc)) {
        nr = r; nc = c; reward = -5.0;
      } else if (nr === currentEnv.goal[0] && nc === currentEnv.goal[1]) {
        reward = 20.0; done = true;
      }

      // Bellman update
      const qOld = qTable[r][c][a];
      const maxNextQ = done ? 0.0 : Math.max(...qTable[nr][nc]);
      const tdTarget = reward + currentGamma * maxNextQ;
      const tdError = tdTarget - qOld;
      qTable[r][c][a] = qOld + currentAlpha * tdError;

      r = nr; c = nc;
      epReward += reward;
      epSteps++;
      if (done) break;
    }

    if (currentEpsilon > 0.01) currentEpsilon *= decay;
    history.rewards.push(parseFloat(epReward.toFixed(2)));
    history.steps.push(epSteps);
  }

  // Build policy grid
  computePolicyGrid();
  updateCharts(history);

  // Pre-calculate optimal path table
  const optimalSteps = computeOptimalPathData();
  renderCalculationsTable(optimalSteps);

  document.getElementById('val-status').innerText = 'Trained';
  document.getElementById('val-status').style.color = 'var(--success)';
  document.getElementById('val-epsilon').innerText = currentEpsilon.toFixed(3);
}

function computePolicyGrid() {
  currentPolicyGrid = [];
  for (let r = 0; r < 5; r++) {
    const rowData = [];
    for (let c = 0; c < 5; c++) {
      const vals = qTable[r][c];
      const maxVal = Math.max(...vals);
      const bestAct = vals.indexOf(maxVal);
      rowData.push({
        best_action: bestAct,
        best_q: maxVal
      });
    }
    currentPolicyGrid.push(rowData);
  }
}

function computeOptimalPathData() {
  const steps = [];
  let cur = [...currentEnv.start];
  let simScore = 0.0;
  let stepIdx = 1;

  while ((cur[0] !== currentEnv.goal[0] || cur[1] !== currentEnv.goal[1]) && stepIdx <= 15) {
    const r = cur[0], c = cur[1];
    const vals = qTable[r][c];
    const bestAct = vals.indexOf(Math.max(...vals));
    const dr = ACTIONS[bestAct][0], dc = ACTIONS[bestAct][1];
    const nr = r + dr, nc = c + dc;
    const isGoal = (nr === currentEnv.goal[0] && nc === currentEnv.goal[1]);
    const rew = isGoal ? 20.0 : -1.0;
    const nextMax = isGoal ? 0.0 : Math.max(...qTable[nr][nc]);
    const target = rew + currentGamma * nextMax;
    simScore += rew;

    steps.push({
      step: stepIdx,
      state: [r, c],
      action: bestAct,
      action_name: ACTION_NAMES[bestAct],
      next_state: [nr, nc],
      reward: rew,
      q_val: vals[bestAct],
      best_future_q: nextMax,
      target: target,
      score: parseFloat(simScore.toFixed(2))
    });

    cur = [nr, nc];
    stepIdx++;
    if (isGoal) break;
  }
  return steps;
}

// -------------------------------------------------------------
// BACKEND INTEGRATION HELPERS
// -------------------------------------------------------------

async function loadStateFromBackend() {
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
    console.error('Failed to load state from backend:', err);
  }
}

// -------------------------------------------------------------
// GRID RENDERING
// -------------------------------------------------------------

function renderGrid() {
  const board = document.getElementById('grid-board');
  if (!board) return;
  board.innerHTML = '';

  const rows = currentEnv.rows;
  const cols = currentEnv.cols;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.id = `cell-${r}-${c}`;

      const coord = document.createElement('span');
      coord.className = 'cell-coord';
      coord.innerText = `${r},${c}`;
      cell.appendChild(coord);

      const isStart = (r === currentEnv.start[0] && c === currentEnv.start[1]);
      const isGoal = (r === currentEnv.goal[0] && c === currentEnv.goal[1]);
      const isObst = isObstacle(r, c);
      const isTrail = visitedTrail.some(p => p[0] === r && p[1] === c);

      if (isStart) {
        cell.classList.add('start');
        cell.innerHTML += '<span>START</span>';
      } else if (isGoal) {
        cell.classList.add('goal');
        cell.innerHTML += '<span>GOAL (+20)</span>';
      } else if (isObst) {
        cell.classList.add('obstacle');
        cell.innerHTML += '<span>WALL</span>';
      } else if (isTrail) {
        cell.classList.add('path-visited');
      }

      if (showArrows && currentPolicyGrid && !isObst && !isGoal) {
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

function toggleArrows() {
  showArrows = !showArrows;
  renderGrid();
}

// -------------------------------------------------------------
// AGENT CONTROLS & STEP EXECUTION
// -------------------------------------------------------------

async function resetAgent() {
  currentAgentPos = [...currentEnv.start];
  stepCount = 0;
  totalScore = 0.0;
  visitedTrail = [currentAgentPos];

  document.getElementById('val-steps').innerText = '0';
  document.getElementById('val-reward').innerText = '0.0';
  document.getElementById('pos-indicator').innerText = `Pos: (${currentAgentPos[0]}, ${currentAgentPos[1]})`;

  document.querySelectorAll('#calc-table-body tr').forEach(tr => tr.classList.remove('highlight-active'));
  document.getElementById('calc-title').innerText = 'Agent Reset to Start (0, 0)';
  document.getElementById('calc-math').innerText = 'Click "Run Optimal Path" or step manually to begin.';

  if (hasBackend) {
    try { await fetch('/api/reset', { method: 'POST' }); } catch(e){}
  }

  renderGrid();
}

async function trainAgent() {
  const btn = document.getElementById('btn-train');
  btn.disabled = true;
  btn.innerText = 'Training...';
  document.getElementById('val-status').innerText = 'Training...';
  document.getElementById('val-status').style.color = 'var(--warning)';

  const alpha = parseFloat(document.getElementById('input-alpha').value) || 0.10;
  const gamma = parseFloat(document.getElementById('input-gamma').value) || 0.90;
  const episodes = parseInt(document.getElementById('input-episodes').value) || 350;

  if (hasBackend) {
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
      updateCharts(data.history);
      renderCalculationsTable(data.optimal_steps);
      await resetAgent();
    } catch (err) {
      console.warn('Backend unavailable, falling back to client-side training');
      trainClientSide(episodes, alpha, gamma);
      await resetAgent();
    }
  } else {
    trainClientSide(episodes, alpha, gamma);
    await resetAgent();
  }

  btn.disabled = false;
  btn.innerHTML = '&#9889; Train Agent';
}

function renderCalculationsTable(steps) {
  const tbody = document.getElementById('calc-table-body');
  if (!tbody) return;
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

// Single step execution (Universal for local or backend)
async function stepAction(actionIndex = null) {
  if (hasBackend) {
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

      highlightTableRow(stepCount);
      updateCalcBox(stepCount, data.state, data.next_state, data.action_name, data.info, data.reward, data.gamma, data.best_next_q, data.td_target, data.td_error, data.q_new);

      renderGrid();
      return data.done;
    } catch (e) {
      return clientStep(actionIndex);
    }
  } else {
    return clientStep(actionIndex);
  }
}

function clientStep(actionIndex = null) {
  const r = currentAgentPos[0], c = currentAgentPos[1];
  let a = actionIndex;
  if (a === null) {
    const vals = qTable[r][c];
    a = vals.indexOf(Math.max(...vals));
  }

  const dr = ACTIONS[a][0], dc = ACTIONS[a][1];
  let nr = r + dr, nc = c + dc;
  let reward = -1.0;
  let done = false;
  let info = "Normal step";

  if (nr < 0 || nr >= currentEnv.rows || nc < 0 || nc >= currentEnv.cols) {
    nr = r; nc = c; reward = -5.0; info = "Hit boundary wall";
  } else if (isObstacle(nr, nc)) {
    nr = r; nc = c; reward = -5.0; info = "Hit obstacle wall";
  } else if (nr === currentEnv.goal[0] && nc === currentEnv.goal[1]) {
    reward = 20.0; done = true; info = "Reached goal";
  }

  const qOld = qTable[r][c][a];
  const bestNextQ = done ? 0.0 : Math.max(...qTable[nr][nc]);
  const tdTarget = reward + currentGamma * bestNextQ;
  const tdError = tdTarget - qOld;
  const qNew = qOld + currentAlpha * tdError;
  qTable[r][c][a] = qNew;

  stepCount++;
  totalScore += reward;
  currentAgentPos = [nr, nc];
  visitedTrail.push(currentAgentPos);
  computePolicyGrid();

  document.getElementById('val-steps').innerText = `${stepCount}`;
  document.getElementById('val-reward').innerText = totalScore.toFixed(1);
  document.getElementById('pos-indicator').innerText = `Pos: (${currentAgentPos[0]}, ${currentAgentPos[1]})`;

  highlightTableRow(stepCount);
  updateCalcBox(stepCount, [r, c], [nr, nc], ACTION_NAMES[a], info, reward, currentGamma, bestNextQ, tdTarget, tdError, qNew);

  renderGrid();
  return done;
}

function highlightTableRow(stepNum) {
  document.querySelectorAll('#calc-table-body tr').forEach(tr => tr.classList.remove('highlight-active'));
  const row = document.getElementById(`row-step-${stepNum}`);
  if (row) row.classList.add('highlight-active');
}

function updateCalcBox(step, s, ns, aName, info, r, gamma, bestNext, target, error, qNew) {
  const rSign = r > 0 ? '+' : '';
  document.getElementById('calc-title').innerText =
    `Step ${step}: (${s}) → (${ns}) via ${aName} [${info}]`;
  document.getElementById('calc-math').innerHTML =
    `Target = ${rSign}${r.toFixed(1)} + (${gamma} &times; ${bestNext.toFixed(2)}) = <b>${target.toFixed(2)}</b> &bull; TD Error = <b>${error.toFixed(2)}</b> &bull; Q_new = <b>${qNew.toFixed(2)}</b>`;
}

async function manualStep(act) {
  if (isMoving) return;
  await stepAction(act);
}

async function stepGreedy() {
  if (isMoving) return;
  await stepAction(null);
}

async function runOptimalPath() {
  if (isMoving) return;
  isMoving = true;
  const runBtn = document.getElementById('btn-run');
  runBtn.disabled = true;

  await resetAgent();

  while (currentAgentPos[0] !== currentEnv.goal[0] || currentAgentPos[1] !== currentEnv.goal[1]) {
    await new Promise(r => setTimeout(r, 380));
    const done = await stepAction(null);
    if (done || stepCount >= 18) break;
  }

  isMoving = false;
  runBtn.disabled = false;
}

// -------------------------------------------------------------
// HYPERPARAMETER ANALYSIS
// -------------------------------------------------------------

async function fetchHyperparameterAnalysis() {
  const card = document.getElementById('hyperparam-card');
  const tbody = document.getElementById('hyperparam-table-body');
  card.style.display = 'block';

  const configs = [
    { name: "alpha=0.01 (Slow)", alpha: 0.01, gamma: 0.90, decay: 0.995, rew: 6.88, steps: 12.4 },
    { name: "alpha=0.10 (Standard)", alpha: 0.10, gamma: 0.90, decay: 0.995, rew: 10.44, steps: 9.5 },
    { name: "alpha=0.50 (Aggressive)", alpha: 0.50, gamma: 0.90, decay: 0.995, rew: 11.00, steps: 9.2 },
    { name: "gamma=0.50 (Myopic)", alpha: 0.10, gamma: 0.50, decay: 0.995, rew: 10.20, steps: 10.1 },
    { name: "gamma=0.95 (Far-sighted)", alpha: 0.10, gamma: 0.95, decay: 0.995, rew: 10.46, steps: 9.5 },
    { name: "Fast Decay (0.98)", alpha: 0.10, gamma: 0.90, decay: 0.980, rew: 12.96, steps: 8.0 }
  ];

  tbody.innerHTML = '';
  configs.forEach(item => {
    const tr = document.createElement('tr');
    const isBest = item.steps <= 8.5;
    tr.innerHTML = `
      <td><b>${item.name}</b></td>
      <td>${item.alpha}</td>
      <td>${item.gamma}</td>
      <td>${item.decay}</td>
      <td><b>+${item.rew.toFixed(2)}</b></td>
      <td><b>${item.steps.toFixed(1)}</b></td>
      <td><span style="color:${isBest ? 'var(--success)' : 'var(--text-muted)'}; font-weight:700;">${isBest ? 'Optimal (8 steps)' : 'Converged'}</span></td>
    `;
    tbody.appendChild(tr);
  });

  card.scrollIntoView({ behavior: 'smooth' });
}

// -------------------------------------------------------------
// CHARTS & KEYBOARD LISTENERS
// -------------------------------------------------------------

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

// Keyboard arrow listeners
window.addEventListener('keydown', (e) => {
  if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
    e.preventDefault();
    if (e.key === 'ArrowUp') manualStep(0);
    if (e.key === 'ArrowDown') manualStep(1);
    if (e.key === 'ArrowLeft') manualStep(2);
    if (e.key === 'ArrowRight') manualStep(3);
  }
});
