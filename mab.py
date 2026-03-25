"""
ex6.py — Multi-Armed Bandit: Six Strategy Comparison
=====================================================
Budget  : $10,000 (= 10,000 Bernoulli pulls, $1 each)
Bandits : A (μ=0.8), B (μ=0.7), C (μ=0.5)

Methods compared:
  1. A/B Testing
  2. Optimistic Initial Values (greedy)
  3. ε-Greedy
  4. Softmax (Boltzmann)
  5. Upper Confidence Bound (UCB1)
  6. Thompson Sampling
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

np.random.seed(42)

# ─── Global Parameters ────────────────────────────────────────
MU        = np.array([0.8, 0.7, 0.5])   # true means: A, B, C
NAMES     = ['A', 'B', 'C']
K         = len(MU)                      # number of arms
T         = 10_000                       # total budget / pulls
N_SIM     = 5_000                        # Monte Carlo runs per method
OPTIMAL   = T * MU.max()                 # 10,000 × 0.8 = 8,000

# Hyper-parameters
AB_EACH   = 1_000    # A/B test: pulls per arm (A and B only)
EPSILON   = 0.1      # ε-greedy exploration rate
TAU       = 0.1      # Softmax temperature
UCB_C     = 1.0      # UCB confidence level
OPT_INIT  = 1.0      # Optimistic initial Q value
OPT_ALPHA = 0.1      # Optimistic: constant step size

COLORS    = ['#e41a1c','#ff7f00','#4daf4a','#984ea3','#377eb8','#a65628']

# ══════════════════════════════════════════════════════════════
#  SIMULATION HELPERS
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
#  VECTORIZED METHODS — each returns (n_sim, T) rewards array
#  All N_SIM simulations run in parallel via numpy broadcasting
# ══════════════════════════════════════════════════════════════

def run_ab_test(n_sim):
    rewards = np.zeros((n_sim, T))
    t = 0
    r_arm = np.zeros((n_sim, K))
    # Exploration phase: pull A and B each AB_EACH times
    for arm in [0, 1]:
        r = (np.random.random((n_sim, AB_EACH)) < MU[arm]).astype(float)
        rewards[:, t:t + AB_EACH] = r
        r_arm[:, arm] = r.sum(axis=1)
        t += AB_EACH
    # Exploitation phase
    best = np.argmax(r_arm[:, :2], axis=1)          # (n_sim,)
    rewards[:, t:] = (np.random.random((n_sim, T - t)) < MU[best][:, None]).astype(float)
    return rewards


def run_optimistic(n_sim):
    rewards = np.zeros((n_sim, T))
    Q   = np.full((n_sim, K), OPT_INIT, dtype=float)
    idx = np.arange(n_sim)
    for t in range(T):
        arms = np.argmax(Q, axis=1)
        r = (np.random.random(n_sim) < MU[arms]).astype(float)
        rewards[:, t] = r
        Q[idx, arms] += OPT_ALPHA * (r - Q[idx, arms])
    return rewards


def run_epsilon_greedy(n_sim):
    rewards = np.zeros((n_sim, T))
    Q   = np.zeros((n_sim, K))
    N   = np.zeros((n_sim, K))
    idx = np.arange(n_sim)
    for t in range(T):
        explore = np.random.random(n_sim) < EPSILON
        arms = np.where(explore, np.random.randint(K, size=n_sim), np.argmax(Q, axis=1))
        r = (np.random.random(n_sim) < MU[arms]).astype(float)
        rewards[:, t] = r
        N[idx, arms] += 1
        Q[idx, arms] += (r - Q[idx, arms]) / N[idx, arms]
    return rewards


def run_softmax(n_sim):
    rewards = np.zeros((n_sim, T))
    Q   = np.zeros((n_sim, K))
    N   = np.zeros((n_sim, K))
    idx = np.arange(n_sim)
    for t in range(T):
        logits = Q / TAU
        logits -= logits.max(axis=1, keepdims=True)
        # Gumbel-max trick: vectorised categorical sampling
        gumbel = -np.log(-np.log(np.random.random((n_sim, K)) + 1e-20))
        arms = np.argmax(logits + gumbel, axis=1)
        r = (np.random.random(n_sim) < MU[arms]).astype(float)
        rewards[:, t] = r
        N[idx, arms] += 1
        Q[idx, arms] += (r - Q[idx, arms]) / N[idx, arms]
    return rewards


def run_ucb(n_sim):
    rewards = np.zeros((n_sim, T))
    Q   = np.zeros((n_sim, K))
    N   = np.zeros((n_sim, K))
    idx = np.arange(n_sim)
    t = 0
    # Initialise: pull each arm once
    for arm in range(K):
        r = (np.random.random(n_sim) < MU[arm]).astype(float)
        rewards[:, t] = r
        N[:, arm] = 1
        Q[:, arm] = r
        t += 1
    for t in range(t, T):
        bonus = UCB_C * np.sqrt(np.log(t + 1) / (N + 1e-9))
        arms = np.argmax(Q + bonus, axis=1)
        r = (np.random.random(n_sim) < MU[arms]).astype(float)
        rewards[:, t] = r
        N[idx, arms] += 1
        Q[idx, arms] += (r - Q[idx, arms]) / N[idx, arms]
    return rewards


def run_thompson(n_sim):
    rewards = np.zeros((n_sim, T))
    alpha = np.ones((n_sim, K))
    beta  = np.ones((n_sim, K))
    idx   = np.arange(n_sim)
    for t in range(T):
        arms = np.argmax(np.random.beta(alpha, beta), axis=1)
        r = (np.random.random(n_sim) < MU[arms]).astype(float)
        rewards[:, t] = r
        alpha[idx, arms] += r
        beta[idx, arms]  += 1 - r
    return rewards


# ══════════════════════════════════════════════════════════════
#  RUN ALL METHODS — Monte Carlo Average
# ══════════════════════════════════════════════════════════════
methods = [
    ("A/B Testing",              run_ab_test),
    ("Optimistic Init (Q=1.0)",  run_optimistic),
    ("ε-Greedy (ε=0.10)",        run_epsilon_greedy),
    ("Softmax (τ=0.10)",         run_softmax),
    ("UCB1 (c=1.0)",             run_ucb),
    ("Thompson Sampling",        run_thompson),
]

print("Running Monte Carlo simulations …")
print(f"  {N_SIM:,} runs × {T:,} steps × {len(methods)} methods\n")

all_mean_rewards = {}  # method → (T,) mean reward per step
all_mean_regret  = {}  # method → (T,) mean regret per step
all_total_reward = {}  # method → scalar total reward
all_total_regret = {}  # method → scalar total regret

for name, fn in methods:
    mean_r = fn(N_SIM).mean(axis=0)  # vectorised: no inner loop
    running_avg = np.cumsum(mean_r) / np.arange(1, T + 1)
    all_mean_rewards[name] = running_avg
    all_mean_regret[name]  = MU.max() - running_avg
    all_total_reward[name] = mean_r.sum()
    all_total_regret[name] = np.arange(1, T + 1).size * MU.max() - mean_r.sum()
    print(f"  ✓  {name}")

print()

# ══════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════
print("=" * 62)
print(f"{'Method':<30} {'Total Reward':>14} {'Regret':>10}")
print("=" * 62)
print(f"{'Optimal (always A)':<30} {'$'+f'{OPTIMAL:,.0f}':>14} {'$0':>10}")
print("-" * 62)
for name, _ in methods:
    total_r = all_total_reward[name]
    regret  = all_total_regret[name]
    print(f"{name:<30} {'$'+f'{total_r:,.0f}':>14} {'$'+f'{regret:,.0f}':>10}")
print("=" * 62)

# ══════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "Multi-Armed Bandit — Six Strategy Comparison\n"
    r"Bandits: A($\mu$=0.8), B($\mu$=0.7), C($\mu$=0.5)  |  Budget=$10,000",
    fontsize=14, fontweight='bold'
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

steps = np.arange(1, T + 1)

# ── Plot 1: Average Reward per Step ──────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.axhline(y=MU.max(), color='k', linestyle='--', linewidth=1.5, label='Optimal', alpha=0.6)
for (name, _), color in zip(methods, COLORS):
    ax1.plot(steps, all_mean_rewards[name], color=color, linewidth=1.8, label=name)
ax1.set_xlabel('Step')
ax1.set_ylabel('Running Average Reward')
ax1.set_title('Running Average Reward (cumulative mean)')
ax1.legend(fontsize=8, loc='lower right')
ax1.grid(True, alpha=0.3)

# ── Plot 2: Average Regret per Step ──────────────────────────
ax2 = fig.add_subplot(gs[1, :2])
for (name, _), color in zip(methods, COLORS):
    ax2.plot(steps, all_mean_regret[name], color=color, linewidth=1.8, label=name)
ax2.set_xlabel('Step')
ax2.set_ylabel('Running Average Regret')
ax2.set_title('Running Average Regret (cumulative mean)')
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(True, alpha=0.3)

# ── Plot 3: Final Total Reward Bar ────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
labels_bar = [n.split(' (')[0] for n, _ in methods]
totals     = [all_total_reward[n] for n, _ in methods]
bars = ax3.barh(labels_bar, totals, color=COLORS, alpha=0.8, edgecolor='black')
ax3.axvline(x=OPTIMAL, color='black', linestyle='--', linewidth=1.5, label=f'Optimal ${OPTIMAL:.0f}')
ax3.set_xlabel('Total Reward ($)')
ax3.set_title('Final Total Reward')
ax3.legend(fontsize=8)
for bar, val in zip(bars, totals):
    ax3.text(val + 5, bar.get_y() + bar.get_height() / 2,
             f'${val:,.0f}', va='center', fontsize=8, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# ── Plot 4: Final Regret Bar ──────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
regrets = [all_total_regret[n] for n, _ in methods]
bars4   = ax4.barh(labels_bar, regrets, color=COLORS, alpha=0.8, edgecolor='black')
ax4.set_xlabel('Total Regret ($)')
ax4.set_title('Final Regret (lower = better)')
for bar, val in zip(bars4, regrets):
    ax4.text(val + 1, bar.get_y() + bar.get_height() / 2,
             f'${val:,.0f}', va='center', fontsize=8, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

plt.savefig('ex6_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved → ex6_comparison.png")

# ══════════════════════════════════════════════════════════════
#  ANALYTICAL SUPPLEMENT — A/B Testing
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("ANALYTICAL RESULTS — A/B Testing")
print("=" * 62)
n          = AB_EACH
diff_mean  = MU[0] - MU[1]
diff_std   = np.sqrt((MU[0]*(1-MU[0]) + MU[1]*(1-MU[1])) / n)
z          = diff_mean / diff_std
p_A_wins   = stats.norm.cdf(z)
ab_phase   = n * MU[0] + n * MU[1]
exploit    = (T - 2*n) * (p_A_wins * MU[0] + (1-p_A_wins) * MU[1])
total_ab   = ab_phase + exploit
print(f"  A/B test phase reward (analytical) : ${ab_phase:,.0f}")
print(f"  P(A selected after A/B test)        : {p_A_wins:.8f}  (Z={z:.2f})")
print(f"  Exploitation phase reward           : ${exploit:,.2f}")
print(f"  Total reward (analytical)           : ${total_ab:,.2f}")
print(f"  Regret vs optimal                   : ${OPTIMAL - total_ab:,.2f}")
print(f"    ├─ A/B phase (forced B pulls)     : ${n*(MU[0]-MU[1]):,.0f}")
print(f"    └─ Exploit phase (wrong arm risk) : ${OPTIMAL-total_ab - n*(MU[0]-MU[1]):,.4f}")

# ══════════════════════════════════════════════════════════════
#  EXPLANATION
# ══════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════╗
║  WHY BANDIT ALGORITHMS OUTPERFORM A/B TESTING           ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  A/B Testing                                             ║
║    Fixed exploration phase wastes budget on B even       ║
║    after evidence accumulates that A is better.          ║
║    Regret ≈ $100 (structural, unavoidable).              ║
║                                                          ║
║  Optimistic Initial Values                               ║
║    High Q=1.0 forces early exploration; once arms are    ║
║    "deflated" by real pulls, greedily sticks to A.       ║
║    Simple but sensitive to α choice.                     ║
║                                                          ║
║  ε-Greedy                                                ║
║    Explores uniformly at rate ε=10%.  Wastes ~1,000      ║
║    pulls randomly throughout; linear regret O(T).        ║
║                                                          ║
║  Softmax                                                  ║
║    Probabilistic: pulls A more once Q_A > Q_B but        ║
║    never fully stops exploring weaker arms.              ║
║                                                          ║
║  UCB1                                                    ║
║    Explores arms with high uncertainty; confidence       ║
║    bounds shrink as n_i grows → logarithmic regret.      ║
║                                                          ║
║  Thompson Sampling                                       ║
║    Bayesian posteriors update after every pull.          ║
║    Naturally concentrates on A; O(ln T) regret.          ║
║    Empirically best in most Bernoulli bandit settings.   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")
