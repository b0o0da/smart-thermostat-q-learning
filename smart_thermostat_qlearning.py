"""
Assignment 1 - Smart Room Temperature Control Using Q-Learning
Custom environment (no Gymnasium) + tabular Q-Learning with epsilon-greedy policy.
"""
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------
# 1) State space, action space
# ---------------------------------------------------------------
COLD, COMFORTABLE, HOT = 0, 1, 2
OFF, ON = 0, 1
STATES = {COLD: "Cold (15C)", COMFORTABLE: "Comfortable (20C)", HOT: "Hot (25C)"}
ACTIONS = {OFF: "Heater OFF", ON: "Heater ON"}
N_STATES, N_ACTIONS = len(STATES), len(ACTIONS)

# ---------------------------------------------------------------
# 2) Transition function
# ---------------------------------------------------------------
TRANSITIONS = {
    (COLD, ON): COMFORTABLE,
    (COLD, OFF): COLD,
    (COMFORTABLE, ON): HOT,
    (COMFORTABLE, OFF): COMFORTABLE,
    (HOT, ON): HOT,
    (HOT, OFF): COMFORTABLE,
}

def transition(state, action):
    return TRANSITIONS[(state, action)]

# ---------------------------------------------------------------
# 3) Reward function (based on the state reached)
# ---------------------------------------------------------------
def reward(next_state):
    if next_state == COMFORTABLE:
        return 10
    if next_state == HOT:
        return -5
    return -1  # cold

def step(state, action):
    next_state = transition(state, action)
    return next_state, reward(next_state)

# ---------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------
ALPHA = 0.1          # learning rate
GAMMA = 0.9          # discount factor
EPISODES = 500
STEPS_PER_EPISODE = 20
EPS_START, EPS_MIN, EPS_DECAY = 1.0, 0.01, 0.99

# 4) Q-table initialised with zeros
Q = np.zeros((N_STATES, N_ACTIONS))

def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)       # explore
    best = np.flatnonzero(Q[state] == Q[state].max())
    return int(random.choice(best))              # exploit (random tie-break)

# ---------------------------------------------------------------
# 5) Q-Learning training
# ---------------------------------------------------------------
rewards_per_episode = []
epsilon = EPS_START
first_update_log = None

for ep in range(EPISODES):
    state = COLD                                  # initial state
    total = 0
    for t in range(STEPS_PER_EPISODE):
        action = choose_action(state, epsilon)
        next_state, r = step(state, action)

        old_q = Q[state, action]
        max_next = Q[next_state].max()
        # Bellman Q-learning update
        Q[state, action] = old_q + ALPHA * (r + GAMMA * max_next - old_q)

        if first_update_log is None and r == 10 and old_q == 0:
            first_update_log = (STATES[state], ACTIONS[action], r, old_q,
                                max_next, Q[state, action])
        total += r
        state = next_state
    rewards_per_episode.append(total)
    epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

# ---------------------------------------------------------------
# 6) Results: Q-table + optimal policy
# ---------------------------------------------------------------
print("=" * 60)
print("Final Q-table (rows = states, cols = [OFF, ON])")
print("=" * 60)
print(f"{'State':<20}{'OFF':>12}{'ON':>12}")
for s in range(N_STATES):
    print(f"{STATES[s]:<20}{Q[s, OFF]:>12.3f}{Q[s, ON]:>12.3f}")

print("\nLearned optimal policy:")
for s in range(N_STATES):
    print(f"  {STATES[s]:<20} -> {ACTIONS[int(np.argmax(Q[s]))]}")

# ---------------------------------------------------------------
# 7) Hand calculation of the Bellman update
#    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Hand calculation (Bellman / Q-learning update)")
print("=" * 60)
print("""
Example 1 (very first update, Q-table all zeros):
  s = Cold, a = Heater ON, r = +10, s' = Comfortable
  Q(Cold,ON) = 0 + 0.1 * [ 10 + 0.9 * max Q(Comfortable,.) - 0 ]
             = 0 + 0.1 * [ 10 + 0.9 * 0 - 0 ]
             = 0.1 * 10
             = 1.0

Example 2 (same transition again, Q(Cold,ON)=1.0 and max Q(Comfortable,.)=1.0):
  Q(Cold,ON) = 1.0 + 0.1 * [ 10 + 0.9 * 1.0 - 1.0 ]
             = 1.0 + 0.1 * 9.9
             = 1.0 + 0.99
             = 1.99
""")
if first_update_log:
    s, a, r, oq, mn, nq = first_update_log
    print(f"Check from actual training run: s={s}, a={a}, r={r}, "
          f"old Q={oq:.3f}, max Q(s')={mn:.3f} -> new Q={nq:.3f}")

# ---------------------------------------------------------------
# 8) Plot cumulative reward per episode
# ---------------------------------------------------------------
window = 20
moving = np.convolve(rewards_per_episode, np.ones(window) / window, mode="valid")
plt.figure(figsize=(9, 5))
plt.plot(rewards_per_episode, alpha=0.4, label="Cumulative reward per episode")
plt.plot(range(window - 1, EPISODES), moving, color="red", lw=2,
         label=f"Moving average ({window})")
plt.xlabel("Episode")
plt.ylabel("Cumulative reward")
plt.title("Q-Learning: Smart Thermostat")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("reward_plot.png", dpi=150)
print("\nPlot saved to reward_plot.png")
print(f"Avg reward first 50 episodes: {np.mean(rewards_per_episode[:50]):.1f}")
print(f"Avg reward last 50 episodes : {np.mean(rewards_per_episode[-50:]):.1f}")

print("""
Discussion:
The cumulative reward starts low because epsilon is high and the agent acts
randomly, often overheating the room (-5) or staying cold (-1). As epsilon
decays and the Q-values converge, the agent learns to turn the heater ON when
Cold and then keep it OFF when Comfortable (+10 each step), and to turn it OFF
when Hot. The reward rises quickly and stabilizes near the maximum
(max = 200: +10 in each of the 20 steps), with small drops caused by the
remaining exploration. This shows the agent learned the optimal policy.
""")
