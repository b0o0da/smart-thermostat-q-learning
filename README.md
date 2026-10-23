# Smart Room Temperature Control Using Q-Learning

**Course:** Reinforcement Learning – Assignment 1

**Team members**
| Member | Part |
|---|---|
| Yehya Mohamed | Part 1 – Environment |
| Loay Hany | Part 2 – Q-Learning algorithm & Bellman update |
| Abdelrahman Ahmed | Part 3 – Results, plot & discussion |

## Project Overview
A smart thermostat agent learns, by trial and error, whether to turn a heater **ON** or **OFF** so the room stays **comfortable (20°C)** and never overheats. The problem is solved with **tabular Q-Learning** on a custom environment written from scratch (no Gymnasium or prebuilt RL libraries).

**Files**
- `smart_thermostat_qlearning.py` – full implementation (run it with `python smart_thermostat_qlearning.py`)
- `reward_plot.png` – cumulative reward per episode
- `README.md` – this file

**Requirements:** Python 3, `numpy`, `matplotlib`

---

## Part 1 – Environment (Yehya Mohamed)

The environment is built with plain Python functions and dictionaries.

**State space (3 states)**

| Index | State |
|---|---|
| 0 | Cold (15°C) |
| 1 | Comfortable (20°C) |
| 2 | Hot (25°C) |

**Action space (2 actions):** `0 = Heater OFF`, `1 = Heater ON`

**Transition function** – stored in the dictionary `TRANSITIONS` and accessed by `transition(state, action)`:

| Current State | Action | Next State |
|---|---|---|
| Cold | ON | Comfortable |
| Cold | OFF | Cold |
| Comfortable | ON | Hot |
| Comfortable | OFF | Comfortable |
| Hot | ON | Hot |
| Hot | OFF | Comfortable |

**Reward function** – `reward(next_state)` depends on the state reached:
- Comfortable → **+10**
- Hot → **−5**
- Cold → **−1**

**`step(state, action)`** combines both: it returns `(next_state, reward)`, just like an environment step in RL.

**Hyperparameters:** learning rate α = 0.1, discount factor γ = 0.9, 500 episodes, initial state = Cold. We chose 20 steps per episode (the assignment does not specify it), so the maximum possible reward per episode is 200.

---

## Part 2 – Q-Learning Algorithm (Loay Hany)

**Q-table:** a 3×2 NumPy array (3 states × 2 actions) initialised with **zeros**. `Q[s, a]` estimates the total future reward of taking action `a` in state `s`.

**Epsilon-greedy policy** (`choose_action`):
- With probability ε → **explore**: pick a random action.
- Otherwise → **exploit**: pick the action with the highest Q-value.
- ε starts at 1.0 (fully random) and decays by ×0.99 each episode down to a minimum of 0.01, so the agent explores first and exploits later.

**Training loop (500 episodes)**
1. Reset the state to Cold.
2. For each of the 20 steps: choose an action, call `step()`, get the next state and reward.
3. Update the Q-table with the Bellman equation.
4. Move to the next state and accumulate the episode reward.

**Bellman Q-Learning update**

```
Q(s,a) ← Q(s,a) + α · [ r + γ · max_a' Q(s',a') − Q(s,a) ]
```

- `r + γ · max Q(s',a')` is the **target** (immediate reward + discounted best future value).
- `target − Q(s,a)` is the **error**, and α controls how much of it we correct each time.

**Hand calculation**

*Example 1 – first update (Q-table all zeros).* State = Cold, action = ON, r = +10, next state = Comfortable:

```
Q(Cold,ON) = 0 + 0.1 × [10 + 0.9 × 0 − 0]
           = 0.1 × 10
           = 1.0
```

*Example 2 – same transition repeated,* with Q(Cold,ON) = 1.0 and max Q(Comfortable,·) = 1.0:

```
Q(Cold,ON) = 1.0 + 0.1 × [10 + 0.9 × 1.0 − 1.0]
           = 1.0 + 0.1 × 9.9
           = 1.99
```

The script also prints the first real update from training, and it matches Example 1 (new Q = 1.0).

---

## Part 3 – Results, Plot & Discussion (Abdelrahman Ahmed)

**Final Q-table**

| State | OFF | ON | Best action |
|---|---|---|---|
| Cold (15°C) | 86.6 | 100.0 | **Heater ON** |
| Comfortable (20°C) | 100.0 | 85.0 | **Heater OFF** |
| Hot (25°C) | 100.0 | 85.0 | **Heater OFF** |

**Learned optimal policy:** turn the heater ON when Cold, and OFF when Comfortable or Hot.

**Why these numbers make sense**
- Staying Comfortable forever gives +10 every step, so its value is 10 / (1 − 0.9) = **100**. That is Q(Comfortable, OFF) and Q(Hot, OFF).
- Q(Cold, ON) = 10 + 0.9 × 100 = **100**.
- Q(Comfortable, ON) = −5 + 0.9 × 100 = **85**, and Q(Hot, ON) = −5 + 0.9 × 100 = **85** (they lead to or keep the Hot state, which is penalised).
- Q(Cold, OFF) would converge to −1 + 0.9 × 100 ≈ 89 (staying cold once, then moving on). It shows 86.6 because this bad action is rarely tried once ε becomes small, so it has not fully converged. It is still lower than ON, so the policy is unaffected.

**Plot:** `reward_plot.png` shows the cumulative reward per episode (blue) and a 20-episode moving average (red).

**Learning performance**
- Average reward in the first 50 episodes: **≈ 83**. In the last 50 episodes: **≈ 199**, very close to the maximum of 200.
- At the start, ε is high, so the agent acts almost randomly. It often overheats the room (−5) or stays cold (−1), and the reward is low and noisy.
- As the Q-values converge and ε decays, the agent turns the heater ON once when Cold and then keeps it OFF, collecting +10 per step.
- The small drops that remain are caused by the minimum exploration rate (ε = 0.01).
- Conclusion: the agent learned the optimal policy, which keeps the room comfortable and avoids overheating.

---

## How to Run
```
pip install numpy matplotlib
python smart_thermostat_qlearning.py
```
The program prints the Q-table, the optimal policy, the hand calculation and the discussion, and saves `reward_plot.png`.
