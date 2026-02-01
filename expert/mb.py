# %% [markdown]
# # Model-Based Sepsis RL
# 
# ---

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import sys
sys.path.append('/content/drive/MyDrive/sepsisrl')

# %% [markdown]
# ## Setup

# %%
import autograd.numpy as np
import numpy as onp
from autograd import grad
import autograd.numpy.random as npr
import pandas as pd
from autograd.misc.optimizers import adam
from bayesian_neural_net import make_nn_funs
from PPO import PPO

# %% [markdown]
# Load training data and combine validation and test sets. Use state features from `state_features.txt`.

# %%
state_features = open('/content/drive/MyDrive/sepsisrl/data/state_features.txt').read().splitlines()
train_df = pd.read_csv('/content/drive/MyDrive/sepsisrl/data/rl_train_data_final_cont.csv')
val_df = pd.read_csv('/content/drive/MyDrive/sepsisrl/data/rl_val_data_final_cont.csv')
test_df = pd.read_csv('/content/drive/MyDrive/sepsisrl/data/rl_test_data_final_cont.csv')
eval_df = pd.concat([val_df, test_df], ignore_index=True)
state_dim = len(state_features)
action_bins = 5
action_dim = action_bins * action_bins
print('Train samples:', len(train_df))
print('Eval samples:', len(eval_df))
print('Number of state features:', state_dim)

# %% [markdown]
# Helper function to build transitions where the next state is the following bloc in the same ICU stay, matching the logic used in `cql_q_network.ipynb`.

# %%
action_map = {(iv, vaso): iv * action_bins + vaso for iv in range(action_bins) for vaso in range(action_bins)}


def build_transitions(df):
    states, actions, rewards, next_states, done_flags = [], [], [], [], []
    N = len(df)
    print(f"[data] building transitions over {N} rows…")

    for i in range(N):
        cur_state = onp.asarray(df.loc[i, state_features])
        iv = int(df.loc[i, 'iv_input'])
        vaso = int(df.loc[i, 'vaso_input'])
        iv = int(onp.clip(iv, 0, action_bins - 1))
        vaso = int(onp.clip(vaso, 0, action_bins - 1))
        action_id = action_map[(iv, vaso)]
        action = onp.zeros(action_dim, dtype=onp.float32)
        action[action_id] = 1.0

        if i != N - 1 and df.loc[i, 'icustayid'] == df.loc[i + 1, 'icustayid']:
            nxt = onp.asarray(df.loc[i + 1, state_features])
            reward = float(df.loc[i + 1, 'reward'])
            done = 0
        else:
            nxt = onp.zeros(len(state_features), dtype=onp.float32)
            reward = float(df.loc[i, 'reward'])
            done = 1

        states.append(cur_state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(nxt)
        done_flags.append(done)

    out = (onp.vstack(states), onp.vstack(actions), onp.asarray(rewards, dtype=onp.float32),
           onp.vstack(next_states), onp.asarray(done_flags, dtype=onp.int8))
    print("[data] shapes  states", out[0].shape, "actions", out[1].shape,
          "rewards", out[2].shape, "next_states", out[3].shape, "dones", out[4].shape)
    return out


# %% [markdown]
# ## Bayesian Neural Network Environment

# %%
train_states, train_actions, train_rewards, train_next_states, train_done = build_transitions(train_df)
mask = train_done == 0
inputs = onp.hstack([train_states[mask], train_actions[mask]])
state_deltas = train_next_states[mask] - train_states[mask]
print('Transitions used for training:', inputs.shape[0])
print('Input dimension:', inputs.shape[1])

layer_sizes = [inputs.shape[1], 32, 32, state_deltas.shape[1]]
L2_reg, noise_var = 1.0, 0.1
num_weights, pred_fun, logprob = make_nn_funs(layer_sizes, L2_reg, noise_var)
print("Calculated number of weights:", num_weights)

def black_box_vi(logprob, num_weights, num_samples=20, rs=None):
    """
    Returns (objective, gradient, unpack_params).
    Everything on the gradient path uses autograd.numpy (np).
    """
    if rs is None:
        rs = npr.RandomState(0)

    def objective(params, t):
        # Everything here must use np.*, not onp.*
        mean, log_std = np.split(params, 2)
        eps = rs.randn(num_samples, num_weights)          # OK to sample with non-autograd RNG
        samples = mean + np.exp(log_std) * eps            # np.exp so ArrayBox is handled
        # log q(w)
        log_q = -0.5 * np.sum(
            ((samples - mean) / np.exp(log_std))**2 + 2*log_std + np.log(2*np.pi),
            axis=1
        )
        # log p(w | D) supplied by make_nn_funs
        # Expectation over samples; logprob should be vectorized over samples.
        log_p = logprob(samples, t)
        return np.mean(log_q - log_p)

    gradient = grad(objective)
    unpack = lambda p: np.split(p, 2)
    return objective, gradient, unpack


log_posterior = lambda w, t: logprob(w, inputs, state_deltas)
print('log posterior function ready')
objective, gradient, unpack_params = black_box_vi(log_posterior, num_weights)
print('blackbox ready')
rs = npr.RandomState(0)
init_mean = rs.randn(num_weights)
init_log_std = -5*onp.ones(num_weights)
init_params = onp.concatenate([init_mean, init_log_std])
variational_params = adam(gradient, init_params, step_size=0.01, num_iters=100)
print('variatinal params ready')
mean, log_std = unpack_params(variational_params)
print('Trained BNN with', num_weights, 'parameters')

# %% [markdown]
# Evaluate model on the combined validation and test sets.

# %%

eval_states, eval_actions, _, eval_next_states, eval_done = build_transitions(eval_df)
eval_mask = eval_done == 0
eval_inputs = onp.hstack([eval_states[eval_mask], eval_actions[eval_mask]])
true_deltas = eval_next_states[eval_mask] - eval_states[eval_mask]
weights = mean[None, :]
pred_deltas = pred_fun(weights, eval_inputs)[0]
mse = onp.mean((pred_deltas - true_deltas)**2)
print('Evaluation transitions:', eval_inputs.shape[0])
print('MSE on val+test:', mse)


# %% [markdown]
# ## Policy Search with PPO

# %%

class SepsisBNNEnv:
    def __init__(self, mean, log_std, pred_fun):
        self.mean = mean
        self.log_std = log_std
        self.pred_fun = pred_fun
        self.rs = npr.RandomState(0)
        self.reset()
    def reset(self):
        self.idx = self.rs.randint(0, len(train_states))
        self.state = train_states[self.idx]
        return self.state
    def step(self, action):
        if onp.ndim(action) > 0:
            action_id = int(onp.argmax(action))
        else:
            action_id = int(action)
        action_id = int(onp.clip(action_id, 0, action_dim - 1))
        action_vec = onp.zeros(action_dim, dtype=onp.float32)
        action_vec[action_id] = 1.0
        weights = self.mean + onp.exp(self.log_std) * self.rs.randn(1, len(self.mean))
        delta = self.pred_fun(weights, onp.hstack([self.state, action_vec])[None, :])[0]
        next_state = self.state + delta
        reward = train_rewards[self.idx]
        done = bool(train_done[self.idx])
        self.idx = (self.idx + 1) % len(train_states)
        self.state = next_state
        return next_state, reward, done, {}

env = SepsisBNNEnv(mean, log_std, pred_fun)
ppo = PPO(state_dim=state_dim, action_dim=action_dim, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, K_epochs=5, eps_clip=0.2, has_continuous_action_space=False)
for episode in range(2):
    state = env.reset()
    for t in range(5):
        action = ppo.select_action(state)
        next_state, reward, done, _ = env.step(action)
        ppo.buffer.rewards.append(reward)
        ppo.buffer.is_terminals.append(done)
        state = next_state
        if done:
            print(f"[ppo] episode {episode} terminated at step {t}")
            break
    print(f"[ppo] update after episode {episode}")
    ppo.update()
print('[ppo] training loop finished')


# %% [markdown]
# ## Combining PPO with Clinician Policy based on SOFA
# 
# Placeholder for blending strategies that rely on clinician policy when SOFA indicates low-confidence regions.
