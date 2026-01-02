# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)

# %%
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')

# %%
import os

# Root of your project in Drive
PROJECT_DIR = "/content/drive/MyDrive/sepsisrl"
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "outputs")

# Ensure dirs exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Project:", PROJECT_DIR)
print("Data:", DATA_DIR)
print("Outputs:", OUTPUT_DIR)

# %%
import random
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

# %%
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# %%
state_features_path = os.path.join(DATA_DIR, "state_features.txt")
with open(state_features_path) as f:
    state_features = f.read().split()
print((state_features))
print(len(state_features))

# %%
train_path = os.path.join(DATA_DIR, "rl_train_data_final_cont.csv") # used no-term
df = pd.read_csv(train_path)

# %%
df.head()

# %%
val_path = os.path.join(DATA_DIR, "rl_val_data_final_cont.csv")
val_df = pd.read_csv(val_path)

# %%
test_path = os.path.join(DATA_DIR, "rl_test_data_final_cont.csv")
test_df = pd.read_csv(test_path)

# %%
REWARD_THRESHOLD = 20
reg_lambda = 5

# %%
# PER important weights and params
per_flag = True
beta_start = 0.9
df['prob'] = abs(df['reward'])
temp = 1.0/df['prob']
#temp[temp == float('Inf')] = 1.0
df['imp_weight'] = pow((1.0/len(df) * temp), beta_start)

# %%
# SET THIS TO FALSE
clip_reward = False # was set to true

# %%
hidden_1_size = 128
hidden_2_size = 128

class DuelingQNetwork(tf.keras.Model):
    def __init__(self, input_dim, num_actions=25, leaky_alpha=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions

        # Dense -> BatchNorm -> LeakyReLU (no bias in Dense to match BN)
        self.fc1 = tf.keras.layers.Dense(hidden_1_size, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(center=True, scale=True)
        self.act1 = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)

        self.fc2 = tf.keras.layers.Dense(hidden_2_size, use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(center=True, scale=True)
        self.act2 = tf.keras.layers.LeakyReLU(alpha=leaky_alpha)

        # Heads after channel split (linear maps)
        kinit = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
        self.adv_head = tf.keras.layers.Dense(num_actions, kernel_initializer=kinit, bias_initializer="zeros")
        self.val_head = tf.keras.layers.Dense(1,          kernel_initializer=kinit, bias_initializer="zeros")

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)

        # split last dimension into two streams
        streamA, streamV = tf.split(x, num_or_size_splits=2, axis=-1)
        A = self.adv_head(streamA)  # [B, A]
        V = self.val_head(streamV)  # [B, 1]
        q = V + (A - tf.reduce_mean(A, axis=1, keepdims=True))
        return q


# %%
def soft_update(main_model, target_model, tau=0.001):
    for w_main, w_tgt in zip(main_model.weights, target_model.weights):
        w_tgt.assign((1.0 - tau) * w_tgt + tau * w_main)

# %%
# define an action mapping - how to get an id representing the action from the (iv,vaso) tuple
action_map = {}
count = 0
for iv in range(5):
    for vaso in range(5):
        action_map[(iv,vaso)] = count
        count += 1

# %%
def process_train_batch(size):
    if per_flag:
        # uses prioritised exp replay
        a = df.sample(n=size, weights=df['prob'])
    else:
        a = df.sample(n=size)
    states = None
    actions = None
    rewards = None
    next_states = None
    done_flags = None
    for i in a.index:
        cur_state = a.loc[i,state_features]
        iv = int(a.loc[i, 'iv_input'])
        vaso = int(a.loc[i, 'vaso_input'])
        action = action_map[iv,vaso]
        reward = a.loc[i,'reward']

        if clip_reward:
            if reward > 1: reward = 1
            if reward < -1: reward = -1

        if i != df.index[-1]:
            # if not terminal step in trajectory
            if df.loc[i, 'icustayid'] == df.loc[i+1, 'icustayid']:
                next_state = df.loc[i + 1, state_features]
                done = 0
            else:
                # trajectory is finished
                next_state = np.zeros(len(cur_state))
                done = 1
        else:
            # last entry in df is the final state of that trajectory
            next_state = np.zeros(len(cur_state))
            done = 1

        if states is None:
            states = copy.deepcopy(cur_state)
        else:
            states = np.vstack((states,cur_state))

        if actions is None:
            actions = [action]
        else:
            actions = np.vstack((actions,action))

        if rewards is None:
            rewards = [reward]
        else:
            rewards = np.vstack((rewards,reward))

        if next_states is None:
            next_states = copy.deepcopy(next_state)
        else:
            next_states = np.vstack((next_states,next_state))

        if done_flags is None:
            done_flags = [done]
        else:
            done_flags = np.vstack((done_flags,done))

    return (states, np.squeeze(actions), np.squeeze(rewards), next_states, np.squeeze(done_flags), a)

# %%
# extract chunks of length size from the relevant dataframe, and yield these to the caller
def process_eval_batch(size, eval_type = None):
    if eval_type is None:
        raise Exception('Provide eval_type to process_eval_batch')
    elif eval_type == 'train':
        a = df.copy()
    elif eval_type == 'val':
        a = val_df.copy()
    elif eval_type == 'test':
        a = test_df.copy()
    else:
        raise Exception('Unknown eval_type')
    count = 0
    while count < len(a.index):
        states = None
        actions = None
        rewards = None
        next_states = None
        done_flags = None

        start_idx = count
        end_idx = min(len(a.index), count+size)
        segment = a.index[start_idx:end_idx]

        for i in segment:
            cur_state = a.loc[i,state_features]
            iv = int(a.loc[i, 'iv_input'])
            vaso = int(a.loc[i, 'vaso_input'])
            action = action_map[iv,vaso]
            reward = a.loc[i,'reward']

            if clip_reward:
                if reward > 1: reward = 1
                if reward < -1: reward = -1

            if i != a.index[-1]:
                # if not terminal step in trajectory
                if a.loc[i, 'icustayid'] == a.loc[i+1, 'icustayid']:
                    next_state = a.loc[i + 1, state_features]
                    done = 0
                else:
                    # trajectory is finished
                    next_state = np.zeros(len(cur_state))
                    done = 1
            else:
                # last entry in df is the final state of that trajectory
                next_state = np.zeros(len(cur_state))
                done = 1

            if states is None:
                states = copy.deepcopy(cur_state)
            else:
                states = np.vstack((states,cur_state))

            if actions is None:
                actions = [action]
            else:
                actions = np.vstack((actions,action))

            if rewards is None:
                rewards = [reward]
            else:
                rewards = np.vstack((rewards,reward))

            if next_states is None:
                next_states = copy.deepcopy(next_state)
            else:
                next_states = np.vstack((next_states,next_state))

            if done_flags is None:
                done_flags = [done]
            else:
                done_flags = np.vstack((done_flags,done))

        yield (states, np.squeeze(actions), np.squeeze(rewards), next_states, np.squeeze(done_flags), a)

        count += size
#         if count >= 3000:
#             break


# %%
def do_eval(eval_type, mainQN, targetQN, gamma):
    if eval_type == 'train':
        gen = (process_train_batch(32) for _ in range(len(df)//32))
    elif eval_type == 'val':
        gen = (process_train_batch(32) for _ in range(len(val_df)//32))
    else:
        gen = (process_train_batch(32) for _ in range(len(test_df)//32))

    phys_q_ret, actions_ret, agent_q_ret, actions_taken_ret, error_ret = [], [], [], [], 0.0

    for states, actions, rewards, next_states, done_flags, _ in gen:
        states_tf      = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions_tf     = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards_tf     = tf.convert_to_tensor(rewards, dtype=tf.float32)
        done_tf        = tf.convert_to_tensor(done_flags, dtype=tf.float32)

        # Double Q: argmax from main, value from target
        q_next_main  = mainQN(next_states_tf, training=False)     # [B, A]
        actions_from_q1 = tf.argmax(q_next_main, axis=1, output_type=tf.int32)  # [B]

        q_next_tgt   = targetQN(next_states_tf, training=False)   # [B, A]
        idx_next     = tf.stack([tf.range(tf.shape(actions_from_q1)[0]), actions_from_q1], axis=1)
        double_q_val = tf.gather_nd(q_next_tgt, idx_next)         # [B]
        double_q_val = tf.clip_by_value(double_q_val, -REWARD_THRESHOLD, REWARD_THRESHOLD)

        targets = rewards_tf + gamma * double_q_val * (1.0 - done_tf)

        q_now = mainQN(states_tf, training=False)                 # [B, A]
        idx_now = tf.stack([tf.range(tf.shape(actions_tf)[0]), actions_tf], axis=1)
        phys_q = tf.gather_nd(q_now, idx_now)                     # [B]

        actions_taken = tf.argmax(q_now, axis=1, output_type=tf.int32)
        agent_q = tf.gather_nd(q_now, tf.stack([tf.range(tf.shape(actions_taken)[0]), actions_taken], axis=1))

        abs_err = tf.abs(targets - phys_q)
        error_ret += float(tf.reduce_mean(abs_err).numpy())

        phys_q_ret.extend(phys_q.numpy())
        actions_ret.extend(actions_tf.numpy())
        agent_q_ret.extend(agent_q.numpy())
        actions_taken_ret.extend(actions_taken.numpy())

    return phys_q_ret, actions_ret, agent_q_ret, actions_taken_ret, error_ret


# %%
phys_q_train = []
agent_q_train = []
phys_actions_tr = []
agent_actions_tr = []

def train_set_performance(mainQN):
    global phys_q_train, agent_q_train, phys_actions_tr, agent_actions_tr
    phys_q_train, agent_q_train, phys_actions_tr, agent_actions_tr = [], [], [], []
    for r in df.index:
        cur_state = np.asarray([df.loc[r, state_features]], dtype=np.float32)
        q = np.squeeze(mainQN(cur_state, training=False).numpy())
        iv = int(df.loc[r, 'iv_input'])
        vaso = int(df.loc[r, 'vaso_input'])
        action = action_map[(iv, vaso)]
        phys_q_train.append(q[action])
        agent_q_train.append(np.max(q))
        agent_actions_tr.append(int(np.argmax(q)))
        phys_actions_tr.append(action)


# %%
def do_save_results(mainQN, save_dir):
    _, _, agent_q_train_l, agent_actions_train, _ = do_eval('train', mainQN, mainQN, 0.0)
    joblib.dump(agent_actions_train, os.path.join(save_dir, 'dqn_normal_actions_train.pkl'))

    _, _, agent_q_val, agent_actions_val, _ = do_eval('val', mainQN, mainQN, 0.0)
    _, _, agent_q_test, agent_actions_test, _ = do_eval('test', mainQN, mainQN, 0.0)

    joblib.dump(agent_actions_val,  os.path.join(save_dir, 'dqn_normal_actions_val.pkl'))
    joblib.dump(agent_actions_test, os.path.join(save_dir, 'dqn_normal_actions_test.pkl'))
    joblib.dump(agent_q_train_l,    os.path.join(save_dir, 'dqn_normal_q_train.pkl'))
    joblib.dump(agent_q_val,        os.path.join(save_dir, 'dqn_normal_q_val.pkl'))
    joblib.dump(agent_q_test,       os.path.join(save_dir, 'dqn_normal_q_test.pkl'))


# %%
pwd

# %%
# === CHANGED FOR TF2: optimizer, checkpointing, and training loop (no Sessions/Savers) ===
# Hyperparameters (keep your originals)
per_alpha = 0.6
per_epsilon = 0.01
batch_size = 32
gamma = 0.99
num_steps = 80000
load_model = True
save_dir = os.path.join(OUTPUT_DIR, "cql_phys")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
ckpt_dir = os.path.join(save_dir, "ckpt")

tau = 0.001

# === CHANGED FOR TF2: add a small helper near the top of your training cell ===
def _head(arr, k=32):
    """Return a small numpy array preview for logging."""
    a = np.asarray(arr)
    return a[:k] if a.ndim else a

print("Init done")  # matches original vibe after restore/build


# Build models
input_dim = len(state_features)
mainQN   = DuelingQNetwork(input_dim=input_dim, num_actions=25)
targetQN = DuelingQNetwork(input_dim=input_dim, num_actions=25)

# Materialize variables
_ = mainQN(tf.zeros((1, input_dim), dtype=tf.float32), training=False)
_ = targetQN(tf.zeros((1, input_dim), dtype=tf.float32), training=False)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Checkpoints
ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=mainQN, target_model=targetQN)
manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)

# Try restore PER/IS weights
try:
    per_weights = joblib.load(os.path.join(save_dir, "per_weights.pkl"))
    imp_weights = joblib.load(os.path.join(save_dir, "imp_weights.pkl"))
    df['prob'] = per_weights
    df['imp_weight'] = imp_weights
    print("PER and Importance weights restored")
except Exception:
    print("No PER/IS weights found - using current defaults")

# Restore model
if load_model and manager.latest_checkpoint:
    print(f"Restoring from {manager.latest_checkpoint} ...")
    ckpt.restore(manager.latest_checkpoint).expect_partial()
else:
    print("No checkpoint found or load_model=False. Starting fresh.")

cql_alpha = 0.1  # keep your CQL scaling

@tf.function
def train_step(states, actions, rewards, next_states, done_flags, imp_sampling_weights):
    with tf.GradientTape() as tape:
        # Q(s, a)
        q_now_all = mainQN(states, training=True)  # [B, A]
        idx_now   = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
        q_sa      = tf.gather_nd(q_now_all, idx_now)         # [B]

        # Double Q target
        q_next_main  = mainQN(next_states, training=False)
        next_best    = tf.argmax(q_next_main, axis=1, output_type=tf.int32)
        q_next_tgt   = targetQN(next_states, training=False)
        idx_next     = tf.stack([tf.range(tf.shape(next_best)[0]), next_best], axis=1)
        q_next       = tf.gather_nd(q_next_tgt, idx_next)
        q_next       = tf.clip_by_value(q_next, -REWARD_THRESHOLD, REWARD_THRESHOLD)

        targets = rewards + gamma * q_next * (1.0 - done_flags)

        # TD error and PER-weighted Huber loss
        td_error = targets - q_sa
        abs_error = tf.abs(td_error)
        base_loss = tf.keras.losses.huber(targets, q_sa, delta=1.0)  # per-sample
        w = imp_sampling_weights / (tf.reduce_mean(imp_sampling_weights) + 1e-8)
        loss = tf.reduce_mean(w * base_loss)

        # Regularizer: penalize |Q| beyond threshold (on chosen action)
        reg_vector = tf.nn.relu(tf.abs(q_sa) - REWARD_THRESHOLD)
        reg_term = tf.reduce_sum(reg_vector)
        loss += reg_lambda * reg_term

        # CQL term: log-sum-exp over actions minus Q(s,a_chosen)
        cql_term = tf.reduce_mean(tf.reduce_logsumexp(q_now_all, axis=1) - q_sa)
        loss += cql_alpha * cql_term

    grads = tape.gradient(loss, mainQN.trainable_variables)
    optimizer.apply_gradients(zip(grads, mainQN.trainable_variables))
    return loss, abs_error

# Main training
os.makedirs(save_dir, exist_ok=True)
net_loss = 0.0

for i in range(num_steps):
    ckpt.step.assign_add(1)

    # sample batch
    states, actions_batch, rewards, next_states, done_flags, sampled_df = process_train_batch(batch_size)

    # Tensors
    states_tf      = tf.convert_to_tensor(states, dtype=tf.float32)
    next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
    actions_tf     = tf.convert_to_tensor(actions_batch, dtype=tf.int32)
    rewards_tf     = tf.convert_to_tensor(rewards, dtype=tf.float32)
    done_tf        = tf.convert_to_tensor(done_flags, dtype=tf.float32)

    # PER IS weights
    max_imp = float(max(df['imp_weight'])) if len(df) > 0 else 1.0
    imp_sampling_weights = np.array(sampled_df['imp_weight'] / max(max_imp, 1e-8))
    imp_sampling_weights = np.clip(np.nan_to_num(imp_sampling_weights, nan=1.0, posinf=1.0, neginf=1.0), 1e-3, None)
    imp_tf = tf.convert_to_tensor(imp_sampling_weights, dtype=tf.float32)

    # one optimization step
    loss, abs_err = train_step(states_tf, actions_tf, rewards_tf, next_states_tf, done_tf, imp_tf)

    # current-batch agent actions (authors' "chosen actions")
    cur_act = tf.argmax(mainQN(states_tf, training=False), axis=1).numpy()

    # soft target update
    soft_update(mainQN, targetQN, tau)

    # update PER priorities and IS weights in dataframe
    error_np = abs_err.numpy()
    net_loss += float(np.sum(error_np))
    new_weights = np.power((error_np + per_epsilon), per_alpha)
    df.loc[df.index.isin(sampled_df.index), 'prob'] = new_weights
    temp = 1.0 / new_weights
    df.loc[df.index.isin(sampled_df.index), 'imp_weight'] = np.power(((1.0/len(df)) * temp), beta_start)

    step = int(ckpt.step.numpy())


    if step % 1000 == 0:
        manager.save()
        av_loss = net_loss / (1000.0 * batch_size)
        print(f"Saved Model, step is {step}")
        print("Average loss is ", av_loss)
        net_loss = 0.0
        print("Saving PER and importance weights")
        joblib.dump(df['prob'],       os.path.join(save_dir, 'per_weights.pkl'))
        joblib.dump(df['imp_weight'], os.path.join(save_dir, 'imp_weights.pkl'))

    if step % 5000 == 0:
        # Validation snapshot (DDQN-style target with CQL training)
        phys_q, phys_actions, agent_q, agent_actions, mean_abs_error = do_eval('val', mainQN, targetQN, gamma)

        # Match the authors’ intermediate prints
        print("physactions ", _head(phys_actions, 32))
        print("chosen actions ", _head(agent_actions, 32))
        print(float(mean_abs_error))                 # mean abs TD error (accumulated average across mini-batches)
        print(np.mean(phys_q))                       # avg Q for physician-chosen actions
        print(np.mean(agent_q))                      # avg Q for agent-chosen actions
        print("Saving results")
        do_save_results(mainQN, save_dir)
        print("length IS ", len(df['imp_weight']))

# Final save
do_save_results(mainQN, save_dir)


# %%
pd.Series(phys_actions).value_counts()

# %%
pd.Series(agent_actions).value_counts()


