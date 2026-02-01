# %%
# Colab quickstart (optional)
# from google.colab import drive
# drive.mount("/content/drive")
# %cd /content/drive/MyDrive/sepsisrl
# !pip install -r requirements.txt
# python expert/moe.py --config final_config.yaml --train_or_load --eval_split val
#
# --- CLI entrypoint for reproducible reporting ---
import sys
from pathlib import Path

if "--config" in sys.argv:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from expert.policy_runner import run_policy_cli

    run_policy_cli("moe")
    raise SystemExit(0)

# %%
import pandas as pd
import pickle as pkl
import string
import numpy as np;
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Note that if it not worthwhile to run the following code on GPU
# since even if we try to vectorize it as much as possible
# the objective function (computational heavy) still contains many loops

# %%
train_set = pd.read_csv('path/to/trainset')
test_set = pd.read_csv('path/to/testset')

# %%
train_actions = train_set['vaso_input'].values + 5 * train_set['iv_input'].values
test_actions = test_set['vaso_input'].values + 5 * test_set['iv_input'].values

# %%
# patient state characteristics, cirtical features for expert selection
attr_cols = ['age', 'elixhauser', 'SOFA', 'GCS', 'FiO2_1', 'BUN', 'Albumin']

# %%
# get ICU-stay block for each patient, say, row 1-4 are the records for patient 1, row 5-10 are for patient 2,
# then, fence_post returns [0, 4, 10 ...] so on and so forth, this is used for WDR estimate.
def get_fence_post(df):
    fence_posts = []
    bloc = df['bloc'].values
    for i, idx in enumerate(bloc):
        if idx == 0:
            fence_posts += [ i ]

    return np.array(fence_posts)

def get_traj_len(df):
    traj_len = []
    fence_posts = get_fence_post(df)
    for i in range(len(fence_posts)-1):
        traj_len += list(range(fence_posts[i+1] - fence_posts[i] ))
    
    traj_len += list(range(df.shape[0] - fence_posts[-1]))
    
    return np.array(traj_len) + 1

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def actions_2_probs(ind, src_actions, num_actions=25, expert='kernel'):
    
    selected_actions = src_actions[ind]
    selected_actions[selected_actions == 0] = -1
    
    action_probs = np.zeros((selected_actions.shape[0], num_actions))
    
    if expert == 'kernel':
        actor_actions = selected_actions * np.isin(ind, train_survivors)
    else:
        actor_actions = selected_actions
    
    for i in range(actor_actions.shape[0]):
        actions = actor_actions[i]
        a, c = np.unique(actions[actions != 0], return_counts=True)
        a[a == -1] = 0
        action_probs[i, a] = c / np.sum(c)
    
    return action_probs

def restrict_actions(target, src, th):
    restricted_target = (target * (src > th))
    return restricted_target / np.sum(restricted_target, axis=1, keepdims=True)

def get_ir_diff(df):
    uids = np.unique(df['icustayid'])
    irs = np.zeros(df.shape[0])
    counter = 0
    for uid in uids:
        u_rewards = df[ df['icustayid'] == uid]['neg_mortality_logodds']
        irs[counter: counter + u_rewards.shape[0] - 1] = u_rewards[1:].values - u_rewards[:-1].values
        counter += u_rewards.shape[0]
    return irs

def get_expert_dist(pi_e, pi_k, pi_d):
    
    a_pi_e = np.argmax(pi_e, axis=1)
    a_pi_k = np.argmax(pi_k, axis=1)
    a_pi_d = np.argmax(pi_d, axis=1)
    
    s_k = np.where( a_pi_e == a_pi_k )[0]
    s_d = np.where( a_pi_e == a_pi_d )[0]
    
    ns = get_moe_unique_action(pi_e, pi_k, pi_d)
    
    print ( 's_k: ', s_k.shape[0] / pi_e.shape[0], 's_d:', s_d.shape[0] / pi_e.shape[0], 'ns: ', ns / pi_e.shape[0])

# %%
class MOE(nn.Module):
    
    def __init__(self, input_size, logits=2):
        
        super(MOE, self).__init__()
        
        self.logits = logits
        self.linear = nn.Linear(input_size, logits)

        if logits == 2:
            self.activation = nn.Softmax()
        elif logits == 1: 
            self.activation = nn.Sigmoid()

    def forward(self, x):
        wx = self.linear(x)
        return self.activation(wx)

# %%
def WDR(
    actions_sequence, rewards_sequence, fence_posts, gamma,
    pi_evaluation, pi_behavior, V = None, Q = None, num_of_actions = None ):

    num_of_trials = len( fence_posts )
    # get weight table
    whole_rho = Variable(torch.zeros((num_of_trials, 21)))
    
    for trial_i in range( num_of_trials ):
        
        rho = 1
        trial_rho = torch.zeros(21)
        trial_rho[0] = rho
        trial_rho = Variable(trial_rho) 
        
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = actions_sequence.shape[0] - fence_posts[-1]
        for t in range(
                fence_posts[ trial_i], fence_posts[ trial_i ] + steps_in_trial ):
            previous_rho = rho
            rho = rho * (pi_evaluation[ t, actions_sequence[t]] / \
                pi_behavior[ t, actions_sequence[t]])
            trial_aux = torch.zeros(21)
            trial_aux[t - fence_posts[ trial_i] + 1] = 1
            trial_aux = Variable(trial_aux)
            trial_rho = trial_rho + trial_aux * rho
        
        if steps_in_trial < 20:
            for t in range(fence_posts[ trial_i ] + steps_in_trial, fence_posts[ trial_i ] + 20):
                trial_aux = torch.zeros(21)
                trial_aux[t - fence_posts[ trial_i]+1] = 1
                trial_aux = Variable(trial_aux)
                trial_rho = trial_rho + trial_aux * rho
    
        whole_aux = torch.zeros((num_of_trials, 21))
        whole_aux[trial_i, :] = 1
        whole_rho = whole_rho + Variable(whole_aux) * trial_rho
        
    weight_table = whole_rho / torch.sum(whole_rho, dim = 0)
    
    estimator = 0
    # calculate the doubly robust estimator of the policy
    for trial_i in range(num_of_trials):
        discount = 1 / gamma
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = actions_sequence.shape[0] - fence_posts[-1]
        for t in range(
                fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):
            previous_weight = weight_table[trial_i, t - fence_posts[ trial_i]]
            weight = weight_table[trial_i, t - fence_posts[ trial_i]+1]
            discount = discount * gamma
            r = rewards_sequence[ t ]
            Q_value = Q[ t, actions_sequence[t]]
            V_value = V[t]
            estimator = estimator + weight * discount * r - discount * ( weight * Q_value - previous_weight * V_value ) 
    
    return estimator

# %%
# WDR in numpy version
def np_WDR(
    actions_sequence, rewards_sequence, fence_posts, gamma,
    pi_evaluation, pi_behavior, V = None, Q = None, num_of_actions = None, return_weights=False):

    num_of_trials = len( fence_posts )
    # get weight table
    whole_rho = np.zeros((num_of_trials, 21))
    for trial_i in range( num_of_trials ):
        rho = 1
        trial_rho = np.zeros(21)
        trial_rho[0] = rho
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len( actions_sequence) - fence_posts[-1]
        for t in range(
                fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):
            previous_rho = rho
            rho *= pi_evaluation[ t, actions_sequence[t]] / \
                pi_behavior[ t, actions_sequence[ t]]
            trial_aux = np.zeros(21)
            trial_aux[t - fence_posts[ trial_i]+1] = 1
            trial_rho = trial_rho + trial_aux*rho
        
        if steps_in_trial < 20:
            for t in range(fence_posts[ trial_i] + steps_in_trial, fence_posts[trial_i] + 20):
                
                trial_aux = np.zeros(21)
                trial_aux[t - fence_posts[ trial_i]+1] = 1
                trial_rho = trial_rho + trial_aux*rho
    
        whole_aux = np.zeros((num_of_trials, 21))
        whole_aux[trial_i, :] = 1
        whole_rho += whole_aux*trial_rho
        
    weight_table = whole_rho/np.sum(whole_rho, axis = 0)
    
    estimator = 0
    #pa, pb = 0, 0
    # calculate the doubly robust estimator of the policy
    for trial_i in range(num_of_trials):
        discount = 1/gamma
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len(actions_sequence) - fence_posts[-1]
        for t in range(
                fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):
            previous_weight = weight_table[trial_i, t - fence_posts[ trial_i]]
            weight = weight_table[trial_i, t - fence_posts[ trial_i]+1]
            discount *= gamma
            r =  rewards_sequence[ t ]
            Q_value=  Q[ t, actions_sequence[ t ] ] 
            V_value =  V[t]
            estimator =  estimator + weight * discount * r - discount * ( weight * Q_value - previous_weight * V_value )
    
    if return_weights:
        return estimator, whole_rho
    else:
        return estimator

# %%
def evaluate(pi_e, phase='train', VQ='phy'):
    
    
    if phase == 'train':
        
        train_fence_posts = get_fence_post(train_df)
        
        if VQ == 'phy':
            V, Q = phy_train_V, phy_train_Q
        elif VQ == 'dqn':
            V, Q = dqn_train_V, dqn_train_Q
            
        wdr = np_WDR(train_actions, train_rewards, train_fence_posts, .99, pi_e, train_pi_b, V, Q)

    elif phase == 'test':
        
        test_fence_posts = get_fence_post(test_df)
        
        if VQ == 'phy':
            V, Q = phy_test_V, phy_test_Q
        elif VQ == 'dqn':
            V, Q = dqn_test_V, dqn_test_Q
        
        wdr = np_WDR(test_actions, test_rewards, test_fence_posts, .99, pi_e, test_pi_b, V, Q)
        
    return wdr

# %%
def objective(action_seq, rewards, fence_posts, pi_e, pi_b, V, Q):
    return -WDR(action_seq, rewards, fence_posts, .99, pi_e, pi_b, V, Q)

# %%
def do_eval_test(moe=None, pi_e_type='moe'):
    
    x = Variable(torch.FloatTensor(test_df[attr_cols + ['traj_len', 'dist']].values))

    action_prob_k = Variable(torch.FloatTensor(test_df.values[:,:25]))
    action_prob_d = Variable(torch.FloatTensor(test_df.values[:,25:50]))

    if pi_e_type == 'moe':
        probs = moe(x)
        # print ('expert dist:', np.unique(np.argmax(probs.data.numpy(), axis=1), return_counts=True))
        if moe.logits == 2:
            pi_e = torch.unsqueeze(probs[:,0], 1) * action_prob_k + torch.unsqueeze(probs[:,1], 1) * action_prob_d
        elif moe.logits == 1:
            pi_e = probs * action_prob_k + (1-probs) * action_prob_d
    elif pi_e_type == 'kernel':
        pi_e = action_prob_k
    else:
        pi_e = action_prob_d
    
    pi_e = pi_e.data.numpy()
    fence_posts = get_fence_post(test_df)
    
    return np_WDR(test_actions, test_rewards, fence_posts, .99, pi_e, test_pi_b, phy_test_V, phy_test_Q)

# %%
def get_moe_policies(moe, df):
    
    x = Variable(torch.FloatTensor(df[attr_cols + ['traj_len', 'dist']].values))
    action_prob_k = Variable(torch.FloatTensor(df.values[:,:25]))
    action_prob_d = Variable(torch.FloatTensor(df.values[:,25:50]))

    probs = moe(x)
    if moe.logits == 2:
        pi_e = torch.unsqueeze(probs[:,0], 1) * action_prob_k + torch.unsqueeze(probs[:,1], 1) * action_prob_d
    elif moe.logits == 1:
        pi_e = probs * action_prob_k + (1-probs) * action_prob_d
    
    return probs.data.numpy(), pi_e.data.numpy()

# %%
def pretrained(moe, pretrained_weight, pretrained_bias):
    moe.linear.weight.data = torch.FloatTensor(pretrained_weight)
    moe.linear.bias.data = torch.FloatTensor(pretrained_bias)

# %%
def train(df, moe, batch_size=128, lr=0.001, num_epoch=10):
    
    uids = np.unique(df['icustayid'].values)
    np.random.shuffle(uids)
    num_batch = uids.shape[0] // batch_size
    
    optimizer = torch.optim.Adam(moe.parameters(), lr=lr)
    prev_obj = 0
    prev_obj_train = 0
    stop_counter = 0
    
    for epoch in range(num_epoch):
        
        for batch_idx in range(num_batch):
            
            batch_uids = uids[batch_idx*batch_size: (batch_idx+1)*batch_size]
            batch_user = df[df['icustayid'].isin(batch_uids)]
            batch_user_idx = batch_user.index.values
            
            x = Variable(torch.FloatTensor(batch_user[attr_cols + ['traj_len', 'dist']].values))
            
            action_prob_k = Variable(torch.FloatTensor(batch_user.values[:,:25]))
            action_prob_d = Variable(torch.FloatTensor(batch_user.values[:,25:50]))
            
            probs = moe(x)
            if moe.logits == 2:
                pi_e = torch.unsqueeze(probs[:,0], 1) * action_prob_k + \
                    torch.unsqueeze(probs[:,1], 1) * action_prob_d
            elif moe.logits == 1:
                pi_e = probs * action_prob_k + (1-probs) * action_prob_d
            
            fence_posts = get_fence_post(batch_user)
            
            action_seq = train_actions[batch_user_idx]
            rewards = batch_user['reward'].values
            
            pi_b = Variable(torch.FloatTensor(train_pi_b[batch_user_idx]))

            
            Q = Variable(torch.FloatTensor(phy_train_Q[batch_user_idx]))
            V = torch.max(Q, dim=1)[0]
            
            obj = objective(action_seq, rewards, fence_posts, pi_e, pi_b, V, Q)
            
            if np.isnan(obj.data[0]):
                return 0
            
            optimizer.zero_grad()
            obj.backward()
            optimizer.step()

        print ('********************')
        print('Epoch:{}/{}, wdr:{}'.format(epoch + 1, num_epoch, -obj.data[0]))
        wdr = do_eval_test(moe)
        print ('********************')
        print('Eval: epoch:{}/{}, wdr:{}'.format(epoch + 1, num_epoch, wdr))
        for param in moe.linear.parameters():
            print(param.data.numpy().tolist())
        print ('********************')
        
#         if prev_obj > wdr or prev_obj_train > -obj.data[0]:
#             stop_counter += 1
#         else:
#             stop_counter = 0
#         prev_obj = wdr
#         prev_obj_train = -obj.data[0]
        
#         if stop_counter == 1:
#             return 1

# %%
'''
    Note that when deriving **kernel policy** on test set, 
    one should look at the similar patient states from trainset.

    Note that when deriving **physician policy** for test set, 
    one should look at the similar patient states within the test set
'''
# indices of neighbors can be obtained through "kernel.ipynb"
# or call the code below
'''
##############################################################

for 1) deriving kernel policy over train and test sets; 
    2) deriving physician policy over train set;

knn = KNN(300)
knn.fit(train_embeddings)

train_dist, train_ind = knn.kneighbors(train_embeddings)
test_from_train_dist, test_from_train_ind = knn.kneighbors(test_embeddings)

##############################################################

for 1) deriving physician policy over test sets; 
knn_phy = KNN(300)
knn_phy.fit(test_embeddings)
phy_test_dist, phy_test_ind = knn_phy.kneighbors(test_embeddings)

##############################################################
'''
dist_train, ind_train = pkl.load(open('kernel_knn_train.pkl', 'rb'))
# for kernel policy derviation over testset, indices of neighbors shall from the trainset
# however, for obtaining the physician policy over testset,
# need to find neighbors from the testset instead of trainset.!!!!
dist_test, ind_test = pkl.load(open('kernel_knn_test_from_train.pkl', 'rb'))

# physician policy over test sets
_, ind_test_from_test_for_pi_b = pkl.load(open('kernel_knn_test.pkl', 'rb'))
# survivors in train set
train_survivors = np.where(train_set['died_in_hosp'].values == 0)[0]

# collect kernel and dqn policy, pi_e
## kernel
# action_probs_k_train = actions_2_probs(ind_train, train_actions)
# action_probs_k_test = actions_2_probs(ind_test, train_actions)
action_probs_k_train, action_probs_k_test = pkl.load(open('kernel_actions.pkl','rb'))
## dqn, restrict actions
### dqn training results are dumped as tuple (actions, q-values)
dqn_res_train = pkl.load(open('../outcome_le/results_train.pkl','rb'))
dqn_res_test = pkl.load(open('../outcome_le/results_test.pkl','rb'))

# collect physician policy, pi_b
train_pi_b = actions_2_probs(ind_train, train_actions, expert='phy')
# to compute ind_test_from_test_for_pi_b, train knn and predict using test set solely.
test_pi_b = actions_2_probs(ind_test_from_test_for_pi_b, test_actions, expert='phy')
# test phy examined by trainset
test_pi_b_from_train = actions_2_probs(ind_test, train_actions, expert='phy')

# restrict dqn policies by physician polcies
action_probs_d_train = restrict_actions(softmax(dqn_res_train[0]), train_pi_b, 3/300)
action_probs_d_test = restrict_actions(softmax(dqn_res_test[0]), test_pi_b_from_train, 3/300)

# %%
# V, Q
## obtained from qnetwork_solve_phy_VQ.py
phy_train_Q = pkl.load(open('../outcome_phy/results_train.pkl','rb'))[0]
phy_test_Q = pkl.load(open('../outcome_phy/results_test.pkl','rb'))[0]

phy_train_V = phy_train_Q.max(axis = 1)
phy_test_V = phy_test_Q.max(axis = 1)

## obtained from qnetwork.py
dqn_train_Q = dqn_res_train[0]
dqn_test_Q = dqn_res_test[0]

dqn_train_V = dqn_train_Q.max(axis = 1)
dqn_test_V = dqn_test_Q.max(axis = 1)

# rewards, R_(t) - R_{t-1}
train_rewards = get_ir_diff(train_set)
test_rewards = get_ir_diff(test_set)

# %%
# prepare for training MoE
## train set 
train_df = pd.DataFrame(np.hstack((action_probs_k_train, action_probs_d_train)))
train_df[attr_cols] = train_set[attr_cols]
train_df['reward'] = train_rewards
train_df['traj_len'] = get_traj_len(train_set)
train_df['dist'] = np.max(dist_train[:,1:], axis=1)
train_df['icustayid'] = train_set['icustayid']
train_df['bloc'] = train_set['bloc']

## test set
test_df = pd.DataFrame(np.hstack((action_probs_k_test, action_probs_d_test)))
test_df[attr_cols] = test_set[attr_cols]
test_df['reward'] = test_rewards
test_df['traj_len'] = get_traj_len(test_set) 
test_df['dist'] = np.max(dist_test[:,1:], axis=1)
test_df['icustayid'] = test_set['icustayid']
test_df['bloc'] = test_set['bloc']

# %%
# examine the phy, kernel, and dqn
## Train
print ('pi_b on train:', evaluate(train_pi_b))
print ('kernel policy on train:', evaluate(action_probs_k_train))
print ('dqn policy on train:', evaluate(action_probs_d_train, VQ='dqn'))

# %%
## Test
print ('pi_b on test:', evaluate(test_pi_b, phase='test'))
print ('kernel policy on test:', evaluate(action_probs_k_test, phase='test'))
print ('dqn policy on test:', evaluate(action_probs_d_test, phase='test', VQ='dqn'))

# %%
moe = MOE(input_size=9, logits=1)

# %%
# moe = MOE(input_size=9, logits=1)
train(train_df, moe, batch_size=256, lr=0.0001, num_epoch=500)
# Epoch:37/100, wdr:4.2721781730651855
# ********************
# Eval: epoch:37/100, wdr:3.9994751696702746
# [[7.278773409780115e-05, -0.3777479827404022, 0.08824396878480911, 0.013657244853675365, 0.11063016206026077, -0.3569730818271637, 0.17868568003177643, -0.15885134041309357, -0.34712275862693787]]
# [0.24018655717372894]
# Epoch:52/100, wdr:4.322579860687256
# ********************
# Eval: epoch:52/100, wdr:4.063299313263504
# [[0.17618609964847565, -0.7344371676445007, 0.11212635040283203, 0.4373891353607178, 0.5129995346069336, -0.522505521774292, 0.6723778247833252, -0.2311212420463562, -0.5206314325332642]]
# [0.34241819381713867]

# Epoch:100/100, wdr:4.509039402008057
# ********************
# Eval: epoch:100/100, wdr:4.169190939357189
# [[1.2300101518630981, -1.352432131767273, -0.3300890624523163, 0.9182170033454895, 1.240336298942566, -0.5786391496658325, 1.7389116287231445, -0.2638624310493469, -0.8485985994338989]]
# [0.7569210529327393]
# ********************

# Epoch:29/100, wdr:4.870425224304199
# ********************
# Eval: epoch:29/100, wdr:4.24502183234042
# [[1.2412420511245728, -0.8054572939872742, -0.4281856119632721, 0.7711164355278015, 0.3763531744480133, -0.10058163851499557, 1.7026106119155884, -0.7935341596603394, -0.26787468791007996]]
# [0.7318970561027527]
# dqn:3.799225081366078, phy:4.24502183234042

# Epoch:23/100, wdr:5.080012321472168
# ********************
# Eval: epoch:23/100, wdr:3.882480572462679
# [[-0.000850573880597949, -0.3292688727378845, -0.008397337049245834, 0.14008520543575287, -0.4858025014400482, -0.006549003068357706, -0.3450920879840851, -0.07146909832954407, -0.07110099494457245]]
# [0.018589148297905922]

# Epoch:36/100, wdr:3.622654676437378
# ********************
# Eval: epoch:36/100, wdr:4.314914156055653
# [[1.3287606239318848, -0.7961893677711487, -0.5077809691429138, 0.8471158742904663, 0.3134307861328125, -0.07832232117652893, 1.8573822975158691, -0.6942882537841797, -0.3460248112678528]]
# [0.7956991195678711]

# %%
# pre_w = [[-0.12163831293582916, 0.08448589593172073, -0.37054142355918884, 0.09086832404136658, -0.3181394934654236, 0.16706600785255432, -0.2878364026546478, 0.13344427943229675, -0.18734200298786163]]
# pre_b = [-0.35878944396972656]
# pre_w = [[0.18300466239452362, -0.28264403343200684, -0.16560478508472443, -0.3295769989490509, -0.07599536329507828, -0.23663748800754547, -0.06759240478277206, 0.21431511640548706, -0.2873593866825104]]
# pre_b = [-0.11591021716594696]
# pre_w = [[1.3287606239318848, -0.7961893677711487, -0.5077809691429138, 0.8471158742904663, 0.3134307861328125, -0.07832232117652893, 1.8573822975158691, -0.6942882537841797, -0.3460248112678528]]
# pre_b = [0.7956991195678711]
pre_w = [[1.3282166719436646, -0.7970498204231262, -0.5085808038711548, 0.8464760184288025, 0.3131825625896454, -0.07899399101734161, 1.8568652868270874, -0.6949451565742493, -0.3468712866306305]]
pre_b = [0.7950364351272583]
pretrained(moe, pre_w, pre_b)

# %%
pi_expert, pi_moe = get_moe_policies(moe, test_df)

# %%
evaluate(pi_moe, phase='test', VQ='dqn')

# %%
evaluate(pi_moe, phase='test', VQ='phy')

# %%
# objective function is complicated, need randomize start points and simulate many times
# global maxima is not guaranteed. 
# Maybe use "basinhopping" to find global maxima, but very slow.
for i in range(1000):
    print ('sim. n. ', i)
    moe = MOE(input_size=9, logits=1)
    train(train_df, moe, batch_size=128, lr=0.0001, num_epoch=50)

# %%
do_eval_test(moe)

# %%
_, pi_moe = get_moe_policies(moe, test_df)

# %%
train_fence_post = get_fence_post(train_df)

# %%
test_fence_post = get_fence_post(test_df)

# %%
