import cf.counterfactual as cf
import h5py
import numpy as np
import os
import pickle
import random
from scipy.linalg import block_diag
import sys
import time
from bound_calculators import MultiStepCFBoundCalculator, MultiStepCFBoundCalculatorCSOnly, MultiStepCFBoundCalculatorNoAssumptions, ParallelMultiStepCFBoundCalculator
from gumbel_multi_threaded import CounterfactualSampler
from sepsis_utils import convert

class Action(object):
    NUM_ACTIONS_TOTAL = 8
    ANTIBIOTIC_STRING = "antibiotic"
    VENT_STRING = "ventilation"
    VASO_STRING = "vasopressors"
    ACTION_VEC_SIZE = 3

    def __init__(self, selected_actions = None, action_idx = None):
        
        # Actions can be specified in two ways: by providing a list of selected actions (as strings) or by an action index.
        assert (selected_actions is not None and action_idx is None) \
            or (selected_actions is None and action_idx is not None), \
            "must specify either set of action strings or action index"
            
        if selected_actions is not None:
            if Action.ANTIBIOTIC_STRING in selected_actions:
                self.antibiotic = 1
            else:
                self.antibiotic = 0
            if Action.VENT_STRING in selected_actions:
                self.ventilation = 1
            else:
                self.ventilation = 0
            if Action.VASO_STRING in selected_actions:
                self.vasopressors = 1
            else:
                self.vasopressors = 0
                
        else:
            mod_idx = action_idx
            term_base = Action.NUM_ACTIONS_TOTAL/2
            self.antibiotic = np.floor(mod_idx/term_base).astype(int)
            mod_idx %= term_base
            term_base /= 2
            self.ventilation = np.floor(mod_idx/term_base).astype(int)
            mod_idx %= term_base
            term_base /= 2
            self.vasopressors = np.floor(mod_idx/term_base).astype(int)
            
            '''
            There are three treatments (A, E, V) and thus 2^3 = 8 possible action combinations. 
            The binary representation of action_idx from 0 to 7 can be thought of as the action combinations:

                000 -> No treatments
                001 -> V
                010 -> E
                011 -> E, V
                100 -> A
                101 -> A, V
                110 -> A, E
                111 -> A, E, V
                
            The code block breaks down action_idx to understand which treatments are being used and initializes the three attributes (self.antibiotic, self.ventilation, self.vasopressors) accordingly.
            '''
            
    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.antibiotic == other.antibiotic and \
            self.ventilation == other.ventilation and \
            self.vasopressors == other.vasopressors

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def get_action_idx(self):
        assert self.antibiotic in (0, 1)
        assert self.ventilation in (0, 1)
        assert self.vasopressors in (0, 1)
        return 4*self.antibiotic + 2*self.ventilation + self.vasopressors
    '''
    The weighted sum effectively encodes the three binary values into a single integer (form 0 to 7; NUM_ACTIONS_TOTAL = 8 in total). 
    The weights (4, 2, and 1) were chosen to uniquely identify each combination of the three treatments.
    
    For example:

        If only antibiotic is used: action_idx = 4*1 + 2*0 + 0*1 = 4.
        If only ventilation is used: action_idx = 4*0 + 2*1 + 0*1 = 2.
        If antibiotic and ventilation are used: action_idx = 4*1 + 2*1 + 0*1 = 6.
        If all three are used: action_idx = 4*1 + 2*1 + 1*1 = 7.
    '''

    def __hash__(self):
        return self.get_action_idx()
    
    def get_selected_actions(self):
        selected_actions = set()
        if self.antibiotic == 1:
            selected_actions.add(Action.ANTIBIOTIC_STRING)
        if self.ventilation == 1:
            selected_actions.add(Action.VENT_STRING)
        if self.vasopressors == 1:
            selected_actions.add(Action.VASO_STRING)
        return selected_actions
    
    def get_abbrev_string(self):
        '''
        AEV: antibiotics, ventilation, vasopressors
        '''
        output_str = ''
        if self.antibiotic == 1:
            output_str += 'A'
        if self.ventilation == 1:
            output_str += 'E'
        if self.vasopressors == 1:
            output_str += 'V'
        return output_str
    
    def get_action_vec(self):
        return np.array([[self.antibiotic], [self.ventilation], [self.vasopressors]])

class State(object):
    NUM_OBS_STATES = 720
    NUM_HID_STATES = 2  # Diabetic status is hidden.
    NUM_PROJ_OBS_STATES = int(720 / 5)  # Marginalising over glucose
    NUM_FULL_STATES = int(NUM_OBS_STATES * NUM_HID_STATES)

    def __init__(self, state_idx = None, idx_type = 'obs', diabetic_idx = None, state_categs = None):
        # Initialises the state either by its index or by passing specific categories for each state variable.
        assert state_idx is not None or state_categs is not None
        assert ((diabetic_idx is not None and diabetic_idx in [0, 1]) or
                (state_idx is not None and idx_type == 'full'))

        assert idx_type in ['obs', 'full', 'proj_obs']

        if state_idx is not None:
            self.set_state_by_idx(
                    state_idx, idx_type=idx_type, diabetic_idx=diabetic_idx)
        elif state_categs is not None:
            assert len(state_categs) == 7, "must specify 7 state variables"
            self.hr_state = state_categs[0]
            self.sysbp_state = state_categs[1]
            self.percoxyg_state = state_categs[2]
            self.glucose_state = state_categs[3]
            self.antibiotic_state = state_categs[4]
            self.vaso_state = state_categs[5]
            self.vent_state = state_categs[6]
            self.diabetic_idx = diabetic_idx

    def check_absorbing_state(self):
        num_abnormal = self.get_num_abnormal()
        if num_abnormal >= 3:
            return True
        elif num_abnormal == 0 and not self.on_treatment():
            return True
        return False
    
    def state_rewards(self):
        num_abnormal = self.get_num_abnormal()
        if num_abnormal >= 3:
            return (-1000)
        elif num_abnormal == 2:
            return (-50)
        elif num_abnormal == 1:
            return (+50)
        elif num_abnormal == 0 and self.on_treatment():
            return (+70)
        elif num_abnormal == 0 and not self.on_treatment():
            return (+1000)

    def set_state_by_idx(self, state_idx, idx_type, diabetic_idx=None):
        """set_state_by_idx

        The state index is determined by using "bit" arithmetic, with the
        complication that not every state is binary

        :param state_idx: Given index
        :param idx_type: Index type, either observed (720), projected (144) or
        full (1440)
        :param diabetic_idx: If full state index not given, this is required
        """
        
        if idx_type == 'obs':
            term_base = State.NUM_OBS_STATES/3
        elif idx_type == 'proj_obs':
            term_base = State.NUM_PROJ_OBS_STATES/3
        elif idx_type == 'full':
            term_base = State.NUM_FULL_STATES/2
        
        mod_idx = state_idx

        if idx_type == 'full':           
            self.diabetic_idx = np.floor(mod_idx/term_base).astype(int)
            mod_idx %= term_base
            term_base /= 3
        else:
            assert diabetic_idx is not None
            self.diabetic_idx = diabetic_idx

        self.hr_state = np.floor(mod_idx/term_base).astype(int)

        mod_idx %= term_base
        term_base /= 3
        self.sysbp_state = np.floor(mod_idx/term_base).astype(int)

        mod_idx %= term_base
        term_base /= 2
        self.percoxyg_state = np.floor(mod_idx/term_base).astype(int)

        if idx_type == 'proj_obs':
            self.glucose_state = 2
        else:
            mod_idx %= term_base
            term_base /= 5
            self.glucose_state = np.floor(mod_idx/term_base).astype(int)

        mod_idx %= term_base
        term_base /= 2
        self.antibiotic_state = np.floor(mod_idx/term_base).astype(int)

        mod_idx %= term_base
        term_base /= 2
        self.vaso_state = np.floor(mod_idx/term_base).astype(int)

        mod_idx %= term_base
        term_base /= 2
        self.vent_state = np.floor(mod_idx/term_base).astype(int)


    def get_state_idx(self, idx_type='obs'):
        '''
        returns integer index of state: significance order as in categorical array
        '''
        
        if idx_type == 'obs':
            categ_num = np.array([3,3,2,5,2,2,2])
            state_categs = [
                    self.hr_state,
                    self.sysbp_state,
                    self.percoxyg_state,
                    self.glucose_state,
                    self.antibiotic_state,
                    self.vaso_state,
                    self.vent_state]
        elif idx_type == 'proj_obs':
            categ_num = np.array([3,3,2,2,2,2])
            state_categs = [
                    self.hr_state,
                    self.sysbp_state,
                    self.percoxyg_state,
                    self.antibiotic_state,
                    self.vaso_state,
                    self.vent_state]
        elif idx_type == 'full':
            categ_num = np.array([2,3,3,2,5,2,2,2])
            state_categs = [
                    self.diabetic_idx,
                    self.hr_state,
                    self.sysbp_state,
                    self.percoxyg_state,
                    self.glucose_state,
                    self.antibiotic_state,
                    self.vaso_state,
                    self.vent_state]

        sum_idx = 0
        prev_base = 1
        for i in range(len(state_categs)):
            idx = len(state_categs) - 1 - i
            sum_idx += prev_base*state_categs[idx]
            prev_base *= categ_num[idx]
        return sum_idx
    
    def __eq__(self, other):
        '''
        override equals: two states equal if all internal states same
        '''
        return isinstance(other, self.__class__) and \
            self.hr_state == other.hr_state and \
            self.sysbp_state == other.sysbp_state and \
            self.percoxyg_state == other.percoxyg_state and \
            self.glucose_state == other.glucose_state and \
            self.antibiotic_state == other.antibiotic_state and \
            self.vaso_state == other.vaso_state and \
            self.vent_state == other.vent_state

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.get_state_idx()

    def get_num_abnormal(self):
        '''
        returns number of abnormal conditions
        '''
        num_abnormal = 0
        if self.hr_state != 1:
            num_abnormal += 1
        if self.sysbp_state != 1:
            num_abnormal += 1
        if self.percoxyg_state != 1:
            num_abnormal += 1
        if self.glucose_state != 2:
            num_abnormal += 1
        return num_abnormal
    
    def on_treatment(self):
        '''
        returns True iff any of 3 treatments active
        '''
        if self.antibiotic_state == 0 and \
            self.vaso_state == 0 and self.vent_state == 0:
            return False
        return True

    def on_antibiotics(self):
        '''
        returns True iff antibiotics active
        '''
        return self.antibiotic_state == 1

    def on_vasopressors(self):
        '''
        returns True iff vasopressors active
        '''
        return self.vaso_state == 1

    def on_ventilation(self):
        '''
        returns True iff ventilation active
        '''
        return self.vent_state == 1

    def copy_state(self):
        return State(state_categs = [
            self.hr_state,
            self.sysbp_state,
            self.percoxyg_state,
            self.glucose_state,
            self.antibiotic_state,
            self.vaso_state,
            self.vent_state],
            diabetic_idx=self.diabetic_idx)

    def get_state_vector(self):
        return np.array([self.hr_state,
            self.sysbp_state,
            self.percoxyg_state,
            self.glucose_state,
            self.antibiotic_state,
            self.vaso_state,
            self.vent_state]).astype(int)


class MDP:
    def __init__(self):
        self.default_init_state = 777
        self.states = range(1440)
        self.actions = range(8)
        self.transition_matrix, self.rewards, self.state_rewards = self.load_data()
        self.values = np.zeros(len(self.states))

        # Hyperparameters
        self.discount = 0.9

        # Policies
        self.optimal_policy = self.value_iteration()
        self.suboptimal_policy = self.policy_with_randomization(self.optimal_policy, 0.2)


    def load_data(self):
        with open("data/diab_txr_mats-replication.pkl", "rb") as f:
            mdict = pickle.load(f)

        tx_mat = mdict["tx_mat"]
        r_mat = mdict["r_mat"]

        tx_mat_full = np.zeros((len(self.actions), State.NUM_FULL_STATES, State.NUM_FULL_STATES))
        r_mat_full = np.zeros((len(self.actions), State.NUM_FULL_STATES, State.NUM_FULL_STATES))
        
        for a in range(len(self.actions)):
            tx_mat_full[a, ...] = block_diag(tx_mat[0, a, ...], tx_mat[1, a,...])
            r_mat_full[a, ...] = block_diag(r_mat[0, a, ...], r_mat[1, a, ...])

        # Modify transition probabilities and rewards to account for absorbing states.
        all_absorbing_states = []
        all_absorbing_rewards = []
        self.non_absorbing_states = []
        all_rewards = []

        for s in range(1440):
            get_states = State(state_idx=s, idx_type = 'full')
            abs = get_states.check_absorbing_state()
            if abs == True: 
                all_absorbing_states.append(s)
                rew = get_states.state_rewards()
                all_absorbing_rewards.append(rew)
                
            if abs == False:
                self.non_absorbing_states.append(s)

            rew = get_states.state_rewards()
            all_rewards.append(rew)

        for s in range(1440):
            for a in range(8):
                if s in all_absorbing_states:
                    tx_mat_full[a, s, :] = np.zeros(1440)
                    tx_mat_full[a, s, s] = 1 

        for s in range(1440):
            for a in range(8):
                if s in all_absorbing_states:
                    reward_idx = all_absorbing_states.index(s)
                    r_mat_full[a, s, :] = np.full((1440,), (all_absorbing_rewards[reward_idx]))
                else:
                    for s_p in np.where(tx_mat_full[a, s, :]!=0)[0]:
                        r_mat_full[a, s, s_p] = all_rewards[s_p]

        rewards_pi = np.zeros((1440, 8)) 

        for s in range(1440):
            for a in range(8):
                s_p = (np.where(tx_mat_full[a, s, :] == (np.max(tx_mat_full[a, s, :]))))[0][0]
                rewards_pi[s, a] = r_mat_full[a, s, s_p]

        transition_matrix = np.swapaxes(tx_mat_full, 0, 1)
        self.tx_mat_full = tx_mat_full
        self.r_mat_full = r_mat_full

        return transition_matrix, rewards_pi, all_rewards
    
    def value_iteration(self):
        DISCOUNT_Pol = 0.99
        fullMDP = cf.MatrixMDP(self.tx_mat_full, self.r_mat_full)
        
        return fullMDP.policyIteration(discount=DISCOUNT_Pol, eval_type=1)

    def policy_with_randomization(self, policy, randomization_probability):
        policy_matrix = policy
        random_policy = randomization_probability*np.ones((len(self.states), len(self.actions)))
        random_policy = random_policy + policy_matrix
        random_policy = random_policy / (randomization_probability*len(self.actions) + 1)
        assert np.allclose(np.sum(random_policy, axis=1), 1)
        
        return random_policy

    def translating_policy_to_matrix(self, policy):
        print(policy)
    
    # Samples a random trajectory from a suboptimal policy.
    def sample_random_trajectory(self, n_steps=10):
        n_state = 4
        trajectory = np.zeros((n_steps, n_state))

        current_state = random.choice(self.non_absorbing_states)

        for time_idx in range(n_steps):
            action = np.random.choice(8, size=1, p=self.suboptimal_policy[current_state])[0]
            next_state = np.random.choice(len(self.states), size=1, p=self.transition_matrix[current_state, action, :])[0] 
            reward = self.state_rewards[current_state]
            trajectory[time_idx, :] = np.array([current_state, next_state, action, reward])
            current_state = next_state

        return trajectory.astype(int)
    
    # Samples trajectory produced by optimal policy.
    def sample_optimal_trajectory(self, n_steps=10):
        n_state = 4
        trajectory = np.zeros((n_steps, n_state))

        current_state = random.choice(self.non_absorbing_states)

        for time_idx in range(n_steps):
            action = np.random.choice(8, size=1, p=self.optimal_policy[current_state])[0]
            next_state = np.random.choice(len(self.states), size=1, p=self.transition_matrix[current_state, action, :])[0] 
            reward = self.state_rewards[current_state]
            trajectory[time_idx, :] = np.array([current_state, next_state, action, reward])
            current_state = next_state

        return trajectory.astype(int)
    
    # Returns an example suboptimal trajectory that enters a dangerous, terminal state.
    def sample_suboptimal_trajectory(self, n_steps=10):
        return [[  777,   939,     3,   -50],
        [  939,   941,     6,   -50],
        [  941,   949,     6, -1000],
        [  949,   949,     0, -1000],
        [  949,   949,     0, -1000],
        [  949,   949,     0, -1000],
        [  949,   949,     0, -1000],
        [  949,   949,     0, -1000],
        [  949,   949,     0, -1000],
        [  949,   949,     0, -1000]]


def format_transition_matrix_for_julia(interval_CF_MDP, n_timesteps, n_states, n_actions):
    # We have to treat each (t, s) as a separate state, and only allow the transitions to the next time step.
    transition_matrix = {}

    for t in range(n_timesteps):
        for s in range(n_states):
            lower_transition_probs = np.zeros(shape=(n_states * (n_timesteps+1), n_actions))
            upper_transition_probs = np.zeros(shape=(n_states * (n_timesteps+1), n_actions))

            for a in range(n_actions):
                for s_prime in range(n_states):
                    bounds = interval_CF_MDP[t][s, a, s_prime]
                    lower_transition_probs[((t+1)*n_states) + s_prime, a] = bounds[0]
                    upper_transition_probs[((t+1)*n_states) + s_prime, a] = bounds[1]

            transition_matrix[(t, s)] = (lower_transition_probs, upper_transition_probs)

    # Make last states sink states.
    for s in range(n_states):
        lower_transition_probs = np.zeros(shape=(n_states * (n_timesteps+1), n_actions))
        upper_transition_probs = np.zeros(shape=(n_states * (n_timesteps+1), n_actions))

        lower_transition_probs[(n_timesteps*n_states) + s, :] = 1.0
        upper_transition_probs[(n_timesteps*n_states) + s, :] = 1.0

        transition_matrix[(t+1, s)] = (lower_transition_probs, upper_transition_probs)

    return transition_matrix


def load_generated_policy(filename, n_timesteps, n_states):
    pi = np.zeros(shape=(n_timesteps, n_states))
    with h5py.File(filename, "r") as file:
        data = file["data"]

        for t in range(n_timesteps):
            ref = data[t]
            res = np.array(file[ref])     
            int_array = [int(byte_str) for byte_str in res]
            
            for s in range(n_states):
                pi[t, s] = int_array[(t*n_states)+s]
        
    return pi.astype(int)


def load_value_function(filename, n_timesteps, n_states):
    V = np.zeros(shape=(n_timesteps, n_states))

    with h5py.File(filename, "r") as file:
        data = np.array(file["data"])

        for t in range(1, n_timesteps+1):
            ref = data[n_timesteps-t]
            res = np.array(file[ref])
            
            float_array = res

            for s in range(n_states):
                V[t-1, s] = float_array[((t-1) * n_states)+s]
    
    return V.astype(float)


def rescale_probabilities(probabilities, lower, upper):
    # Project onto the interval.
    probabilities = np.clip(probabilities, lower, upper)
    
    # Calculate sum of probabilities.
    total = np.sum(probabilities)

    # If sum is 1, no further adjustment needed.
    if np.isclose(total, 1.0):
        return probabilities
    
    # If sum is not 1, we need to adjust proportionally.
    diff = 1.0 - total
    
    # Try to distribute the difference to the free variables (not already at the bounds).
    free_indices = np.where((probabilities > lower) & (probabilities < upper))[0]
    
    if len(free_indices) > 0:
        adjustment = diff / len(free_indices)
        probabilities[free_indices] += adjustment
        
        # Clip again to ensure no values fall out of bounds.
        probabilities = np.clip(probabilities, lower, upper)
        
        # Recalculate sum after adjustment.
        return rescale_probabilities(probabilities, lower, upper)
    
    return probabilities


def sample_constrained_dirichlet(lower, upper):
    # Generate a Dirichlet sample.
    alpha = np.ones(len(lower))
    dirichlet_sample = np.random.dirichlet(alpha, 1)[0]
    
    # Scale the Dirichlet sample to fit the range [lower, upper].
    scaled_sample = lower + dirichlet_sample * (upper - lower)
    
    # Rescale the probabilities to ensure they sum to 1 while staying in bounds.
    final_sample = rescale_probabilities(scaled_sample, lower, upper)

    return final_sample


def sample_probabilities_within_intervals(lower, upper):
    while True:
        sampled_probs = sample_constrained_dirichlet(lower, upper)

        # Allowed error due to floating-point errors.
        epsilon = 1e-8 # allowed error

        # Check if the normalized probabilities still lie within the bounds.
        assert(np.isclose(np.sum(sampled_probs), 1.0) and np.all(sampled_probs >= lower-epsilon) and np.all(sampled_probs <= upper+epsilon))
        
        return sampled_probs


def sample_CFMDP(interval_CF_MDP, n_timesteps=10, n_states=1440, n_actions=8):
    CFMDP = np.zeros(shape=(n_timesteps, n_states, n_actions, n_states))

    for t in range(n_timesteps):
        print(f"t={t}")
        for s in range(n_states):
            for a in range(n_actions):
                CFMDP[t, s, a] = sample_probabilities_within_intervals(interval_CF_MDP[t, s, a, :, 0], interval_CF_MDP[t, s, a, :, 1])

    return CFMDP


def evaluate_policies(MDP_rewards, pi, pi_gumbel, interval_CF_MDP, n_steps = 10, init_state=777):
    n_state = 4
    N_CFMDPS = 200
    N_TRAJECTORIES = 10000

    all_rewards_gumbel = []
    all_rewards_icfmdp = []
    
    for k in range(N_CFMDPS):
        print(f"{k}/{N_CFMDPS}")
        CFMDP = sample_CFMDP(interval_CF_MDP)

        # Test with ICFMDP policy
        for j in range(N_TRAJECTORIES):
            print(f"{j}/{N_TRAJECTORIES}")
            trajectory = np.zeros((n_steps, n_state))
            current_state = init_state

            for time_idx in range(n_steps):
                action = pi[time_idx, current_state]
                next_state = np.random.choice(1440, size=1, p=CFMDP[time_idx, current_state, action])[0] 
                reward = MDP_rewards[current_state]
                trajectory[time_idx, :] = np.array([current_state, next_state, action, reward])
                current_state = next_state
            
            rewards = trajectory[:, 3]
            all_rewards_icfmdp.append(rewards)

        # Test with Gumbel-max policy
        for j in range(N_TRAJECTORIES):
            print(f"{j}/{N_TRAJECTORIES}")
            trajectory = np.zeros((n_steps, n_state))
            current_state = init_state

            for time_idx in range(n_steps):
                action = pi_gumbel[time_idx, current_state]
                next_state = np.random.choice(1440, size=1, p=CFMDP[time_idx, current_state, action])[0] 
                reward = MDP_rewards[current_state]
                trajectory[time_idx, :] = np.array([current_state, next_state, action, reward])
                current_state = next_state

            rewards = trajectory[:, 3]
            all_rewards_gumbel.append(rewards)

    return all_rewards_icfmdp, all_rewards_gumbel


def generate_interval_CFMDP(mdp, bound_calculator, trajectory=None):
    if trajectory is None:
        trajectory = np.array(mdp.sample_random_trajectory()).astype(int)       

    n_timesteps = len(trajectory)
    n_states = mdp.transition_matrix.shape[0]
    n_actions = mdp.transition_matrix.shape[1]

    interval_CFMDP = bound_calculator.calculate_bounds(trajectory)

    return mdp, interval_CFMDP, trajectory, n_timesteps, n_states, n_actions


def generate_interval_CFMDP_parallel(trajectory=None):
    mdp = MDP()

    if trajectory is None:
        trajectory = np.array(mdp.sample_random_trajectory()).astype(int)

    n_timesteps = len(trajectory)
    n_states = mdp.transition_matrix.shape[0]
    n_actions = mdp.transition_matrix.shape[1]

    bound_calculator = ParallelMultiStepCFBoundCalculator(mdp.transition_matrix)
    interval_CF_MDP = bound_calculator.calculate_bounds(trajectory)

    return mdp, interval_CF_MDP, trajectory, n_timesteps, n_states, n_actions


def run_gumbel_sampler(mdp, trajectory, n_state=1440, n_actions=8):
    gumbel_sampler = CounterfactualSampler(mdp, n_state, n_actions)
    gumbel_CF_MDP = gumbel_sampler.run_parallel_sampling(mdp.tx_mat_full, np.array([trajectory]))

    return gumbel_CF_MDP


def get_gumbel_max_optimal_policy(P_cf, n_timesteps, n_states, n_actions, rewards):
    h_fun = np.zeros((n_timesteps+1, n_states)) 
    pi = np.zeros((n_timesteps+1, n_states), dtype=int) 

    for s in range(n_states):
        h_fun[n_timesteps, s] = rewards[s]

    for t in range(n_timesteps-1, -1, -1):
        for s in range(n_states):
            best_act = 0
            max_val = -np.inf
            
            for a in range(n_actions):
                val = rewards[s]

                for s_p in range(n_states):
                    if P_cf[a, t][s, s_p] != 0:
                        val += 1.0 * (P_cf[a, t][s, s_p] * h_fun[t+1, s_p])

                if val > max_val:
                    max_val = val
                    best_act = a
        
            h_fun[t, s] = max_val
            pi[t, s] = best_act


    return pi, h_fun


def calculate_average_probability_width_ICFMDP(interval_CF_MDP):
    num_transitions = 10 * 1440 * 8 * 1440
    num_non_zero_transitions = 0
    total_prob_bounds = 0

    for t in range(10):
        for s in range(1440):
            for a in range(8):
                for s_prime in range(1440):
                    lb = interval_CF_MDP[t, s, a, s_prime, 0]
                    ub = interval_CF_MDP[t, s, a, s_prime, 1]

                    if not (ub == 0.0):
                        num_non_zero_transitions += 1
                        total_prob_bounds += ub - lb

    total_non_zero_prob_bounds = total_prob_bounds / num_non_zero_transitions
    total_prob_bounds /= num_transitions

    return total_prob_bounds, total_non_zero_prob_bounds


def normalise(interval_CF_MDP, n_timesteps, n_states, n_actions):
    def trunc_to_decimal_places(value, decimal_places):
        factor = 10 ** decimal_places
        return np.trunc(value * factor) / factor
    
    def ceil_to_decimal_places(value, decimal_places):
        factor = 10 ** decimal_places
        return np.ceil(value * factor) / factor
    
    interval_CF_MDP[:, :, :, :, 0] = trunc_to_decimal_places(interval_CF_MDP[:, :, :, :, 0], 12)
    interval_CF_MDP[:, :, :, :, 1] = ceil_to_decimal_places(interval_CF_MDP[:, :, :, :, 1], 12)

    # Normalising

    for t in range(n_timesteps):
        for s in range(n_states):
            for a in range(n_actions):
                ub_threshold = 1.0000000000000000
                lb_threshold = 1.0000000000000000

                i = 0

                while sum(interval_CF_MDP[t, s, a, :, 0]) > lb_threshold:
                    print(f"iter i {i} s={s} a={a} lb={interval_CF_MDP[t, s, a, :, 0]} sum={sum(interval_CF_MDP[t, s, a, :, 0])}")
        
                    interval_CF_MDP[t, s, a, :, 0] = interval_CF_MDP[t, s, a, :, 0] / sum(interval_CF_MDP[t, s, a, :, 0])
                    
                    print(f"iter i {i} s={s} a={a} lb={interval_CF_MDP[t, s, a, :, 0]} sum={sum(interval_CF_MDP[t, s, a, :, 0])}")

                    if sum(interval_CF_MDP[t, s, a, :, 0]) > lb_threshold:
                        print(f"iter i {i} s={s} a={a} lb={interval_CF_MDP[t, s, a, :, 0]} sum={sum(interval_CF_MDP[t, s, a, :, 0])}")
                    i += 1
                
                i=0
                while sum(interval_CF_MDP[t, s, a, :, 1]) < ub_threshold:
                    print(f"iter {i} s={s} a={a} ub={interval_CF_MDP[t, s, a, :, 1]} sum={sum(interval_CF_MDP[t, s, a, :, 1])}")
            
                    interval_CF_MDP[t, s, a, :, 1] = interval_CF_MDP[t, s, a, :, 1] / sum(interval_CF_MDP[t, s, a, :, 1])
                    
                    print(f"iter {i} s={s} a={a} ub={interval_CF_MDP[t, s, a, :, 1]} sum={sum(interval_CF_MDP[t, s, a, :, 1])}")

                    if sum(interval_CF_MDP[t, s, a, :, 1]) < ub_threshold:
                        print(f"iter {i} s={s} a={a} ub={interval_CF_MDP[t, s, a, :, 1]} sum={sum(interval_CF_MDP[t, s, a, :, 1])}")
                    i += 1

                if sum(interval_CF_MDP[t, s, a, :, 1]) < ub_threshold:
                    print(f"s={s} a={a} ub={interval_CF_MDP[t, s, a, :, 1]} sum={sum(interval_CF_MDP[t, s, a, :, 1])}")

                if sum(interval_CF_MDP[t, s, a, :, 0]) > lb_threshold:
                    print(f"s={s} a={a} lb={interval_CF_MDP[t, s, a, :, 0]} sum={sum(interval_CF_MDP[t, s, a, :, 0])}")

                assert(sum(interval_CF_MDP[t, s, a, :, 1]) >= ub_threshold)
                assert(sum(interval_CF_MDP[t, s, a, :, 0]) <= lb_threshold)

    epsilon = 1e-16
    nonzero_mask = interval_CF_MDP[:, :, :, :, 0] != 0.0
    # This adjustment ensure the probs are valid (i.e., that all UBs are > LBs)
    interval_CF_MDP[nonzero_mask, 0] -= epsilon
    nonzero_mask = interval_CF_MDP[:, :, :, :, 1] != 0.0
    interval_CF_MDP[nonzero_mask, 1] += epsilon

    return interval_CF_MDP


def normalise2(interval_CF_MDP, n_timesteps, n_states, n_actions):
    interval_CF_MDP[:, :, :, :, 0] = np.round(interval_CF_MDP[:, :, :, :, 0], 13)

    for t in range(n_timesteps):
        for s in range(n_states):
            for a in range(n_actions):
                ub_threshold = 1.0000000000000000
                lb_threshold = 1.0000000000000000

                i = 0

                while sum(interval_CF_MDP[t, s, a, :, 0]) > lb_threshold:
                    print(f"iter i {i} s={s} a={a} lb={interval_CF_MDP[t, s, a, :, 0]} sum={sum(interval_CF_MDP[t, s, a, :, 0])}")
        
                    interval_CF_MDP[t, s, a, :, 0] = interval_CF_MDP[t, s, a, :, 0] / sum(interval_CF_MDP[t, s, a, :, 0]) # max(1.000000000000001, sum(interval_CF_MDP[t, s, a, :, 0]))
                    
                    print(f"iter i {i} s={s} a={a} lb={interval_CF_MDP[t, s, a, :, 0]} sum={sum(interval_CF_MDP[t, s, a, :, 0])}")

                    if sum(interval_CF_MDP[t, s, a, :, 0]) > lb_threshold:
                        print(f"iter i {i} s={s} a={a} lb={interval_CF_MDP[t, s, a, :, 0]} sum={sum(interval_CF_MDP[t, s, a, :, 0])}")
                    i += 1
                
                i=0
                while sum(interval_CF_MDP[t, s, a, :, 1]) < ub_threshold:
                    print(f"iter {i} s={s} a={a} ub={interval_CF_MDP[t, s, a, :, 1]} sum={sum(interval_CF_MDP[t, s, a, :, 1])}")
            
                    interval_CF_MDP[t, s, a, :, 1] = interval_CF_MDP[t, s, a, :, 1] / sum(interval_CF_MDP[t, s, a, :, 1]) # min(0.999999999999999, sum(interval_CF_MDP[t, s, a, :, 1]))
                    
                    print(f"iter {i} s={s} a={a} ub={interval_CF_MDP[t, s, a, :, 1]} sum={sum(interval_CF_MDP[t, s, a, :, 1])}")

                    if sum(interval_CF_MDP[t, s, a, :, 1]) < ub_threshold:
                        print(f"iter {i} s={s} a={a} ub={interval_CF_MDP[t, s, a, :, 1]} sum={sum(interval_CF_MDP[t, s, a, :, 1])}")
                    i += 1

                if sum(interval_CF_MDP[t, s, a, :, 1]) < ub_threshold:
                    print(f"s={s} a={a} ub={interval_CF_MDP[t, s, a, :, 1]} sum={sum(interval_CF_MDP[t, s, a, :, 1])}")

                if sum(interval_CF_MDP[t, s, a, :, 0]) > lb_threshold:
                    print(f"s={s} a={a} lb={interval_CF_MDP[t, s, a, :, 0]} sum={sum(interval_CF_MDP[t, s, a, :, 0])}")

                assert(sum(interval_CF_MDP[t, s, a, :, 1]) >= ub_threshold)
                assert(sum(interval_CF_MDP[t, s, a, :, 0]) <= lb_threshold)

    epsilon = 1e-16
    nonzero_mask = interval_CF_MDP[:, :, :, :, 0] != 0.0
    # This adjustment ensure the probs are valid (i.e., that all UBs are > LBs)
    interval_CF_MDP[nonzero_mask, 0] -= epsilon
    nonzero_mask = interval_CF_MDP[:, :, :, :, 1] != 0.0
    interval_CF_MDP[nonzero_mask, 1] += epsilon

    return interval_CF_MDP


def main():
    if len(sys.argv) < 2:
        print("Usage: python sepsis.py <function_name>")
        sys.exit(1)
    
    function_name = sys.argv[1]

    if function_name == "generate_icfmdps":
        mdp = MDP()

        NUM_OBS_PATHS = 4

        optimal_trajectory = np.array([[1348, 1109,    6,  -50],
            [1109, 1096,    0,   50],
            [1096, 1096,    0, 1000],
            [1096, 1096,    0, 1000],
            [1096, 1096,    0, 1000],
            [1096, 1096,    0, 1000],
            [1096, 1096,    0, 1000],
            [1096, 1096,    0, 1000],
            [1096, 1096,    0, 1000],
            [1096, 1096,    0, 1000]]).astype(int)

        close_to_optimal_trajectory = np.array([[ 1348,  1337,     2,   -50],
             [ 1337,  1348,     4,    50],
             [ 1348,  1416,     0,   -50],
             [ 1416,  1408,     0,   -50],
             [ 1408,  1408,     5, -1000],
             [ 1408,  1408,     4, -1000],
             [ 1408,  1408,     6, -1000],
             [ 1408,  1408,     5, -1000],
             [ 1408,  1408,     0, -1000],
             [ 1408,  1408,     6, -1000]]).astype(int)
        
        close_to_catastrophic_trajectory = np.array([[777, 859,   3, -50],
            [859, 857,   2,  50],
            [857, 861,   6,  50],
            [861, 853,   6,  50],
            [853, 861,   6, -50],
            [861, 869,   6,  50],
            [869, 861,   6, -50],
            [861, 853,   6,  50],
            [853, 861,   6, -50],
            [861, 861,   6,  50]]).astype(int)

        catastrophic_trajectory = np.array(mdp.sample_suboptimal_trajectory()).astype(int)

        print(optimal_trajectory)
        print(close_to_optimal_trajectory)
        print(close_to_catastrophic_trajectory)
        print(catastrophic_trajectory)

        obs_trajectories = [optimal_trajectory, close_to_optimal_trajectory, close_to_catastrophic_trajectory, catastrophic_trajectory]
        interval_CFMDPs = []

        if not os.path.exists(f"transition_matrices"):
            os.makedirs(f"transition_matrices")

        if not os.path.exists(f"ICFMDPs"):
            os.makedirs(f"ICFMDPs")

        for i, obs_trajectory in enumerate(obs_trajectories):
            mdp, interval_CFMDP, observed_trajectory, n_timesteps, n_states, n_actions = generate_interval_CFMDP_parallel(obs_trajectory)
            
            n_timesteps = 10
            n_states = 1440
            n_actions = 8

            interval_CFMDP = normalise(interval_CFMDP, n_timesteps, n_states, n_actions)
            interval_CFMDPs.append(interval_CFMDP)

            transition_matrix = format_transition_matrix_for_julia(interval_CFMDP, n_timesteps, n_states, n_actions)
            convert(transition_matrix, i, mdp.state_rewards, tra_filename = f"transition_matrices/sepsis_tra_{i}.pickle", filename = f"ICFMDPs/sepsis_{i}.jl")

        with open(f"sepsis_interval_CFMDPs.pickle", "wb") as f:
            pickle.dump(interval_CFMDPs, f)

        with open(f"sepsis_obs_trajectories.pickle", "wb") as f:
            pickle.dump(obs_trajectories, f)


    elif function_name == "evaluate_performance":
        mdp = MDP()

        with open(f"sepsis_interval_CFMDPs.pickle", "rb") as f:
            interval_CFMDPs = pickle.load(f)

        with open(f"sepsis_obs_trajectories.pickle", "rb") as f:
            obs_trajectories = pickle.load(f)

        for i, trajectory in enumerate(obs_trajectories):
            print(trajectory)

            interval_CFMDP = interval_CFMDPs[i]
            n_timesteps = len(trajectory)
            n_states = 1440
            n_actions = 8

            pi = load_generated_policy(f"ICFMDPs/sepsis_policy_{i}.jld2", n_timesteps, n_states)

            # Min and max value functions are also generated for analysis.
            # V_min = load_value_function(f"ICFMDPs/sepsis_value_pessimistic_{i}.jld2", n_timesteps, n_states)
            # V_max = load_value_function(f"ICFMDPs/sepsis_value_optimistic_{i}.jld2", n_timesteps, n_states)

            gumbel_CF_MDP = run_gumbel_sampler(mdp, trajectory, n_states, n_actions)
            pi_gumbel, V_gumbel = get_gumbel_max_optimal_policy(gumbel_CF_MDP, n_timesteps, n_states, n_actions, mdp.state_rewards)

            all_icfmdp_rewards, all_gumbel_rewards = evaluate_policies(mdp.state_rewards, pi, pi_gumbel, interval_CFMDP, init_state = trajectory[0,0])
            all_icfmdp_rewards = np.array(all_icfmdp_rewards).reshape(2000000, 10)
            all_gumbel_rewards = np.array(all_gumbel_rewards).reshape(2000000, 10)

            mean_icfmdp_rewards = np.mean(np.array(all_icfmdp_rewards), axis=0)
            std_icfmdp_rewards = np.std(np.array(all_icfmdp_rewards), axis=0)
            mean_gumbel_rewards = np.mean(np.array(all_gumbel_rewards), axis=0)
            std_gumbel_rewards = np.std(np.array(all_gumbel_rewards), axis=0)

            # Clip upper and lower error bars to within reward range [-1000, 1000].
            upper_icfmdp_errors = np.clip(mean_icfmdp_rewards + std_icfmdp_rewards, None, 1000) - mean_icfmdp_rewards
            upper_gumbel_errors = np.clip(mean_gumbel_rewards + std_gumbel_rewards, None, 1000) - mean_gumbel_rewards
            lower_icfmdp_errors = mean_icfmdp_rewards - np.clip(mean_icfmdp_rewards - std_icfmdp_rewards, -1000, None)
            lower_gumbel_errors = mean_gumbel_rewards - np.clip(mean_gumbel_rewards - std_gumbel_rewards, -1000, None)

            with open(f"Sepsis Results.txt", "a") as file:
                file.write(f"Observed trajectory: {trajectory}\n\n")
                file.write(f"Average Results: \n\n")
                file.write(f"Mean ICFMDP rewards: {mean_icfmdp_rewards}\n")
                file.write(f"Upper bounds: {upper_icfmdp_errors}\n")
                file.write(f"Lower bounds: {lower_icfmdp_errors}\n")
                file.write(f"Mean gumbel rewards: {mean_gumbel_rewards}\n")
                file.write(f"Upper bounds: {upper_gumbel_errors}\n")
                file.write(f"Lower bounds: {lower_gumbel_errors}\n\n")

            # Look for the CF paths with the lowest cumulative reward for both policies.
            sum_rewards_icfmdp = np.sum(np.array(all_icfmdp_rewards), axis=1)
            sum_rewards_gumbel = np.sum(np.array(all_gumbel_rewards), axis=1)
            worst_icfmdp_path = all_icfmdp_rewards[np.argmin(sum_rewards_icfmdp)]
            worst_gumbel_path = all_gumbel_rewards[np.argmin(sum_rewards_gumbel)]

            with open(f"Sepsis Results.txt", "a") as f:
                f.write(f"CF path with lowest cumulative reward, ICFMDP = {worst_icfmdp_path}\n")
                f.write(f"CF path with lowest cumulative reward, Gumbel-max CFMDP = {worst_gumbel_path}\n")


    elif function_name == "measure_execution_time":
        # Measure the average time it takes to calculate interval MDP.
        mdp = MDP()
        mdp.value_iteration()

        NUM_OBS_PATHS = 20
        total_execution_time_in_seconds_interval_MDP = 0.0
        total_execution_time_in_seconds_gumbel = 0.0

        trajectories = []

        for i in range(NUM_OBS_PATHS):
            trajectories.append(mdp.sample_random_trajectory())

        for i in range(NUM_OBS_PATHS):
            start = time.time()

            mdp, interval_CFMDP, _, n_timesteps, n_states, n_actions = generate_interval_CFMDP_parallel(trajectory=trajectories[i])
            
            end = time.time()
            total_execution_time_in_seconds_interval_MDP += end - start

            start = time.time()

            gumbel_CF_MDP = run_gumbel_sampler(mdp, trajectories[i], n_states, n_actions)

            end = time.time()

            total_execution_time_in_seconds_gumbel += end - start

        with open(f"Sepsis Execution Times.txt", "a") as f:
            f.write(f"Interval CFMDP = {total_execution_time_in_seconds_interval_MDP / NUM_OBS_PATHS}\n")
            f.write(f"Gumbel CFMDP = {total_execution_time_in_seconds_gumbel / NUM_OBS_PATHS}\n")


    elif function_name == "measure_avg_width":
        # Measures the average width of the counterfactual/bounding-cfs probability bounds in the interval CFMDP.
        mdp = MDP()
        mdp.value_iteration()

        NUM_OBS_PATHS = 20
        trajectories = []

        interval_CFMDPs_both_assumptions = []
        interval_CFMDPs_only_CS = []
        interval_CFMDPs_no_assumptions = []        

        for i in range(NUM_OBS_PATHS):
            trajectories.append(mdp.sample_random_trajectory())

        for i in range(NUM_OBS_PATHS):
            # Generate ICFMDP given both CS and monotonicity assumptions.
            bound_calculator = MultiStepCFBoundCalculator(mdp.transition_matrix)
            mdp, interval_CFMDP, trajectory, n_timesteps, n_states, n_actions = generate_interval_CFMDP(mdp, bound_calculator, trajectory=trajectories[i])

            epsilon = 1e-16
            nonzero_mask = interval_CFMDP[:, :, :, :, 0] != 0.0

            # This adjustment ensure the probs are valid (i.e., that all UBs are > LBs), accounting for floating-point errors.
            interval_CFMDP[nonzero_mask, 0] -= epsilon
            nonzero_mask = interval_CFMDP[:, :, :, :, 1] != 0.0
            interval_CFMDP[nonzero_mask, 1] += epsilon

            interval_CFMDPs_both_assumptions.append(interval_CFMDP)

            # Generate ICFMDP given only CS assumption.
            bound_calculator = MultiStepCFBoundCalculatorCSOnly(mdp.transition_matrix)
            mdp, interval_CFMDP, trajectory, n_timesteps, n_states, n_actions = generate_interval_CFMDP(mdp, bound_calculator, trajectory=trajectories[i])

            epsilon = 1e-16
            nonzero_mask = interval_CFMDP[:, :, :, :, 0] != 0.0

            # This adjustment ensure the probs are valid (i.e., that all UBs are > LBs), accounting for floating-point errors.
            interval_CFMDP[nonzero_mask, 0] -= epsilon
            nonzero_mask = interval_CFMDP[:, :, :, :, 1] != 0.0
            interval_CFMDP[nonzero_mask, 1] += epsilon

            interval_CFMDPs_only_CS.append(interval_CFMDP)

            # Generate ICFMDP given no assumptions.
            bound_calculator = MultiStepCFBoundCalculatorNoAssumptions(mdp.transition_matrix)
            mdp, interval_CFMDP, trajectory, n_timesteps, n_states, n_actions = generate_interval_CFMDP(mdp, bound_calculator, trajectory=trajectories[i])

            epsilon = 1e-16
            nonzero_mask = interval_CFMDP[:, :, :, :, 0] != 0.0

            # This adjustment ensure the probs are valid (i.e., that all UBs are > LBs), accounting for floating-point errors.
            interval_CFMDP[nonzero_mask, 0] -= epsilon
            nonzero_mask = interval_CFMDP[:, :, :, :, 1] != 0.0
            interval_CFMDP[nonzero_mask, 1] += epsilon

            interval_CFMDPs_no_assumptions.append(interval_CFMDP)

        with open(f"Sepsis probability width results.txt", "a") as file:
            file.write(f"counterfactual/bounding-cfs stability + monotonicity:\n")

            total_probs = 0
            total_non_zero_probs = 0

            for interval_CFMDP in interval_CFMDPs_both_assumptions:
                probs, non_zero_probs = calculate_average_probability_width_ICFMDP(interval_CFMDP)
                total_probs += probs
                total_non_zero_probs += non_zero_probs

            avg_probs = total_probs / len(interval_CFMDPs_both_assumptions)
            avg_non_zero_probs = total_non_zero_probs / len(interval_CFMDPs_both_assumptions)

            file.write(f"Avg prob bounds = {avg_probs}\n")
            file.write(f"Avg non-zero prob bounds = {avg_non_zero_probs}\n\n")

            file.write(f"Only counterfactual/bounding-cfs stability:\n")

            total_probs = 0
            total_non_zero_probs = 0

            for interval_CFMDP in interval_CFMDPs_only_CS:
                probs, non_zero_probs = calculate_average_probability_width_ICFMDP(interval_CFMDP)
                total_probs += probs
                total_non_zero_probs += non_zero_probs

            avg_probs = total_probs / len(interval_CFMDPs_only_CS)
            avg_non_zero_probs = total_non_zero_probs / len(interval_CFMDPs_only_CS)

            file.write(f"Avg prob bounds = {avg_probs}\n")
            file.write(f"Avg non-zero prob bounds = {avg_non_zero_probs}\n\n")

            file.write(f"No assumptions:\n")

            total_probs = 0
            total_non_zero_probs = 0

            for interval_CFMDP in interval_CFMDPs_no_assumptions:
                probs, non_zero_probs = calculate_average_probability_width_ICFMDP(interval_CFMDP)
                total_probs += probs
                total_non_zero_probs += non_zero_probs

            avg_probs = total_probs / len(interval_CFMDPs_no_assumptions)
            avg_non_zero_probs = total_non_zero_probs / len(interval_CFMDPs_no_assumptions)

            file.write(f"Avg prob bounds = {avg_probs}\n")
            file.write(f"Avg non-zero prob bounds = {avg_non_zero_probs}\n\n")

    else:
        print(f"Function '{function_name}' is not recognized.")

main()
