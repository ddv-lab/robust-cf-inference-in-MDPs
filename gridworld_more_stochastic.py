import h5py
import numpy as np
import os
import pickle
import sys
import time
from bound_calculators import MultiStepCFBoundCalculator, MultiStepCFBoundCalculatorCSOnly, MultiStepCFBoundCalculatorNoAssumptions
from gridworld_utils import convert_transition_matrix_to_julia_imdp
from gumbel_single_threaded import CounterfactualSampler

class MDP:
    def __init__(self):
        self.init_state = 0
        self.states = range(16)
        self.actions = range(4)
        self.transition_matrix = np.zeros((len(self.states), len(self.actions), len(self.states)))
        self.rewards = np.zeros((len(self.states), len(self.actions)))
        self.optimal_policy = np.zeros(len(self.states), dtype=int)
        self.values = np.zeros(len(self.states))

        # Hyperparameters
        self.discount = 0.9

        for state in self.states:
            for action in self.actions:
                next_state = state

                if action == 0:  # up
                    if state not in [0, 1, 2, 3]:
                        next_state = state - 4
                elif action == 1:  # right
                    if state not in [3, 7, 11, 15]:
                        next_state = state + 1
                elif action == 2:  # down
                    if state not in [12, 13, 14, 15]:
                        next_state = state + 4
                elif action == 3:  # left
                    if state not in [0, 4, 8, 12]:
                        next_state = state - 1

                # Set the transition probabilities and rewards
                self.transition_matrix[state, action, next_state] = 1

                # Unsafe absorbing state
                if state in [6]:
                    self.transition_matrix[state, action, :] = 0.0
                    self.transition_matrix[state, action, state] = 1.0
                    self.rewards[state, action] = -100
                
                # Absorbing goal state
                elif state == 15:
                    self.transition_matrix[state, action, :] = 0.0
                    self.transition_matrix[state, action, state] = 1.0
                    self.rewards[state, action] = 100

                elif state in [1, 4, 5]:
                    self.rewards[state, action] = 1.0
                elif state in [2, 8, 9]:
                    self.rewards[state, action] = 2.0
                elif state in [3, 10, 12]:
                    self.rewards[state, action] = 3.0
                elif state in [7, 13]:
                    self.rewards[state, action] = 4.0
                elif state in [11, 14]:
                    self.rewards[state, action] = 5.0 
        
        s_transition_probabilities = np.zeros((len(self.states), len(self.actions), len(self.states)));
        p_r = 0.4

        for action in self.actions:
            other_actions = [a for a in self.actions if a != action]

            for a in other_actions:
                s_transition_probabilities[:, action, :] += (1 - p_r)/3 * self.transition_matrix[:, a, :]

            s_transition_probabilities[:, action, :] += p_r * self.transition_matrix[:, action, :]
        
        assert np.allclose(np.sum(s_transition_probabilities, axis=2), 1.0)
        self.transition_matrix = s_transition_probabilities

    
    def value_iteration(self):
        self.values = np.zeros(len(self.states))

        while True:
            new_values = np.zeros(len(self.states))
            for state in self.states:
                values = np.zeros(len(self.actions))
                for action in self.actions:
                    for next_state in self.states:
                        values[action] += self.transition_matrix[state, action, next_state] * (self.rewards[state, action] + self.discount * self.values[next_state])
                
                new_values[state] = np.max(values)
                self.optimal_policy[state] = np.argmax(values)

            if np.sum(np.abs(new_values - self.values)) < 1e-4:
                break

            self.values = new_values

            
    def policy_with_randomization(self, policy, randomization_probability):
        policy_matrix = self.translating_policy_to_matrix(policy)
        random_policy = randomization_probability*np.ones((len(self.states), len(self.actions)))
        random_policy = random_policy + policy_matrix
        random_policy = random_policy / (randomization_probability*len(self.actions) + 1)
        assert np.allclose(np.sum(random_policy, axis=1), 1)
        return random_policy
    

    def translating_policy_to_matrix(self, policy):
        policy_matrix = np.zeros((len(self.states), len(self.actions)))
        for i in range(len(policy)):
            policy_matrix[i][policy[i]] = 1
        return policy_matrix
    

    # Samples a random trajectory from a suboptimal policy.
    def sample_random_trajectory(self, n_steps=10, randomization=1.0):
        suboptimal_policy = self.policy_with_randomization(self.optimal_policy, randomization)

        n_state = 4
        trajectory = np.zeros((n_steps, n_state))

        current_state = self.init_state

        for time_idx in range(n_steps):
            action = np.random.choice(4, size=1, p=suboptimal_policy[current_state])[0]
            next_state = np.random.choice(len(self.states), size=1, p=self.transition_matrix[current_state, action, :])[0] 
            reward = self.rewards[current_state, action]
            trajectory[time_idx, :] = np.array([current_state, next_state, action, reward])
            current_state = next_state

        return trajectory.astype(int)
    
    
    # Samples trajectory produced by optimal policy.
    def sample_optimal_trajectory(self, n_steps=10):
        n_state = 4
        trajectory = np.zeros((n_steps, n_state))

        current_state = self.init_state

        for time_idx in range(n_steps): 
            action = self.optimal_policy[current_state]
            next_state = np.random.choice(len(self.states), size=1, p=self.transition_matrix[current_state, action, :])[0] 
            reward = self.rewards[current_state, action]
            trajectory[time_idx, :] = np.array([current_state, next_state, action, reward])
            current_state = next_state

        return trajectory
    
    
    # Returns an example suboptimal trajectory that enters the dangerous, terminal state.
    def sample_suboptimal_trajectory(self, n_steps=10):
        return [[0, 1, 1, 1],  [1, 2, 1, 2], [2, 6, 2, -100],  [6, 6, 0, -100], [6, 6, 0, -100], [6, 6, 0, -100], [6, 6, 0, -100], [6, 6, 0, -100], [6, 6, 0, -100], [6, 6, 0, -100]]


def format_transition_matrix_for_julia(interval_CFMDP, n_timesteps, n_states, n_actions):
    # We have to treat each (t, s) as a separate state, and only allow the transitions to the next time step
    transition_matrix = {}

    for t in range(n_timesteps):
        for s in range(n_states):
            lower_transition_probs = np.zeros(shape=(n_states * (n_timesteps+1), n_actions))
            upper_transition_probs = np.zeros(shape=(n_states * (n_timesteps+1), n_actions))

            for a in range(n_actions):
                for s_prime in range(n_states):
                    bounds = interval_CFMDP[t][s, a, s_prime]
                    lower_transition_probs[((t+1)*16) + s_prime, a] = bounds[0]
                    upper_transition_probs[((t+1)*16) + s_prime, a] = bounds[1]

            transition_matrix[(t, s)] = (lower_transition_probs, upper_transition_probs)

    # Make last states sink states
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
        print(data)

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
        data = file["data"]

        for t in range(1, n_timesteps+1):
            ref = data[n_timesteps-t]
            res = np.array(file[ref])
            float_array = [float(byte_str) for byte_str in res]

            for s in range(n_states):
                V[t-1, s] = float_array[((t-1) * n_states)+s]
    
    return V.astype(float)


def sample_probabilities_within_intervals(lower, upper):    
    while True:
        # Sample probabilities from within the given bounds.
        sampled_probs = np.random.uniform(lower, upper)
        
        # Normalise the sampled probabilities so that they sum to 1.
        sampled_probs /= np.sum(sampled_probs)
        sampled_probs /= np.sum(sampled_probs)

        # Allowed error due to floating-point errors.
        epsilon = 1e-13

        # Check if the normalised probabilities still lie within the bounds.
        if np.all(sampled_probs >= lower-epsilon) and np.all(sampled_probs <= upper+epsilon):
            return sampled_probs


def sample_CFMDP(interval_CFMDP, n_timesteps=10, n_states=16, n_actions=4):
    CFMDP = np.zeros(shape=(n_timesteps, n_states, n_actions, n_states))

    for t in range(n_timesteps):
        for s in range(n_states):
            for a in range(n_actions):
                CFMDP[t, s, a] = sample_probabilities_within_intervals(interval_CFMDP[t, s, a, :, 0], interval_CFMDP[t, s, a, :, 1])

    return CFMDP


def sample_optimal_CF_trajectory(rewards, pi, interval_CFMDP, n_steps = 10):
    n_state = 4
    trajectory = np.zeros((n_steps, n_state))
    current_state = 0

    for time_idx in range(n_steps):
        action = pi[time_idx, current_state]
        prob_bounds = interval_CFMDP[time_idx][current_state, action]
        lower_bounds = prob_bounds[:, 0]
        upper_bounds = prob_bounds[:, 1]

        next_state = np.random.choice(16, size=1, p=sample_probabilities_within_intervals(lower_bounds, upper_bounds))[0] 
        reward = rewards[current_state, action]
        trajectory[time_idx, :] = np.array([current_state, next_state, action, reward])
        current_state = next_state

    return trajectory


def evaluate_policies(MDP_rewards, pi, pi_gumbel, interval_CFMDP, n_steps = 10):
    n_state = 4
    N_CFMDPS = 200
    N_TRAJECTORIES = 10000

    all_rewards_gumbel = []
    all_rewards_icfmdp = []

    for k in range(N_CFMDPS):
        print(f"{k}/{N_CFMDPS}")
        CFMDP = sample_CFMDP(interval_CFMDP)

        # Test with ICFMDP policy.   
        for j in range(N_TRAJECTORIES):
            print(f"{j}/{N_TRAJECTORIES}")
            trajectory = np.zeros((n_steps, n_state))
            current_state = 0

            for time_idx in range(n_steps):
                action = pi[time_idx, current_state]
                next_state = np.random.choice(16, size=1, p=CFMDP[time_idx, current_state, action])[0] 
                reward = MDP_rewards[current_state, action]
                trajectory[time_idx, :] = np.array([current_state, next_state, action, reward])
                current_state = next_state

            rewards = trajectory[:, 3]
            all_rewards_icfmdp.append(rewards)

        # Test with Gumbel-max policy.
        for j in range(N_TRAJECTORIES):
            print(f"{j}/{N_TRAJECTORIES}")
            trajectory = np.zeros((n_steps, n_state))
            current_state = 0

            for time_idx in range(n_steps):
                action = pi_gumbel[time_idx, current_state]
                next_state = np.random.choice(16, size=1, p=CFMDP[time_idx, current_state, action])[0] 
                reward = MDP_rewards[current_state, action]
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


def run_gumbel_sampler(mdp, trajectory, n_states, n_actions):
    gumbel_sampler = CounterfactualSampler(mdp, n_states, n_actions)
    gumbel_CFMDP = gumbel_sampler.run_gumbel_sampling(np.array([trajectory]))

    return gumbel_CFMDP


def get_gumbel_max_optimal_policy(P_cf, n_timesteps, n_states, n_actions, rewards):
    h_fun = np.zeros((n_timesteps+1, n_states)) 
    pi = np.zeros((n_timesteps+1, n_states), dtype=int)

    for s in range(n_states):
        h_fun[n_timesteps, s] = rewards[s, 0]

    for t in range(n_timesteps-1, -1, -1):
        for s in range(n_states):
            best_act = 0
            max_val = -np.inf
            
            for a in range(n_actions):
                val = rewards[s, a]

                for s_p in range(n_states):
                    if P_cf[s, a, s_p, t] != 0:
                        val += 1.0 * (P_cf[s, a, s_p, t] * h_fun[t+1, s_p])

                if val > max_val:
                    max_val = val
                    best_act = a
        
            h_fun[t, s] = max_val
            pi[t, s] = best_act

    return pi, h_fun


def calculate_average_probability_width_ICFMDP(interval_CFMDP):
    num_transitions = 10 * 16 * 4 * 16
    num_non_zero_transitions = 0
    total_prob_bounds = 0

    for t in range(10):
        for s in range(16):
            for a in range(4):
                for s_prime in range(16):
                    lb = interval_CFMDP[t, s, a, s_prime, 0]
                    ub = interval_CFMDP[t, s, a, s_prime, 1]

                    if not (ub == 0.0):
                        num_non_zero_transitions += 1
                        total_prob_bounds += ub - lb

    total_non_zero_prob_bounds = total_prob_bounds / num_non_zero_transitions
    total_prob_bounds /= num_transitions

    return total_prob_bounds, total_non_zero_prob_bounds


def main():
    if len(sys.argv) < 2:
        print("Usage: python gridworld.py <function_name>")
        sys.exit(1)
    
    function_name = sys.argv[1]

    if function_name == "generate_icfmdps":
        mdp = MDP()
        mdp.value_iteration()

        NUM_OBS_PATHS = 4

        optimal_trajectory = np.array([[  0,   0,   2,   0],
            [  0,   4,   2,   0],
            [  4,   5,   2,   1],
            [  5,   9,   2,   1],
            [  9,  10,   2,   2],
            [ 10,  14,   2,   3],
            [ 14,  15,   1,   5],
            [ 15,  15,   0, 100],
            [ 15,  15,   0, 100],
            [ 15,  15,   0, 100]]).astype(int)
        
        slightly_suboptimal_trajectory = np.array([[ 0,  4,  2,  0],
            [ 4,  4,  2,  1],
            [ 4,  4,  2,  1],
            [ 4,  8,  2,  1],
            [ 8, 12,  2,  2],
            [12, 12,  3,  3],
            [12,  8,  1,  3],
            [ 8,  9,  0,  2],
            [ 9, 13,  2,  2],
            [13,  9,  1,  4]]).astype(int)
        
        almost_catastrophic_trajectory = np.array([[   0,    1,    1,    0],
            [   1,    2,    0,    1],
            [   2,    1,    1,    2],
            [   1,    0,    3,    1],
            [   0,    4,    1,    0],
            [   4,    8,    2,    1],
            [   8,    9,    3,    2],
            [   9,   10,    2,    2],
            [  10,    9,    0,    3], # Almost goes to state 6 (the dangerous state) at this step.
            [   9,    10,   2,    2]]
        ).astype(int)

        # Enters dangerous, terminal state, receiving -100 reward.
        catastrophic_trajectory = np.array(mdp.sample_suboptimal_trajectory()).astype(int)

        print(optimal_trajectory)
        print(slightly_suboptimal_trajectory)
        print(almost_catastrophic_trajectory)
        print(catastrophic_trajectory)

        obs_trajectories = [optimal_trajectory, slightly_suboptimal_trajectory, almost_catastrophic_trajectory, catastrophic_trajectory]
        interval_CFMDPs_both_assumptions = []

        for i, obs_trajectory in enumerate(obs_trajectories):
            bound_calculator = MultiStepCFBoundCalculator(mdp.transition_matrix)
            mdp, interval_CFMDP, _, n_timesteps, n_states, n_actions = generate_interval_CFMDP(mdp, bound_calculator, trajectory=obs_trajectory)
            
            epsilon = 1e-16
            nonzero_mask = interval_CFMDP[:, :, :, :, 0] != 0.0

            # This adjustment is necessary due to floating-point errors, to ensure the probs are valid (i.e., that all UBs are > LBs).
            interval_CFMDP[nonzero_mask, 0] -= epsilon
            nonzero_mask = interval_CFMDP[:, :, :, :, 1] != 0.0
            interval_CFMDP[nonzero_mask, 1] += epsilon

            interval_CFMDPs_both_assumptions.append(interval_CFMDP)

            transition_matrix = format_transition_matrix_for_julia(interval_CFMDP, n_timesteps, n_states, n_actions)
        
            if not os.path.exists(f"transition_matrices"):
                os.makedirs(f"transition_matrices")

            with open(f"transition_matrices/gridworld_tra_{i}.pickle", "wb") as f:
                pickle.dump(transition_matrix, f)

            if not os.path.exists(f"ICFMDPs"):
                os.makedirs(f"ICFMDPs")

            convert_transition_matrix_to_julia_imdp(i, tra_filename=f"transition_matrices/gridworld_tra_{i}.pickle", filename=f"ICFMDPs/gridworld_{i}.jl")

        with open(f"interval_CFMDPs.pickle", "wb") as f:
            pickle.dump(interval_CFMDPs_both_assumptions, f)

        with open(f"obs_trajectories.pickle", "wb") as f:
            pickle.dump(obs_trajectories, f)

    
    elif function_name == "evaluate_performance":
        mdp = MDP()
        mdp.value_iteration()

        with open(f"interval_CFMDPs.pickle", "rb") as f:
            interval_CFMDPs_both_assumptions = pickle.load(f)

        with open(f"obs_trajectories.pickle", "rb") as f:
            obs_trajectories = pickle.load(f)

        for i, trajectory in enumerate(obs_trajectories):
            print(trajectory)

            interval_CFMDP = interval_CFMDPs_both_assumptions[i]
            n_timesteps = len(trajectory)
            n_states = 16
            n_actions = 4

            pi = load_generated_policy(f"ICFMDPs/gridworld_policy_{i}.jld2", n_timesteps, n_states)
            
            # Min and max value functions are also generated for analysis.
            # V_min = load_value_function(f"ICFMDPs/gridworld_value_pessimistic_{i}.jld2", n_timesteps, n_states)
            # V_max = load_value_function(f"ICFMDPs/gridworld_value_optimistic_{i}.jld2", n_timesteps, n_states)

            gumbel_CFMDP = run_gumbel_sampler(mdp, trajectory, n_states, n_actions)
            pi_gumbel, V_gumbel = get_gumbel_max_optimal_policy(gumbel_CFMDP, n_timesteps, n_states, n_actions, mdp.rewards)

            all_icfmdp_rewards, all_gumbel_rewards = evaluate_policies(mdp.rewards, pi, pi_gumbel, interval_CFMDP)
            all_icfmdp_rewards = np.array(all_icfmdp_rewards).reshape(2000000, 10)
            all_gumbel_rewards = np.array(all_gumbel_rewards).reshape(2000000, 10)

            mean_icfmdp_rewards = np.mean(np.array(all_icfmdp_rewards), axis=0)
            std_icfmdp_rewards = np.std(np.array(all_icfmdp_rewards), axis=0)
            mean_gumbel_rewards = np.mean(np.array(all_gumbel_rewards), axis=0)
            std_gumbel_rewards = np.std(np.array(all_gumbel_rewards), axis=0)

            # Clip upper and lower error bars to within reward range [-100, 100].
            upper_icfmdp_errors = np.clip(mean_icfmdp_rewards + std_icfmdp_rewards, None, 100) - mean_icfmdp_rewards
            upper_gumbel_errors = np.clip(mean_gumbel_rewards + std_gumbel_rewards, None, 100) - mean_gumbel_rewards
            lower_icfmdp_errors = mean_icfmdp_rewards - np.clip(mean_icfmdp_rewards - std_icfmdp_rewards, -100, None)
            lower_gumbel_errors = mean_gumbel_rewards - np.clip(mean_gumbel_rewards - std_gumbel_rewards, -100, None)

            with open(f"GridWorld (p=0.4) Results.txt", "a") as file:
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

            with open(f"GridWorld (p=0.4) Results.txt", "a") as f:
                f.write(f"CF path with lowest cumulative reward, ICFMDP = {worst_icfmdp_path}\n")
                f.write(f"CF path with lowest cumulative reward, Gumbel-max CFMDP = {worst_gumbel_path}\n")


    elif function_name == "measure_execution_time":
        # Measure the average time it takes to calculate interval CFMDP.
        mdp = MDP()
        mdp.value_iteration()

        NUM_OBS_PATHS = 20
        total_execution_time_in_seconds_interval = 0.0
        total_execution_time_in_seconds_gumbel = 0.0

        trajectories = []

        for i in range(NUM_OBS_PATHS):
            trajectories.append(mdp.sample_random_trajectory())

        # with open(f"executiontimegridworld4trajectories.pickle", "rb") as f:
        #     trajectories = pickle.load(f)

        for i in range(NUM_OBS_PATHS):
            trajectory = trajectories[i]

            start = time.time()

            bound_calculator = MultiStepCFBoundCalculator(mdp.transition_matrix)
            mdp, interval_CFMDP, _, n_timesteps, n_states, n_actions = generate_interval_CFMDP(mdp, bound_calculator, trajectory=trajectory)

            end = time.time()

            total_execution_time_in_seconds_interval += end - start

            start = time.time()
            
            gumbel_CFMDP = run_gumbel_sampler(mdp, trajectory, n_states, n_actions)
            
            end = time.time()

            total_execution_time_in_seconds_gumbel += end - start

        print(f"Execution Times for GridWorld (p=0.4):")
        print(f"Interval CFMDP = {total_execution_time_in_seconds_interval / NUM_OBS_PATHS}\n")
        print(f"Gumbel CFMDP = {total_execution_time_in_seconds_gumbel / NUM_OBS_PATHS}\n")


    elif function_name == "measure_avg_width":
        # Measures the average width of the counterfactual probability bounds in the interval CFMDP.
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


        with open(f"Gridworld (p=0.4) probability width results.txt", "a") as file:
            file.write(f"Counterfactual stability + monotonicity:\n")

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

            file.write(f"Only counterfactual stability:\n")

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
        print(f"Function '{function_name}' is not recognised.")

main()
