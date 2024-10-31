import numpy as np
from decimal import Decimal

class CFBoundCalculatorNoAssumptions:
    def __init__(self, observed_state, observed_action, observed_next_state, transition_matrix):
        self.observed_state = observed_state
        self.observed_action = observed_action
        self.observed_next_state = observed_next_state
        self.transition_matrix = transition_matrix


    def lower_bound(self, s, a, s_prime):
        if s == self.observed_state and a == self.observed_action:
            # (s_t, a_t) -> s'
            if s_prime == self.observed_next_state:
                return 1.0
            else:
                return 0.0
            
        # (s, a) has disjoint support with (s_t, a_t).
        if self.transition_matrix[s, a, s_prime] > 1 - self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state]:
            return (self.transition_matrix[s, a, s_prime] - (1 - self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state])) / self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state]
        else:
            return 0.0
            

    def upper_bound(self, s, a, s_prime):
        if s == self.observed_state and a == self.observed_action:
            if s_prime == self.observed_next_state:
                return 1.0
            else:
                return 0.0
            
        # (s, a) has a completely disjoint support from (s_t, a_t).
        return min(self.transition_matrix[s, a, s_prime], self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state]) / self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state]


    def calculate_all_bounds(self):
        n_states = self.transition_matrix.shape[0]
        n_actions = self.transition_matrix.shape[1]

        interval_cf_transition_matrix = np.zeros(shape=(n_states, n_actions, n_states, 2))

        for s in range(n_states):
            for a in range(n_actions):
                for s_prime in range(n_states):
                    lb = Decimal(self.lower_bound(s, a, s_prime))
                    ub = Decimal(self.upper_bound(s, a, s_prime))
                    interval_cf_transition_matrix[s, a, s_prime] = [lb, ub]
                
        return interval_cf_transition_matrix
    

class MultiStepCFBoundCalculatorNoAssumptions:
    def __init__(self, transition_matrix):
        self.transition_matrix = transition_matrix
    

    def calculate_bounds(self, trajectory):
        n_timesteps = len(trajectory)
        n_states = self.transition_matrix.shape[0]
        n_actions = self.transition_matrix.shape[1]

        interval_cf_transition_matrix = np.zeros(shape=(n_timesteps, n_states, n_actions, n_states, 2))

        for t in range(n_timesteps):
            print(f"Calculating bounds at time t={t}")
            bound_calculator = CFBoundCalculatorNoAssumptions(trajectory[t][0], trajectory[t][2], trajectory[t][1], self.transition_matrix)
            interval_cf_transition_matrix[t] = bound_calculator.calculate_all_bounds()

        return interval_cf_transition_matrix