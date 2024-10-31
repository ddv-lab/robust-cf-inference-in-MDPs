import numpy as np
from decimal import Decimal

class CFBoundCalculatorCSOnly:
    def __init__(self, observed_state, observed_action, observed_next_state, transition_matrix):
        self.observed_state = observed_state
        self.observed_action = observed_action
        self.observed_next_state = observed_next_state
        self.transition_matrix = transition_matrix


    def _get_support(self, s, a):
        return set(np.nonzero(self.transition_matrix[s, a, :])[0])


    def _check_counterfactual_stability(self, s, a, s_prime):
        if self.transition_matrix[self.observed_state, self.observed_action, s_prime] > 0.0 and not s_prime == self.observed_next_state:
            if (self.transition_matrix[s, a, self.observed_next_state] / self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state]) >= (self.transition_matrix[s, a, s_prime] / self.transition_matrix[self.observed_state, self.observed_action, s_prime]):
                return False
            
        return True


    def _lower_bound_helper(self, s, a, s_prime):
        support_of_cf = self._get_support(s, a)
        support_of_cf.discard(s_prime)

        upper_bounds = sum([self.transition_matrix[s, a, s_prime_prime] - (self.upper_bound(s, a, s_prime_prime) * self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state]) for s_prime_prime in support_of_cf])

        return (self.transition_matrix[s, a, s_prime] - (1 - self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state] - upper_bounds)) / self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state]


    def lower_bound(self, s, a, s_prime):
        if s == self.observed_state and a == self.observed_action:
            # (s_t, a_t) -> s'
            if s_prime == self.observed_next_state:
                return 1.0
            else:
                return 0.0
            
        support_of_observed = self._get_support(self.observed_state, self.observed_action)
        support_of_cf = self._get_support(s, a)
        overlapping_support = support_of_observed.intersection(support_of_cf)

        if len(overlapping_support) == 0:
            # (s, a) has disjoint support with (s_t, a_t).
            if self.transition_matrix[s, a, s_prime] > 1 - self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state]:
                return (self.transition_matrix[s, a, s_prime] - (1 - self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state])) / self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state]
            else:
                return 0.0
            
        # (s, a) has overlapping support with (s_t, a_t).
        if not self._check_counterfactual_stability(s, a, s_prime):
            # CF prob must be 0 to satisfy counterfactual stability.
            return 0.0
        
        # Transition vacuously satisfies counterfactual stability.
        return max(0.0, self._lower_bound_helper(s, a, s_prime))

    def upper_bound(self, s, a, s_prime):
        if s == self.observed_state and a == self.observed_action:
            if s_prime == self.observed_next_state:
                return 1.0
            else:
                return 0.0
            
        support_of_observed = self._get_support(self.observed_state, self.observed_action)
        support_of_cf = self._get_support(s, a)
        overlapping_support = support_of_observed.intersection(support_of_cf)

        if len(overlapping_support) > 0:
            if s_prime != self.observed_next_state:
                if not self._check_counterfactual_stability(s, a, s_prime):
                    return 0
                
        # (s, a) has disjoint support from (s_t, a_t).
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
    

class MultiStepCFBoundCalculatorCSOnly:
    def __init__(self, transition_matrix):
        self.transition_matrix = transition_matrix
    

    def calculate_bounds(self, trajectory):
        n_timesteps = len(trajectory)
        n_states = self.transition_matrix.shape[0]
        n_actions = self.transition_matrix.shape[1]

        interval_cf_transition_matrix = np.zeros(shape=(n_timesteps, n_states, n_actions, n_states, 2))

        for t in range(n_timesteps):
            print(f"Calculating bounds at time t={t}")
            bound_calculator = CFBoundCalculatorCSOnly(trajectory[t][0], trajectory[t][2], trajectory[t][1], self.transition_matrix)
            interval_cf_transition_matrix[t] = bound_calculator.calculate_all_bounds()

        return interval_cf_transition_matrix