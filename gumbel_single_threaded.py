import numpy as np

def truncated_gumbel(logit, truncation):
    assert not np.isneginf(logit)

    gumbel = np.random.gumbel(size=(truncation.shape[0])) + logit
    trunc_g = -np.log(np.exp(-gumbel) + np.exp(-truncation))
    return trunc_g


def topdown(obs_logits, obs_state, nsamp=1):
    poss_next_states = obs_logits.shape[0]
    gumbels = np.zeros((nsamp, poss_next_states))

    # Sample top gumbels.
    topgumbel = np.random.gumbel(size=(nsamp))

    for next_state in range(poss_next_states):
        # This is the observed next state.
        if (next_state == obs_state) and not(np.isneginf(obs_logits[next_state])):
            gumbels[:, obs_state] = topgumbel - obs_logits[next_state]

        # These were the other feasible options (p > 0).
        elif not(np.isneginf(obs_logits[next_state])):
            gumbels[:, next_state] = truncated_gumbel(obs_logits[next_state], topgumbel) - obs_logits[next_state]

        # These had zero probability to start with, so are unconstrained.
        else:
            gumbels[:, next_state] = np.random.gumbel(size=nsamp)

    return gumbels


class CounterfactualSampler(object):
    def __init__(self, mdp, n_states, n_actions):
        self.mdp = mdp
        self.sprtb_theta = 0.9
        self.sprtb_delta = 0.05
        self.sprtb_r = 0.9
        self.n_states = n_states
        self.n_actions = n_actions
    
    def cf_posterior(self, obs_prob, intrv_prob, state, n_mc):
        obs_logits = np.log(obs_prob)
        next_state = state
        intrv_logits = np.log(intrv_prob)
        gumbels = topdown(obs_logits, next_state, n_mc)
        posterior = intrv_logits + gumbels
        intrv_posterior = np.argmax(posterior, axis=1)
        posterior_prob = np.zeros(np.size(intrv_prob, 0))
        
        for i in range(np.size(intrv_prob, 0)):
            posterior_prob[i] = np.sum(intrv_posterior == i) / n_mc

        return posterior_prob, intrv_posterior

    def cf_sample_prob(self, trajectories, all_actions, T, n_cf_samps=1): 
        n_obs = trajectories.shape[0] 
        n_mc = 100000

        P_cf = np.zeros(shape=(self.n_states, all_actions, self.n_states, T))
        
        for a in range(all_actions):
            for t in range(T):
                for obs_idx in range(n_obs):
                    for _ in range(n_cf_samps):
                        obs_state = trajectories[obs_idx, t, :]
                        obs_current_state = int(obs_state[0])
                        obs_next_state = int(obs_state[1])
                        obs_action = int(obs_state[2])

                        for s in range(self.n_states):
                            obs_intrv = self.mdp.transition_matrix[obs_current_state, obs_action, :]
                            cf_intrv = self.mdp.transition_matrix[s, a, :]
                            cf_prob, s_p = self.cf_posterior(obs_intrv, cf_intrv, obs_next_state, n_mc)
                            
                            for s_p in range(len(cf_prob)):
                                P_cf[s, a, s_p, t] = cf_prob[s_p]

        return P_cf


    def run_gumbel_sampling(self, trajectories):
        n_steps = trajectories.shape[1]
        n_actions = self.n_actions

        P_cf = self.cf_sample_prob(trajectories, n_actions, n_steps)

        return P_cf
