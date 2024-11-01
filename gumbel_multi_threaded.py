from multiprocessing import Process, Manager
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
        # This is the observed outcome.
        if (next_state == obs_state) and not(np.isneginf(obs_logits[next_state])):
            gumbels[:, obs_state] = topgumbel - obs_logits[next_state]

        # These were the other feasible options (p > 0).
        elif not(np.isneginf(obs_logits[next_state])):
            gumbels[:, next_state] = truncated_gumbel(obs_logits[next_state], topgumbel) - obs_logits[next_state]
        
        # These have zero probability to start with, so are unconstrained.
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

        gumbels = topdown(obs_logits, next_state, n_mc);

        posterior = intrv_logits + gumbels
        intrv_posterior = np.argmax(posterior, axis=1)

        posterior_prob = np.zeros(np.size(intrv_prob, 0))
        for i in range(np.size(intrv_prob, 0)):
            posterior_prob[i] = np.sum(intrv_posterior == i) / n_mc

        return posterior_prob, intrv_posterior
    

    def cf_sample_prob(self, tx_mat_full, trajectories, a, time_idx, P_cf_save, n_cf_samps=1): 
        n_obs = trajectories.shape[0] 
        n_mc = 10000
        
        for obs_idx in range(n_obs):
            P_cf = {}

            for _ in range(n_cf_samps):
                obs_state = trajectories[obs_idx, time_idx, :]
                obs_current_state = int(obs_state[0])
                obs_next_state = int(obs_state[1])
                obs_action = int(obs_state[2])

                P_cf[a, time_idx] = np.zeros((int(self.n_states),int(self.n_states)))

                for s in range(self.n_states):                    
                    obs_intrv = tx_mat_full[obs_action, obs_current_state, :]
                    cf_intrv =  tx_mat_full[a, s, :]
                    
                    cf_prob, s_p = self.cf_posterior(obs_intrv, cf_intrv, obs_next_state, n_mc)
            
                    for s_p in range(len(cf_prob)):
                        P_cf[a,time_idx][s,s_p] = cf_prob[s_p]

        P_cf_save[(a,time_idx)] = P_cf


    def run_sample(self, tx_mat_full, inp, trajectories, P_cf):
        P_cf_save = {}

        for i in inp:
            self.cf_sample_prob(tx_mat_full, trajectories, i[0], i[1], P_cf_save)

        for i in inp:
            P_cf.update(P_cf_save[i])


    def run_parallel_sampling(self, tx_mat_full, trajectories):
        n_steps = trajectories.shape[1]
        n_actions = 8
        
        inp = [(a, time_idx) for time_idx in range(n_steps) for a in range(n_actions)]

        # Run with n threads.
        def split(a, n):
            k, m = divmod(len(a), n)
            return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
        
        split_work = split(inp, 32)
        processes = []

        with Manager() as manager:
            P_cf = manager.dict()
            
            for chunk in split_work:
                process = Process(target=self.run_sample, args=(tx_mat_full, chunk, trajectories, P_cf))
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

            return P_cf.copy()