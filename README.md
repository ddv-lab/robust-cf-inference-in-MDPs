# Robust Counterfactual Inference in MDPs

This paper addresses a key limitation in existing counterfactual inference methods for Markov Decision Processes (MDPs). Current approaches often assume a specific causal model, e.g., the Gumbel-max structural causal model (SCM), to make counterfactuals identifiable. However, there are typically multiple causal models that align with the observational and interventional distributions of an MDP, each yielding different counterfactual distributions. Fixing a particular mechanism is a strong assumption that limits the validity (and usefulness) of counterfactual inference. We propose a novel approach that computes tight bounds on counterfactual transition probabilities, encompassing all compatible causal models. A major contribution of this work is to provide closed-form expressions for such bounds which make them highly efficient to compute, unlike previous approaches which require solving prohibitively large optimisation problems. Once such an _interval_ counterfactual MDP is constructed, our method identifies robust counterfactual policies that optimise the worst-case reward w.r.t.\ the uncertain MDP probabilities. We evaluate our approach on various MDP case studies, demonstrating improved robustness compared to existing methods.

## Instructions
The data for the Sepsis experiment can be found [here](https://github.com/GuyLor/gumbel_max_causal_gadgets_part2/tree/main/data). Create a folder called `data/` and add the unzipped `diab_txr_mats-replication.pkl.zip` file to the folder.

To replicate the experiments, you can either run the `.sh` scripts that are provided for each of the MDP environments, or run the Python code directly.

The `.sh` scripts are: `run_gridworld_experiments_less_stochastic.sh` (GridWorld (p=0.9)), `run_gridworld_experiments_more_stochastic.sh` (GridWorld (p=0.4)) and `run_sepsis_experiments.sh` (Sepsis). These run the policy evaluation experiments from our paper.

The Python files can be run as follows:

* `python <experiment.py> generate_icfmdps`: Generates the ICFMDPs for the 4 observed paths from the paper.
* `python <experiment.py> evaluate_performance`: Compares the performance of the ICFMDP policy vs. the Gumbel-max SCM policy.
* `python <experiment.py> measure_execution_time`: Measures the average execution time for generating the ICFMDP vs. the Gumbel-max SCM CFMDP.
* `python <experiment.py> measure_avg_width`: Measures the average width of probability bounds in the generated ICFMDPs, with and without assumptions.

## Acknowledgements
The code for the Sepsis MDP is taken from [1], and can be found [here](https://github.com/clinicalml/gumbel-max-scm). This is licensed under the MIT License.

### References
[1] Oberst, M., & Sontag, D. (2019, May). Counterfactual off-policy evaluation with gumbel-max structural causal models. In _International Conference on Machine Learning_ (pp. 4881-4890). PMLR.
