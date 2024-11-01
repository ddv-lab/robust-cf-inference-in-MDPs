import numpy as np

TRANSITION_PROB_COUNTER = 1

def generate_julia_file(filename, interval_mdp_code, transition_probs_code):
    with open(filename, 'w') as file:
        file.write(transition_probs_code)
        file.write(interval_mdp_code)

def create_transition_prob_string(lower_probs, upper_probs):
    global TRANSITION_PROB_COUNTER

    lower_str = "\n        ".join([" ".join(map(str, row)) for row in lower_probs])
    upper_str = "\n        ".join([" ".join(map(str, row)) for row in upper_probs])

    prob_str = f"""
    prob{TRANSITION_PROB_COUNTER} = IntervalProbabilities(;
        lower = [
            {lower_str}
        ],
        upper = [
            {upper_str}
        ],
    )
    """

    TRANSITION_PROB_COUNTER += 1

    return prob_str

def create_transition_prob_string_compressed(lower_probs, upper_probs):
    global TRANSITION_PROB_COUNTER

    lower_str_parts = []
    upper_str_parts = []

    for col in range(lower_probs.shape[1]):
        non_zero_probs_indices = np.nonzero(upper_probs[:, col])[0] # use upper bound indicies because some states might have LB=0

        non_zero_probs_lower = ", ".join(map(str, lower_probs[non_zero_probs_indices, col]))
        non_zero_probs_upper = ", ".join(map(str, upper_probs[non_zero_probs_indices, col]))
        non_zero_probs_indices = [idx+1 for idx in non_zero_probs_indices]
        non_zero_probs_indices = ", ".join(map(str, non_zero_probs_indices))
        
        lower_str_parts.append(f"\n\t\t\tSparseVector({lower_probs.shape[0]}, [{non_zero_probs_indices}], [{non_zero_probs_lower}]),")

        upper_str_parts.append(f"\n\t\t\tSparseVector({upper_probs.shape[0]}, [{non_zero_probs_indices}], [{non_zero_probs_upper}]),")

    lower_str = "".join(lower_str_parts)
    upper_str = "".join(upper_str_parts)

    prob_str = f"""
    prob{TRANSITION_PROB_COUNTER} = IntervalProbabilities(;
        lower = sparse_hcat({lower_str}
        ),
        upper = sparse_hcat({upper_str}
        ),
    )
    """

    TRANSITION_PROB_COUNTER += 1

    return prob_str


def convert(transition_matrix, iter, rewards, tra_filename = f"transition_matrices/sepsis_tra.pickle", filename = f"ICFMDPs/sepsis.jl"):
    global TRANSITION_PROB_COUNTER
    TRANSITION_PROB_COUNTER = 1

    transition_probs_str = ""
    actions_str = ""

    rewards = np.tile(np.array(rewards), 11)
    rewards = rewards.astype(np.float32)

    i = 1

    for t in range(11):
        for s in range(1440):
            probs = transition_matrix[(t,s)]
            transition_probs_str += f"{create_transition_prob_string_compressed(probs[0], probs[1])}\n"
            actions_str += f"""["0", "1", "2", "3", "4", "5", "6", "7"] => prob{i}, """
            i += 1

    string_array = [str(x) for x in rewards]
    rewards_str = "[" + ", ".join(string_array) + "]"

    transition_probs_code = f"""
    using IntervalMDP
    using JLD2
    using SparseArrays

    {transition_probs_str}

    transition_probs = [{actions_str}]

    initial_states = [Int32(778)]

    mdp = IntervalMarkovDecisionProcess(transition_probs, initial_states)
    """

    interval_mdp_code = f"""
    discount_factor = 1.0

    time_horizon = 10
    println("Running value iteration...")

    V_mins = []
    V_maxs = []

    for i in 1:10
        println(i)
        prop = FiniteTimeReward({rewards_str}, discount_factor, i)
        
        spec = Specification(prop, Pessimistic, Maximize)
        problem = Problem(mdp, spec)
        V_min, k, residual = value_iteration(problem)
        V_min = Array(V_min)
        push!(V_mins, V_min)

        spec = Specification(prop, Optimistic, Maximize)
        problem = Problem(mdp, spec)
        V_max, k, residual = value_iteration(problem)
        V_max = Array(V_max)
        push!(V_maxs, V_max)   
    end

    JLD2.save("ICFMDPs/sepsis_value_pessimistic_{iter}.jld2", "data", V_mins)
    JLD2.save("ICFMDPs/sepsis_value_optimistic_{iter}.jld2", "data", V_maxs)

    prop = FiniteTimeReward({rewards_str}, discount_factor, time_horizon)

    spec = Specification(prop, Pessimistic, Maximize)
    problem = Problem(mdp, spec)

    policy = control_synthesis(problem)

    JLD2.save("ICFMDPs/sepsis_policy_{iter}.jld2", "data", policy)

    """

    generate_julia_file(filename, interval_mdp_code, transition_probs_code)
    print(f"Julia file '{filename}' generated successfully.")
