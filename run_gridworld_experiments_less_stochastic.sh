#!/bin/bash

# Generating the ICFMDPs
echo "Generating the ICFMDPs..."
python gridworld_less_stochastic.py generate_icfmdps

# Run robust value iteration on generated ICFMDPs
echo "Running robust value iteration..."
for i in {0..3}; do
    julia "ICFMDPs/gridworld_$i.jl"
done

# Evaluating policy performance
echo "Evaluating policy performance..."
python3 gridworld_less_stochastic.py evaluate_performance
