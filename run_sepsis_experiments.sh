#!/bin/bash

# Generating the ICFMDPs
echo "Generating the ICFMDPs..."
python sepsis.py generate_icfmdps

# Run robust value iteration on generated ICFMDPs
echo "Running robust value iteration..."
for i in {0..3}; do
    julia "ICFMDPs/sepsis_$i.jl"
done

# Evaluating policy performance
echo "Evaluating policy performance..."
python3 sepsis.py evaluate_performance
