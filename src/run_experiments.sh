#!/bin/bash

# Define common variables
models=("gpt-4o" "gemma-3-27b-it")
shots=(5 10)

# # PEAK IDENTIFICATION
# echo "Running PEAK IDENTIFICATION experiments"
# for model in "${models[@]}"; do
#     echo "Zero shot with $model"
#     python main.py --config-path=../config/waves --config-name=zero_shot model.model_name=$model
#     for shot in "${shots[@]}"; do
#         echo "Few shot ($shot) with $model"
#         python main.py --config-path=../config/waves --config-name=few_shot data.num_shots=$shot model.model_name=$model
#     done
# done

# # ABNORMAL DETECTION
# echo "Running ABNORMAL DETECTION experiments"
# for model in "${models[@]}"; do
#     echo "Zero shot with $model"
#     python main.py --config-path=../config/abnormal --config-name=zero_shot model.model_name=$model
#     for shot in "${shots[@]}"; do
#         echo "Few shot ($shot) with $model"
#         python main.py --config-path=../config/abnormal --config-name=few_shot data.num_shots=$shot model.model_name=$model
#     done
# done

# BRUGADA DETECTION
brugada_models=("gpt-4o-2024-11-20" "medgemma-4b-it")
brugada_configs=("zero_shot" "few_shot" "few_shot_waves" "few_shot_diagnostics" "few_shot_waves_diagnostics")
representations=("" "data.representation=full")

echo "Running BRUGADA DETECTION experiments"
for model in "${brugada_models[@]}"; do
    for config in "${brugada_configs[@]}"; do
        if [ "$config" = "zero_shot" ]; then
            echo "Zero shot with $model"
            python main.py --config-path=../config/brugada --config-name=$config model.model_name=$model
            echo "Zero shot (full) with $model"
            python main.py --config-path=../config/brugada --config-name=$config model.model_name=$model data.representation=full
        elif [ "$config" = "few_shot_waves" ] && [ "$model" = "gpt-4o-2024-11-20" ]; then
            # Skip few_shot_waves for gpt-4o-2024-11-20 as per original
            continue
        else
            for shot in "${shots[@]}"; do
                echo "$config ($shot) with $model"
                python main.py --config-path=../config/brugada --config-name=$config data.num_shots=$shot model.model_name=$model
                if [ "$config" != "few_shot_waves" ]; then
                    echo "$config ($shot, full) with $model"
                    python main.py --config-path=../config/brugada --config-name=$config data.num_shots=$shot model.model_name=$model data.representation=full
                fi
            done
        fi
    done
done
