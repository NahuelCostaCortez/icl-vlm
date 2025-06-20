# PEAK IDENTIFICATION
#python main.py --config-path=../config/waves --config-name=zero_shot model.model_name=gpt-4o
#python main.py --config-path=../config/waves --config-name=few_shot data.num_shots=5 model.model_name=gpt-4o
#python main.py --config-path=../config/waves --config-name=few_shot data.num_shots=10 model.model_name=gpt-4o
#python main.py --config-path=../config/waves --config-name=zero_shot model.model_name=gemma-3-27b-it
#python main.py --config-path=../config/waves --config-name=few_shot data.num_shots=5 model.model_name=gemma-3-27b-it
#python main.py --config-path=../config/waves --config-name=few_shot data.num_shots=10 model.model_name=gemma-3-27b-it

# ABNORMAL DETECTION
#python main.py --config-path=../config/abnormal --config-name=zero_shot model.model_name=gpt-4o
#python main.py --config-path=../config/abnormal --config-name=few_shot data.num_shots=5 model.model_name=gpt-4o
#python main.py --config-path=../config/abnormal --config-name=few_shot data.num_shots=10 model.model_name=gpt-4o
#python main.py --config-path=../config/abnormal --config-name=zero_shot model.model_name=gemma-3-27b-it
#python main.py --config-path=../config/abnormal --config-name=few_shot data.num_shots=5 model.model_name=gemma-3-27b-it
#python main.py --config-path=../config/abnormal --config-name=few_shot data.num_shots=10 model.model_name=gemma-3-27b-it

# BRUGADA DETECTION
#echo "Zero shot"
#python main.py --config-path=../config/brugada --config-name=zero_shot model.model_name=gpt-4o-2024-11-20

#echo "Few shot, just labels"
#python main.py --config-path=../config/brugada --config-name=few_shot data.num_shots=5 model.model_name=gpt-4o-2024-11-20
#python main.py --config-path=../config/brugada --config-name=few_shot data.num_shots=10 model.model_name=gpt-4o-2024-11-20

#echo "Few shot, labels and waves"
#python main.py --config-path=../config/brugada --config-name=few_shot_waves data.num_shots=5 model.model_name=gpt-4o-2024-11-20
#python main.py --config-path=../config/brugada --config-name=few_shot_waves data.num_shots=10 model.model_name=gpt-4o-2024-11-20

#echo "Few shot, labels and diagnostics"
#python main.py --config-path=../config/brugada --config-name=few_shot_diagnostics data.num_shots=5 model.model_name=gpt-4o-2024-11-20
#python main.py --config-path=../config/brugada --config-name=few_shot_diagnostics data.num_shots=10 model.model_name=gpt-4o-2024-11-20

#echo "Few shot, labels, waves and diagnostics"
#python main.py --config-path=../config/brugada --config-name=few_shot_waves_diagnostics data.num_shots=5 model.model_name=gpt-4o-2024-11-20
#python main.py --config-path=../config/brugada --config-name=few_shot_waves_diagnostics data.num_shots=10 model.model_name=gpt-4o-2024-11-20

# now with 12 leads
#echo "Zero shot"
#python main.py --config-path=../config/brugada --config-name=zero_shot model.model_name=gpt-4o-2024-11-20 data.representation=full

#echo "Few shot, just labels"
#python main.py --config-path=../config/brugada --config-name=few_shot data.num_shots=5 model.model_name=gpt-4o-2024-11-20 data.representation=full
#python main.py --config-path=../config/brugada --config-name=few_shot data.num_shots=10 model.model_name=gpt-4o-2024-11-20 data.representation=full

#echo "Few shot, labels and diagnostics"
#python main.py --config-path=../config/brugada --config-name=few_shot_diagnostics data.num_shots=5 model.model_name=gpt-4o-2024-11-20 data.representation=full
#python main.py --config-path=../config/brugada --config-name=few_shot_diagnostics data.num_shots=10 model.model_name=gpt-4o-2024-11-20 data.representation=full



# ----- medgemma-4b-it -----
#echo "Zero shot"
#python main.py --config-path=../config/brugada --config-name=zero_shot model.model_name=medgemma-4b-it

#echo "Few shot, just labels"
#python main.py --config-path=../config/brugada --config-name=few_shot data.num_shots=5 model.model_name=medgemma-4b-it
#python main.py --config-path=../config/brugada --config-name=few_shot data.num_shots=10 model.model_name=medgemma-4b-it

#echo "Few shot, labels and waves"
#python main.py --config-path=../config/brugada --config-name=few_shot_waves data.num_shots=5 model.model_name=medgemma-4b-it
#python main.py --config-path=../config/brugada --config-name=few_shot_waves data.num_shots=10 model.model_name=medgemma-4b-it

#echo "Few shot, labels and diagnostics"
#python main.py --config-path=../config/brugada --config-name=few_shot_diagnostics data.num_shots=5 model.model_name=medgemma-4b-it
#python main.py --config-path=../config/brugada --config-name=few_shot_diagnostics data.num_shots=10 model.model_name=medgemma-4b-it

#echo "Few shot, labels, waves and diagnostics"
#python main.py --config-path=../config/brugada --config-name=few_shot_waves_diagnostics data.num_shots=5 model.model_name=medgemma-4b-it
#python main.py --config-path=../config/brugada --config-name=few_shot_waves_diagnostics data.num_shots=10 model.model_name=medgemma-4b-it
