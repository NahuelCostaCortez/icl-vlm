import hydra
import utils
import data
import llm
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
import base64
import io
from PIL import Image
import openai
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def display_message_with_image(message_json):
    # Parse the message
    message = json.loads(message_json) if isinstance(message_json, str) else message_json
    
    # Print the text content
    for content_item in message["content"]:
        if content_item["type"] == "text":
            print(content_item["text"])
        elif content_item["type"] == "image_url":
            # Extract the base64 string (remove the data:image/jpeg;base64, prefix)
            img_data = content_item["image_url"]["url"].split(",")[1]
            
            # Decode base64 string
            img_bytes = base64.b64decode(img_data)
            
            # Open image with PIL
            img = Image.open(io.BytesIO(img_bytes))
            
            # Display with matplotlib
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.axis('off')
            plt.show()

@hydra.main(config_path="../config/brugada", config_name="few_shot", version_base=None)
def main(cfg):

	# ----------- Experiment conditions -----------
	# Data settings
	task = cfg.project
	name = cfg.name
	representation = cfg.data.representation
	shots = cfg.data.num_shots
	datafile_path = cfg.data.datafile_path
	diagnosis_column = 'diagnosis' if 'brugada' in datafile_path else 'diagnosis'
	label_replacements = cfg.data.label_replacements
	label_predictions = cfg.data.label_predictions
	save_path = cfg.data.save_path
	system_prompt_path = cfg.user_args.system_prompt_path
	user_query_path = cfg.user_args.user_query_path
	descriptions = True if "diagnostics" in name else False

	# Model settings
	model_name = cfg.model.model_name
	
	# system_prompt
	if representation == "full":
		user_query_path = user_query_path.split(".txt")[0] + "_12.txt"
		system_prompt_path = system_prompt_path.split(".txt")[0] + "_12.txt"
	system_prompt = utils.load_text_file(system_prompt_path)
	user_query = utils.load_text_file(user_query_path)
    # --------------------------------------------
    
	# ---------------- Load data -----------------
	metadata = utils.load_and_prepare_metadata(datafile_path, representation)

	# ICL: few-shot samples
	few_shot_samples = None
	few_shot_samples_metadata = None

	if name != "zero_shot":
		# Select samples for few-shot learning
		few_shot_samples_metadata = data.select_samples(metadata, task, shots)
		if name == "few_shot_waves" or name == "few_shot_waves_diagnostics":
			# change path to images with segments
			few_shot_samples_metadata['path'] = few_shot_samples_metadata.apply(
				lambda row: f"{row['path'].split('.png')[0]}_segments.png", 
				axis=1
			)
		few_shot_mappings = data.get_few_shot_mappings(few_shot_samples_metadata, label_replacements, descriptions)
		few_shot_samples = llm.encode_few_shot_samples(few_shot_mappings)
	# remove ICL samples from metadata
	metadata = metadata[~metadata['path'].str.contains('icl')]
	# keep only columns patient_id, brugada, path
	metadata = metadata[['patient_id', diagnosis_column, 'path']]
	# ---------------------------------------------

	# ----------------- LLM client ----------------
	client = llm.initialize_client(model_name)
	# ---------------------------------------------

	# ----------------- Run model -----------------
	for _, patient in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing patients"):
		# encode to base64
		query_img = llm.encode_image(patient['path'])

		# process messages
		message_for_sample = llm.process_messages(system_prompt, user_query, query_img, few_shot_samples)

		'''
		print(json.dumps(message_for_sample[0], indent=2))
		display_message_with_image(message_for_sample[1]) # if zero shot
		print(json.dumps(message_for_sample[1], indent=2))
		for i in range(2, len(message_for_sample)-1):
			display_message_with_image(message_for_sample[i])
		display_message_with_image(message_for_sample[-1])
		# Validate token count for GPT models
		#if "gpt" in model_name:
		#	llm.validate_token_count(message_for_sample, model_name, img_quality)
		'''

		# Get model prediction with error handling
		try:
			response = llm.get_model_prediction(client, model_name, message_for_sample)
		except openai.BadRequestError as e:
			logger.error(f"OpenAI BadRequestError for patient {patient['patient_id']}: {e}")
			response = None
		except openai.RateLimitError as e:
			logger.error(f"OpenAI RateLimitError for patient {patient['patient_id']}: {e}")
			response = None
		except openai.APIError as e:
			logger.error(f"OpenAI APIError for patient {patient['patient_id']}: {e}")
			response = None
		except Exception as e:
			logger.error(f"Unexpected error for patient {patient['patient_id']}: {e}")
			response = None
        
		#print(response)

		# Include response in metadata
		# Update the metadata dataframe with the model's response
		# Handle the case where response is not correctly generated
		if (
			response is not None
			and isinstance(response, dict)
			and 'thoughts' in response
			and 'answer' in response
			and 'score' in response
			and response['answer'] in label_predictions
		):
			metadata.loc[metadata['patient_id'] == patient['patient_id'], 'thoughts'] = response['thoughts']
			metadata.loc[metadata['patient_id'] == patient['patient_id'], 'answer'] = response['answer']
			metadata.loc[metadata['patient_id'] == patient['patient_id'], 'correct'] = patient[diagnosis_column] == label_predictions[response['answer']]
			metadata.loc[metadata['patient_id'] == patient['patient_id'], 'score'] = response['score']
		else:
			# If response is missing or malformed, fill with default/failure values
			metadata.loc[metadata['patient_id'] == patient['patient_id'], 'thoughts'] = None
			metadata.loc[metadata['patient_id'] == patient['patient_id'], 'answer'] = None
			metadata.loc[metadata['patient_id'] == patient['patient_id'], 'correct'] = None
			metadata.loc[metadata['patient_id'] == patient['patient_id'], 'score'] = None
		#break
	# ---------------------------------------------

	# Save results
	os.makedirs(save_path, exist_ok=True)
	metadata.to_csv(f"{save_path}/results_{model_name}.csv", index=False)


if __name__ == "__main__":
    main()