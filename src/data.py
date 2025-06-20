import pandas as pd
from transformers import AutoModel
import torch

SAMPLES_IN_5_SECONDS_AT_100HZ = 500


# ----------------- DATA SELECTION ------------------ 
def select_samples(metadata, task, n_samples, seed=42):
	"""
	Select n_samples random samples from the metadata
	"""

	if task == 'waves':
		patients = metadata[metadata['diagnosis'] == -1].sample(n=n_samples, random_state=seed)
	elif task == 'abnormal':
		patients = metadata[metadata['path'].str.contains('icl')].sample(n=n_samples, random_state=seed)
	elif task == 'brugada':
		metadata = metadata[metadata['path'].str.contains('icl')] # filter ICL samples
		
		# Sample n_samples from each class
		sampled_by_class = {}
		for diagnosis in metadata['brugada'].unique():
			sampled_by_class[diagnosis] = metadata[metadata['brugada'] == diagnosis].sample(n=n_samples, random_state=seed)
		
		patients = pd.concat([sampled_by_class[diagnosis] for diagnosis in metadata['brugada'].unique()], ignore_index=True)
		
		# Reset index to have continuous numbering
		patients = patients.reset_index(drop=True)

	return patients

def get_few_shot_mappings(few_shot_samples_metadata, label_replacements, descriptions=False):
	"""
	Create few-shot mappings.
	"""

	# Dictionary with the description as key and the path as value
	if descriptions:
		few_shot_mappings = {
			label_replacements[str(sample['brugada'])]+'.'+sample['description']: sample['path'] for _, sample in few_shot_samples_metadata.iterrows()
		}
	else:
		few_shot_mappings = {
			label_replacements[str(diagnosis)]: few_shot_samples_metadata[few_shot_samples_metadata['diagnosis'] == diagnosis]['path'].tolist()
			for diagnosis in few_shot_samples_metadata['diagnosis'].unique()
		}

	return few_shot_mappings

# --------------------------------------------------

# ---------------------- RAG -----------------------

def setup_knn(metadata, datafile_path, prepare_data_fn):
	"""
	Set up KNN for similar sample selection.

	Args:
		metadata: Metadata
		datafile_path: Path to the datafile

	Returns:
		embeddings_model: Embeddings model
		collection: Chroma collection
	"""

	size = 'small'
	embeddings_model = AutoModel.from_pretrained(f"Edoardo-BS/hubert-ecg-{size}", trust_remote_code=True)
	samples, metadatas, ids = prepare_data_fn(metadata, datafile_path)
	embeddings_model.eval()
	embeddings = []
	with torch.no_grad():
		embeddings = embeddings_model(samples, 
							attention_mask=None, 
							output_attentions=False, 
							output_hidden_states=False, 
							return_dict=True)['last_hidden_state']

	# Mean pooling over sequence length dimension to get (bs, 512)
	embeddings = embeddings.mean(axis=1)
	embeddings = embeddings.cpu().numpy()
	collection = create_chroma_collection(embeddings, metadatas, ids)

	return embeddings_model, collection

def create_chroma_collection(embeddings, metadatas, ids):
	"""
	Create a Chroma collection with the embeddings, metadatas and ids.
	"""
	import chromadb

	chroma_client = chromadb.Client()
	try:
		chroma_client.delete_collection(name="ecg_embeddings")
	except:
		pass

	collection = chroma_client.create_collection(
		name="ecg_embeddings",
		metadata={"hnsw:space": "cosine"}  # Using cosine similarity for embeddings
	)

	# Split into batches of 5000 to avoid exceeding max batch size
	batch_size = 5000
	for i in range(0, len(embeddings), batch_size):
		batch_end = min(i + batch_size, len(embeddings))
		collection.add(
			embeddings=embeddings[i:batch_end].tolist(),
			metadatas=metadatas[i:batch_end],
			ids=ids[i:batch_end]
    )
		
	return collection

def query_similar_ecgs(model, collection, patient_id, query_ecg, n_results=5):
	"""
	Find similar ECGs to the query ECG.

	Args:
		model: Model to use for embedding
		collection: Chroma collection
		patient_id: Patient ID
		query_ecg: Preprocessed ECG tensor
		n_results: Number of results to return
		
	Returns:
		List of similar ECGs with metadata
	"""

	with torch.no_grad():
		query_embedding = model(query_ecg.unsqueeze(0)).last_hidden_state.mean(dim=1)
		query_embedding = query_embedding.cpu().numpy()

	results_final = []
	for class_label in ['normal', 'brugada']:

		results = collection.query(
			query_embeddings=query_embedding.tolist(),
			n_results=n_results+1,
			where={"label": class_label},
		)

		# Check if query patient_id is in the results
		if any(result['patient_id'] == patient_id for result in results['metadatas'][0]):
			# remove first result since it's the query patient
			results['metadatas'][0] = results['metadatas'][0][1:]
		
		results_final.extend(results['metadatas'][0][:n_results])

	return results_final
# --------------------------------------------------