import os
import numpy as np
import pandas as pd


def load_text_file(file_path):
    """Load text from a file."""
    with open(file_path, mode="r") as f:
        return f.read()

def load_and_prepare_metadata(datafile_path, representation):
	"""Load and prepare the metadata dataframe."""

	metadata = pd.read_csv(datafile_path + '/metadata.csv')

	if 'brugada' in datafile_path:
		metadata = metadata[metadata['missing'] == False].reset_index(drop=True)
		# Convert brugada=2 to brugada=1 for consistency, they both indicate brugada
		metadata.loc[metadata['brugada'] == 2, 'brugada'] = 1

	# need to add datafile_path to get the complete path
	if representation is not None:
		metadata['path'] = metadata.apply(
			lambda row: f"{datafile_path}/{representation}/{row['path']}", 
			axis=1
		)
	else:
		metadata['path'] = metadata.apply(
			lambda row: f"{datafile_path}/{row['path']}", 
			axis=1
		)

	return metadata