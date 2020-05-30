# taus-qe

Find a linear transformation between two monolingual vector spaces using either the Least Squares or the Orthogonal Procrustes method
and calculate a semantic similarity score between source and target sentences based on the resulting transformation.

## Usage

From the parent directory, run semscore.py, which takes the following arguments:

	-mtd: mapping method; "lstsq" for Least Squres or "orth" for Orthogonal Procrustes
	-n: integer value for shortening your input data, useful for running quick tests (optional, None if not provided)
	
Required files:
	
	-data/pairs.json: file containing source and target sentences, follow the example in data/sample_pairs.json
	-data/seed_dictionary.json: a high-quality seed dictionary for learning the transformation matrix
	-vectors/source_vectors.bin: source language word embeddings, word2vec format
	-vectors/target_vectors.bin: target language word embeddings, word2vec format
		
The following files are generated:

	-data/transformation_matrix.csv
	-data/results.csv: a file containing the source and target sentences along with their calculated semantic similarity score
	
## Find out more
	
Information about word2vec word embeddings:

	-https://radimrehurek.com/gensim/models/word2vec.html
	
Read more about the transformation methods here:

	-https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
	-https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.linalg.orthogonal_procrustes.html