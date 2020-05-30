# taus-qe
Linear Mapping between two monolingual vector spaces using the Least Squares or Orthogonal Procrustes methods
Calculate a linear transformation between two monolingual vector spaces using either the Least Squares or Orthogonal Procrustes method and calculate a semantic similarity score between source and target sentences.

From the parent directory, run semscore.py, which takes the following arguments:

	-mtd: mapping method; "lstsq" for Least Squres or "orth" for Orthogonal Procrustes
	-n: integer value for shortening your input data, useful for running quick tests (optional, None if not provided)