Genre Classifier
========================

Optional: Store the below files in data/ 
[DataSet](https://www.dropbox.com/s/3shzbk6kyuo5akm/data.zip?dl=0)


Setup
=====================================

1. Clone the Repository.
```shell
git clone https://github.com/anshulshah96/Python-Genre-Classifier
```

2. Install [Anaconda](https://www.continuum.io/downloads#linux). Preferably Python 2.7.

3. Create Conda Environment
```bash
conda env create -f environment.yml
```

4. Start the Environment
```bash
source activate genre
```

5. Run the program
```python
python genre_classifier.py

```

JAudio Music Extraction Features
=====================================

	- Saved Features ONLY for Each window
	- Spectral Centroid
	- Spectral Rolloff Point
	- Zero Crossings
	- Strongest Frequency Via Spectral Centroid
	- Strongest Frequency Via FFT Maximum
	- MFCC
	- LPC
	- Window Size = 131072
	- Window Overlap = 0.06


License
=========
[MIT License](https://anshul.mit-license.org/)
