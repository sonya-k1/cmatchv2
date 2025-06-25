# cMatch 

## Prerequisite

You need Python 3.9 and pip the package installer for Python [pip](https://pip.pypa.io/en/stable/)


## Install

Install a virtual environment if you want.

```
    $ python3 -m venv --prompt cmatch venv
    $ source ./venv/bin/activate
```

Install the dependencies

```
    $ pip install -r requirements.txt
```

## cMatch command line tool

run cmatch.py

```
    $ ./cmatch.py --help 
```


## Streamlit integration for visualisation

streamlit run cmatch.py [template file] [output file path ending in '.parquet'] [similarity threshold: float (0-1)] [overlap_tolerance: int (bases)] [sequence files (.seq or .fasta)]

# Notes

Some visualisation functions in visualisation.py were developed specifically for GFP construct analysis with 3 parts and therefore may not be compatible with other cosntructs. This applies to plot_3d_multiple_scores. Further development is needed to generalise this to other constructs.
 However functions such as visualise_parts and plot_error_types should still be compatible for any construct for debugging and visualisation.


The main python files used are cmatch.py, reconstruction.py and matching.py.
Input files are processed into Seq objects by functions in matching.py.
Library matching is performed by functions in cmatch.py. 
Reconstruction is performed by functions in reconstruction.py.