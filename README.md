# cMatch 

## Version 2 updates:
- Hybrid alignment algorithm in `matching.py`
- Storing Outputs of cMatch in `store_outputs_to_parquet.py`
- Visualisations in `visualisation.py`

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

streamlit run cmatch.py [template file] [Output File path for results ending in '.parquet' :str ] [match_results path : str][direction ('forward' or 'reverse'): str] [similarity threshold: float (0-1)] [overlap_tolerance: int (bases)] [sequence files (.seq or .fasta/.fastq)]

# Notes

Some visualisation functions in visualisation.py were developed specifically for GFP construct analysis with 3 parts and therefore may not be compatible with other cosntructs. This applies to plot_3d_multiple_scores. Further development is needed to generalise this to other constructs.
 However functions such as visualise_parts and plot_error_types should still be compatible for any construct for debugging and visualisation.


The main python files used are cmatch.py, reconstruction.py and matching.py.
Input files are processed into Seq objects by functions in matching.py.
Library matching is performed by functions in cmatch.py. 
Reconstruction is performed by functions in reconstruction.py.

## Simple Example 

Run cMatch on a simple example, matching the violacein template (template.json) to a test sequence. 


```
    streamlit run cmatch.py simple_example/template.json test_output.parquet ./test_match.json True 0.9 0 simple_example/vioA-test.seq
```

The cmatch tool will output Parquet and JSON for the results. A Streamlit browser will open automatically with the visualisations. Run without streamlit for debugging. 


## Run the tests

### Run the tests for algorithm CM_0

```
    $ python Testing_Algorithm_CM_0.py
```

### Run the tests for algorithm CM_1

```
    $ python Testing_Algorithm_CM_1.py
```

### Run the tests for algorithm CM_2 First Example

```
    $ python test_cm2_vio_easy_1vs1_th75.py
    $ python test_cm2_vio_easy_1vs1_th99.py
    $ python test_cm2_vio_hard_1vs1_th75.py
    $ python test_cm2_vio_hard_1vs1_th99.py
```

### Run the tests for algorithm CM_2 Second Example

```
    $ python test_cm2_vio_easy_1vsAll_th75.py
    $ python test_cm2_vio_easy_1vsAll1_th99.py
    $ python test_cm2_vio_hard_1vsAll1_th75.py
    $ python test_cm2_vio_hard_1vsAll1_th99.py
```

### Run the tests for algorithm CM_2 Real life example Lycopene Operon 

```
    $ python test_cm2_lycopene_sanger_10.py
    $ python test_cm2_lycopene_sanger_100.py

```

### Run the tests for algorithms CM_1, CM_2 Violacein-0000 cat x2 and cat x3

```
    $ python cm1_cat2.py
    $ python cm1_cat2min.py
```

```
    $ python cm1_cat3.py
    $ python cm1_cat3min.py
```

```
    $ python cm2_cat2.py
    $ python cm2_cat2min.py
```

```
    $ python cm2_cat3.py
    $ python cm2_cat3min.py
```
