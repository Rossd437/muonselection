# How-to-Guides

## How To Obtain Track and Segment Information

After you ran the selection, there will be two csv files in the `results/csvs` directory. If you would like to read through these csv files, pandas is the recommended option. You should already have pandas installed into your conda environment if you are the selection already. So, just activate your environment with: <br>

`conda activate selection_env`

If you didn't run the selection, you can install pandas in a python virtual environment or conda environment with the commands below. <br>

Python Virtual Environment: `pip install pandas` <br>
Conda Environment: `conda install conda-forge::pandas` <br>

To load in the csv files,

Load in csv file: `df = pd.read_csv(<file_name>)`

If you would like to now what values are stored in the csv file:

Csv columns: `df.columns`

You can do all kinds of data analysis with these dataframes for examples:

Segment dQ/dx: `df['dQ']/df['dx']`
Segment dN/dx: `df['dQ']/df['dx']`

and so on. Pretty much just do `df['<column_name>']`.

