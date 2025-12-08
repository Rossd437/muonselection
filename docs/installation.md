# How-to-Install and Run Muon Selection and Lifetime

Before creating a conda environment, you will need to clone the github repository to your local machine with the command below. <br>

For SSH: ```git clone git@github.com:Rossd437/project-CMSE-602.git``` <br>
For HTTPS: ```git clone https://github.com/Rossd437/project-CMSE-602.git```

Make sure your change into the correct directory with: <br>
```cd project-CMSE-602```

Now we can start on the conda environment. The only dependency that you will need to install yourself is snakemake. The best way to do this is to make a conda environment with snakemake already installed. Run the command below: <br>
```conda create -c conda-forge -c bioconda -c nodefaults -n muon_selection_env snakemake```

Then run ```conda activate muon_selection_env```

Now that you have snakemake installed into your conda environment, running should go smooth..... hopefully. There is already an environment.yaml file in the git repository that you cloned, and snakemake will automatically run this yaml file when the workflow is run.

To run the complete workflow: <br>

```snakemake --use-conda --cores 1```

AAAANNNNND finished!!! There should be plots of the purity, efficiency, and lifetime of the muon selection in a ```results/plots``` directory. Also, there should be two csv files of the track and segment information in ```results/csvs```.