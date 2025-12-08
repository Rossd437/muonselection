# How-to-Install and Run Muon Selection and Lifetime

Before creating a conda environment, you will need to clone the github repository to your local machine with the command below. <br>

For SSH: ```git clone git@github.com:Rossd437/project-CMSE-602.git``` <br>
For HTTPS: ```git clone https://github.com/Rossd437/project-CMSE-602.git```

Make sure your change into the correct directory with: <br>
```cd project-CMSE-602```

Now we can start on the conda environment. There is a script in the `scripts` directory ready to be run that will install all the dependecies for you. Switch to the scripts directory with: <br>

`cd scripts`

You may need to change the permission to be able to execute the script by running: <br>

`chmod 700 install_env.sh`

Then run: <br>

`source install_env.sh`

This will install all the dependencies from conda, such as, python, pip, snakemake, numpy, pandas, matplotlib, scipy, sccikit-learn, and seaborn. It will then activate the conda environment that it just made. After activating the conda environment, it still needs to pip install pylandau and h5flow. After it installs those last two dependencies with pip, your conda environemnt is now setup and activated for you and you will automatically `cd` back to `project-CMSE-602`. 

Now that you have your conda environment, running should go smooth..... hopefully. Since you should already be in the `project-CMSE-602` directory, just run: <br>

```snakemake --use-conda --cores 1```

AAAANNNNND finished!!! There should be plots of the purity, efficiency, and lifetime of the muon selection in a ```results/plots``` directory. Also, there should be two csv files of the track and segment information in ```results/csvs```.