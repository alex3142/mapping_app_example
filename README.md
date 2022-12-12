# Mapping App README #


### What is this repository for? ###

This repo contains code to deploy an mapping streamlit app.

### Project Structure and Info ###

This project is structured in two parts, the analysis, and the app. the analysis code can be found in 
the 'artefact_creation' folder and the app in the 'app' folder. The analysis code produces the data 
for the app. The analysis has already been run so there is no need to run the analysis code unless there
is an update. The app is deployed in a docker container with the dockerfile in the app directory.

### How do I get set up? ###


This code was developed with a conda environment using python 3.9 on the Windows 11 operating system 
and may differ on other operating system/python distributions.
Information on installing anaconda can be found at the [anaconda website](https://www.anaconda.com/).

**Note - this set up assumes a non-powershell terminal window**

Once downloaded a conda environment can be set up by running the following in a conda command prompt
(in the following example called 'fever-env'): `conda create -n mapping-env python=3.9`.
If asked to proceed press `y` and then enter.

Then the environment can be activated (instructions should be displayed in the terminal), 
and the dependencies can be installed by navigating to the directory containing this
readme file and running the following command: `conda install --file requirements.txt`.
If asked to proceed press `y` and then enter.

A jupyer kernel can be created with the following command: `python -m ipykernel install --user --name mapping-env` 
if desired.

Once the above has been performed the virtual environment/jupyter kernel can be set in whichever
IDE is preferred by the user.

