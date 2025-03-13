# MATSCI 176 Final Project: Exploratory Analysis of Using Synthetically Generated Differential Reflectance Data to train a Convolutional Neural Network to Predict Oxidation in WS2
Authors: Emma White and Yemi Uthman

## Project Overview

<img width="887" alt="image" src="https://github.com/user-attachments/assets/0a0a33f6-226c-4307-8d6a-4cdce761f224" />

In this project, we demonstrate the efficacy of using synthetically generated datasets as a powerful method for data augmentation when training multi-task convolutional neural networks (CNNs) to analyze Hyperspectral Optical Microscopy (HOM) data cubes. Specifically, our model classifies HOM data cubes into substrate versus material categories and predicts the oxidation time of samples using regression.

Two-dimensional (2D) semiconductors have shown substantial promise for next-generation very large-scale integration (VLSI), low-power computing, optoelectronics, and high-efficiency photovoltaics. Their atomically thin structure, exceptional carrier mobility, and high on/off current ratios offer considerable advantages. However, scaling synthesis processes and managing contamination remain challenging due to defects frequently observed in Chemical Vapor Deposition (CVD)-grown samples. Currently, device measurements serve as the gold standard for evaluating material quality, providing essential insights into electrical properties. Despite their usefulness, these measurements are costly, time-intensive, and disruptive to rapid exploratory research.

Accelerating the evaluation of electronic properties in 2D materials without environmentally harmful and wasteful device fabrication would significantly enhance semiconductor research efficiency at universities. Hyperspectral Optical Microscopy (HOM) addresses this need by enabling rapid, broadband differential reflectance spectroscopy (DRS) measurements with sub-micron spatial resolution. HOM produces three-dimensional datasets composed of 2D DRS images spanning wavelengths from deep ultraviolet to near-infrared (DUV-NIR). These spectral datasets highlight critical excitonic features linked directly to band structure, defect density, and optoelectronic properties, enabling swift, millimeter-scale assessment of material quality. Furthermore, HOM datasets are particularly suitable for analysis using machine learning (ML) methods.

To assess material quality by measuring defect concentration, we exposed a WS₂ sample to a UV-O₃ (UV-Ozone) surface oxidation process for a total of 15 minutes, taking measurements every minute. Differential reflectance spectra (DRS) were recorded within a 550-700 nm wavelength range, generating 15 TIFF image stacks corresponding to each oxidation time step. Two overlapping flakes located in the top-left region of each image were selected, forming a dataset of differential reflectance spectra to guide synthetic data generation. Given that oxidation occurs uniformly across the WS₂ sample, each oxidation interval yields one averaged spectrum. To substantially augment our dataset from the original 15 spectra to 50,000 spectra, we generated synthetic data using Lorentzian models tailored to mimic real spectral data. By introducing controlled artificial noise and slight variations in peak parameters, we rapidly expanded the dataset size, enhancing our model’s training capacity.

This study contains the main files for performing non-linear least squares curve fitting to HOM data, training the multi-task model, testing the model, as well as two ablation studies that explore the effects of using a much simpler logistic regression model on the classification performance, and another that looks at the effects of removing pre-processing steps of the DRS data on model performance.

Ultimately, this study validates synthetic data augmentation as an effective approach for training ML models on HOM data, paving the way for broader applications. This method holds potential for extracting additional physical properties, including electronic mobility, dopant density, and alloy composition, further expanding the scope and efficiency of HOM analysis.

## Guide on Using This Approach

## Getting Started 
All of the necessary files for training and testing the model are included within this repository, and the tiff stacks containing the HOM data can be found from this link: https://drive.google.com/file/d/1Nl9k6N7_o8a3MekAxHAgIa5Y8L2XjtYQ/view?usp=drive_link

Firstly, you will want to clone this repository into your own workspace in VSCode, and place the unzipped HOM tiff stacks folder directly in the work space as well. You will also want to create a virtual environment to download the necessary libraries listed in the "requirements.txt" file. A step by step guide to creating a virtual environment can be found here: https://code.visualstudio.com/docs/python/environments

## Step 1: Lorentzian Curve Fitting and Extracting Peak Fit Parameters

The first part of this project aims to perform a non-linear least squares lorentzian fit to the existing and noisy HOM spectra. To do this, you'll want to open the notebook titled "Parallel_Curve_Fitting.ipynb." This notebook (along with most of these) has been organized to be able to run without any editing to test on our model UV-Ozone dataset.

Stepping through each code block, you will be able to perform the curve fit routine on every pixel in each of the 15 300x300 pixel tiff images. Running the code block in cell 5 allows you to visualize what the original noisy data looks like, as well as the background subtracted data and the curve fitting result for one pixel in a test image.

![image](https://github.com/user-attachments/assets/ae901e5a-d268-4e69-8956-8f1af4860a2b)

Running cell 7 allows you to perform a parallelized version of the curve fitting routine. Performing the peak fitting routine in parallel cuts the total analysis time down from over 10 hours to just around 30 minutes. This section will save the peak fit parameters (peak width, height, and amplitude) into .csv files within a newly created folder titled "Peak Parameter CSV Files." 

Proceeding forward, the next few cells can all be run as they are, with their main purpose being loading in these .csv files into a readable format for doing further analysis. Saving them in this manner also allows you to come back to the data without neeeding to redo the curve fitting routine.

Next, we take a region of 20x20 pixels located at the center of the triangle 










