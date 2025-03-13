# MATSCI 176 Final Project: Exploratory Analysis of Using Synthetically Generated Differential Reflectance Data to Train a Convolutional Neural Network to Predict Oxidation in WS2
Authors: Emma White and Yemi Uthman

## Project Overview

<img width="887" alt="image" src="https://github.com/user-attachments/assets/0a0a33f6-226c-4307-8d6a-4cdce761f224" />

In this project, we demonstrate the efficacy of using synthetically generated datasets as a powerful method for data augmentation when training multi-task convolutional neural networks (CNNs) to analyze Hyperspectral Optical Microscopy (HOM) data cubes. Specifically, our model classifies HOM data cubes into substrate versus material categories and predicts the oxidation time of samples using regression.

Two-dimensional (2D) semiconductors have shown substantial promise for next-generation very large-scale integration (VLSI), low-power computing, optoelectronics, and high-efficiency photovoltaics. Their atomically thin structure, exceptional carrier mobility, and high on/off current ratios offer considerable advantages. However, scaling synthesis processes and managing contamination remain challenging due to defects frequently observed in Chemical Vapor Deposition (CVD)-grown samples. Currently, device measurements serve as the gold standard for evaluating material quality, providing essential insights into electrical properties. Despite their usefulness, these measurements are costly, time-intensive, and disruptive to rapid exploratory research.

Accelerating the evaluation of electronic properties in 2D materials without environmentally harmful and wasteful device fabrication would significantly enhance semiconductor research efficiency at universities. Hyperspectral Optical Microscopy (HOM) addresses this need by enabling rapid, broadband differential reflectance spectroscopy (DRS) measurements with sub-micron spatial resolution. HOM produces three-dimensional datasets composed of 2D DRS images spanning wavelengths from deep ultraviolet to near-infrared (DUV-NIR). These spectral datasets highlight critical excitonic features linked directly to band structure, defect density, and optoelectronic properties, enabling swift, millimeter-scale assessment of material quality. Furthermore, HOM datasets are particularly suitable for analysis using machine learning (ML) methods.

To assess material quality by measuring defect concentration, we exposed a WS₂ sample to a UV-O₃ (UV-Ozone) surface oxidation process for a total of 15 minutes, taking measurements every minute. Differential reflectance spectra (DRS) were recorded within a 550-700 nm wavelength range, generating 15 TIFF image stacks corresponding to each oxidation time step. Two overlapping flakes located in the top-left region of each image were selected, forming a dataset of differential reflectance spectra to guide synthetic data generation. Given that oxidation occurs uniformly across the WS₂ sample, each oxidation interval yields one averaged spectrum. To substantially augment our dataset from the original 15 spectra to 50,000 spectra, we generated synthetic data using Lorentzian models tailored to mimic real spectral data. By introducing controlled artificial noise and slight variations in peak parameters, we rapidly expanded the dataset size, enhancing our model’s training capacity.

This study contains the main files for performing non-linear least squares curve fitting to HOM data, training the multi-task model, testing the model, as well as two ablation studies that explore the effects of using a much simpler logistic regression model on the classification performance, and another that looks at the effects of removing pre-processing steps of the DRS data on model performance.

Ultimately, this study validates synthetic data augmentation as an effective approach for training ML models on HOM data, paving the way for broader applications. This method holds potential for extracting additional physical properties, including electronic mobility, dopant density, and alloy composition, further expanding the scope and efficiency of HOM analysis.

## Getting Started 
All of the necessary files for training and testing the model are included within this repository, and the tiff stacks containing the HOM data can be found from this link: https://drive.google.com/file/d/1Nl9k6N7_o8a3MekAxHAgIa5Y8L2XjtYQ/view?usp=drive_link

Firstly, you will want to clone this repository into your own workspace in VSCode, and place the unzipped HOM tiff stacks folder directly in the work space as well. You will also want to create a virtual environment to download the necessary libraries listed in the "requirements.txt" file. A step by step guide to creating a virtual environment can be found here: https://code.visualstudio.com/docs/python/environments

## Step 1: Lorentzian Curve Fitting and Extracting Peak Fit Parameters

The first part of this project aims to perform a non-linear least squares lorentzian fit to the existing and noisy HOM spectra. To do this, you'll want to open the notebook titled "Parallel_Curve_Fitting.ipynb." This notebook (along with most of these) has been organized to be able to run without any editing to test on our model UV-Ozone dataset.

Stepping through each code block, you will be able to perform the curve fit routine on every pixel in each of the 15 300x300 pixel tiff images. Running the code block in cell 5 allows you to visualize what the original noisy data looks like, as well as the background subtracted data and the curve fitting result for one pixel in a test image.

![image](https://github.com/user-attachments/assets/ae901e5a-d268-4e69-8956-8f1af4860a2b)

Running cell 7 allows you to perform a parallelized version of the curve fitting routine. Performing the peak fitting routine in parallel cuts the total analysis time down from over 10 hours to just around 30 minutes. This section will save the peak fit parameters (peak width, height, and amplitude) into .csv files within a newly created folder titled "Peak Parameter CSV Files." 

Proceeding forward, the next few cells can all be run as they are, with their main purpose being loading in these .csv files into a readable format for doing further analysis. Saving them in this manner also allows you to come back to the data without neeeding to redo the curve fitting routine.

Next, we take a region of 20x20 pixels located at the center of the triangle that we will use to extract the average peak parameters for this sample. We don't want to use every pixel in the sample, as even though we believe it should oxidize uniformly across the sample, we still observe some edge effects between the material and substrate junction which may alter these parameters. The arrays of the average parameters are also saved into a .csv file in a newly generated folder titled "Average Parameters" to be used for later.

The last step is to then fit an allometric function to the average parameter evolution across these 15 minutes of oxidation, which again, can be run without making any alterations to the code. This will generate a plot of the peak evolution over time, as well as a function relating the parameter value to some oxidation time.

![image](https://github.com/user-attachments/assets/ff0d2637-460e-4c21-a23a-25f2435077cb)

We now have three functions the express how we expect our DRS curve to evolve with increased oxidation time, which we then use as the inputs to create our synthetic data.

## Step 2: Generating Synthetic Data and Training the Multi-Task Convolutional Neural Network

The next notebook to use will be titled "Train Oxidation and Regression Model.ipynb." In this notebook, you will need to edit lines 23 - 25 that contain the functions describing the peak parameter evolution to match the functions generated from the first notebook.

    amplitude = 0.435 - 0.15 * time**(0.26)
    width = 9.452 + 3.89 * time**0.26
    position = 621.06 - 16.16 * time**0.08

Where all the constants shown above will match the values for a, b, and c from the allometric fitting. You can also play around with the parameters that generate a random value of noise for both the substrate and material spectra, as well as the uncertainty value in the peak fit parameters (currently set to +/- 5%).

    amplitude *= np.random.uniform(0.95, 1.05)
    width *= np.random.uniform(0.95, 1.05)
    position *= np.random.uniform(0.995, 1.005)

    spectrum = lorentzian(wavelengths, amplitude, width, position)

    # Add noise
    noise = np.random.normal(0, 0.005, size=spectrum.shape)
    spectrum += noise

The remaining parts of this cell are used for training the multi-task model, and includes the current architecture of the model (which contains shared feature extraction layers with seperate regression and classification heads) shown below. We also used a masked MAE loss function for measuring the regression results during training. This function is designed to only calculate the error on material spectra, as we dont expect to see any change in the substrate spectra with increased oxidation.

<img width="878" alt="image" src="https://github.com/user-attachments/assets/4f6a37cd-de71-4364-ac09-c38a95eb1d62" />

At the end of this cell, the model will report some characteristic synthetic spectra as a sanity check, as well as save the tensorflow model into a new folder titled "Oxidation_Model."

![image](https://github.com/user-attachments/assets/4bbd9d95-d86b-4bfc-b0ec-c468f5cea9a8)

## Step 3: Evaluating the Accuracy of the Model

In this section, we will be using the notebook titled "Spectrum Processing and Model Testing." Cell 4 contains the bulk of the pre-processing functions. These include the hyperspy function (the library used to load in the 3D HOM dataset), three functions to remove a numerical issue point that arises from a miscalculation with the software used to generate the HOM data, a background subtraction function remove reflection effects from the substrate, a Savitzky-Golay filter function to help smooth the noisy HOM curve, and lastly a function to perform all these pre-processing steps at once, utilizing the helper functions define above. Here, we also import a function titled "getGT" from the "Ground_Truth_Creator.py" script, which essentially just transforms the original HOM image into binary and uses a threshold to set a ground truth for classification (substrate (0) and material (1)). This allows us to evaluate our accuracy results later in this notebook.

We can run cells 4 and 5 to load in and pre-process the spectra and compare what an example spectrum looks like before and after pre-processing.

![image](https://github.com/user-attachments/assets/3dc1467e-d063-4c10-a0b9-4b71c4f39a61)

The next part of this notebook is where we begin to evaluate the model performance on our real, experimental data. The "visualizeModel" function loads in the saved tensorflow model from the 'Oxidation_Model" folder and applies it to each of the 16 files in our dataset (from time = 0 to time = 15 minutes). Here is a visualization of the model's prediction for the t=0 sample.

![image](https://github.com/user-attachments/assets/7c002a3c-9855-4e30-8f30-0a0a50c0eb73)

The rest of the cells in the notebook can also be run as they are written, and will output the accuracy scores and regression results for all 16 time steps. The oxidation prediction time was reported to be the median value of the predicted values, as we see the edge effects at the substrate/material junction begin to play a role. In this work, we are more interested in what the median pixel prediction is telling us, rather than analyzing the edge effects, although that may be valuable in future work.

Finally, we plot the predicted oxidation values against the ground truth values and report the R<sup>2</sup>, RMSE, and MAE values for the fit as shown below.

<img width="501" alt="image" src="https://github.com/user-attachments/assets/60ea1c2d-cfbe-4690-afbc-ba6a449f1f0a" />

## Step 4: Ablation Study 1 - Evaluating Performance of Simple Logistic Regression Model

In order to show the importance of using a convolutional neural network for feature extraction, we employed a simple logisitc regression model in the notebook titled "Ablation_Study_1_Regression_Model.ipynb" that looks to find a trend in the peak paramaters (width, height, and position) to classify the spectra into material vs substrate. In order to do this, divide the total dataset of 16 tiff images each containing 90,000 spectra (for a 300x300 pixel image) into a training and testing set. Here, we use Sci-Kit Learn's implementation of a logistic regression model.

We first begin by importing time series oxidation data from the "Time Series Oxidation Files" folder in our workspace, and perform the ground truth results again using the 'getGT' function. We also loop through the .csv files that contain the curve fit parameters, located in the folder titled "Peak Parameter CSV Files" also within our workspace. This cell will combine the individual arrays containing the parameters into one larger array, which will then be converted into a Pandas dataframe in the next cell. Lastly, we perform the logistic regression study, first using a standard scaler to scale the original parameters down, as well as utilizing a class weighting to account for the imbalance in the number of substrate vs material pixels. We also used a train/test split of 80/20 and reported the logistic regression accuracy and confusion matrix as shown below.

![image](https://github.com/user-attachments/assets/4098e7bd-d252-4975-bfe7-81a17b0b5c8c)

## Step 5: Ablation Study 2 - Removing Pre-processing Steps

The novelty of this approach is using synthetically generated data to train a model, and to prove the importance of creating synthetic data that closely mimics our experimental data, we studied how our pre-processing methods affect the accuracy of the model. In the notebook titled "Ablation_Study_2_Removing_Preprocessing.ipynb," we remove the pre-processing functions described in step three (except for the numerical issue correction) to evaluate how our model performs on a dataset with noise that is much greater than we simulated in the synthetic dataset. 

This notebook can again be run exactly as is, as the methods directly mimic those in the notebook titled "Spectrum Processing and Model Testing.ipynb." This notebook will output an unprocessed example spectrum, the prediction for time=0, and the final regression results all shown below.

![image](https://github.com/user-attachments/assets/e6b4f3c3-1ad0-4955-b6f5-82ac10d5a5d7) 
![image](https://github.com/user-attachments/assets/34ef260a-8588-40f0-8b0f-b16d1a733414) 
<img width="530" alt="image" src="https://github.com/user-attachments/assets/886e2eb4-fe36-42ba-bfaa-c813c5ab81ff" />

## Conclusins and Important Notes
This readme file is designed to walk you through, step-by-step, how to use this analysis platform, as well as describe our thought process behind the methods done in this study. It is important to organize your files exactly how this repository is organized, which is why we highly suggest cloning this repository into your VSCode workspace to perform the analysis. 

Remember, you must download the zipped data file from the shared document on google drive available above and again here at this link: https://drive.google.com/file/d/1Nl9k6N7_o8a3MekAxHAgIa5Y8L2XjtYQ/view?usp=drive_link

For even more instructions or confusion, each notebook has a header that describes in detail the purpose of the code contained within the notebook, as well as definitions for the inputs and outputs of each function with the description of what the intended purpose of the function is. 

This code is also highly amenable to minor adjustments in the train/test size, the parameters to create the synthetic dataset, and methods to optimize the curve fitting routine. We do stress again, however, that it is HIGHLY RECOMMENDED to keep the file organization as has been shown here to limit any unforseen errors.

Lastly, please view the references.txt file for any further information on the methods done in this study, and for more of a background on the experimental techniques used to gather the data.





















