# MATSCI-176-Final-Project

## Project Overview

In this project, we demonstrate the efficacy of using synthetically generated datasets as a powerful method for data augmentation when training multi-task convolutional neural networks (CNNs) to analyze Hyperspectral Optical Microscopy (HOM) data cubes. Specifically, our model classifies HOM data cubes into substrate versus material categories and predicts the oxidation time of samples using regression.

Two-dimensional (2D) semiconductors have shown substantial promise for next-generation very large-scale integration (VLSI), low-power computing, optoelectronics, and high-efficiency photovoltaics. Their atomically thin structure, exceptional carrier mobility, and high on/off current ratios offer considerable advantages. However, scaling synthesis processes and managing contamination remain challenging due to defects frequently observed in Chemical Vapor Deposition (CVD)-grown samples. Currently, device measurements serve as the gold standard for evaluating material quality, providing essential insights into electrical properties. Despite their usefulness, these measurements are costly, time-intensive, and disruptive to rapid exploratory research.

Accelerating the evaluation of electronic properties in 2D materials without environmentally harmful and wasteful device fabrication would significantly enhance semiconductor research efficiency at universities. Hyperspectral Optical Microscopy (HOM) addresses this need by enabling rapid, broadband differential reflectance spectroscopy (DRS) measurements with sub-micron spatial resolution. HOM produces three-dimensional datasets composed of 2D DRS images spanning wavelengths from deep ultraviolet to near-infrared (DUV-NIR). These spectral datasets highlight critical excitonic features linked directly to band structure, defect density, and optoelectronic properties, enabling swift, millimeter-scale assessment of material quality. Furthermore, HOM datasets are particularly suitable for analysis using machine learning (ML) methods.

To assess material quality by measuring defect concentration, we exposed a WS₂ sample to a UV-O₃ (UV-Ozone) surface oxidation process for a total of 15 minutes, taking measurements every minute. Differential reflectance spectra (DRS) were recorded within a 550-700 nm wavelength range, generating 15 TIFF image stacks corresponding to each oxidation time step. Two overlapping flakes located in the top-left region of each image were selected, forming a dataset of differential reflectance spectra to guide synthetic data generation. Given that oxidation occurs uniformly across the WS₂ sample, each oxidation interval yields one averaged spectrum. To substantially augment our dataset from the original 15 spectra to 50,000 spectra, we generated synthetic data using Lorentzian models tailored to mimic real spectral data. By introducing controlled artificial noise and slight variations in peak parameters, we rapidly expanded the dataset size, enhancing our model’s training capacity.

Ultimately, this study validates synthetic data augmentation as an effective approach for training ML models on HOM data, paving the way for broader applications. This method holds potential for extracting additional physical properties, including electronic mobility, dopant density, and alloy composition, further expanding the scope and efficiency of HOM analysis.

## Guide on Using This Approach

All of the necessary files for training and testing the model are included within this repository, and the tiff stacks containing the HOM data can be found from this link: *** ADD LINK ***

## Getting Started 

Firstly, you will want to clone this repository into your own workspace, and place the HOM tiff stacks folder directly in the work space as well. In 
