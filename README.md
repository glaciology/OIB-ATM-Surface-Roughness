# 🧊 Kriging + Along-track Analyses of Operation IceBridge's ILATM2

## Overview

- This repository uses NASA's Airborne Topographic Mapper (ATM) surface roughness product (ILATM2) to create an ice sheet-wide gridded roughness product, characterize its spatiotemporal patterns, and identify the major climatic controls affecting the observed patterns.
- See Repository Contents for more information on acquisition, processing, and analyses. 
- See Background for more info on surface roughness and the OIB ATM. 

## Repository Contents

* [**/ILATM2AcquisitionAndPreprocessing**](./ILATM2AcquisitionAndPreprocessing) -  Script for downloading and pre-processing the ILATM2 product.  

* [**/VariographyAndKriging**](./VariographyAndKriging) - Script for interpolating and mapping gridded roughness for each year. 

* [**/alongTrackAnalyses**](./alongTrackAnalyses) - Script for analyzing the spatial and temporal trends in the along-track data. 

* [**/FilterILATM2byAWS**](./FilterILATM2byAWS) - Script to prepare ILATM2 for random forest regression, filtering data to within 5 km of on-ice automated weather stations. 

* [**/RandomForestRegressionForILATM2AndAWS**](./RandomForestRegressionForILATM2AndAWS) - Script for random forest regression, determining the primary climatic controls on surface roughness. 

## Background

Surface roughness is a key factor in the turbulent heat flux of the Greenland Ice Sheet (GrIS), significantly impacting the energy input to the upper snow surface and, consequently, the ice sheet mass balance. Despite its importance, predictive models often oversimplify surface roughness, which is essential for an improved understanding of GrIS sensitivity to climate change. Surface roughness is oversimplified largely due to the scale-dependency of roughness and the lack of standardized roughness parameters (for additional reading on these topics, I recommend 'Roughness in the Earth Sciences' by Mark W. Smith). 

The ILATM2 product is a single-scale roughness estimate (via root-mean-square) of 30 x 80 meter segments of  along-track elevation shots (ILATM2 documentation can be found here: https://nsidc.org/data/ilatm2/versions/2). Using the springtime (March, April, May) campaigns from 2009 to 2019, the included scripts produce and map yearly gridded products for meter-scale roughness, supplemented with along-track analyses. With daily climate variables from PROMICE and GC-NET (https://promice.org) automated weather stations (AWS), the scripts also use machine learning (random forest regressions) to examine potential climatic drivers of observed roughness patterns.  