# Assessing convergent EEG and fMRI biomarkers following accelerated intermittent theta-burst stimulation (aiTBS) protocol in cases of treatment-resistant depression

This repository contains the code for the Master's thesis project at [Freier Universität Berlin](https://www.fu-berlin.de/) by Elvira Shemena, conducted at the [Clinical Neurotechnology Lab](http://www.clinical-neurotechnology.com/) at [Charité – Universitätsmedizin Berlin](https://www.charite.de/).

The work was supervised by Surjo Soekadar and Miro Grundei.

## Content of the repository
- [README.md](./README.md) This readme
- [questionnaire/](./questionnaire) Clinical outcomes (HAM-D17, BDI-II)
  - `01_convert_questionnaires.py` Converts raw scores to percent change and responder status
  - `02_questionnaire_stats.py` Paired pre/post statistics with effect sizes
  - `03_figures.py` Clinical response figures
- [mri/](./mri) Coil targeting and resting-state fMRI connectivity of the default mode network (DMN)
  - `00_capsule_projection_3d_slicer.py` Projects the TMS target onto the pial surface (3D Slicer)
  - `01_ras_to_mni_yfield_batch.py` Transforms native RAS coordinates to MNI space
  - `02_roi_maker.py` Builds the spherical DMN and individual dlPFC ROI masks
  - `03_conn_connectivity_extractor.py` Extracts ROI-to-ROI connectivity from CONN
  - `04_correlation.py` Planned DMN composite test against symptom change
  - `05_exploratory.py` Per-pair pre/post, stratified and correlation analyses
  - `06_summary.py` Collects significant and trend-level results
  - `fmri_stats.py`, `fmri_plots.py`, `stats_helpers.py` Shared statistics and plotting modules
- [eeg/](./eeg) Resting-state EEG preprocessing and biomarkers (eyes open and eyes closed)
  - `00_preprocessing_pipeline.py` Filtering, ICA, interpolation and EO/EC split
  - `01_spectral_power.py` Band power
  - `02_asymmetry.py` Frontal alpha asymmetry
  - `03_connectivity.py` Band-wise wPLI connectivity
  - `04_statistics_pre_post.py` Pre/post and stratified tests
  - `05_correlations.py` Correlations of feature change with symptom change
  - `06_plots.py` Figures from the cached results
  - `07_summary.py` Collects significant and trend-level results
  - `eeg_compute.py` Shared engine for epoching, features and statistics
  - `utils.py`, `helpers.py` Preprocessing helpers
- [classifier/](./classifier) Prediction of treatment response from baseline EEG and fMRI features
  - `00_nested_cv.py` Nested cross-validation for EEG, fMRI and early fusion

Scripts are numbered in execution order within each folder. Paths and parameters are set at the top of each script.

## Requirements
Python 3.13.7 with numpy, pandas, scipy, statsmodels, matplotlib, scikit-learn, joblib, tqdm, mne, mne-connectivity, mne-icalabel, pyxdf, pyyaml, nibabel and nilearn. fMRI preprocessing and first-level connectivity were done in the CONN toolbox (MATLAB/SPM), and coil targeting in 3D Slicer.

## Abstract
### Background
Accelerated intermittent theta burst stimulation (aiTBS) has shown rapid response rate for treatment-resistant depression (TRD). However the neurophysiological changes associated with clinical improvement remain poorly understood, limiting the development of biomarkers for monitoring and predicting treatment response. 

### Methods
In this naturalistic, open-label study, we analyzed twenty patients with TRD who received 67 aiTBS sessions targeting the left prefrontal dorsolateral cortex over one month. Resting-state EEG and fMRI were recorded separately and sequentially before and after treatment. 

### Results
Depressive symptoms improved significantly, with response rates of 35% (HAM-D17) and 25% (BDI-II). Although no group-level EEG or fMRI changes survived correction, changes in theta, beta and gamma power were associated with clinical improvement. Exploratory analysis on stratified groups further revealed that sex and psychiatric comorbidity may influence the direction of alpha- and theta-band effects. Among pre-treatment measures, higher baseline theta connectivity was associated with less HAM-D17 improvement. 

### Discussion
Together, these findings demonstrate significant clinical improvement following intensive aiTBS and identify preliminary EEG markers associated with individual differences in treatment response. However, the absence of robust group-level neurophysiological changes in the small, heterogeneous sample limit strong conclusions regarding their biomarker potential. Larger controlled studies are therefore needed to determine whether these electrophysiological associations generalize and can reliably support treatment monitoring or response prediction. Combining EEG and fMRI, ideally through simultaneous acquisition, may further help characterize the complementary temporal and spatial dynamics underlying response to aiTBS.

**Keywords:** accelerated intermittent theta-burst stimulation, treatment-resistant depression, EEG, fMRI, biomarkers

## Data Availability
Due to patient privacy regulations at Charité – Universitätsmedizin Berlin, the individual-level data and associated graphics cannot be shared through GitHub or any other open repository. Only group-level analysis results and the associated code are presented in this work.
