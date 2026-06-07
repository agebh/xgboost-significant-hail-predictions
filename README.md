# xgboost-significant-hail-predictions
This repository contains the development of the XGBoost classification model for predicting significant hail probabilities.

### `model_dev_xgbGlobal_severeHail`
Global XGBoost training workflow for severe hail prediction using a hail-size threshold of **2.5 cm**.  
It combines US NOAA/SPC and European ESSL observations with ERA5 predictors, applies land-sea masking, builds cyclic two-year cross-validation folds, performs negative downsampling, trains XGBoost models, evaluates performance metrics, and saves models, predictions, statistics, plots, and SHAP-based diagnostics.

### `model_dev_xgbGlobal_sigHail`
Global XGBoost training workflow for significant hail prediction using a higher hail-size threshold of **4.4 cm**.  
It follows the same US–Europe combined modelling structure as the severe-hail workflow, but uses different class-balancing settings and regional contribution weights to better represent rarer significant-hail events.

### `model_dev_xgbGlobal_sigHail_tuning`
Global XGBoost hyperparameter-tuning workflow for significant hail prediction over the US and Europe.  
The script uses NOAA/SPC and ESSL hail observations, ERA5 hail predictors, land-sea masks, and cyclic two-year cross-validation folds. It applies a **4.4 cm hail-size threshold**, performs region-specific negative downsampling, tunes XGBoost hyperparameters with Optuna, compares tuned and baseline models, saves fold-specific models and predictions, and exports validation/test metrics, PR/ROC curves, AUCPR learning curves, and Optuna trial summaries.

### `model_dev_xgbUS_sigHail`
US-only XGBoost training workflow for significant hail prediction using NOAA/SPC observations and ERA5 CONUS predictors.  
This notebook focuses only on the US domain, applies a 4.4 cm hail-size threshold, prepares binary hail targets, performs cyclic two-year cross-validation, downsamples negative samples, trains XGBoost models, and produces prediction, evaluation, and interpretation outputs for the US region.

### `XGBoost_analysis_visualization.ipynb`
Analysis and visualization notebook for evaluating the XGBoost hail prediction models.  
It compares observations with `xgbGlobal` and `xgbUS` predictions using seasonal cycles, spatial diagnostics, annual time series, city-level 3×3 grid-box averages, cross-validated fold summaries, ERA5 predictor analyses, outlier diagnostics, and model-performance visualizations such as PR/ROC curves and SHAP-based interpretation plots. 

### `predictions_xgbGlobal_1959-2024.ipynb`
Prediction and visualization notebook for the final `xgbGlobal_sigHail` model over **1959–2024**.  
It applies the trained global XGBoost model year by year to ERA5 predictor fields, saves annual prediction outputs for the global grid, CONUS, and Europe, and creates mean annual frequency maps, global/regional inset maps, and annual time series for global, regional, and selected city-level hail frequency.
