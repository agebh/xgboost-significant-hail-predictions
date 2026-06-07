# xgboost-significant-hail-predictions
This repository contains the development of the XGBoost classification model for predicting significant hail probabilities.

### `model_dev_xgbGlobal_severeHail`
Global XGBoost training workflow for severe hail prediction using a hail-size threshold of **2.5 cm**.  
It combines US NOAA/SPC and European ESSL observations with ERA5 predictors, applies land-sea masking, builds cyclic two-year cross-validation folds, performs negative downsampling, trains XGBoost models, evaluates performance metrics, and saves models, predictions, statistics, plots, and SHAP-based diagnostics.

### `model_dev_xgbGlobal_sigHail`
Global XGBoost training workflow for significant hail prediction using a higher hail-size threshold of **4.4 cm**.  
It follows the same US–Europe combined modelling structure as the severe-hail workflow, but uses different class-balancing settings and regional contribution weights to better represent rarer significant-hail events.

### `model_dev_xgbUS_sigHail`
US-only XGBoost training workflow for significant hail prediction using NOAA/SPC observations and ERA5 CONUS predictors.  
This notebook focuses only on the US domain, applies a 4.4 cm hail-size threshold, prepares binary hail targets, performs cyclic two-year cross-validation, downsamples negative samples, trains XGBoost models, and produces prediction, evaluation, and interpretation outputs for the US region.
