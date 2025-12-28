#  Inverse Model for Frustrated Composites

##  Overview
This project builds an inverse model for **frustrated composites** using deep learning. It consists of three main steps: **data preparation**, **model training**, and **prediction**. 
it supports two seperate options to get the inverse solution: 
1. train a direct inverse model and predict the fiber orientation of a surface.
2. train a forward model and then optimize to find the fiber orientation of a surface.


## What’s inside


1. "Loop-for-Dataset.gh": Grasshopper dataset generation
   - Runs simulation in loop(calibrated) and exports to XLS files
   - This can be used for a single simulation by inserting the genepool as an input to fiber orientation and disabling the loop components.
2. "prep_dataset.py": dataset processing
   - Transforms/cleans XLS exports into training-ready datasets. It uses the modules in the modules folder. 
3. Python training code (two models)
   - Direct prediction model/"inverse_model.py": target geometry → predicted fiber layout (inverse mapping)
   - Surrogate forward model/"forward_model.py": design parameters → predicted shape/metrics (forward mapping for optimization)
4. Grasshopper inference + optimization
   - "Inverse_Prediction.gh": Run direct prediction in GH
   - "Forward_Surrogate.gh": Optimization loop using the surrogate forward model (or other objective-driven search)
   - "Loop-for-Testing-Predictions.gh": has all options and is capable of running in loop to test multiple options and document the results. 
5. Plugin
    - "Plugin-components" folder contains elements needed for the plugin. This includes a seperate "Simulation.gh" that can be used as-is
    - "rh8" contains the plugin itself
    - "Plugin-example.gh" is an example for using the plugin
6. "Dataset_organizer.xlsx" is a table of all the excel files generated in this research by the "Loop-for-Dataset.gh" code and used for the training of the models.
7. utils folder contains different utility codes used in the research that are not directly needed for the inverse predictions.



Training runs and experiments were logged using Weights & Biases.
Inverse model runs: [link](https://wandb.ai/kapon-gal-technion/inverse_model_regression?nw=nwuserkapongal)
Forward surrogate model runs: [link](https://wandb.ai/kapon-gal-technion/forward_model?nw=nwuserkapongal)