## GB_PV_Capacity_Model

This repository contains code for estimating solar photovoltaic (PV) capacity in Great Britain (GB) regions.

## Using the Docker Image (Recommended)

To run the Solar PV Capacity Estimation code within a Docker container, follow these steps:

### 1. Clone the Repository

Clone the repository and navigate to its root directory:

```bash
git clone <repo-url>
cd <repo-directory>
```

2. Build and Launch the Docker Container
Run the following commands to build and start the containerized development environment:

```bash
docker compose build --build-arg "buildmode=dev"
docker compose up -d
```

3. Access the Services
Once the container is running, you can access the following services:

JupyterLab: https://127.0.0.1:5000   
Authentication Token: gbpvcapacitymodel   
MLFlow Tracking Server: https://127.0.0.1:5001   


## Data Processing Workflow
Run the following Jupyter notebooks to process the data:

`clc.ipynb`
`climate.ipynb`
`gb_pv_capacity.ipynb`
`gva.ipynb`
`ROC.ipynb`

Next, run `merging_data.ipynb` to create the dataset for training the model.

Finally, run `feature_engineering.ipynb` to normalize the dataset for model preparation.

## Data analysis
cd data_analysis   
Run `data_analysis_per.ipynb` to analyze the normalized data or `data_analysis_abs.ipynb` to analyze the absolute data.


## Model training and selection 
cd models   
Run `model_training_norm.ipynb` to train the model.    
Run `model_selection_norm` to evaluate the models.    





