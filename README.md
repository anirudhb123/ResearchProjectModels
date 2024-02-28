# Anomaly Detection in Proton-Proton Collision Data from the LHC

This repository contains implementations of supervised and unsupervised anomaly detection models tailored for analyzing proton-proton collision data collected from the Large Hadron Collider (LHC). These models are essential for identifying rare and potentially significant events that may indicate phenomena beyond the Standard Model of particle physics.

## Introduction
Proton-proton collision data obtained from experiments at the LHC are vast and complex. Among this data, anomalies or rare events are of particular interest as they might unveil new physics phenomena. Anomaly detection techniques are crucial for sifting through this data to uncover events that deviate from the expected behavior.

## Models
The repository includes both supervised and unsupervised anomaly detection models:

### Supervised Models:

### Unsupervised Models:
1. **Autoencoders**: Deep learning models trained to reconstruct normal data, identifying anomalies as deviations from the reconstructed data.
2. **Variational Autoencoders (VAEs)**: A variant of autoencoders that learn the underlying probability distribution of the data, enabling the detection of outliers based on the reconstruction error and learned distribution.

## Usage
Each model is implemented in a separate Python script within the `models` directory. Follow the instructions below to utilize these models:

### Prerequisites
- Python 3.x
- NumPy
- pandas
- scikit-learn
- TensorFlow 

### Instructions
1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/your-username/ResearchProjectModels.git
    ```

2. Navigate to the repository directory:

    ```bash
    cd ResearchProjectModels
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Prepare your proton-proton collision data in a suitable format. 

5. Choose the model you want to use and execute its corresponding Python script. For example, to run a variational autoencoder model that uses mse as the loss metric, run

    ```bash
    python ResearchProjectModels/'Variational Autoencoder Models'/vae_mse.py
    ```
