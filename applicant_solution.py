import json
#import gdown

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers

# Download the dataset
#url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
#downloaded_file = "challenge.mat"
#gdown.download(url, "challenge.mat", quiet=True, fuzzy=True)

import os

# Find dataset file
if os.path.exists("challenge.mat"):
    dataset_path = "challenge.mat"

else:
    raise FileNotFoundError(
        "Please place challenge.mat in the repository root.")

# Load dataset
data = loadmat(dataset_path, simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, ntx = tx.shape
_, nrx = rx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def your_canceller(tx_n, rx):

    #### Step 1. Baseline Cancellation ####

    # Use baseline function written for us
    tx_linear = baseline(tx_n, rx, helpers["fit_tx_prediction"])
    tx_nonlinear = rx - tx_linear

    #### Step 2. Non-linear Feature Engineering ####

    # Parameters for nonlinear delayed features
    delays_number = 8
    delays_list = list(range(delays_number))

    # Parameters for nonlinear polynomial features
    orders = [1, 3, 5]
    block = 200000

    def make_features(start, end):
        """
        This function creates nonlinear feature matrix for regression
          * Create delayed versions
          * Create nonlinear polynomial features
        Combine everything into one big feature matrix X. 
        """
        rows_number = end - start
        # All feature matrix
        feature_matrices = []

        # Step 2.1 For nonlinear delayed features
        for delay in delays_list:
          # Creates an empty matrix of shape (rows, number_of_tx_channels)   
            A = np.zeros((rows_number, ntx), dtype=np.complex128)

            # Compute shifted TX indices
            start1 = max(0, start - delay)
            end1 = max(0, end - delay)

            # Handle boundary conditions
            valid_start = max(0, delay - start)
            valid_len = end1 - start1

            # Fill matrix A with delayed TX samples
            if valid_len > 0:
                A[valid_start : valid_start + valid_len] = tx_n[start1:end1]

            # Step 2.2 For nonlinear polynomial features
            for degree in orders:
                feature_matrices.append(A * (np.abs(A) ** (degree - 1)))

        # combines all features into one big matrix X
        X = np.concatenate(feature_matrices, axis=1)
        return X

    #### Step 3. Ridge regression #### 

    # Create a feature matrix and determine
    # how many total nonlinear features exist
    X0 = make_features(0, min(10, N))
    numb_feat = X0.shape[1]

    # Create matrix A = traspose(X)X
    A = np.zeros((numb_feat, numb_feat), dtype=np.complex128)

    # Create matrix B = traspose(X)Y
    B = np.zeros((numb_feat, nrx), dtype=np.complex128)

    # Loop through the dataset in chunks
    # to avoid huge memory usage and populate A and B
    for start in range(0, N, block):
        end = min(N, start + block)
        X = make_features(start, end)
        Y = tx_nonlinear[start:end]
        A += X.conj().T @ X
        B += X.conj().T @ Y

    # Set regularization value
    ridge = 1e-2

    # Solve for W 
    W = np.linalg.solve(A + ridge * np.eye(numb_feat),B)

    #### Step 4. Interference prediction and subtraction #### 

    # Empty matrix for predicted interference
    # Same shape as rx
    predicted = np.zeros_like(rx)

    # Loop through blocks again
    # to generate predictions
    for start in range(0, N, block):

        # End index of current block
        end = min(N, start + block)

        # Rebuild nonlinear features
        X = make_features(start, end)

        # Predict interference: pred = XW
        predicted[start:end] = X @ W

    # Found from grid search 
    # Controls how strongly predicted nonlinear interference is removed
    alpha = 0.58

    # Remove predicted nonlinear interference
    residual = tx_linear - alpha * predicted

    #### Step 4. Singular Value Decomposition ####

    # Found from grid search 
    # Controls how strongly the dominant
    # SVD component is removed
    svd_alpha = 0.9

    # For SVD, find three matrices U, S, Vh
    U, S, V = np.linalg.svd(residual, full_matrices=False)

    # Construct strongest rank-1 component
    rank1 = S[0] * np.outer(U[:, 0], V[0, :])

    # Remove rank-1 component
    rx_hat = residual - svd_alpha * rank1

    # Return final cleaned signal
    return rx_hat


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
