# Report: SMILES-2026 Signal Interference Cancellation

## Abstract 
This solution implements a nonlinear interference cancellation 3-stage pipeline for eliminating structured interference in multi-channel received signals. By combining baseline cancellation, nonlinear polynomial feature regression and a Singular Value Decomposition (SVD) spatial cleanup stage, the method models both nonlinear hardware distortion and temporal memory effects [1] across all transmit channels jointly. The proposed approach achieved 9.6 dB interference suppression. 

## Reproducibility Instructions

### Environment
The solution was tested using:

- Python 3.8+
- numpy
- scipy

No additional external libraries are required.

### Repository Setup
Clone the repository and enter the project folder:
- git clone https://github.com/On-Point-RND/SMILES-2026-Signal.git
- cd SMILES-2026-Signal

### Install required packages
- python -m pip install --upgrade pip
- python -m pip install numpy scipy

### Dataset
Before running the solution, place the dataset file (challenge.mat (original name) or challenge_challenge.mat (sometimes changed during downloading) into the repository root directory. Dataset file should be in the same folder as applicant_solution.py

### Running the Solution
Run python3 applicant_solution.py

### Output 
After execution, the repository root will contain results.json, where results are stored. 


## Final Solution Description
The final solution is based on the idea that the received signal still have residual structured TX-driven interference even after applying the provided baseline cancellation method. I assumed that remaining interference concists of two things. First one is nonlinear hardware effects from all transmitted signals jointly and temporal memory behavior of the physical system. The temporal memory effects refer to the fact that the interference at a given time sample is not determined only by the current transmitted signal, but also by previously transmitted samples [1]. Second interference compoinent comes from external interference term. It is not a function of tx but it is spatially coherent — the same source appears (with different amplitude and phase) across all 4 receive channels.

To address this, the final solver pipeline consists of the following steps:
- Step 1. Baseline linear cancellation
- Step 2. Nonlinear feature engineering
  - Step 2.1 Nonlinear delayed feature engineering
  - Step 2.2 Nonlinear polynomial feature engineering
- Step 3. Ridge regression
- Step 4. Interference prediction and subtraction
- Step 5. Final SVD-based spatial cleanup step

Also, on the side, a grid search was conducted over several hyperparameters of the nonlinear cancellation pipeline, including the number of delays, nonlinear polynomial orders, ridge regularization strength, nonlinear subtraction scaling factor, and SVD cleanup scaling coefficient. The final parameter configuration was selected based on the achieved interference suppression while maintaining valid explainability behavior. 

## Experiments and Failed Attempts
Before arriving at the final solution, I experimented with several different approaches and parameter settings outlined below:  

### 1. Different Nonlinear Polynomial Orders
Several nonlinear polynomial configurations were tested. Initially, I tried using only first-order terms, and first- plus third-order terms.
Adding third-order nonlinear features improved performance noticeably, but adding adding fourth-order nonlinear feature ([1, 3, 5, 7]) didn't impove the performance at all, so it stayed at 9.49dB. 
The final solution used orders = [1, 3, 5]

### 2. Different Numbers of Delays
I tested several delay configurations to model temporal memory effects I expleined above.
Using very few delays limited the model ability to capture longer temporal interference behavior.However, using too many delays increased the size of the nonlinear feature matrix, and therefore increased computation time. The final solution used num_delays = 8, which provided a reasonable balance between performance and complexity. For example, num_delays = 12 decreased inference compression to 9.45dB. 

### 3. Ordinary Least Squares Without Regularization
I initially tested ordinary least squares regression without regularization, which gave 9.49dB. Adding ridge regularization (using cross-validation to choose the tuning parameter 'ridge') produced better result of 9.6dB. 

### 4. No Singular Value Decomposition (SVD) Step
I tested versions of the pipeline without the final SVD stage. Even the nonlinear regression removed some amount of TX-dependent interference, there was still a leftover interference structure remaining across receive channels. Adding the rank-1 SVD cleanup step improved the final suppression performance by removing this dominant shared component.
No SVD Step produced 3.47 dB and with SVD step produced 9.6dB. 

### 5. Grid Search Instead of Manual Parameter Selection
One experiment I tried was running a grid search on a specified by me range over the main hyperparameters instead of choosing them manually. I tested different values for the nonlinear subtraction strength 'alpha', the SVD cleanup strength 'svd_alpha', the number of delays, and the ridge regularization value.

## References
[1] Landin PN, Barbé K, Van Moer W, Isaksson M, Händel P. Two novel memory polynomial models for modeling of RF power amplifiers. International Journal of Microwave and Wireless Technologies. 2015;7(1):19-29. doi:10.1017/S1759078714000397
