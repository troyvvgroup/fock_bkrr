# fock_bkrr
A custom implementation of Kernel Ridge Regression for predicting converged Fock matrices from SAD guesses.

The general goal here is to to predict converged Fock matrices from SAD guesses for a specific system.
Our training set consists of some number SAD/Fock pairs from an AIMD run.

Repository Structure
--------------------

blocked_krr.py:    The python script that loads the relevant matrices, performs KRR, and computes the error in the output
    
README.txt:     This file.


Example Usage
-----

python3 blocked_krr.py OUTPUT_DIRECTORY_NAME REGULARIZATION_STRENGTH NUMBER_OF_BLOCKS TOTAL_TIMESTEPS_TO_USE FRACTION_SPAN_CORRECTION TRAIN_SET_SIZE SUBSAMPLING_FRACTION

Within the code, one can also toggle the following parameters:
- The set of train sizes to be considered
- Whether the predicted Focks should be solved for their energies
- The kernel type
- Whether to apply standard scaling
- Whether to use different regularization strengths for the linear prediction and span correction
- Whether to average matrix elements above and below the diagonal
- Whether to print the full matrix's MAE or just the predicted portion's MAE
- Whether to include lower triangular blocks
- Whether to print block-by-block error


Input Data Format
-----------------
This code assumes the following file structure:
- Three subdirectories, one titled "converged_fock" and one titled "sad_guess" and one titled "xyz_coords"
- Files in the converged_fock subdirectory are formatted "fock_TIMESTEP.npy" (1-indexed)
- Files in the sad_guess subdirectory are formatted "sad_TIMESTEP.npy" (1-indexed)
- Files in the xyz_coords subdirectory are formatted "TIMESTEP.xyz" (1-indexed)

The code currently assumes the molecule or system of interest is charge neutral with spin 1 when computing energies from fock matrices
