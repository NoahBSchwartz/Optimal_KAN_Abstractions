================================================================================
KAN SEGMENT FITTING SETTINGS
================================================================================
Input Dimensions: 4
Hidden Dimensions: 15
Output Dimensions: 2
Input Bounds: [(-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45)]
Max Error Threshold: 1.0
Vanilla Segments Per Curve: 20
Max Segments Lipschitz: 60
Min X: -5.0
Max X: 5.0
Time Limit (seconds): 35000
Number of Validation Samples: 5000
Absolute Tolerance: 0.001
Relative Tolerance: 0.01
Random Seed: 42
================================================================================
COMPARING KAN SEGMENT FITTING METHODS
================================================================================

--------------------------------------------------
METHOD 1: VANILLA FITTING (EQUAL SEGMENTS)
--------------------------------------------------
Fitting Layer 0 (4 -> 15)
Fitting Layer 1 (15 -> 2)
Generated data for 90 connections.
Vanilla Fitting Time: 0.2098 seconds
Total Segments: 1800
Max Error: 0.017048
Connections with Infinite Error: 0

--------------------------------------------------
METHOD 2: LIPSCHITZ-WEIGHTED FITTING
--------------------------------------------------
Max Error Threshold Changed to:  0.012424031554502335
Analyzing Layer 0: 4 inputs -> 15 outputs
Analyzing Layer 1: 15 inputs -> 2 outputs
Analyzing Layer 0: 4 inputs -> 15 outputs
Analyzing Layer 1: 15 inputs -> 2 outputs
Weighing Layer 0: 4 inputs -> 15 outputs
Weighing Layer 1: 15 inputs -> 2 outputs
  Achieved error: 0.012418857680462914
  Total segments allocated: 1518
Lipschitz DP Computation Time: 11.3765 seconds
Weighting Time: 11.3216 seconds
Allocation Time: 5.5130 seconds
Total Lipschitz Fitting Time: 28.2111 seconds
Total Segments: 1518
Achieved Error: 0.012419
Connections with Infinite Error: 0

--------------------------------------------------
VERIFICATION COMPARISON
--------------------------------------------------

Running MIP verification for vanilla segments...
Setting up MIP for KAN [4, 15, 2]...
Input bounds: [(-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45)]
Defined 7307 base constraints.

--- Optimizing for Output Dimension 0 ---
Solving for Min Output 0 using Gurobi...
Min Output 0 found: -1.008523
Solving for Max Output 0 using Gurobi...
Max Output 0 found: 1.237919

--- Optimizing for Output Dimension 1 ---
Solving for Min Output 1 using Gurobi...
Min Output 1 found: -1.158423
Solving for Max Output 1 using Gurobi...
Max Output 1 found: 0.836654
Vanilla Verification Time: 432.9222 seconds
Vanilla Total Time (conversion + verification): 432.9224 seconds

Running MIP verification for Lipschitz segments...
Setting up MIP for KAN [4, 15, 2]...
Input bounds: [(-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45)]
Defined 6179 base constraints.

--- Optimizing for Output Dimension 0 ---
Solving for Min Output 0 using Gurobi...
Min Output 0 found: -1.170007
Solving for Max Output 0 using Gurobi...
Max Output 0 found: 1.455343

--- Optimizing for Output Dimension 1 ---
Solving for Min Output 1 using Gurobi...
Min Output 1 found: -1.265953
Solving for Max Output 1 using Gurobi...
Max Output 1 found: 0.953473
Lipschitz Verification Time: 130.5677 seconds
Lipschitz Total Time (conversion + verification): 130.5682 seconds

LIPSCHITZ error was valid
