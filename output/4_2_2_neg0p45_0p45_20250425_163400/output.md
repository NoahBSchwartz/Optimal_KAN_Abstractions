================================================================================
KAN SEGMENT FITTING SETTINGS
================================================================================
Input Dimensions: 4
Hidden Dimensions: 2
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
Fitting Layer 0 (4 -> 2)
Fitting Layer 1 (2 -> 2)
Generated data for 12 connections.
Vanilla Fitting Time: 0.0460 seconds
Total Segments: 240
Max Error: 0.024602
Connections with Infinite Error: 0

--------------------------------------------------
METHOD 2: LIPSCHITZ-WEIGHTED FITTING
--------------------------------------------------
Max Error Threshold Changed to:  0.08129261248122926
Analyzing Layer 0: 4 inputs -> 2 outputs
Analyzing Layer 1: 2 inputs -> 2 outputs
Analyzing Layer 0: 4 inputs -> 2 outputs
Analyzing Layer 1: 2 inputs -> 2 outputs
Weighing Layer 0: 4 inputs -> 2 outputs
Weighing Layer 1: 2 inputs -> 2 outputs
  Achieved error: 0.08079723220121668
  Total segments allocated: 188
Lipschitz DP Computation Time: 1.7489 seconds
Weighting Time: 1.7167 seconds
Allocation Time: 0.1108 seconds
Total Lipschitz Fitting Time: 3.5764 seconds
Total Segments: 188
Achieved Error: 0.080797
Connections with Infinite Error: 0

--------------------------------------------------
VERIFICATION COMPARISON
--------------------------------------------------

Running MIP verification for vanilla segments...
Setting up MIP for KAN [4, 2, 2]...
Input bounds: [(-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45)]
Defined 976 base constraints.

--- Optimizing for Output Dimension 0 ---
Solving for Min Output 0 using Gurobi...
Min Output 0 found: -1.283883
Solving for Max Output 0 using Gurobi...
Max Output 0 found: 2.235353

--- Optimizing for Output Dimension 1 ---
Solving for Min Output 1 using Gurobi...
Min Output 1 found: -0.928467
Solving for Max Output 1 using Gurobi...
Max Output 1 found: 0.729237
Vanilla Verification Time: 0.3861 seconds
Vanilla Total Time (conversion + verification): 0.3861 seconds

Running MIP verification for Lipschitz segments...
Setting up MIP for KAN [4, 2, 2]...
Input bounds: [(-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45)]
Defined 768 base constraints.

--- Optimizing for Output Dimension 0 ---
Solving for Min Output 0 using Gurobi...
Min Output 0 found: -1.547812
Solving for Max Output 0 using Gurobi...
Max Output 0 found: 2.459497

--- Optimizing for Output Dimension 1 ---
Solving for Min Output 1 using Gurobi...
Min Output 1 found: -1.010963
Solving for Max Output 1 using Gurobi...
Max Output 1 found: 0.813703
Lipschitz Verification Time: 0.2324 seconds
Lipschitz Total Time (conversion + verification): 0.2326 seconds

LIPSCHITZ error was valid
