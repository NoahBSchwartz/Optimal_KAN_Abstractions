================================================================================
KAN SEGMENT FITTING SETTINGS
================================================================================
Input Dimensions: 4
Hidden Dimensions: 5
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
Fitting Layer 0 (4 -> 5)
Fitting Layer 1 (5 -> 2)
Generated data for 30 connections.
Vanilla Fitting Time: 0.1077 seconds
Total Segments: 600
Max Error: 0.022329
Connections with Infinite Error: 0

--------------------------------------------------
METHOD 2: LIPSCHITZ-WEIGHTED FITTING
--------------------------------------------------
Max Error Threshold Changed to:  0.05110094630601236
Analyzing Layer 0: 4 inputs -> 5 outputs
Analyzing Layer 1: 5 inputs -> 2 outputs
Analyzing Layer 0: 4 inputs -> 5 outputs
Analyzing Layer 1: 5 inputs -> 2 outputs
Weighing Layer 0: 4 inputs -> 5 outputs
Weighing Layer 1: 5 inputs -> 2 outputs
  Achieved error: 0.05049472762168885
  Total segments allocated: 417
Lipschitz DP Computation Time: 4.4041 seconds
Weighting Time: 4.1823 seconds
Allocation Time: 0.6997 seconds
Total Lipschitz Fitting Time: 9.2860 seconds
Total Segments: 417
Achieved Error: 0.050495
Connections with Infinite Error: 0

--------------------------------------------------
VERIFICATION COMPARISON
--------------------------------------------------

Running MIP verification for vanilla segments...
Setting up MIP for KAN [4, 5, 2]...
Input bounds: [(-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45)]
Defined 2437 base constraints.

--- Optimizing for Output Dimension 0 ---
Solving for Min Output 0 using Gurobi...
Min Output 0 found: -1.726762
Solving for Max Output 0 using Gurobi...
Max Output 0 found: 1.177724

--- Optimizing for Output Dimension 1 ---
Solving for Min Output 1 using Gurobi...
Min Output 1 found: -0.859621
Solving for Max Output 1 using Gurobi...
Max Output 1 found: 0.887337
Vanilla Verification Time: 2.6112 seconds
Vanilla Total Time (conversion + verification): 2.6114 seconds

Running MIP verification for Lipschitz segments...
Setting up MIP for KAN [4, 5, 2]...
Input bounds: [(-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45)]
Defined 1705 base constraints.

--- Optimizing for Output Dimension 0 ---
Solving for Min Output 0 using Gurobi...
Min Output 0 found: -1.946008
Solving for Max Output 0 using Gurobi...
Max Output 0 found: 1.527459

--- Optimizing for Output Dimension 1 ---
Solving for Min Output 1 using Gurobi...
Min Output 1 found: -1.091625
Solving for Max Output 1 using Gurobi...
Max Output 1 found: 1.175026
Lipschitz Verification Time: 1.6408 seconds
Lipschitz Total Time (conversion + verification): 1.6410 seconds

LIPSCHITZ error was valid
