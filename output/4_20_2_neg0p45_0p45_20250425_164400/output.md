================================================================================
KAN SEGMENT FITTING SETTINGS
================================================================================
Input Dimensions: 4
Hidden Dimensions: 20
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
Fitting Layer 0 (4 -> 20)
Fitting Layer 1 (20 -> 2)
Generated data for 120 connections.
Vanilla Fitting Time: 0.3253 seconds
Total Segments: 2400
Max Error: 0.016863
Connections with Infinite Error: 0

--------------------------------------------------
METHOD 2: LIPSCHITZ-WEIGHTED FITTING
--------------------------------------------------
Max Error Threshold Changed to:  0.010552648053936669
Analyzing Layer 0: 4 inputs -> 20 outputs
Analyzing Layer 1: 20 inputs -> 2 outputs
Analyzing Layer 0: 4 inputs -> 20 outputs
Analyzing Layer 1: 20 inputs -> 2 outputs
Weighing Layer 0: 4 inputs -> 20 outputs
Weighing Layer 1: 20 inputs -> 2 outputs
  Achieved error: 0.01053333005017645
  Total segments allocated: 2108
Lipschitz DP Computation Time: 17.1884 seconds
Weighting Time: 17.0081 seconds
Allocation Time: 11.7091 seconds
Total Lipschitz Fitting Time: 45.9057 seconds
Total Segments: 2108
Achieved Error: 0.010533
Connections with Infinite Error: 0

--------------------------------------------------
VERIFICATION COMPARISON
--------------------------------------------------

Running MIP verification for vanilla segments...
Setting up MIP for KAN [4, 20, 2]...
Input bounds: [(-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45)]
Defined 9742 base constraints.

--- Optimizing for Output Dimension 0 ---
Solving for Min Output 0 using Gurobi...
Min Output 0 found: -1.202284
Solving for Max Output 0 using Gurobi...
Max Output 0 found: 1.251123

--- Optimizing for Output Dimension 1 ---
Solving for Min Output 1 using Gurobi...
Min Output 1 found: -1.071661
Solving for Max Output 1 using Gurobi...
Max Output 1 found: 1.133252
Vanilla Verification Time: 2721.1787 seconds
Vanilla Total Time (conversion + verification): 2721.1790 seconds

Running MIP verification for Lipschitz segments...
Setting up MIP for KAN [4, 20, 2]...
Input bounds: [(-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45), (-0.45, 0.45)]
Defined 8574 base constraints.

--- Optimizing for Output Dimension 0 ---
Solving for Min Output 0 using Gurobi...
Min Output 0 found: -1.343442
Solving for Max Output 0 using Gurobi...
