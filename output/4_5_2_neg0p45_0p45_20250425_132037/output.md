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
Vanilla Fitting Time: 0.0954 seconds
Total Segments: 600
Max Error: 0.019407
Connections with Infinite Error: 0

--------------------------------------------------
METHOD 2: LIPSCHITZ-WEIGHTED FITTING
--------------------------------------------------
Max Error Threshold Changed to:  0.03744604530296194
Analyzing Layer 0: 4 inputs -> 5 outputs
Analyzing Layer 1: 5 inputs -> 2 outputs
Analyzing Layer 0: 4 inputs -> 5 outputs
Analyzing Layer 1: 5 inputs -> 2 outputs
Weighing Layer 0: 4 inputs -> 5 outputs
Weighing Layer 1: 5 inputs -> 2 outputs
  Achieved error: 0.0371701717376709
  Total segments allocated: 517
Lipschitz DP Computation Time: 4.2505 seconds
Weighting Time: 4.2112 seconds
Allocation Time: 0.7213 seconds
Total Lipschitz Fitting Time: 9.1830 seconds
Total Segments: 517
Achieved Error: 0.037170
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
