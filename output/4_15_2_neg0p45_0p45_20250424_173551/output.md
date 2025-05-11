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
Vanilla Fitting Time: 0.2306 seconds
Total Segments: 1800
Max Error: 0.021364
Connections with Infinite Error: 0

--------------------------------------------------
METHOD 2: LIPSCHITZ-WEIGHTED FITTING
--------------------------------------------------
Max Error Threshold Changed to:  0.027583402469015448
Analyzing Layer 0: 4 inputs -> 15 outputs
Analyzing Layer 1: 15 inputs -> 2 outputs
Analyzing Layer 0: 4 inputs -> 15 outputs
Analyzing Layer 1: 15 inputs -> 2 outputs
Weighing Layer 0: 4 inputs -> 15 outputs
Weighing Layer 1: 15 inputs -> 2 outputs
  Achieved error: 0.027548089623451233
  Total segments allocated: 1213
Lipschitz DP Computation Time: 12.4199 seconds
Weighting Time: 12.5647 seconds
Allocation Time: 6.4544 seconds
Total Lipschitz Fitting Time: 31.4390 seconds
Total Segments: 1213
Achieved Error: 0.027548
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
