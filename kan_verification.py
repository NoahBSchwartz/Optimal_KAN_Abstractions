from src.kan_verification_grapher import *
from src.kan_verification_trainer import *
from src.kan_verification_fitter import *
from src.kan_verification_milp import *
from src.kan_verification_experiment_runner import *
from src.custom_fastkan import FastKAN
import argparse
import ast

def main(args):
    size = args.size
    experiment_type = args.experiment_type
    input_bounds = args.input_bounds
    segments_per_curve = args.segments_per_curve
    timeout_seconds = args.timeout_seconds
    if experiment_type == "input_range" and segments_per_curve is None:
        raise ValueError("--segments_per_curve is required for experiment_type 'input_range'")
    if experiment_type == "bound_tightness" and input_bounds is None:
        raise ValueError("--input_bounds is required for experiment_type 'bound_tightness'")
    if experiment_type == "single_run":
        if segments_per_curve is None:
            raise ValueError("--segments_per_curve is required for experiment_type 'single_run'")
        if input_bounds is None:
            raise ValueError("--input_bounds is required for experiment_type 'single_run'")

    kan = None
    if not isinstance(size, list):
        try:
            size = ast.literal_eval(size)
            if not isinstance(size, list):
                raise ValueError
        except (ValueError, SyntaxError):
            print(f"Error: --size argument must be a list (e.g., '[4,15,2]'). Received: {size}")
            return

    if input_bounds and not isinstance(input_bounds, list):
        try:
            input_bounds = ast.literal_eval(input_bounds)
            if not (isinstance(input_bounds, list) and len(input_bounds) == 2 and all(isinstance(x, (int, float)) for x in input_bounds)):
                raise ValueError
        except (ValueError, SyntaxError):
            print(f"Error: --input_bounds argument must be a list of two floats (e.g., '[-1.0, 1.0]'). Received: {input_bounds}")
            return

    if size == [4,15,2]:
        print(f"Training KAN of size: {size} on the Duffing Oscillator ODE. \nThen running experiment: {experiment_type}")
        kan = train_duffing_oscillator()
    elif size == [8,24,3]:
        print(f"Training KAN of size: {size} on the Dampened Pendulum ODE. \nThen running experiment: {experiment_type}")
        kan = train_dampened_pendulum()
    elif size == [784, 16, 10]:
        print(f"Training KAN of size: {size} on the MNIST dataset. \nThen running experiment: {experiment_type}")
        kan = train_mnist()
    elif size == [3072, 16, 10]:
        print(f"Training KAN of size: {size} on the CIFAR10 dataset. \nThen running experiment: {experiment_type}")
        kan = train_cifar10()
    else:
        print(f"Misc. size chosen, randomly instantiating KAN of size: {size}\nThen running experiment: {experiment_type}")
        kan = FastKAN(layers_hidden=size, grid_min=-2., grid_max=2., num_grids=15, use_base_update = False, use_layernorm=False)

    if kan is None:
        print("Error: KAN model was not initialized.")
        return
    if not kan.layers:
        print("Error: KAN model layers are not properly initialized for dimension access.")
        if hasattr(kan, 'layers') and not kan.layers and size:
             kan.layers.append(type('Layer', (), {'input_dim': size[0] if size else 1, 'output_dim': size[-1] if len(size)>1 else 10})())
        else:
            return

    # Convert input_bounds to the required format: [(-1.0, 1.0)] * input_dim
    if input_bounds is not None:
        input_dim = kan.layers[0].input_dim
        input_bounds = [(input_bounds[0], input_bounds[1])] * input_dim
        print(f"Converted input_bounds to format: {input_bounds}")

    if experiment_type == "input_range":
        results_df, output_dir = run_kan_input_range_experiment(
            kan_model=kan,
            starting_range=0.05,
            range_increment=0.1,
            max_range=10.0,
            timeout_seconds=timeout_seconds, # Usually timeout_seconds is ~4000 for paper tests
            segments_per_curve=segments_per_curve,
            validate_bounds=True,
            num_validation_samples=5000
        )
        print(f"\nExperiment complete!")
        print(f"Results saved to: {output_dir}")
        print(f"Results overview:")

    elif experiment_type == "bound_tightness":
        results_df, output_dir = run_kan_bound_tightness_experiment(
            kan_model=kan,
            input_bounds=input_bounds,
            timeout_seconds=timeout_seconds,
            starting_segments=3,
            segment_increment=3,
            max_segments=30,
            validate_bounds=True,
            num_validation_samples=5000
        )

        print(f"\nExperiment complete!")
        print(f"Results saved to: {output_dir}")
        print(f"Results overview:")
        print(results_df[["segments", "vanilla_total_time", "vanilla_max_tightness",
                            "lipschitz_total_time", "lipschitz_max_tightness", "timed_out"]].to_string(index=False))

    elif experiment_type == "single_run": #
        print("Running verification once using Vanilla fitting method and once using Dynamic Programming fitting method.")
        if not kan.layers:
            print("Error: KAN model layers are not defined for single_run. Cannot determine dimensions.")
            return
        input_dim = kan.layers[0].input_dim
        hidden_dim = kan.layers[-1].input_dim
        output_dim = kan.layers[-1].output_dim

        if size == [784, 16, 10] or size == [3072, 16, 10]:
            print("Running with memory efficient fitting")
            fitting_results = run_kan_fitting_only(
                kan_model=kan,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                vanilla_segments_per_curve=segments_per_curve,
                max_segments_lipschitz=segments_per_curve*3,
                min_x=-5.0,
                max_x=5.0,
                memory_efficient=True
            )
        else:
            fitting_results = run_kan_fitting_only(
                kan_model=kan,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                vanilla_segments_per_curve=segments_per_curve,
                max_segments_lipschitz=segments_per_curve*3,
                min_x=-5.0,
                max_x=5.0,
                memory_efficient=True
            )

        run_kan_verification_with_fitting(
            kan_model=kan,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            input_bounds=input_bounds,
            fitting_results=fitting_results,
            output_dir=None
        )
    else:
        print(f"Error: Unknown experiment_type '{experiment_type}'. "
              "Choose from 'input_range', 'bound_tightness', 'single_run'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KAN experiments.")

    parser.add_argument("--size", type=str, required=True,
                        help="List of integers defining KAN layer sizes (e.g., '[4,15,2]'). String format.")
    parser.add_argument("--experiment_type", type=str, required=True,
                        choices=["input_range", "bound_tightness", "single_run"],
                        help="Type of experiment to run.")
    parser.add_argument("--timeout_seconds", type=int, default=None,
                        help="Timeout in seconds for some experiments.")
    parser.add_argument("--input_bounds", type=str, default=None,
                        help="List of two floats for input bounds (e.g., '[-1.0, 1.0]'). String format Required for 'bound_tightness' and 'single_run'.")
    parser.add_argument("--segments_per_curve", type=int, default=None,
                        help="Number of segments per curve. Required for 'input_range' and 'single_run' experiment.")

    parsed_args = parser.parse_args()

    processed_args = {}
    processed_args['experiment_type'] = parsed_args.experiment_type
    processed_args['timeout_seconds'] = parsed_args.timeout_seconds

    try:
        evaluated_size = ast.literal_eval(parsed_args.size)
        if not isinstance(evaluated_size, list) or not all(isinstance(x, int) for x in evaluated_size):
            raise ValueError("Size must be a list of integers.")
        processed_args['size'] = evaluated_size
    except (ValueError, SyntaxError) as e:
        parser.error(f"Invalid format for --size: {parsed_args.size}. Expected a list of integers (e.g., '[4,15,2]'). Error: {e}")

    if parsed_args.input_bounds:
        try:
            evaluated_bounds = ast.literal_eval(parsed_args.input_bounds)
            if not (isinstance(evaluated_bounds, list) and len(evaluated_bounds) == 2 and all(isinstance(x, (float, int)) for x in evaluated_bounds)):
                raise ValueError("Input bounds must be a list of two numbers.")
            processed_args['input_bounds'] = [float(x) for x in evaluated_bounds] # Ensure float
        except (ValueError, SyntaxError) as e:
            parser.error(f"Invalid format for --input_bounds: {parsed_args.input_bounds}. Expected a list of two floats (e.g., '[-1.0,1.0]'). Error: {e}")
    else:
        processed_args['input_bounds'] = None

    processed_args['segments_per_curve'] = parsed_args.segments_per_curve

    args_namespace = argparse.Namespace(**processed_args)

    main(args_namespace)