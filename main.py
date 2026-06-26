from network_design_test import NetworkDesignProblem
from pathlib import Path
import sys, json
import concurrent.futures

# Determine whether supply or demand are endogenous
uncertainty_configurations = ["endogenous_supply", "endogenous_demand"]

# Determine which combinations of approaches to test
experiment_combinations = [
        [],                             # Approach (i) -- baseline         
        ["JC", "DC"],                   # Approach (ii) 
        ["DFC"],                        # Approach (iii) 
        ["DFC", "DFC-S", "DFC-D"],      # Approach (iv) 
        ["DFC", "DFC-D"],               # Approach (v)
        ["DFC", "DFC-S"]                # Approach (vi)
    ]

# Load experiments
load_path = Path(f"instances/section_6_1/")
index_range = range(1, 2)      # Range of instances to load based on indices

instances = {}
# Get all JSON instances sorted by index
for file in sorted(load_path.glob("*.json")):
    index_str = int(file.stem.split("_")[0])
    if index_str not in index_range:
        continue
    with open(file, "r", encoding="utf-8") as f:
        instances[index_str] = json.load(f)
        f.close()
instances = dict(sorted(instances.items()))  # Sort instances by index
if not instances:
    print("\33[41mNo instances found in the specified range.\33[0m")
    sys.exit(0)

# Define the output path to log experiment results
output_path = Path(f"results/")

# Execution Setting
time_limit = 1800           # Time limit for optimization (in seconds)
optimality_gap = 0.0        # Optimality gap for optimization
compute_big_m = True        # Compute M using Jensen's inequality
pre_sampling = False        # Pre-sample scenarios before optimization
unique_seed = True          # Use unique seed for each distribution
log = {
    "res": True, 
    "lazy": False,       # Print lazy constraints 
    "vis_dis": False,    # Print visited distributions 
    "un_vis_dis": False, # Print unvisited distributions
    "solution": True,    # Print solution
    }
evaluation_sample_size = 1000  # Sample size for post-optimization evaluation
combined_tariff_rate = 0.11     # Tariff rate applied in the even of mixed-sourcing/defaults to average of the zones when not specified

# Multi-processing settings
multi_processing = True     # Use multi-processing for optimizing sub-problems
num_processes = 5           # Number of processes for multi-processing
same_workers = True         # Use same worker pool over all instances
if same_workers:
    workers_configuration = {
        "number_of_processes": num_processes if multi_processing else 1,  # For single processing, set multi-processing = False
        "executor": concurrent.futures.ProcessPoolExecutor(max_workers=num_processes)
    } 

# Console output preperation
all_cuts =["JC", "DC", "DFC", "DFC-S", "DFC-D"] 

def print_flags(flags, total_width=20):
    # Collect active cuts
    active = [key for key in flags.keys() if flags[key]]
    if not active:
        active = ["BL"]
    # Join with commas
    cut_str = ", ".join(active)
    cut_str = f"{cut_str}," if cut_str else ""  # trailing comma
    print(f"{cut_str:<{total_width}}", end="", flush=True)  # pad right

def tf(flag: bool) -> str:
    return "T" if flag else "F"

# Test method
def test():
    endogenous_supply = True if "endogenous_supply" in uncertainty_configurations else False
    endogenous_demand = True if "endogenous_demand" in uncertainty_configurations else False

    for _, instance in instances.items():
        print(
            f" #id: {instance['name']:<18} "
            + (
                f"Endogenous: Supply, Demand" if endogenous_supply and endogenous_demand else
                f"Endogenous: Supply" if endogenous_supply else
                f"Endogenous: Demand" if endogenous_demand else
                f"Endogenous: None"
            ),
            end="\n \t",
            flush=True
        )

        for experiment in experiment_combinations:
            flags = {cut: (cut in experiment) for cut in all_cuts}
            add_jensen                  = flags["JC"]
            add_ghost_scenario          = flags["DC"]
            add_dominant_flow           = flags["DFC"]
            add_dominant_flow_sup       = flags["DFC-S"]
            add_dominant_flow_dem       = flags["DFC-D"]
            print_flags(flags)

            problem = NetworkDesignProblem(
                nodes=instance["nodes"],
                profits=instance["profits"],
                investment_costs=instance["investment_costs"],
                processing_times=instance["processing_times"],
                supply_uncertainty=instance["supply_uncertainty"],
                demand_uncertainty=instance["demand_uncertainty"],
                plant_zones=instance["plant_zones"],
                product_tariffs=instance["product_tariffs"],
                sample_size=instance["sample_size"],
                seed=instance.get("seed", None),
                compute_big_m=compute_big_m,
                unique_seed=unique_seed,
                pre_sampling=pre_sampling,
                evaluation_sample_size=evaluation_sample_size,
                endogenous_supply=endogenous_supply,
                endogenous_demand=endogenous_demand,
                add_jensen=add_jensen,
                add_ghost_scenario=add_ghost_scenario,
                add_dominant_flow=add_dominant_flow,
                add_dominant_flow_sup=add_dominant_flow_sup,
                add_dominant_flow_dem=add_dominant_flow_dem,
                instance_name=instance["name"],
                multi_processing=multi_processing,
                num_processes=num_processes,
                same_workers=same_workers,
                workers_configuration=workers_configuration if same_workers else None,
                output_path=output_path,
                log=log,
                time_limit=time_limit,
                optimality_gap=optimality_gap,
                combined_tariff_rate=combined_tariff_rate,
            )
            obj, time, cuts, is_optimal, bound, mip_gap = problem.optimize()

            print(
                    f"obj: {obj:<15.3f}"
                    f"time: {time:<10.3f}"
                    f"itr: {cuts:<9}"
                    f"opt: {tf(is_optimal):<5}"
                    f"bound: {bound:<15.3f}"
                    f"gap: {mip_gap:<15.3f}",
                    end="\n \t"
                )
        print("\r" ,"-"*125)

if __name__ == "__main__":
    test()
    if same_workers:
       workers_configuration["executor"].shutdown(wait=False, cancel_futures=True)