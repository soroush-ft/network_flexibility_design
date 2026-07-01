from typing import Mapping, Sequence, Optional, Any
import hashlib
import os
from random import seed
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from network_flow import MaxFlow
import itertools
from pprint import pprint
import time
import datetime
import concurrent.futures
import hashlib
import csv, os
import matplotlib.pyplot as plt
import math

# Initiating an empty Gurobi environment.
env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

class NetworkDesignProblem:
    """
        - This class solves the network design problem under endogenous supply and demand uncertainty.
        - Due to the changeovers, the number of edges connected to each supply node affects the distribution of supply.
        - Due to the tariff's effect, the zones from which a product can be sourced affect the distribution of demand.
        - The solution approach is based on the L-shaped method, equipped with valid cuts to improve the master problem's relaxation.
        - The type of uncertainty (endogenous/exogenous) can be configured for both supply and demand.
        - The code supports multi-processing to solve the second-stage problems in parallel.
        - The code supports pre-sampling of all scenarios at the beginning or sampling on-the-fly during the optimization.
        - This python implementation uses Gurobi as the optimization solver.
    """

    def __init__(
        self,
        *,
        nodes: Mapping[str, Sequence[Any]],
        profits: Mapping[Any, Mapping[Any, float]],
        investment_costs: Mapping[Any, Mapping[Any, float]],
        processing_times: Mapping[Any, Mapping[Any, float]],
        supply_uncertainty: Mapping[Any, Mapping[str, float | str]],
        demand_uncertainty: Mapping[Any, Mapping[str, float | str]],
        plant_zones: Mapping[Any, Any] | None = None,
        product_tariffs: Mapping[Any, Mapping[Any, float]] | None = None,

        sample_size: int = 1000,
        compute_big_m: bool = True,
        endogenous_supply: bool = True,
        endogenous_demand: bool = False,
        add_jensen: bool = True,
        add_ghost_scenario: bool = True,
        add_dominant_flow: bool = True,
        add_dominant_flow_sup: bool = True,
        add_dominant_flow_dem: bool = True,

        seed: Optional[int] = None,
        unique_seed: bool = True,
        pre_sampling: bool = True,
        evaluation_sample_size: int = 10000,

        instance_name: str,
        multi_processing: bool = True,
        num_processes: int = 8,
        same_workers: bool = False,
        workers_configuration: Optional[Mapping[str, Any]] = None,

        log: Mapping[str, bool],
        output_path: Optional[str] = None,
        time_limit: Optional[int] = None,
        optimality_gap: float = 0.0,
        combined_tariff_rate: float = None
    ) -> None:

        # --- Problem Configuration ---
        self.endogenous_supply = endogenous_supply
        self.endogenous_demand = endogenous_demand

        # --- Problem Data ---
        self.nodes = nodes
        self.plants = tuple(nodes["plants"])
        self.products = tuple(nodes["products"])
        self.len_plants, self.len_products = len(nodes["plants"]), len(nodes["products"])

        self.profits = profits
        self.investment_costs = investment_costs
        self.processing_times = processing_times
        self.supply_uncertainty = supply_uncertainty
        self.demand_uncertainty = demand_uncertainty

        #-- Zone and Tariff Information ---
        if self.endogenous_demand:
            self.plant_zones = plant_zones
            self.product_tariffs = product_tariffs
            self.zones = tuple(sorted(set(self.plant_zones.values())))  # Set of all zones
            self.get_all_zone_combinations()

        # --- Generating All Distributions ---
        self.combined_tariff_rate = combined_tariff_rate
        if log["vis_dis"] or log["un_vis_dis"] or pre_sampling:
            if self.endogenous_supply:
                self.get_all_supply_distributions()
            if self.endogenous_demand:
                self.get_all_demand_distributions()

        # --- RMP Initialization ---
        self.model = gp.Model("rmp", env=env)

        # --- Second-stage Sampling ---
        self.sample_size = sample_size  # Number of scenarios
        self.samples = None
        self.seed = seed
        self.pre_sampling = pre_sampling  # If True, all samples are generated at the beginning
        if self.pre_sampling:
            self.create_samples_collection()
        else:
            self.rng = None
            self.unique_seed = unique_seed  # If True, each distribution has unique seed for sampling
        self.evaluation_sample_size = evaluation_sample_size  # Sample size for evaluation after optimization

        # --- Valid Cuts Configuration ---
        self.add_jensen = add_jensen  # Add Jensen's cut to the RMP
        self.add_ghost_scenario = add_ghost_scenario  # Add ghost scenario cut
        if self.add_ghost_scenario:
            self.network_flow = MaxFlow(self.nodes, self.profits, self.processing_times, env)
        self.add_dominant_flow = add_dominant_flow  # Add dominant flow problem to the RMP
        self.add_dominant_flow_sup = add_dominant_flow_sup  # Add dominant flow for subsets of supply distributions
        self.add_dominant_flow_dem = add_dominant_flow_dem  # Add dominant flow for subsets of demand distributions

        # --- Logging Configuration ---
        self.log_entries = []
        self.log = log  
        self.instance_name = instance_name
        self.added_distribution_cuts = {}
        self.visited_supply_distributions = {}
        self.visited_demand_distributions = {}
        self.total_subproblem_time = 0
        self.total_subproblem_cut_time = 0
        self.total_time = 0

        # --- Multi-Processing Setup ---
        self.same_workers = same_workers
        if not self.same_workers:
            # self.number_of_processes = min((os.cpu_count(), num_processes)) if multi_processing else 1  # For single processing, set multi-processing = False
            self.number_of_processes = num_processes if multi_processing else 1  # For single processing, set multi-processing = False
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.number_of_processes)
        else:
            self.number_of_processes = workers_configuration["number_of_processes"]
            self.executor = workers_configuration["executor"]

        # --- Constants ---
        self.TOL = 1e-6
        self.Big_M = self.get_big_m() if compute_big_m else 10_000
        self.output_path = output_path
        self.time_limit = time_limit
        self.optimality_gap = optimality_gap
        
    def get_rmp(self):
        """
        - Generate the initial Relaxed Master Problem (RMP) for the network design problem.
        """
        # Binary variable y_{i,j}=1, if product j is produced at plant i
        self.edges = {
            (plant, product): self.model.addVar(vtype=GRB.BINARY, name=f"y_{plant},{product}")
            for plant in self.plants
            for product in self.products
        }

        if self.endogenous_supply:
            # Binary variable w_{i,d}=1, if plant i has degree d
            self.degree_assignments = {
                (plant, degree): self.model.addVar(vtype=GRB.BINARY, name=f"w_{plant},{degree}")
                for plant in self.plants
                for degree in range(0, len(self.products) + 1)
            }
            # Constraints connecting variables y_{i,j} and w_{i,d}
            self.model.addConstrs(
                (gp.quicksum(self.edges[(plant, product)] for product in self.products) - gp.quicksum(degree * self.degree_assignments[(plant, degree)] for degree in range(0, len(self.products) + 1)) == 0
                 for plant in self.plants),
                name=f"w&y"
            )
            # Constraints setting degree of each node to 1
            self.model.addConstrs(
                    (gp.quicksum(self.degree_assignments[(plant, degree)] for degree in range(0, len(self.products) + 1)) == 1
                    for plant in self.plants),
                    name=f"w=1"
                )

        if self.endogenous_demand:
            # Binary variable u_{j,z}=1, if product j is assigned to tariff zone z
            self.zone_assignments = {
                (product, zone): self.model.addVar(vtype=GRB.BINARY, name=f"u_{product},{zone}")
                for product in self.products
                for zone in self.zones
            }
            # Connecting y_{i,j} and zone assignment variables u_{j,z}
            self.model.addConstrs(
                (
                    self.edges[(plant, product)] <= self.zone_assignments[(product, self.plant_zones[plant])]
                    for plant in self.plants
                    for product in self.products
                ),
                name="y&u"
            )
            self.model.addConstrs(
                (
                    self.zone_assignments[(product, zone)] <= 
                    gp.quicksum(self.edges[(plant, product)] 
                        for plant in self.plants
                        if self.plant_zones[plant] == zone)
                    for product in self.products
                    for zone in self.zones
                ),
                name="u&y"
            )

        # Continuous variable mu, approximates expected 2nd-stg cost 
        self.mu = self.model.addVar(vtype=GRB.CONTINUOUS, name="mu", lb=0, ub=self.Big_M)      

        # Objective Function is to maximize the total profit
        self.model.setObjective(
            - gp.quicksum(
                self.investment_costs[plant][product] * self.edges[(plant, product)]
                for plant in self.plants for product in self.products) 
            + self.mu,
            GRB.MAXIMIZE
        )        
        
        self.model.update()
        self.get_varlist()
    
    def add_dominant_flow_problem(self):
        """
            - Adds the dominant flow problem to the RMP as a set of constraints.
        """
        # Adding flow variables
        self.flow = {
            (i,j): self.model.addVar(lb=0)
            for i in self.nodes["plants"]
            for j in self.nodes["products"]
        }

        # Adding capacity constraints
        self.model.addConstrs(
            gp.quicksum(self.processing_times[plant][product] * self.flow[(plant, product)] for product in self.nodes["products"]) 
            <= self.supply_uncertainty[plant]["mean"] for plant in self.plants)
        
        # Adding demand constraints
        self.model.addConstrs(
            gp.quicksum(self.flow[(plant, product)] for plant in self.nodes["plants"]) 
            <= self.demand_uncertainty[product]["mean"] for product in self.products)
        
        # Adding edge constraints
        self.model.addConstrs(
            self.flow[plant, product] <= min(self.supply_uncertainty[plant]["mean"]/self.processing_times[plant][product], self.demand_uncertainty[product]["mean"]) * self.edges[(plant, product)] for plant in self.plants for product in self.products)
        
        # Adding upper bound on objective
        self.model.addConstr(
            self.mu <= gp.quicksum(self.profits[plant][product] * self.flow[(plant, product)] 
                        for plant in self.nodes["plants"] 
                        for product in self.nodes["products"]))
        self.model.update()

    def add_dominant_flow_supply_problem(self):
        """
            - Adds the dominant flow problem for subsets of supply distributions to the RMP as a set of constraints.
        """
        self.flow_sup = {}
        for specific_plant in self.nodes["plants"]:
            for degree in range(2, len(self.nodes["products"]) + 1):
                
                # Define a dominant max flow for the subset of supply distributions
                self.flow_sup[(specific_plant, degree)] = {
                    (i,j): self.model.addVar(lb=0)
                    for i in self.nodes["plants"]
                    for j in self.nodes["products"]
                }

                # Adding capacity constraints
                self.model.addConstrs(
                    gp.quicksum(self.processing_times[plant][product] * self.flow_sup[(specific_plant, degree)][(plant, product)] for product in self.nodes["products"]) 
                    <= self.supply_uncertainty[plant]["mean"] for plant in self.plants if plant != specific_plant)
                
                # Adding capacity constraint for the specific plant
                self.model.addConstr(
                    gp.quicksum(self.processing_times[specific_plant][product] * self.flow_sup[(specific_plant, degree)][(specific_plant, product)] for product in self.nodes["products"])
                    <= self.supply_uncertainty[specific_plant]["mean"] * (1 - (degree - 1) * self.supply_uncertainty[specific_plant]["mean_tilt"])
                )
                
                # Adding demand constraints
                self.model.addConstrs(
                    gp.quicksum(self.flow_sup[(specific_plant, degree)][(plant, product)] for plant in self.nodes["plants"]) 
                    <= self.demand_uncertainty[product]["mean"] for product in self.products)
                
                # Adding edge constraints
                self.model.addConstrs(
                    self.flow_sup[(specific_plant, degree)][plant, product] <= min(
                        self.supply_uncertainty[plant]["mean"]/self.processing_times[plant][product], 
                        self.demand_uncertainty[product]["mean"]) 
                        * self.edges[(plant, product)] 
                        for plant in self.plants 
                        for product in self.products 
                        if plant != specific_plant)
                
                # Adding edge constraints for the specific plant
                self.model.addConstrs(
                    self.flow_sup[(specific_plant, degree)][specific_plant, product] <= min(
                        (self.supply_uncertainty[specific_plant]["mean"] * (1 - (degree - 1) * self.supply_uncertainty[specific_plant]["mean_tilt"]))/self.processing_times[specific_plant][product],
                        self.demand_uncertainty[product]["mean"]) 
                        * self.edges[(specific_plant, product)] 
                        for product in self.products)
                
                # Adding upper bound on objective
                self.model.addConstr(
                    self.mu <= gp.quicksum(self.profits[plant][product] * self.flow_sup[(specific_plant, degree)][(plant, product)] 
                                for plant in self.nodes["plants"] 
                                for product in self.nodes["products"])
                                + gp.quicksum(
                                    self.degree_assignments[(specific_plant, degree_prime)]
                                    for degree_prime in range(0, degree)   
                                ) * self.Big_M
                                )
        self.model.update()

    def add_dominant_flow_demand_problem(self):
        """
            - Adds the dominant flow problem for subsets of demand distributions to the RMP as a set of constraints.
        """
        self.flow_dem = {}
        for specific_product in self.products:
            for zone_combination in self.all_zone_combinations:
                if not zone_combination:
                    continue
                
                # Define a dominant max flow for the subset of demand distributions
                self.flow_dem[(specific_product, zone_combination)] = {
                    (plant, product): self.model.addVar(lb=0)
                    for plant in self.nodes["plants"]
                    for product in self.nodes["products"]
                }

                # Adding capacity constraints
                self.model.addConstrs(
                    gp.quicksum(self.processing_times[plant][product] * self.flow_dem[(specific_product, zone_combination)][(plant, product)] for product in self.nodes["products"]) 
                    <= self.supply_uncertainty[plant]["mean"] for plant in self.plants)
                
                # Adding demand constraints
                self.model.addConstrs(
                    gp.quicksum(self.flow_dem[(specific_product, zone_combination)][(plant, product)] for plant in self.plants) 
                    <= self.demand_uncertainty[product]["mean"] for product in self.products if product != specific_product)
                
                # Adding demand constraint for the specific product
                self.model.addConstr(
                    gp.quicksum(self.flow_dem[(specific_product, zone_combination)][(plant, specific_product)] for plant in self.plants)
                    <= self.demand_uncertainty[specific_product]["mean"] * (1 - self.get_effective_tariff(specific_product, zone_combination) * self.demand_uncertainty[specific_product]["mean_tilt"]))
                
                # Adding edge constraints
                self.model.addConstrs(
                    self.flow_dem[(specific_product, zone_combination)][plant, product] <= min(
                        self.supply_uncertainty[plant]["mean"]/self.processing_times[plant][product], 
                        self.demand_uncertainty[product]["mean"]) 
                        * self.edges[(plant, product)] 
                        for plant in self.plants 
                        for product in self.products 
                        if product != specific_product
                )
                    
                # Adding edge constraints for the specific product
                self.model.addConstrs(
                    self.flow_dem[(specific_product, zone_combination)][plant, specific_product] <= min(
                        self.supply_uncertainty[plant]["mean"]/self.processing_times[plant][specific_product], 
                        self.demand_uncertainty[specific_product]["mean"] * (1 - self.get_effective_tariff(specific_product, zone_combination) * self.demand_uncertainty[specific_product]["mean_tilt"])) 
                        * self.edges[(plant, specific_product)] 
                        for plant in self.plants)

                # Adding upper bound on objective
                self.model.addConstr(
                    self.mu <= gp.quicksum(self.profits[plant][product] * self.flow_dem[(specific_product, zone_combination)][(plant, product)] for plant in self.nodes["plants"] for product in self.nodes["products"]) 
                    + (
                    len(zone_combination) - gp.quicksum(self.zone_assignments[(specific_product, zone)] for zone in zone_combination) +
                    gp.quicksum(self.zone_assignments[(specific_product, zone)] for zone  in self.zones if zone not in zone_combination)
                    ) * self.Big_M)
        self.model.update()

    def get_varlist(self):
        """
            - To speed up callback, we create a var list beforehand
        """
        self.y_keys  = [(plant, product) for plant in self.plants for product in self.products]
        self.y_vars  = [self.edges[key] for key in self.y_keys]

        if self.endogenous_demand:
            self.u_keys     = [(product, zone) for product in self.products for zone in self.zones]
            self.u_vars  = [self.zone_assignments[key] for key in self.u_keys]
        else:
            self.u_vars = []

        if self.endogenous_supply:
            self.w_keys = [(plant, degree) for plant in self.plants for degree in range(0, len(self.products) + 1)]
            self.w_vars = [self.degree_assignments[key] for key in self.w_keys]
        else:
            self.w_vars = []

        self.varlist = self.y_vars + self.u_vars + self.w_vars + [self.mu]
    
    def get_design_info(self):
        """
        - Given the current solution of the RMP, extracts design information, namely, zone_assignments and edges.
        """
        edges = [(plant, product) 
                 for plant in self.plants 
                 for product in self.products 
                 if self.sol_y[(plant, product)] > 0.5]
        
        if self.endogenous_supply:
            node_degrees = {plant: sum(1 if self.sol_y[(plant, product)] > 0.5 
                                       else 0 for product in self.products) 
                                       for plant in self.plants}
        else:
            node_degrees = {}

        if self.endogenous_demand:
            zone_assignments = {
                product: tuple(sorted(zone for zone in self.zones 
                        if self.sol_u[(product, zone)] > 0.5))
                for product in self.products
            }
        else:
            zone_assignments = {}

        return node_degrees, zone_assignments, edges

    def create_samples_collection(self):
        """
            - Determines which method for generating samples to use based on the uncertainty configuration.
        """
        self.samples_collection_supply = {}  # Stores samples for all node_degree combinations
        self.samples_collection_demand = {}  # Stores samples for all zone_assignment combinations

        self.rng = np.random.default_rng(self.seed if self.seed is not None else 0)

        if self.endogenous_supply:
            self.create_endogenous_supply()
        else:
            self.create_exogenous_supply()

        if self.endogenous_demand:
            self.create_endogenous_demand()
        else:
            self.create_exogenous_demand()

    def create_endogenous_supply(self):
        """
            - Creates a collection of plant samples for all node_degree combinations.
            - Order of retrieval: node_degrees -> supply_distribution -> samples 
        """
        for node_degrees in self.all_degree_combinations:
            supply_distribution = self._get_supply_distribution(node_degrees)
            key = tuple(node_degrees[plant] for plant in self.plants)
            self.samples_collection_supply[key] = self.get_supply_samples(supply_distribution)

    def create_endogenous_demand(self):
        """
            - Creates a collection of product samples for all zone assignments.
            - Order of retrieval: zone_assignment -> demand_distribution -> samples
        """
        for zone_assignment in self.all_zone_assignments:
            demand_distribution = self._get_demand_distribution(zone_assignment)
            key = tuple(tuple(sorted(zone_assignment[product])) for product in self.products)
            self.samples_collection_demand[key] = self.get_demand_samples(demand_distribution)

    def create_exogenous_supply(self):
        """
            - Creates a collection of plant samples independently of node degrees.
        """
        supply_distribution = self._get_supply_distribution()
        self.samples_collection_supply[None] = self.get_supply_samples(supply_distribution)

    def create_exogenous_demand(self):
        """
        - Creates a collection of product samples independently of zone assignments.
        """
        demand_distribution = self._get_demand_distribution()
        self.samples_collection_demand[None] = self.get_demand_samples(demand_distribution)

    def get_supply_samples(self, supply_distribution, supply_key=None):
        """
            - Up to sampling, the method is indifferent for endogenous and exogenous supply.
            - If endogenous supply: the corresponding key must be provided to generate rng.
        """
        rng = self.rng
        if not self.pre_sampling:
            if self.endogenous_supply and self.unique_seed:
                if supply_key is None:
                    raise ValueError("Supply key must be provided for endogenous supply.")
                rng = np.random.default_rng(self.get_unique_seed("supply", supply_key))
            elif not self.endogenous_supply and self.unique_seed:
                rng = np.random.default_rng(self.get_unique_seed("supply", None))
            else:   # Not unique seed for either endogenous or exogenous supply
                rng = np.random.default_rng(self.seed if self.seed is not None else 0)

        plant_samples = {}
        for plant in self.plants:
            if supply_distribution[plant]["dist"] == "normal":
                raw = rng.normal(
                    loc=supply_distribution[plant]["mean"], 
                    scale=np.sqrt(supply_distribution[plant]["var"]), 
                    size=self.sample_size
                    ).astype(np.float32, copy=False)
                raw[raw <= 0] = 0
                raw[raw >= 2 * supply_distribution[plant]["mean"]] = 2 * supply_distribution[plant]["mean"]  # Cap supply at 2x mean to avoid extreme outliers
                plant_samples[plant] = raw
            if supply_distribution[plant]["dist"] == "constant":
                plant_samples[plant] = np.full(self.sample_size, supply_distribution[plant]["mean"], dtype=np.float32) 
        return plant_samples

    def get_demand_samples(self, demand_distribution, demand_key=None):
        rng = self.rng
        if not self.pre_sampling:
            if self.endogenous_demand and self.unique_seed:
                if demand_key is None:
                    raise ValueError("Demand key must be provided for endogenous demand.")
                rng = np.random.default_rng(self.get_unique_seed("demand", demand_key))
            elif not self.endogenous_demand and self.unique_seed:
                rng = np.random.default_rng(self.get_unique_seed("demand", None))
            else:   # Not unique seed for either endogenous or exogenous demand
                rng = np.random.default_rng(self.seed if self.seed is not None else 0)
            
        product_samples = {}
        for product in self.products:
            if demand_distribution[product]["dist"] == "normal":
                raw = rng.normal(
                    loc=demand_distribution[product]["mean"],
                    scale=np.sqrt(demand_distribution[product]["var"]),
                    size=self.sample_size
                ).astype(np.float32, copy=False)
                raw[raw <= 0] = 0
                raw[raw >= 2 * demand_distribution[product]["mean"]] = 2 * demand_distribution[product]["mean"]  # Cap demand at 2x mean to avoid extreme outliers
                product_samples[product] = raw
            if demand_distribution[product]["dist"] == "constant":
                product_samples[product] = np.full(self.sample_size, demand_distribution[product]["mean"], dtype=np.float32)
        return product_samples

    def get_unique_seed(self, tag: str, key) -> int:
        base = 0 if self.seed is None else int(self.seed)
        msg  = repr((base, tag, key)).encode("utf-8")  # Ensure key is canonical (tuples, sorted)
        h    = hashlib.sha256(msg).digest()            # Create a hashed code to name the distribution
        return int.from_bytes(h[:8], "little")         # 64-bit seed

    def get_samples(self, node_degrees_key=None, node_degrees=None, zone_assignments_key=None, zone_assignments=None):
        """
            - Depending on the uncertainty configuration, retrieves the appropriate samples.
            - For either of the two cases: pre-sampling or on-the-fly sampling:
                - If endogenous uncertainty: the corresponding key must be provided to retrieve/generate the correct samples.
                - If exogenous uncertainty: the corresponding key is ignored.
        """
        self.samples = {"plants": {}, "products": {}}
        if self.pre_sampling:   # Pre-sampling
            if self.endogenous_supply:
                if node_degrees_key is None:
                    raise ValueError("Node degrees key must be provided for endogenous supply.")
                self.samples["plants"] = self.samples_collection_supply[node_degrees_key]
            else:
                self.samples["plants"] = self.samples_collection_supply[None]

            if self.endogenous_demand:
                if zone_assignments_key is None:
                    raise ValueError("Zone assignments key must be provided for endogenous demand.")
                self.samples["products"] = self.samples_collection_demand[zone_assignments_key]
            else:
                self.samples["products"] = self.samples_collection_demand[None]
        else:   # Sampling on the fly
            if self.endogenous_supply:
                if node_degrees_key is None or node_degrees is None:
                    raise ValueError("Node degrees must be provided for endogenous supply.")
                supply_distribution = self._get_supply_distribution(node_degrees)
                self.samples["plants"] = self.get_supply_samples(supply_distribution, node_degrees_key)
            else:
                supply_distribution = self._get_supply_distribution()
                self.samples["plants"] = self.get_supply_samples(supply_distribution)

            if self.endogenous_demand:
                if zone_assignments_key is None or zone_assignments is None:
                    raise ValueError("Zone assignments must be provided for endogenous demand.")
                demand_distribution = self._get_demand_distribution(zone_assignments)
                self.samples["products"] = self.get_demand_samples(demand_distribution, zone_assignments_key)
            else:
                demand_distribution = self._get_demand_distribution()
                self.samples["products"] = self.get_demand_samples(demand_distribution)

    def _get_supply_distribution(self, node_degrees=None):
        supply_distribution = {}
        if self.endogenous_supply:  # Endogenous supply
            if node_degrees is None:
                raise ValueError("Node degrees must be provided for endogenous supply.")
            for plant in self.plants:
                info = {}
                if node_degrees[plant] == 0:
                    info["dist"] = "constant"
                    info["mean"] = self.supply_uncertainty[plant]["mean"]
                    supply_distribution[plant] = info
                else:
                    info["dist"] = self.supply_uncertainty[plant]["dist"]
                    plant_cv = math.sqrt(self.supply_uncertainty[plant]["var"]) / self.supply_uncertainty[plant]["mean"]
                    
                
                    if self.supply_uncertainty[plant]["mean_change"] == "linear":
                        info["mean"] = self.supply_uncertainty[plant]["mean"] * (1 - (node_degrees[plant]-1) * self.supply_uncertainty[plant]["mean_tilt"])       # For each additional product assigned to the plant, the mean supply decreases by a constant factor
                    elif self.supply_uncertainty[plant]["mean_change"] == "constant":
                        info["mean"] = self.supply_uncertainty[plant]["mean"]

                    info["mean"] = max(info["mean"], 0)  # Ensure the plant has a positive mean supply

                    if self.supply_uncertainty[plant]["var_change"] == "linear":
                        plant_cv = plant_cv * (1 + (node_degrees[plant]-1) * self.supply_uncertainty[plant]["var_tilt"])  # For each additional product assigned to the plant, the variance of supply increases by a constant factor
                        info["var"] = (info["mean"] * plant_cv) **2
                    elif self.supply_uncertainty[plant]["var_change"] == "constant":
                        info["var"] = (info["mean"] * plant_cv) **2

                    info["var"] = max(info["var"], 0)

                    supply_distribution[plant] = info
        else:   # Exogenous supply
            for plant in self.plants:
                info = {}
                info["dist"] = self.supply_uncertainty[plant]["dist"]
                info["mean"] = self.supply_uncertainty[plant]["mean"]
                info["var"] = self.supply_uncertainty[plant]["var"]            
                supply_distribution[plant] = info
        return supply_distribution          

    def _get_demand_distribution(self, zone_assignments=None):
        demand_distribution = {}
        if self.endogenous_demand:  # Endogenous demand
            if zone_assignments is None:
                raise ValueError("Zone assignments must be provided for endogenous demand.")
            for product in self.products:
                info = {}
                if not zone_assignments[product]:
                    info["dist"] = "constant"
                    info["mean"] = self.demand_uncertainty[product]["mean"]
                else:
                    info["dist"] = self.demand_uncertainty[product]["dist"]
                    effective_tariff = self.get_effective_tariff(product, zone_assignments[product])
                    product_cv = math.sqrt(self.demand_uncertainty[product]["var"]) / self.demand_uncertainty[product]["mean"]
                    
                    if self.demand_uncertainty[product]["mean_change"] == "linear":
                        # Assuming a linear demand curve
                        info["mean"] = self.demand_uncertainty[product]["mean"] * (1 - effective_tariff * self.demand_uncertainty[product]["mean_tilt"])
                    if self.demand_uncertainty[product]["mean_change"] == "constant":
                        info["mean"] = self.demand_uncertainty[product]["mean"]
                    
                    info["mean"] = max(info["mean"], 0)  # Ensure mean is non-negative

                    if self.demand_uncertainty[product]["var_change"] == "linear":
                        product_cv =  product_cv * (1 + effective_tariff * self.demand_uncertainty[product]["var_tilt"])
                        info["var"] = (info["mean"] * product_cv) **2
                    if self.demand_uncertainty[product]["var_change"] == "constant":
                        info["var"] = (info["mean"] * product_cv) **2
                    
                    info["var"] = max(info["var"], 0)  # Ensure variance is non-negative
                demand_distribution[product] = info
        else:   # Exogenous demand
            for product in self.products:
                info = {}
                info["dist"] = self.demand_uncertainty[product]["dist"]
                info["mean"] = self.demand_uncertainty[product]["mean"]
                info["var"] = self.demand_uncertainty[product]["var"]
                demand_distribution[product] = info
        return demand_distribution
    
    def _get_splitted_samples(self):
        supply_vectors = [
            {plant: self.samples["plants"][plant][i] for plant in self.nodes["plants"]}
            for i in range(self.sample_size)
        ]
        demand_vectors = [
            {product: self.samples["products"][product][i] for product in self.nodes["products"]}
            for i in range(self.sample_size)
        ]

        supply_chunks = np.array_split(supply_vectors, self.number_of_processes)
        demand_chunks = np.array_split(demand_vectors, self.number_of_processes)

        return supply_chunks, demand_chunks

    @staticmethod
    def worker(args):
        supply_chunk, demand_chunk, nodes, edges, profits, processing_times = args
        network_flow = MaxFlow(nodes, profits, processing_times, env)
        results = []
        for scenario in range(len(supply_chunk)):
            second_stage_profit, dual_solution, flow_values = network_flow.maximize(
                supply=supply_chunk[scenario],
                demand=demand_chunk[scenario],
                edges=edges
            )
            results.append((second_stage_profit, dual_solution, flow_values))
        return results

    def is_feasible(self, edges):
        """
            - Check if the current solution of the RMP is feasible.
        """
        supply_chunks, demand_chunks = self._get_splitted_samples()
        input_args = []
        for chunk_number in range(len(supply_chunks)):
            input_args.append((supply_chunks[chunk_number], demand_chunks[chunk_number], self.nodes, edges, self.profits, self.processing_times))
        chunk_results = self.executor.map(self.__class__.worker, input_args)
        
        self.samples["dual_solution"] = []
        expected_second_stage_profit = 0    
        for chunk_result in chunk_results:
            for result in chunk_result:
                second_stage_profit, dual_solution, _ = result
                self.samples["dual_solution"].append(dual_solution)
                expected_second_stage_profit += second_stage_profit
        expected_second_stage_profit /= self.sample_size

        if self.sol_mu > expected_second_stage_profit + self.TOL:
            return False, expected_second_stage_profit
        else:
            return True, None

    def add_dist_cuts(self, node_degrees, zone_assignments, model):
        """
            - Add distribution-specific optimality cuts to the RMP.
        """

        expr_degrees = 0
        if self.endogenous_supply:
            expr_degrees = gp.LinExpr([1] * len(self.plants), [self.degree_assignments[(plant, node_degrees[plant])] for plant in self.plants])
            expr_degrees = (len(self.plants) - expr_degrees)

        expr_zone_assignments = 0
        if self.endogenous_demand:
            dummy_vector = []
            dummy_counter = 0
            for product in self.products:
                for zone in self.zones:
                    if zone not in zone_assignments[product]:
                        dummy_vector.append(1)
                    else:
                        dummy_vector.append(-1)
                        dummy_counter += 1
            expr_zone_assignments = gp.LinExpr(dummy_vector, self.u_vars) + dummy_counter

        scenario_range = range(self.sample_size)
        right_hand_side = 0
        for plant in self.plants:
            term1 = np.array([self.samples["plants"][plant][scenario] for scenario in scenario_range])
            term2 = np.array([self.samples["dual_solution"][scenario][("alpha", plant)] for scenario in scenario_range])
            right_hand_side += np.dot(term1, term2)
        for product in self.products:
            term1 = np.array([self.samples["products"][product][scenario] for scenario in scenario_range])
            term2 = np.array([self.samples["dual_solution"][scenario][("beta", product)] for scenario in scenario_range])
            right_hand_side += np.dot(term1, term2)
        
        edge_coefficients = []
        for plant in self.plants:
            for product in self.products:
                coefficients1 = np.array(
                    [min(
                        self.samples["plants"][plant][scenario]/self.processing_times[plant][product], 
                        self.samples["products"][product][scenario]
                        ) for scenario in scenario_range]
                    )
                coefficients2 = np.array(
                    [self.samples["dual_solution"][scenario][("rho", plant, product)] 
                     for scenario in scenario_range]
                     )
                edge_coefficients.append(np.dot(coefficients1, coefficients2))
        edge_terms = gp.LinExpr(edge_coefficients, self.y_vars)  # sum_{i,j} min(\ksi^{c}_i, \ksi^{d}_j) * rho_{i,j} * y_{i,j}

        model.cbLazy(
            self.mu <= self.Big_M * (expr_zone_assignments + expr_degrees) + \
            (1/self.sample_size) * (
                right_hand_side + edge_terms
            )
        )       

        if self.log["lazy"]:
            self._log(
                f"Node Deg.: {node_degrees} Zone Assgn.: {zone_assignments}; Distribution cut: mu <= {self.Big_M} * ({expr_zone_assignments} + {expr_degrees}) + (1/{self.sample_size}) * ({right_hand_side} + {edge_terms})"
            )

    def add_jensen_cuts(self):
        self.model.addConstr(
            gp.quicksum(
                self.profits[plant][product] * 
                min(
                    self.supply_uncertainty[plant]["mean"]/self.processing_times[plant][product], 
                    self.demand_uncertainty[product]["mean"]
                ) *
                self.edges[(plant, product)]
                for plant in self.nodes["plants"] for product in self.nodes["products"]
            ) - self.mu >= 0,
            name="jensen_cut"
        )
        self.model.update()
      
    def add_ghost_scenario_cut(self, model, edges):
        _, dual_solution, _ = self.network_flow.maximize(
            supply={plant: self.supply_uncertainty[plant]["mean"] for plant in self.nodes["plants"]},
            demand={product: self.demand_uncertainty[product]["mean"] for product in self.nodes["products"]},
            edges=edges
        )
        
        right_hand_side = 0
        alpha_coefficients = np.array([self.supply_uncertainty[plant]["mean"] for plant in self.nodes["plants"]])
        alpha_values = np.array([dual_solution[("alpha", plant)] for plant in self.nodes["plants"]])
        right_hand_side += np.dot(alpha_coefficients, alpha_values)

        beta_coefficients = np.array([self.demand_uncertainty[product]["mean"] for product in self.nodes["products"]])
        beta_values = np.array([dual_solution[("beta", product)] for product in self.nodes["products"]])
        right_hand_side += np.dot(beta_coefficients, beta_values)

        edge_coefficients = []
        for plant in self.nodes["plants"]:
            for product in self.nodes["products"]:
                edge_coefficients.append(
                    min(
                        self.supply_uncertainty[plant]["mean"]/self.processing_times[plant][product], 
                        self.demand_uncertainty[product]["mean"]
                        ) * dual_solution[("rho", plant, product)]
                    )
        edge_terms = gp.LinExpr(edge_coefficients, self.y_vars)  # sum_{i,j} min(\ksi^{c}_i, \ksi^{d}_j) * rho_{i,j} * y_{i,j}
        model.cbLazy(
            self.mu <= right_hand_side + edge_terms
        )

        if self.log["lazy"]:
            self._log(
                f"Ghost scenario cut: mu <= {right_hand_side} + {edge_terms}"
            )

    def get_big_m(self):
        """
            - Calculate a valid Big M value for the problem.
        """
        total_profit = sum(
            self.profits[plant][product] * 
            min(
                self.supply_uncertainty[plant]["mean"]/self.processing_times[plant][product], 
                self.demand_uncertainty[product]["mean"]
            ) 
            for plant in self.nodes["plants"] for product in self.nodes["products"]
        )
        return total_profit

    def optimize(self):
        """
            - Optimize the network design problem using a callback function.
        """
        self.total_time = time.perf_counter()  # Start timer - measure total solution time
        
        self.get_rmp()

        if self.add_dominant_flow:
            self.add_dominant_flow_problem()

        if self.add_dominant_flow_sup:
            self.add_dominant_flow_supply_problem()
        
        if self.add_dominant_flow_dem:
            self.add_dominant_flow_demand_problem()

        # Add valid cuts to RMP
        if self.add_jensen:
            self.add_jensen_cuts()
        
        self.is_optimal = False
        # Implementation of Benders using callback functions:
        self.model.setParam('TimeLimit', self.time_limit)
        self.model.setParam("MIPGap", self.optimality_gap)
        self.model.setParam('Seed', self.seed if self.seed is not None else 0)
        self.model.Params.LazyConstraints = 1  # Allow lazy constraints
        self.model.optimize(lambda model, where: self.callback(model, where))
        self.total_time = time.perf_counter() - self.total_time  # Update the total time for solving the problem
        
        # Shut down the executor for parallel processing
        if not self.same_workers:
            self.executor.shutdown(wait=False, cancel_futures=True)
        
        if self.model.status == GRB.OPTIMAL:
            self.is_optimal = True
        elif self.model.SolCount == 0:
            return -self.Big_M, self.time_limit, sum(self.added_distribution_cuts.values()), self.is_optimal, getattr(self.model, "ObjBound", None), getattr(self.model, "MIPGap", None)
        
        self.sol_y = {key: value.X for key, value in self.edges.items()}
        if self.endogenous_demand:  
            self.sol_u = {key: var.X for key, var in self.zone_assignments.items()}
        node_degrees, zone_assignments, edges = self.get_design_info()
        self.obj_bound = getattr(self.model, "ObjBound", None)
        self.mip_gap = getattr(self.model, "MIPGap", None)
        
        self._generate_output(edges, node_degrees, zone_assignments)
        self._write_csv_summary(node_degrees, zone_assignments)

        return self.model.ObjVal, self.total_time, sum(self.added_distribution_cuts.values()), self.is_optimal, self.obj_bound, self.mip_gap
    
    def callback(self, model, where):
        """
            - Callback function is called during the optimization process.
            - Every time a new first-stage solution is found, the cbLazy method adds the lazy cuts to the RMP.
        """
        if where == GRB.Callback.MIPSOL:
            
            # best_obj   = model.cbGet(GRB.Callback.MIPSOL_OBJBST )
            # best_bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            # runtime    = model.cbGet(GRB.Callback.RUNTIME)

            current_values = model.cbGetSolution(self.varlist)
            nY, nU = len(self.y_vars), len(self.u_vars)
            self.sol_y  = dict(zip(self.y_keys, current_values[:nY]))
            if self.endogenous_demand:
                self.sol_u  = dict(zip(self.u_keys,  current_values[nY:nY+nU]))
            self.sol_mu = current_values[-1]

            node_degrees, zone_assignments, edges = self.get_design_info()
            # print(node_degrees, zone_assignments, edges)
            if self.endogenous_supply:
                node_degrees_key = tuple(node_degrees[plant] for plant in self.plants)
            else:
                node_degrees_key = None
            if self.endogenous_demand:
                zone_assignments_key = tuple(tuple(sorted(zone_assignments[product])) for product in self.products)
            else:
                zone_assignments_key = None
            self.get_samples(node_degrees_key, node_degrees, zone_assignments_key, zone_assignments)

            if (node_degrees_key, zone_assignments_key) not in self.added_distribution_cuts.keys():
                self.added_distribution_cuts[(node_degrees_key, zone_assignments_key)] = 0
            if node_degrees_key not in self.visited_supply_distributions.keys():
                self.visited_supply_distributions[node_degrees_key] = 0
            if zone_assignments_key not in self.visited_demand_distributions.keys():
                self.visited_demand_distributions[zone_assignments_key] = 0


            time_before_solving_subproblem = time.perf_counter()  # Start timer to measure time spent on cuts
            is_feasible, expected_second_stage_profit = self.is_feasible(edges=edges)

            self.total_subproblem_time += time.perf_counter() - time_before_solving_subproblem
            # Update the total time spent on solving the second-stage problem
            if not is_feasible:
                self.added_distribution_cuts[(node_degrees_key, zone_assignments_key)] += 1
                self.visited_supply_distributions[node_degrees_key] += 1
                self.visited_demand_distributions[zone_assignments_key] += 1
                
                time_before_adding_cuts = time.perf_counter()
                self.add_dist_cuts(node_degrees, zone_assignments, model)
                if self.add_ghost_scenario:
                    self.add_ghost_scenario_cut(model, edges)
                self.total_subproblem_cut_time += time.perf_counter() - time_before_adding_cuts
                
                model.cbSetSolution(self.varlist, current_values[:-1] + [expected_second_stage_profit])
                model.cbUseSolution()
                                
    def get_all_supply_distributions(self):
        """
        - A helper function that generates all possible distributions of the problem.
        - For the case of endogenous supply uncertainty, that is all possible combinations of node degrees.
        - Warning: List size grows exponentially with the number of plants, i.e., |products|^|plants| combinations.
        """
        self.all_distributions = itertools.product(range(len(self.nodes["products"])+1), repeat=len(self.nodes["plants"]))
        self.all_degree_combinations = []
        for degree_comb in self.all_distributions:
            self.all_degree_combinations.append({plant: degree_comb[i] for i, plant in enumerate(self.nodes["plants"])})

    def get_all_demand_distributions(self):
        """
        - A helper function that generates all possible distributions of the problem.
        - For the case of endogenous demand uncertainty, that is all possible combinations of products to zone assignments.
        """
        product_subsets = [
            subset
            for r in range(0, self.len_products + 1)
            for subset in itertools.combinations(self.nodes["products"], r)
        ]
        self.all_distributions = itertools.product(product_subsets, repeat=len(self.zones))
        self.all_zone_assignments = []
        for item in self.all_distributions:
            zone_assignment = {product: [] for product in self.nodes["products"]}
            for zone_index, product_subset in enumerate(item):
                for product in product_subset:
                    zone_assignment[product].append(self.zones[zone_index])
            self.all_zone_assignments.append(zone_assignment)

    def get_effective_tariff(self, product, zone_assignments):
            """
                - Given a subset of zones assigned to a product, computes the effective tariff.
            """
            # For this paper, we assume the average tariff for sourcing from the assigned zones
            if not zone_assignments:
                return 0
            if len(zone_assignments) == 1:
                return self.product_tariffs[product][zone_assignments[0]]
            if self.combined_tariff_rate is None: 
                return sum(
                    self.product_tariffs[product][zone] for zone in zone_assignments
                ) / len(zone_assignments)
            else:
                return self.combined_tariff_rate
    
    def get_all_zone_combinations(self):
        """
        - Generate all possible zone assignment combinations for the products.
        """
        self.all_zone_combinations = [
            tuple(subset)
            for r in range(0, len(self.zones) + 1)
            for subset in itertools.combinations(self.zones, r)
        ]
           
    def _get_average_flow_values(self, edges=None, new_sample_size=False):
        """
            - Computes the average flow values across all scenarios.
            - If edges are provided, uses them; otherwise, retrieves design info from the optimal solution.
            - If new_sample_size is True, updates the sample size to evaluation_sample_size.
        """
        if new_sample_size and self.sample_size != self.evaluation_sample_size:         # Update sample size if provided
            # print(f"\033[1;34mUpdating sample size to {self.evaluation_sample_size}\033[0m")
            self.sample_size = self.evaluation_sample_size
        
        if not self.same_workers:
            # Start the executor for parallel processing again
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.number_of_processes)
        
        # Input preparation
        if edges is None:
            node_degrees, zone_assignments, edges = self.get_design_info()
        else:
            node_degrees, zone_assignments, edges = self.get_non_optimal_design_info(edges)
        
        if self.endogenous_supply:
            node_degrees_key = tuple(node_degrees[plant] for plant in self.plants)
        else:
            node_degrees_key = None
        if self.endogenous_demand:
            zone_assignments_key = tuple(tuple(sorted(zone_assignments[product])) for product in self.products)
        else:
            zone_assignments_key = None
        
        # Samples are always collected with respect to the optimal design
        self.get_samples(node_degrees_key, node_degrees, zone_assignments_key, zone_assignments)

        supply_chunks, demand_chunks = self._get_splitted_samples()
        input_args = []
        for chunk_number in range(len(supply_chunks)):
            input_args.append((supply_chunks[chunk_number], demand_chunks[chunk_number], self.nodes, edges, self.profits, self.processing_times))
        chunk_results = self.executor.map(self.__class__.worker, input_args)
        
        average_flow_values = {(plant, product): 0 for plant in self.nodes["plants"] for product in self.nodes["products"]}
        all_flow_values = []
        for chunk_result in chunk_results:
            for result in chunk_result:
                _, _, flow_values = result
                all_flow_values.append(flow_values)
        
        for plant in self.nodes["plants"]:
            for product in self.nodes["products"]:
                average_flow_values[(plant, product)] = sum(
                    flow_values[(plant, product)] for flow_values in all_flow_values
                ) / self.sample_size
                if (plant, product) not in edges and average_flow_values[(plant, product)] > self.TOL:
                    raise RuntimeError("Infeasible flow values detected in the optimal solution.")
                elif (plant, product) not in edges:
                    average_flow_values.pop((plant, product))
        if not self.same_workers:
            self.executor.shutdown(wait=False, cancel_futures=True)
        return average_flow_values        

    def _display_avg_flow_values(self):
        """
        - Display the average flow values with 2 decimal places.
        """
        avg_flow_values = {}
        
        for key, value in self._get_average_flow_values().items():
            avg_flow_values[key] = round(value, 2)
        return avg_flow_values
    
    def _generate_output(self, edges, node_degrees=None, zone_assignments=None):
        """
            - Creating the output log file based on the logging configuration.
        """
        if self.log["res"]:
            self._log_results()

        if self.log["solution"]:
            if self.endogenous_supply:
                node_degrees_key = tuple(node_degrees[plant] for plant in self.nodes["plants"])
            else:
                node_degrees_key = None
            if self.endogenous_demand:
                zone_assignments_key = tuple(tuple(sorted(zone_assignments[product])) for product in self.nodes["products"])
            else:
                zone_assignments_key = None

            self._log_solution(edges, node_degrees_key, zone_assignments_key)

        if self.log["vis_dis"]:
            self._log_vis_dis()

        if self.log["un_vis_dis"]:
            self._log_un_vis_dis()

        self._log_write()
    
    def _log(self, message):
        """
        - This is a helper function to log messages with timestamps.
        """
        timestamp = datetime.datetime.now().strftime("%m/%d %H:%M:%S")
        self.log_entries.append(f"[{timestamp}] {message}")

    def _log_results(self):
        """
        Log the results of the optimization process.
        """
        # --- Log optimization and uncertainty configuration ---
        self._log(
            f"[INST] {self.instance_name:<12} | "
            f"[CFG] "
            f"S:{self._tf(self.endogenous_supply):<3} "
            f"D:{self._tf(self.endogenous_demand):<3} "
            f"J:{self._tf(self.add_jensen):<3} "
            f"G:{self._tf(self.add_ghost_scenario):<3} "
            f"DFP:{self._tf(self.add_dominant_flow):<3} "
            f"S-DFP:{self._tf(self.add_dominant_flow_sup):<3} "
            f"D-DFP:{self._tf(self.add_dominant_flow_dem):<3} | "
            f"[DATA] "
            f"Seed:{self.seed:<3} "
            f"Samp:{self.sample_size:<6} | "
        )

        # Total number of possible supply and demand distributions
        if self.endogenous_supply:
            if hasattr(self, "all_degree_combinations"):
                all_degree_combinations = len(self.all_degree_combinations)
            else:
                all_degree_combinations = (len(self.products) + 1) ** len(self.plants)
        else:
            all_degree_combinations = None  # not applicable

        if self.endogenous_demand:
            if hasattr(self, "all_zone_assignments"):
                all_zone_assignments = len(self.all_zone_assignments)
            else:
                all_zone_assignments = 2 ** (len(self.zones) * len(self.products))
        else:
            all_zone_assignments = None  # not applicable

        # Number of visited supply and demand distributions
        visited_supply_distributions = len(getattr(self, "visited_supply_distributions", []))
        visited_demand_distributions = len(getattr(self, "visited_demand_distributions", []))

        # Ratios of visited distributions
        ratio_vis_sup = (
            visited_supply_distributions / all_degree_combinations
            if (all_degree_combinations is not None and all_degree_combinations > 0)
            else float("nan")
        )

        ratio_vis_dem = (
            visited_demand_distributions / all_zone_assignments
            if (all_zone_assignments is not None and all_zone_assignments > 0)
            else float("nan")
        )

        # --- Log optimization results ---
        self._log(
            f"[Method]: L-Benders |"
            f"[OBJECTIVE RESULTS] "
            f"obj: {self.model.ObjVal:>8.3f} | "
            f"optimal: {self._tf(self.is_optimal):<5} | "
            f"upperBound: {self.obj_bound:<8.3f} |"
            f"mipGap: {self.mip_gap:<8.3f} | "
            f"[VISITED] "
            f"dist: {len(self.added_distribution_cuts):<8} |"
            f"sup dist: {len(self.visited_supply_distributions):<8} | "
            f"dem dist: {len(self.visited_demand_distributions):<8} | "
            f"ratio sup dist: {ratio_vis_sup:<8.3f} | "
            f"ratio dem dist: {ratio_vis_dem:<8.3f} | "
            f"[ITERATIONS & TIME] "
            f"itr.: {sum(self.added_distribution_cuts.values()):<5} | "
            f"time: {self.total_time:<8.3f} | "
            f"total 2nd-stg: {self.total_subproblem_time:<8.3f} | "
            f"cut 2nd-stg: {self.total_subproblem_cut_time:<8.3f} | "
            f"mixed_tariff_rate: {self.combined_tariff_rate if self.combined_tariff_rate is not None else float('nan'):<8.3f}"
        )

        # --- Log solver settings ---
        self._log(
            f"[SOLVER] "
            f"TimeLimit:{self.time_limit} | "
            f"MIPGap:{self.optimality_gap} | "
            f"Seed:{self.seed if self.seed is not None else 0} | "
            f"LazyConstraints:1 | "
            f"Threads:{self.model.Params.Threads} | "
            f"Heuristics:{self.model.Params.Heuristics} | "
            f"IntFeasTol:{self.model.Params.IntFeasTol} | "
            f"Pre sampling:{self._tf(self.pre_sampling)}"
        )

    def _log_solution(self, edges, node_degrees_key=None, zone_assignments_key=None):
        self._log(f"Optimal Design: {edges}")
        self._log(f"Average Flow Values: {self._display_avg_flow_values()}")
        self._log(f"Objective Value: {round(self.model.ObjVal, 2)}")
        if self.endogenous_supply and self.endogenous_demand:
            self._log(f"Optimal Node Degrees: {node_degrees_key}")
            self._log(f"Optimal Zone Assignment: {zone_assignments_key}")
            self._log(f"# of Cuts: {self.added_distribution_cuts.get((node_degrees_key, zone_assignments_key), 0)}")
        elif self.endogenous_supply:
            self._log(f"Optimal Node Degrees: {node_degrees_key}")
            self._log(f"# of Cuts: {self.added_distribution_cuts.get((node_degrees_key, None), 0)}")
        elif self.endogenous_demand:
            self._log(f"Optimal Zone Assignment: {zone_assignments_key}")
            self._log(f"# of Cuts: {self.added_distribution_cuts.get((None, zone_assignments_key), 0)}")
        else:
            self._log(f"# of Cuts: {self.added_distribution_cuts.get((None, None), 0)}")

    def _log_vis_dis(self):
        self._log(f"List of distributions and cuts per each is as follows:")
        for k, v in sorted(self.added_distribution_cuts.items(), key=lambda item: item[1], reverse=True):
                self._log(f"Distribution: {k},\t # of Cuts: {v}")

    def _log_un_vis_dis(self):
        self._log(f"List of unvisited distributions is as follows:")
        visited_supply_keys = set(self.visited_supply_distributions.keys()) 
        if self.endogenous_supply:
            for degree_comb in self.all_degree_combinations:
                degree_key = tuple(degree_comb[plant] for plant in self.plants)
                if visited_supply_keys is None or degree_key not in visited_supply_keys:
                    self._log(f"Unvisited Supply Distribution: {degree_key}")
        if self.endogenous_demand:
            visited_demand_keys = set(self.visited_demand_distributions.keys()) 
            for zone_assignment in self.all_zone_assignments:
                zone_key = tuple(tuple(sorted(zone_assignment[product])) for product in self.products)
                if visited_demand_keys is None or zone_key not in visited_demand_keys:
                    self._log(f"Unvisited Demand Distribution: {zone_key}") 

    def _tf(self, flag):
        return "T" if flag else "F"

    def _log_write(self):
        # --- Write log entries to file ---
        self._log(f"-"*50 + "\n")
        path = self.output_path / f"{self.instance_name}_{'s' if self.endogenous_supply else 'ex'}_{'d' if self.endogenous_demand else 'ex'}.log"
        with open(path, "a") as f:
            for entry in self.log_entries:
                f.write(entry + "\n")
                
    def _write_csv_summary(self, node_degrees=None,  zone_assignments=None):
        """
            - Write results to a csv file.
        """
        # timestamp
        ts = datetime.datetime.now().strftime("%m/%d %H:%M:%S")

        # Total number of possible supply and demand distributions
        if self.endogenous_supply:
            if hasattr(self, "all_degree_combinations"):
                all_degree_combinations = len(self.all_degree_combinations)
            else:
                all_degree_combinations = (len(self.products) + 1) ** len(self.plants)
        else:
            all_degree_combinations = None  # not applicable

        if self.endogenous_demand:
            if hasattr(self, "all_zone_assignments"):
                all_zone_assignments = len(self.all_zone_assignments)
            else:
                all_zone_assignments = 2 ** (len(self.zones) * len(self.products))
        else:
            all_zone_assignments = None  # not applicable

        # Number of visited supply and demand distributions
        visited_supply_distributions = len(getattr(self, "visited_supply_distributions", []))
        visited_demand_distributions = len(getattr(self, "visited_demand_distributions", []))

        # Ratios of visited distributions
        ratio_vis_sup = (
            visited_supply_distributions / all_degree_combinations
            if (all_degree_combinations is not None and all_degree_combinations > 0)
            else float("nan")
        )

        ratio_vis_dem = (
            visited_demand_distributions / all_zone_assignments
            if (all_zone_assignments is not None and all_zone_assignments > 0)
            else float("nan")
        )

        row = {
            "method": "L-Benders",
            "timestamp": ts,
            "instance": self.instance_name,
            "sup": self._tf(self.endogenous_supply),
            "dem": self._tf(self.endogenous_demand),
            "J": self._tf(self.add_jensen),
            "G": self._tf(self.add_ghost_scenario),
            "DFP": self._tf(self.add_dominant_flow),
            "DFP-S": self._tf(self.add_dominant_flow_sup),
            "DFP-D": self._tf(self.add_dominant_flow_dem),
            "seed": self.seed,
            "sample": self.sample_size,
            "obj": self.model.ObjVal,
            "is_optimal": self.is_optimal,
            "bound": self.obj_bound,
            "mip_gap": self.mip_gap,
            "vis_dist": len(self.added_distribution_cuts),
            "vis_sup_dits": len(self.visited_supply_distributions),
            "vis_dem_dits": len(self.visited_demand_distributions),
            "ratio_vis_sup": ratio_vis_sup,
            "ratio_vis_dem": ratio_vis_dem,
            "mixed_tariff_rate": self.combined_tariff_rate if self.combined_tariff_rate is not None else float("nan"),
            "itr": sum(self.added_distribution_cuts.values()),
            "time": self.total_time,
            "second_stage_comp": self.total_subproblem_time,
            "second_stage_cut": self.total_subproblem_cut_time,
            "node_degrees": node_degrees if self.endogenous_supply else None,
            "zone_assignments": zone_assignments if self.endogenous_demand else None,
            "avg flow": self._display_avg_flow_values()
        }

        path = self.output_path / "benders.csv"
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)