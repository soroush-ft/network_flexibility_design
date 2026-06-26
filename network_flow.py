import gurobipy as gp
from gurobipy import GRB

class MaxFlow:
    """
    - Class to solve a flow problem in a bipartite graph.
    """

    def __init__(self, nodes, profits, processing_times, env):
        """
        - Initialize the Flow class with edges, nodes, and profits.
        - Generate a Gurobi model for the maximum flow problem in a bipartite graph.
        """
        self.nodes = nodes
        self.env = env
        self.processing_times = processing_times
        
        model = gp.Model("max_flow", env=self.env)    

        # Adding flow variables
        self.flow = {
            (i,j): model.addVar(lb=0, name=f"x_{i},{j}")
            for i in self.nodes["plants"]
            for j in self.nodes["products"]
        }

        # Adding capacity constraints
        self.alpha = model.addConstrs(
            gp.quicksum(self.processing_times[i][j] * self.flow[i, j] for j in self.nodes["products"]) <= 0
            for i in self.nodes["plants"]
        )

        # Adding demand constraints
        self.beta = model.addConstrs(
            gp.quicksum(self.flow[(i, j)] for i in self.nodes["plants"]) <= 0
            for j in self.nodes["products"]
        )

        # Adding edge constraints
        self.rho = model.addConstrs(
            self.flow[i, j] <= 0  
            for i in self.nodes["plants"]
            for j in self.nodes["products"]
        )

        # Set the objective to maximize the total flow
        model.setObjective(
            gp.quicksum(profits[i][j] * self.flow[(i, j)] 
                        for i in self.nodes["plants"] 
                        for j in self.nodes["products"]), 
                        GRB.MAXIMIZE
        )
        model.update()

        self.model = model


    def maximize(self, supply, demand, edges):
        """
        - Maximize the flow for a bipartite graph given edges, supply, and demand realizations.
        """   
        self.model.setAttr(GRB.Attr.RHS, 
                           [self.alpha[i] for i in self.nodes["plants"]], 
                           [supply[i] for i in self.nodes["plants"]]
                           )
        
        self.model.setAttr(GRB.Attr.RHS, 
                           [self.beta[j] for j in self.nodes["products"]], 
                           [demand[j] for j in self.nodes["products"]]
                           )
        
        self.model.setAttr(GRB.Attr.RHS, 
                           [self.rho[(i, j)] for i in self.nodes["plants"] for j in self.nodes["products"]], 
                           [min(supply[i]/self.processing_times[i][j], demand[j]) if (i, j) in edges else 0 
                            for i in self.nodes["plants"] 
                            for j in self.nodes["products"]]
                            )
        
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            duals = {("alpha", i): self.alpha[i].Pi for i in self.nodes["plants"]}
            duals.update({("beta", j): self.beta[j].Pi for j in self.nodes["products"]})
            duals.update({("rho", i, j): self.rho[i,j].Pi for i in self.nodes["plants"] for j in self.nodes["products"]})
            flow_values = {(i, j): self.flow[(i, j)].X for i in self.nodes["plants"] for j in self.nodes["products"]}
            return self.model.objVal, duals, flow_values
        else:
            raise ValueError("No optimal solution found.")