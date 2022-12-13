from gurobipy import *
import json
import numpy as np

model = Model('Steelcase_MIP')

file = open('dummy_data.json')
data = json.load(file)

############################################
######### PARAMETERS #######################
# Number of Supplier i
i = len(data["Suppliers"])

# Number of Material types
j = len(data["Material Types"])

# Fixed cose of a order of material j from supplier i. (i x j array)
f = np.array(data["Fixed Cost"])

# Variable cost of a unit of material j from supplier i (i x j array)
v = np.array(data["Variable Cost"])

# Cost of each late day for material j by supplier i (i x j array)
d = np.array(data["Lateness Cost"])

# E[lateness] of material j by supplier i (i x j array)
l = np.array(data["Expected Lateness"])

# Minimum order quantity of material j from supplier i (i x j array)
m = np.array(data["Minimum Order Quantity"])

# Demand for material j (1 x j arraay)
d = np.array(data["Demand"])

# Minimum supplier diversity for material j (1 x j)
n = np.array(data["Minimum Supplier Diversity"])

# some large value to allow linearity
M = 100000

############################################
############################################


# %% Variables
x = model.addVars(i, j, lb = 0, vtype = 'I')
y = model.addVars(i, j, vtype = 'B')

# %% Constraints
model.addConstrs((x[a,b] <= M*y[a,b] for a in range(i)
                                     for b in range(j)), name = 'Linearize Binary Constraint')

model.addConstrs((quicksum(x[a,b] for a in range(i)) >= d[b] for b in range(j)), name = 'Demand')

model.addConstrs((quicksum(y[a,b] for a in range(i)) >= n[b] for b in range(j)), name = 'Material Sourcing Diversity')

model.addConstrs((x[a,b] >= m[a,b]*y[a,b] for a in range(i)
                                        for b in range(j)),
                                        name = "MOQ")

## Potential for additional constraints:
##  - Maximum Order Quantity
##  - Any ad hoc contract stipulations


# %% Objective
model.setObjective(quicksum(f[a][b] * y[a,b] + v[a][b] * x[a,b] + d[b] * l[a][b] * y[a,b] for b in range(j) for a in range(i)), GRB.MINIMIZE)

model.optimize()

# %% Custom print

print("Objective value: " + str(model.objVal))

sup = data['Suppliers']
mat = data['Material Types']
for b in range(j):
    for a in range(i):
        print(f"X({sup[a]},{mat[b]}) = " + str(x[a,b].x))

