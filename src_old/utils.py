from math import ceil


def balanced_budget(G, costsNode):
    totalCost = 0
    for v in G.Nodes():
        node_id = v.GetId()
        c = costsNode[node_id]
        totalCost += c
    budget = ceil(totalCost / 2)
    return budget