import numpy as np

from graph_env.env.utils.core import Agent


# OLSRv1 MPR computation https://www.rfc-editor.org/rfc/rfc3626.html
def mpr_heuristic(
        agent: Agent
):
    """
    OLSRv1 MPR computation as before.
    This is invoked for scripted agents in World.step().
    """
    one_hop_neighbours_ids = agent.one_hop_neighbours_ids
    two_hop_neighbours_ids = agent.two_hop_neighbours_ids
    local_view = agent.local_view
    agent_id = agent.id

    two_hop_neighbours_ids = two_hop_neighbours_ids - one_hop_neighbours_ids
    d_y = dict()
    two_hop_coverage_summary = {
        index: [] for index, value in enumerate(two_hop_neighbours_ids) if value == 1
    }
    covered = np.zeros(len(one_hop_neighbours_ids))
    mpr = np.zeros(len(one_hop_neighbours_ids))

    for neighbor in local_view.neighbors(agent_id):
        clean_neighbor_neighbors = local_view.nodes[neighbor]['one_hop'].copy()
        # Exclude from the list the 2-hop neighbours already reachable by self
        clean_neighbor_neighbors[agent_id] = 0
        for index in np.where(one_hop_neighbours_ids)[0]:
            clean_neighbor_neighbors[index] = 0
        # Calculate the coverage for two hop neighbors
        for index in [i for i, x in enumerate(two_hop_neighbours_ids.astype(
                int) & clean_neighbor_neighbors.astype(int)) if x == 1]:
            two_hop_coverage_summary[index].append(
                int(local_view.nodes[neighbor]['label']))
        d_y[int(local_view.nodes[neighbor]['label'])] = sum(
            clean_neighbor_neighbors)

    # Add to covered if that neighbor is the unique provider
    for key, candidates in two_hop_coverage_summary.items():
        if len(candidates) == 1:
            mpr[candidates[0]] = 1
            covered[key] = 1

    reachable_uncovered = np.array(
        [1 if (value_2h and not value_c) else 0 for value_2h, value_c in
         zip(two_hop_neighbours_ids, covered)])
    while (reachable_uncovered != 0).any():
        reachability = dict()
        for neighbor in local_view.neighbors(agent_id):
            reachability[int(local_view.nodes[neighbor]['label'])] = sum(
                local_view.nodes[neighbor]['one_hop'].astype(int) & reachable_uncovered.astype(int))
        max_reachability = [k for k, v in reachability.items() if v == max(reachability.values())]
        if len(max_reachability) == 1:
            key_to_add = max_reachability[0]
            mpr[key_to_add] = 1
            reachable_uncovered = np.array([
                0 if (value_n and value_unc) else value_unc for
                value_n, value_unc in
                zip(local_view.nodes[key_to_add]['one_hop'], reachable_uncovered)])

        elif len(max_reachability) > 1:
            key_to_add = max({k: d_y[k] for k in max_reachability})
            mpr[key_to_add] = 1
            reachable_uncovered = np.array([
                0 if (value_n and value_unc) else value_unc for
                value_n, value_unc in
                zip(local_view.nodes[key_to_add]['one_hop'], reachable_uncovered)])

    return mpr
