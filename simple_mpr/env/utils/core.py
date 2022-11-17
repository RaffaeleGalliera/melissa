import numpy as np
import logging

from pettingzoo.mpe._mpe_utils.core import AgentState, Agent, World
from .constants import NUMBER_OF_AGENTS


class MprAgentState(AgentState):
    def __init__(self):
        super().__init__()
        self.id = 0
        self.received_from = np.zeros(NUMBER_OF_AGENTS)
        self.transmitted_to = np.zeros(NUMBER_OF_AGENTS)
        self.relays_for = np.zeros(NUMBER_OF_AGENTS)
        self.relayed_by = np.zeros(NUMBER_OF_AGENTS)
        self.message_origin = 0


class MprAgent(Agent):
    def __init__(self):
        super().__init__()
        # state
        self.state = MprAgentState()
        self.one_hop_neighbours_ids = None
        self.two_hop_neighbours_ids = None
        self.one_hop_neighbours_neighbours_ids = None
        self.allowed_actions = None

    def has_received_from_relayed_node(self):
        return sum([received and relay for received, relay in zip(self.state.received_from, self.state.relays_for)])


class MprWorld(World):
    # update state of the world
    def __init__(self):
        super(MprWorld, self).__init__()
        self.messages_transmitted = 0

    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # Set MPRs
        for agent in self.agents:
            self.set_relays(agent)

        # Send message
        for agent in self.agents:
            logging.debug(f"Agent {agent.name} Action: {agent.action.c} with Neigh: {agent.one_hop_neighbours_ids}")
            self.update_agent_state(agent)

    def set_relays(self, agent):
        if agent.action.c is not None:
            agent.state.relayed_by = agent.action.c
            neighbour_indices = [i for i, x in
                                 enumerate(agent.one_hop_neighbours_ids) if
                                 x == 1]
            relay_indices = [i for i, x in enumerate(agent.state.relayed_by)
                             if x == 1]
            # Assert relays are subset of one hope neighbours of the agent
            assert (set(relay_indices) <= set(neighbour_indices))
            for index, value in enumerate(agent.state.relayed_by):
                self.agents[index].state.relays_for[agent.id] = value

    def update_agent_state(self, agent):
        # if it has received from a relay node or is message origin and has not already transmitted the message
        if (agent.has_received_from_relayed_node() or agent.state.message_origin) and not sum(agent.state.transmitted_to):
            logging.debug(f"Agent {agent.name} sending to Neighs: {agent.one_hop_neighbours_ids}")

            agent.state.transmitted_to = agent.one_hop_neighbours_ids
            self.messages_transmitted += 1
            neighbour_indices = [i for i, x in enumerate(agent.one_hop_neighbours_ids) if x == 1]
            for index in neighbour_indices:
                self.agents[index].state.received_from[agent.id] = 1
        else:
            logging.debug(f"Agent {agent.name} could not send")
