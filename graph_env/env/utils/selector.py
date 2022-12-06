from pettingzoo.utils import agent_selector


# UNUSED To be finished
class MprSelector:
    """Outputs an agent in the given order whenever agent_select is called.

    Can reinitialize to a new order
    """
    def __init__(self, agent_order, agents):
        self.reinit(agent_order, agents)

    def reinit(self, agent_order, agents):
        self.agent_order = agent_order
        self.agents = agents
        self._current_agent = -1
        self.selected_agent = 0
        self.selectable = [agent for agent in agents if (agent.has_received_from_relayed_node() or agent.state.message_origin) and not sum(agent.state.transmitted_to)]

    def reset(self):
        self.reinit(self.agent_order, self.agents)
        return self.next()

    def next(self):
        if len(self.selectable):
            self._current_agent = (self._current_agent + 1) % len(self.selectable)
            self.selected_agent = self.selectable[self._current_agent].name

            return self.selected_agent
        else:
            return None

    def is_last(self):
        """Does not work as expected if you change the order."""
        return self.selected_agent == self.selectable[-1].name

    def is_first(self):
        return self.selected_agent == self.selectable[0].name

    def __eq__(self, other):
        if not isinstance(other, MprSelector):
            return NotImplemented

        return (
            self.agent_order == other.agent_order
            and self._current_agent == other._current_agent
            and self.selected_agent == other.selected_agent
        )

