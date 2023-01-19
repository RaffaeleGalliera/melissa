class CustomSelector:
    """Outputs an agent in the given order whenever agent_select is called.

    Can reinitialize to a new order
    """
    def __init__(self, agents):
        self.reinit(agents)

    def reinit(self, agents):
        self.agents = agents
        self._current_agent = 0
        self.selected_agent = 0
        self.selectable = [agent for agent in self.agents if (sum(agent.state.received_from) or agent.state.message_origin) and not sum(agent.state.transmitted_to)]
        self.agent_order = [agent.name for agent in self.selectable]

    def reset(self):
        self.reinit(self.agents)
        return self.next()

    def reset(self):
        self.reinit(self.agents)
        return self.next()

    def next(self):
        self._current_agent = (self._current_agent + 1) % len(self.agent_order)
        self.selected_agent = self.agent_order[self._current_agent - 1]
        return self.selected_agent

    def is_last(self):
        """Does not work as expected if you change the order."""
        return self.selected_agent == self.agent_order[-1]

    def is_first(self):
        return self.selected_agent == self.agent_order[0]

    def __eq__(self, other):
        if not isinstance(other, CustomSelector):
            return NotImplemented

        return (
                self.agent_order == other.agent_order
                and self._current_agent == other._current_agent
                and self.selected_agent == other.selected_agent
        )
