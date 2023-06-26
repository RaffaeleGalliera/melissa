class CustomSelector:
    """Outputs an agent in the given order whenever agent_select is called.

    Can reinitialize to a new order
    """
    def __init__(self, agents):
        self.reinit(agents)

    def reinit(self, agents):
        self.agents = {agent: {
            "steps": 0,
            "active": False,
            "selected_round": False
        } for agent in agents}
        self._current_agent = 0
        self.selected_agent = 0

    def reset(self):
        self.reinit(self.agents)
        return self.next()

    def selectables(self):
        return [key for key, values in self.agents.items() if values["active"] and not values["selected_round"]]

    def next(self):
        self.selectable = self.selectables()
        if len(self.selectable):
            self.selected_agent = self.agents[self.selectable[0]]
            self.selected_agent['steps'] += 1
            self.selected_agent['selected_round'] = True

            return self.selectable[0]
        else:
            return False

    def disable(self, agent):
        self.agents[agent]["active"] = False

    def enable(self, agents):
        for agent in agents:
            self.agents[agent]["active"] = True if self.agents[agent]["steps"] < 4 else False

    def start_new_round(self):
        for key, values in self.agents.items():
            values["selected_round"] = False

    def is_last(self):
        """Does not work as expected if you change the order."""
        return True if not len(self.selectables()) else False
