from spade.behaviour import State

import Config


class SetupState(State):
    """
    State in which the State Machine is initialized.
    """

    async def run(self):
        print("[{}] SETUP".format(self.agent.name))

        # MOVED TO FLAgent init
        # Coalitions
        # coalition_index = -1
        # for i, coalition in enumerate(Config.coalitions):
        #     if self.agent.name in coalition:
        #         self.agent.coalition = [a for a in coalition if a != self.agent.name]
        #         coalition_index = i
        #         break
        # if coalition_index == -1:
        #     self.coalition = None
        #     print(f"{self.agent.name} has not a coalition.")
        # else:
        #     print(f"{self.agent.name} belongs to coalition {coalition_index}, with agents: {self.agent.coalition}")
            
        self.set_next_state(Config.TRAIN_STATE_AG)
