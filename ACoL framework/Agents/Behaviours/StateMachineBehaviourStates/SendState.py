import datetime
import random
import uuid

from spade.behaviour import State
from spade.message import Message

import Config
from Utilities.MultipartHandler import MultipartHandler

class SendState(State):
    """
    State in which the agent sends its weights to one of its neighbours.
    """

    def __init__(self):
        super().__init__()
        self.multipart_handler = MultipartHandler()

    def pick_agent(self, coalition: list[str], agents: list[str], coalition_probability: float) -> str:
        coalition_agents = [agent for agent in agents if agent.split("@")[0] in coalition]
        other_agents = [agent for agent in agents if agent.split("@")[0] not in coalition]
        random_prob = random.random()
        if coalition_agents and (random_prob < coalition_probability or len(other_agents) == 0):
            agent = random.choice(coalition_agents)
            print(f"[{self.agent.name}] picked the coalition agent {agent} with prob: {random_prob:.3f} (< {coalition_probability})")
            return agent
        agent = random.choice(other_agents)
        print(f"[{self.agent.name}] picked the agent {agent}")
        return agent       

    async def send_message(self, recipient):
        id = str(uuid.uuid4())
        msg = Message(to=recipient)  # Instantiate the message
        msg.set_metadata("conversation", "pre_consensus_data")
        msg.set_metadata("message_id", id)

        if recipient in self.agent.message_statistics:
            self.agent.message_statistics[recipient]["send"] += 1
        else:
            self.agent.message_statistics[recipient] = {"send": 1, "receive": 0}
        self.agent.message_logger.write_to_file("SEND,{},{}".format(id, recipient))

        local_weights = self.agent.weights
        local_losses = self.agent.losses

        if local_weights is None or local_losses is None:
            msg.body = "I don't have any weights yet"
            print("[{}] Sending message to {}".format(self.agent.name, recipient))
            msg.set_metadata("timestamp", str(datetime.datetime.now()))
            await self.send(msg)
        else:
            msg_local_weights = str(local_weights).strip()
            msg_local_losses = str(local_losses).strip()
            msg_max_order = str(round(self.agent.max_order, 3))
            content = msg_local_weights + "|" + msg_local_losses + "|" + msg_max_order
            print("[{}] Sending message to {}".format(self.agent.name, recipient))
            print(f"[SEND-fsm] Message length: {len(msg_local_weights)} weights + {len(msg_local_losses)} losses + {len(msg_max_order)} max order = {len(content)}")
            print(f"[SEND-fsm] Message content: {msg_local_weights[:5]}...{msg_local_weights[-5:]} weights, {msg_local_losses[:5]}...{msg_local_losses[-5:]} losses, {msg_max_order} max order")

            multipart_messages = self.multipart_handler.generate_multipart_messages(content, Config.max_message_body_length, msg)            
            if multipart_messages is not None:
                for i, message in enumerate(multipart_messages):
                    print(f"[SEND-fsm]  multipart message ({i + 1}/{len(multipart_messages)}) sent to {recipient}")
                    message.set_metadata("timestamp", str(datetime.datetime.now()))
                    await self.send(message)
            else:
                msg.body = content
                msg.set_metadata("timestamp", str(datetime.datetime.now()))
                await self.send(msg)



    async def run(self):
        print("Send")
        if len(self.agent.available_agents) > 0:
            print("[{}] SEND".format(self.agent.name))
            # receiving_agent_id = random.randint(0, len(self.agent.available_agents) - 1)
            receiving_agent = None
            if Config.coalition_probability >= 0:
                # ACoaL
                receiving_agent = self.pick_agent(coalition=self.agent.coalition, agents=self.agent.available_agents, coalition_probability=Config.coalition_probability)
            else:
                # ACoL
                receiving_agent = random.choice(self.agent.available_agents)

            # Send message
            print(f"receiving agent: {receiving_agent}")
            receiving_agent_name_domain = receiving_agent.split("/")[0]
            await self.send_message(receiving_agent_name_domain)         
            
            t1 = datetime.datetime.now()
            self.agent.message_history.insert(0, "{}:{}:{} : Sent message to {}".format(str(t1.hour), str(t1.minute),
                                                                                        str(t1.second),
                                                                                        receiving_agent_name_domain))
            self.set_next_state(Config.RECEIVE_STATE_AG)
        else:
            self.set_next_state(Config.TRAIN_STATE_AG)