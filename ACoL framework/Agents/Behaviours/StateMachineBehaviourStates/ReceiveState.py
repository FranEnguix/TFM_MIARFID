import codecs
import datetime
import pickle

from spade.behaviour import State
from termcolor import colored

from Utilities.MultipartHandler import MultipartHandler
import Config


class ReceiveState(State):
    """
    State waiting for a response after the agent sends his weights to another agent.
    """

    def __init__(self):
        super().__init__()
        self.multipart_handler = MultipartHandler()

    def consensus(self, msg):
        if self.agent.weights is not None and msg.body.split("|")[0] != "None" and not msg.body.startswith("I don't"):
            # Process message
            weights_and_losses = msg.body.split("|")
            neighbour_max_order = int(weights_and_losses[2])
            print(f"[RECV-fsm] Consensus message: {weights_and_losses[0][:5]}...{weights_and_losses[0][-5:]} weights, {weights_and_losses[1][:5]}...{weights_and_losses[1][-5:]} losses, {neighbour_max_order} max order")
            unpickled_neighbour_weights = pickle.loads(codecs.decode(weights_and_losses[0].encode(), "base64"))
            unpickled_neighbour_losses = pickle.loads(codecs.decode(weights_and_losses[1].encode(), "base64"))
            # print(neighbour_max_order)
            if self.agent.max_order < neighbour_max_order:
                self.agent.max_order = neighbour_max_order

            unpickled_local_weights = pickle.loads(codecs.decode(self.agent.weights.encode(), "base64"))

            # Apply consensus and update model
            consensus_weights = self.agent.consensus.apply_consensus(unpickled_local_weights,
                                                                     unpickled_neighbour_weights,
                                                                     1 / self.agent.max_order)

            #self.agent.weight_logger.write_to_file(
            #    "CONSENSUS,{},{},{},{}".format(consensus_weights[0]['layer_input.weight'].numpy().flatten()[0],
            #                                   consensus_weights[0]['layer_input.bias'].numpy().flatten()[0],
            #                                   consensus_weights[0]['layer_hidden.weight'].numpy().flatten()[0],
            #                                   consensus_weights[0]['layer_hidden.bias'].numpy().flatten()[0]))

            self.agent.federated_learning.add_new_local_weight_local_losses(consensus_weights[0],
                                                                            unpickled_neighbour_losses)
            self.agent.federated_learning.set_model()

            # Update agent properties
            self.agent.weights = codecs.encode(pickle.dumps(consensus_weights), "base64").decode()
            self.agent.losses = codecs.encode(pickle.dumps(unpickled_neighbour_losses), "base64").decode()
            print(colored("[{}] Applied consensus after response and updated model".format(self.agent.name), 'red'))

    # def is_multipart(self, message) -> bool:
    #     return message.body.startswith('multipart#')
    
    # def any_multipart_waiting(self) -> bool:
    #     return len(self.multipart_message.keys()) > 0

    # def is_multipart_complete(self, message) -> bool:
    #     sender = str(message.sender).split("@")[0]
    #     if not sender in self.multipart_message.keys():
    #         return None
    #     for part in self.multipart_message[sender]:
    #         if part is None:
    #             return False
    #     return True

    # def rebuild_multipart(self, message):
    #     # multipart_meta = message.get_metadata("multipart")
    #     multipart_meta = message.body.split('|')[0]
    #     if multipart_meta.startswith('multipart'):
    #         multipart_meta_parts = multipart_meta.split('#')[1]
    #         part_number = int(multipart_meta_parts.split("/")[0])
    #         total_parts = int(multipart_meta_parts.split("/")[1])
    #         sender = str(message.sender).split("@")[0]
    #         if not sender in self.multipart_message.keys():
    #             self.multipart_message[sender] = [None] * total_parts
    #         self.multipart_message[sender][part_number - 1] = message.body[len(multipart_meta + '|'):]
    #         if self.is_multipart_complete(message):
    #             body = ""
    #             for part in self.multipart_message[sender]:
    #                 body += part
    #             del self.multipart_message[sender]
    #             message.body = body
    #             return message
    #     return None

    async def run(self):
        print("[{}] RECEIVE".format(self.agent.name))
        msg = await self.receive(timeout=Config.max_seconds_timeout_response)

        if msg:
            t1 = datetime.datetime.now()
            self.agent.message_history.insert(0, "{}:{}:{} : Received response from {}".format(str(t1.hour),
                                                                                               str(t1.minute),
                                                                                               str(t1.second),
                                                                                               str(msg.sender).split("/")[0]))
            print(colored("[{}] Received response from {}".format(self.agent.name, msg.sender), 'cyan'))
            self.agent.message_logger.write_to_file(
                "RECEIVE_RESPONSE,{},{}".format(msg.get_metadata("message_id"), msg.sender))
            if str(msg.sender).split("@")[0] in self.agent.message_statistics:
                self.agent.message_statistics[str(msg.sender).split("@")[0]]["receive"] += 1
            else:
                self.agent.message_statistics[str(msg.sender).split("@")[0]] = {"send": 0, "receive": 1}


            # multipart = self.rebuild_multipart(message=msg)
            multipart = self.multipart_handler.rebuild_multipart(msg)
            if multipart is not None:
                msg = multipart

            if not self.multipart_handler.is_multipart(msg) or multipart is not None:
                # Apply consensus and update the local model with new weights
                self.consensus(msg)

            if self.multipart_handler.any_multipart_waiting():
                self.set_next_state(Config.RECEIVE_STATE_AG)
            else:
                self.set_next_state(Config.TRAIN_STATE_AG)

        else:
            print("[{}] No response was received".format(self.agent.name))
            self.set_next_state(Config.TRAIN_STATE_AG)
