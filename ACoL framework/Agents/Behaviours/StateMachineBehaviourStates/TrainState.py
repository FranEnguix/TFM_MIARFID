import codecs
import pickle
import random

from spade.behaviour import State
from termcolor import colored

import Config


class TrainState(State):
    """
    State in which the agent trains its model
    """

    def consensus(self, msg):
        if self.agent.weights is not None and msg.body.split("|")[0] != "None":
            # Process message
            weights_and_losses = msg.body.split("|")
            unpickled_neighbour_weights = pickle.loads(codecs.decode(weights_and_losses[0].encode(), "base64"))
            unpickled_neighbour_losses = pickle.loads(codecs.decode(weights_and_losses[1].encode(), "base64"))
            neighbour_max_order = int(weights_and_losses[2])
            # print(neighbour_max_order)
            if self.agent.max_order < neighbour_max_order:
                self.agent.max_order = neighbour_max_order

            unpickled_local_weights = pickle.loads(codecs.decode(self.agent.weights.encode(), "base64"))
            # print(unpickled_neighbour_weights[0]['layer_input.weight'])
            # print(unpickled_local_weights)

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
            print(colored("[{}] Applied consensus after training and updated model".format(self.agent.name), 'red'))

    async def deep_learning(self):
        """
        Train the local model, update the weights and losses
        """
        training_id = random.randint(0, 1000)
        print("Training ID : {}".format(training_id))
        local_weights, local_losses, self.agent.train_acc, self.agent.train_loss, self.agent.test_acc, self.agent.test_loss = await self.agent.federated_learning.train_local_model()
        print("Got weights of training {}".format(training_id))
        self.agent.training_logger.write_to_file(
            "{},{},{},{}".format(self.agent.train_acc, self.agent.train_loss, self.agent.test_acc,
                                 self.agent.test_loss))
        # print(local_weights)
        # federated_learning.average_all_weights(local_weights, local_losses)
        # federated_learning.get_acc()

        self.agent.weights = codecs.encode(pickle.dumps(local_weights), "base64").decode()
        self.agent.losses = codecs.encode(pickle.dumps(local_losses), "base64").decode()

        # unpickled_local_weights = pickle.loads(codecs.decode(self.agent.weights.encode(), "base64"))

        #self.agent.weight_logger.write_to_file(
        #    "TRAINING,{},{},{},{}".format(local_weights[0]['layer_input.weight'].numpy().flatten()[0],
        #                                  local_weights[0]['layer_input.bias'].numpy().flatten()[0],
        #                                  local_weights[0]['layer_hidden.weight'].numpy().flatten()[0],
        #                                  local_weights[0]['layer_hidden.bias'].numpy().flatten()[0]))

        # print(" --- Setting local weights and local losses")

        # self.set('local_weights', serial_local_weights)
        # self.set('local_losses', serial_local_losses)

    async def run(self):
        print("[{}] TRAINING".format(self.agent.name))
        if Config.max_training_iterations >= 0 and self.agent.iterations > Config.max_training_iterations:
            print(f"[{self.agent.name}] STOPPING. {self.agent.iterations} iterations reached.")
            await self.agent.stop_agent()
        self.agent.iterations = self.agent.iterations + 1
        print(f"[{self.agent.name}] iteration number: {self.agent.iterations} / {Config.max_training_iterations}")
        self.agent.training_time_logger.write_to_file("START")
        await self.deep_learning()
        self.agent.training_time_logger.write_to_file("STOP")
        self.agent.test_accuracies.append(self.agent.test_acc)
        self.agent.test_losses.append(self.agent.test_loss)
        self.agent.train_accuracies.append(self.agent.train_acc)
        self.agent.train_losses.append(self.agent.train_loss)
        if len(self.agent.pending_consensus_messages) > 0:
            for msg in self.agent.pending_consensus_messages:
                self.consensus(msg)
                self.agent.pending_consensus_messages.remove(msg)
        self.set_next_state(Config.SEND_STATE_AG)
