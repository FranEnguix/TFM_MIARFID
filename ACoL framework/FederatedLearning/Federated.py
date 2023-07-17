import copy
import os
import time

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from termcolor import colored

from FederatedLearning.Utilities import Utilities
from FederatedLearning.Models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCustom, CIFAR4Model
from FederatedLearning.Options import args_parser
from FederatedLearning.Update import LocalUpdate


class Federated:

    def __init__(self, agent_name, model_path, dataset, model_type):
        self.global_weights = None
        self.agent_name = agent_name
        self.model_path = model_path
        self.dataset = dataset
        self.model_type = model_type
        # ------------------------------------------------------------------------------------------------------------------
        # Training
        self.end_time = None
        self.model = None
        self.train_loss, self.train_accuracy = [], []
        self.val_acc_list, self.net_list = [], []
        self.cv_loss, self.cv_acc = [], []
        self.print_every = 2
        self.val_loss_pre, self.counter = 0, 0

        self.start_time = time.time()
        self.best_valid_loss = 0
        # ------------------------------------------------------------------------------------------------------------------
        # define paths
        self.path_project = os.path.abspath('../..')
        self.logger = SummaryWriter('../logs')

        self.args = args_parser()
        print(colored('=' * 30, 'green'))
        print(self.args)
        print(colored('=' * 30, 'green'))

        # exp_details(self.args)

        self.utilities = Utilities()
        # Define the different parameters to configure the training
        if self.args.gpu:
            torch.cuda.set_device(self.args.gpu)
        self.device = 'cuda' if self.args.gpu else 'cpu'

        # load dataset and user groups
        self.train_dataset, self.test_dataset, self.user_groups = self.utilities.get_dataset(self.args, self.dataset)

    def build_model(self):
        # BUILD MODEL
        if self.model_type == 'cnn':
            # Convolutional neural network
            if self.dataset == 'mnist':
                self.model = CNNMnist(args=self.args)
            elif self.dataset == 'fmnist':
                self.model = CNNFashion_Mnist(args=self.args)
            elif self.dataset == 'cifar':
                self.model = CNNCifar(args=self.args)
            elif self.dataset.startswith('cifar4_coal'):
                self.model = CIFAR4Model()

        elif self.model_type == 'mlp':
            # Multi-layer perceptron
            img_size = self.train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
            print(img_size)
            if self.dataset.startswith('cifar4_coal'):
                self.model = MLP(dim_in=len_in, dim_hidden=32, dim_out=4)
            else:
                self.model = MLP(dim_in=len_in, dim_hidden=32, dim_out=self.args.num_classes)

        elif self.dataset == 'custom':
            self.model = CNNCustom(args=self.args)

        else:
            exit('Error: unrecognized model')

        if self.model_path is not None:
            print("Using model provided in file " + self.model_path)
            self.model.load_state_dict(torch.load(self.model_path))
            print(self.model.state_dict()['layer_input.weight'].numpy().flatten()[0])

        print(f"Selected model {self.model_type} and dataset {self.dataset}")
        print(f"Number of parameters: {self.count_parameters(self.model)}")

    def count_parameters(self, model): 
        return sum(p.numel() for p in model.parameters() if p.requires_grad) 
    
    def get_model(self):
        return self.model

    def set_model(self):
        # ------------------------------------------------------------------------------------------------------------------
        # Set the model to train and send it to device.
        self.model.to(self.device)
        # ------------------------------------------------------------------------------------------------------------------
        # copy weights
        self.global_weights = self.model.state_dict()
        # print(self.model.state_dict())

    def print_model(self):
        print(self.model)

    """
    def train_global_model(self):
        # Train global model
        self.global_model.train()
    """

    async def train_local_model(self, epoch=1):
        self.start_time = time.monotonic()
        local_weights, local_losses = [], []

        # TRAINING
        local_update = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                   idxs=self.user_groups[0], logger=self.logger)
        self.model, w, loss = await local_update.update_weights(
            model=copy.deepcopy(self.model), global_round=epoch)
        # import code
        # code.interact(local=locals())
        torch.save(self.model.state_dict(), f"Saved Models/model.pt")
        print("Saving model")

        local_weights.append(copy.deepcopy(w))
        local_losses.append(copy.deepcopy(loss))
        # print(local_weights[0])
        #print(local_weights[0]['layer_input.weight'].numpy().flatten()[0])

        # Get accuracy
        train_acc, train_loss, test_acc, test_loss = self.get_accuracy(local_update)

        self.end_time = time.monotonic()

        return local_weights, local_losses, train_acc, train_loss, test_acc, test_loss

    def get_accuracy(self, local_update):
        # Calculate avg training accuracy over all users at every epoch
        self.model.eval()

        acc, loss = local_update.inference(model=self.model)

        print("[{}] Train Accuracy : {}%".format(self.agent_name, round(acc*100, 2)))
        print("[{}] Train Loss : {}".format(self.agent_name, round(loss, 4)))

        self.train_accuracy.append(acc)
        self.train_loss.append(loss)

        # Test inference after completion of training
        test_acc, test_loss = local_update.test_inference(self.args, self.model, self.test_dataset)

        # print(f' \n Results after {self.args.epochs} global rounds of training:')
        print("[{}] Test Accuracy: {}%".format(self.agent_name, round(test_acc*100, 2)))
        print("[{}] Test Loss: {}".format(self.agent_name, round(test_loss, 4)))
        # print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
        # print(colored('=' * 30, 'green'))

        # if test_loss < self.best_valid_loss:
        #    self.best_valid_loss = test_loss
        #    dateTimeObj = datetime.now()

            # torch.save(self.global_model.state_dict(),'../TorchModels/FL_' + str(dateTimeObj) + "AgentName: " +
            # AgName + '.pt')

        return [round(acc*100, 2), round(loss, 2), round(test_acc*100, 2), round(test_loss, 4)]

    def average_all_weights(self, local_weights, local_losses, verbose=False):

        # update global weights
        global_weights = self.utilities.average_weights(local_weights)

        # update global weights
        self.model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        self.train_loss.append(loss_avg)

    def add_new_local_weight_local_losses(self, local_weights, local_losses):
        # update global weights
        self.model.load_state_dict(local_weights)
        self.train_loss.append(local_losses)
        # print(local_weights)

    def get_predictions(self, model, iterator, device):

        model.eval()

        images = []
        labels = []
        probs = []

        with torch.no_grad():
            for (x, y) in iterator:
                x = x.to(device)

                y_pred, _ = model(x)

                y_prob = F.softmax(y_pred, dim=-1)
                top_pred = y_prob.argmax(1, keepdim=True)

                images.append(x.cpu())
                labels.append(y.cpu())
                probs.append(y_prob.cpu())

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        probs = torch.cat(probs, dim=0)

        return images, labels, probs
