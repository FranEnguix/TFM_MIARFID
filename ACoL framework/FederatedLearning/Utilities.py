import codecs
import pickle
import sys
import copy
import torch
import numpy as np
from sklearn import metrics
import Config
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from FederatedLearning.Sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from FederatedLearning.Sampling import cifar_iid, cifar_noniid


class Utilities:
    def get_coalition_index(self, agent_name: str):
        coalition_index = -1
        for i, coalition in enumerate(Config.coalitions):
            if agent_name in coalition:
                coalition_index = i
                break
        return coalition_index

    def get_cifar4_datasets(self, root_folder: str, cats: int = 0, dogs: int = 0, deers: int = 0, horses: int = 0, tst_cats: int = 0, tst_dogs: int = 0, tst_deers: int = 0, tst_horses: int = 0):
        trainset = datasets.CIFAR10(root=root_folder, train=True, download=True, transform=transforms.ToTensor())
        testset = datasets.CIFAR10(root=root_folder, train=False, download=True, transform=transforms.ToTensor())

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        class_indices = [classes.index(cls) for cls in ['cat', 'dog', 'deer', 'horse']]

        def filter_classes(dataset, class_indices, class_counts):
            class_indices = set(class_indices)
            class_count = {k: 0 for k in class_indices}
            indices = []
            for i, (_, label) in enumerate(dataset):
                if label in class_indices and class_count[label] < class_counts.get(label, 0):
                    indices.append(i)
                    class_count[label] += 1
            dataset.data = dataset.data[indices]
            dataset.targets = np.array(dataset.targets, dtype='int64')[indices]

        filter_classes(trainset, class_indices, {classes.index('cat'): cats, classes.index('dog'): dogs, classes.index('deer'): deers, classes.index('horse'): horses})
        filter_classes(testset, class_indices, {classes.index('cat'): tst_cats, classes.index('dog'): tst_dogs, classes.index('deer'): tst_deers, classes.index('horse'): tst_horses})

        def remap_labels(dataset, class_indices):
            class_indices = {cls: i for i, cls in enumerate(class_indices)}
            dataset.targets = np.array([class_indices[label] for label in dataset.targets], dtype='int64')

        remap_labels(trainset, class_indices)
        remap_labels(testset, class_indices)

        return trainset, testset


    def get_dataset(self, args, dataset):
        """ Returns train and test datasets and a user group which is a dict where
        the keys are the user index and the values are the corresponding data for
        each of those users.
        """
        train_dataset = ""
        test_dataset = ""
        user_groups = ""

        print("get_dataset args:", args)

        if dataset.startswith('cifar4_coal'):
            data_dir = Config.data_set_path + '/cifar4/'
            agent_name = dataset.split('|')[1]
            coalition_idx = self.get_coalition_index(agent_name)
            train_data = Config.cifar4_dataset_distribution["train"][coalition_idx]
            test_data = Config.cifar4_dataset_distribution["test"][coalition_idx]
            
            if Config.coalition_probability == -1 and not Config.cifar4_dataset_acol_mimics_acoal_distribution: # ACoL
                train_data = Config.cifar4_dataset_distribution["train"][-1]
                test_data = Config.cifar4_dataset_distribution["test"][-1]

            print(f"CIFAR4 number of train samples => cat(0):{train_data['cats']}, dog(1):{train_data['dogs']}, deer(2):{train_data['deers']}, horse(3):{train_data['horses']}")
            print(f"CIFAR4 number of test samples => cat(0):{test_data['tst_cats']}, dog(1):{test_data['tst_dogs']}, deer(2):{test_data['tst_deers']}, horse(3):{test_data['tst_horses']}")
            train_dataset, test_dataset = self.get_cifar4_datasets(data_dir, **train_data, **test_data)
            user_groups = mnist_iid(train_dataset, args.num_users)

        elif dataset == 'cifar':
            data_dir = Config.data_set_path + '/cifar/'
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)

            # sample training data amongst users
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = cifar_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    raise NotImplementedError()
                else:
                    # Chose euqal splits for every user
                    user_groups = cifar_noniid(train_dataset, args.num_users)

        elif dataset == 'mnist' or 'fmnist':
            if dataset == 'mnist':
                data_dir = Config.data_set_path + '/mnist/'
            else:
                data_dir = Config.data_set_path + '/fmnist/'

            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

            if dataset == 'mnist':
                train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                               transform=apply_transform)

                test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                              transform=apply_transform)

            else:
                train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                      transform=apply_transform)

                test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                                     transform=apply_transform)

            # sample training data amongst users
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = mnist_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose unequal splits for every user
                    user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                else:
                    # Chose equal splits for every user
                    user_groups = mnist_noniid(train_dataset, args.num_users)

        # print("user_groups:", user_groups)
        return train_dataset, test_dataset, user_groups

    def plot_confusion_matrix(self, labels, pred_labels):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        cm = metrics.confusion_matrix(labels, pred_labels)
        cm = metrics.ConfusionMatrixDisplay(cm, display_labels=range(10))
        cm.plot(values_format='d', cmap='Blues', ax=ax)

    def average_weights(self, w):
        """
        Returns the average of the weights.
        """
        w_avg = []
        for local_weights in w:
            unpickled_local_weights = pickle.loads(codecs.decode(local_weights.encode(), "base64"))
            w_avg = copy.deepcopy(unpickled_local_weights[0])
            for key in w_avg.keys():
                for i in range(1, len(unpickled_local_weights)):
                    w_avg[key] += unpickled_local_weights[i][key]
                w_avg[key] = torch.div(w_avg[key], len(unpickled_local_weights))

        return w_avg

    def exp_details(self, args):
        print('\nExperimental details:')
        print(f'    Model     : {args.model}')
        print(f'    Optimizer : {args.optimizer}')
        print(f'    Learning  : {args.lr}')
        print(f'    Global Rounds   : {args.epochs}\n')

        print('    Federated parameters:')
        if args.iid:
            print('    IID')
        else:
            print('    Non-IID')
        print(f'    Fraction of users  : {args.frac}')
        print(f'    Local Batch size   : {args.local_bs}')
        print(f'    Local Epochs       : {args.local_ep}\n')
        return
