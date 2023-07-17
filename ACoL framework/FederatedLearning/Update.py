import asyncio

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        ###return image.clone().detach(), label.clone().detach()
        return torch.tensor(image), torch.tensor(label)


# This class updates the local weights

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.train_loader, self.valid_loader, self.test_loader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        train_loader = DataLoader(DatasetSplit(dataset, idxs_train),
                                  batch_size=self.args.local_bs, shuffle=True)
        valid_loader = DataLoader(DatasetSplit(dataset, idxs_val),
                                  batch_size=int(len(idxs_val) / 10), shuffle=False)
        test_loader = DataLoader(DatasetSplit(dataset, idxs_test),
                                 batch_size=int(len(idxs_test) / 10), shuffle=False)
        return train_loader, valid_loader, test_loader

    # Update the weight
    async def update_weights(self, model, global_round, verbose=False):
        # Set mode to train model
        model.train()
        epoch_loss = []
        optimizer = ""

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        # print(self.args.local_ep)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                await asyncio.sleep(0)
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                # import code
                # code.interact(local=locals())

                '''if self.args.verbose and (batch_idx % 10 == 0):
                    if verbose:
                        print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))'''

                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # print(epoch_loss)
        return model, model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(self.test_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            loss = self.criterion(outputs, labels)
            batch_loss.append(loss.item())

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, sum(batch_loss) / len(batch_loss)

    def test_inference(self, args, model, test_dataset):
        """ Returns the test accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        device = 'cuda' if args.gpu else 'cpu'
        criterion = nn.NLLLoss().to(device)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        batch_loss = []
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_loss.append(loss.item())

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, sum(batch_loss) / len(batch_loss)
