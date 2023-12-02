from __future__ import absolute_import
from ..placements_helper.placements import Placements
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

#A climb matrix is a 36 x 36 matrix with 1-2 start holds, more than 2 middle holds, optional footholds, and at least one finish hold

#Model that will classify the difficulty of a climb based off it's matrix (hold placements)
class BoardClassificationModel(torch.nn.Module):
    def __init__(self):
        ...

#Model that will generate a climb matrix based off of X parameters (grade, angle, etc.)
class BoardGenerationModel(torch.nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_classes):
        super(BoardGenerationModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.emb = torch.nn.Embedding(vocab_size, emb_size)
        self.rnn = torch.nn.RNN(emb_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size * 2, num_classes)

    def forward(self, X):
        # Look up the embedding
        wordemb = self.emb(X)
        # Forward propagate the RNN
        h, out = self.rnn(wordemb)
        # combine the hidden features computed from *each* time step of
        # the RNN. we do this by 
        features = torch.cat([torch.amax(h, dim=1),
                              torch.mean(h, dim=1)], dim=-1)
        # Compute the final prediction
        z = self.fc(features)
        return z
    
def accuracy(model, dataset, max=1000):
    """
    Estimate the accuracy of `model` over the `dataset`.
    We will take the **most probable class**
    as the class predicted by the model.

    Parameters:
        `model`   - An object of class nn.Module
        `dataset` - A dataset of the same type as `train_data`.
        `max`     - The max number of samples to use to estimate 
                    model accuracy

    Returns: a floating-point value between 0 and 1.
    """

    correct, total = 0, 0
    dataloader = DataLoader(dataset,
                            batch_size=1,  # use batch size 1 to prevent padding
                            collate_fn=collate_batch)
    for i, (x, t) in enumerate(dataloader):
        z = model(x)
        y = torch.argmax(z, axis=1)
        correct += int(torch.sum(t == y))
        total   += 1
        if i >= max:
            break
    return correct / total

# accuracy(model, train_data_indices)

def train_model(model,                # an instance of MLPModel
                train_data,           # training data
                val_data,             # validation data
                learning_rate=0.001,
                batch_size=100,
                num_epochs=10,
                plot_every=50,        # how often (in # iterations) to track metrics
                plot=True):           # whether to plot the training curve
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               collate_fn=collate_batch,
                                               shuffle=True) # reshuffle minibatches every epoch
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # these lists will be used to track the training progress
    # and to plot the training curve
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    try:
        for e in range(num_epochs):
            for i, (texts, labels) in enumerate(train_loader):
                z = model(texts) # TODO
                loss = criterion(z, labels.long())# TODO

                loss.backward() # propagate the gradients
                optimizer.step() # update the parameters
                optimizer.zero_grad() # clean up accumualted gradients

                iter_count += 1
                if iter_count % plot_every == 0:
                    iters.append(iter_count)
                    ta = accuracy(model, train_data)
                    va = accuracy(model, val_data)
                    train_loss.append(float(loss))
                    train_acc.append(ta)
                    val_acc.append(va)
                    print(iter_count, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va)
    finally:
        # This try/finally block is to display the training curve
        # even if training is interrupted
        if plot:
            plt.figure()
            plt.plot(iters[:len(train_loss)], train_loss)
            plt.title("Loss over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")

            plt.figure()
            plt.plot(iters[:len(train_acc)], train_acc)
            plt.plot(iters[:len(val_acc)], val_acc)
            plt.title("Accuracy over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Accuracy")
            plt.legend(["Train", "Validation"])

# model = MyRNN(len(vocab), 128, 64, 2)

if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent.parent
    data_folder = root_dir / "data"

    placements = Placements(data_folder)
    # placements.plot_matrix_placements(placements.all_matrix_climbs)
