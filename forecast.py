import torch
from torch import nn, optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Choose subset of parts to train
keep_parts = slice(10)

epochs = 500 # ctrl-c to stop early
lr = 0.001

seq_len = 50
batch_size = 2

# Increase hidden_dim depending on number of parts being trained
# since it encodes forecast for all parts
hidden_dim = 512
layers = 2

# How many weeks to forecast
predict_steps = 50
# Forecast starts immediately after this slice of data
predict_input = slice(-predict_steps//2)

array = pd.read_csv("order-data.csv", skiprows=[0]).to_numpy()[keep_parts]


# Uncomment below to test predicting sin waves to make sure everything is working...
# epochs = 10000
# lr = .001
# seq_len = 150
# batch_size = 1
# hidden_dim = 128
# layers = 2
# predict_steps = 100
# predict_input = slice(-seq_len, None)
# # Sequence needs to see at least a full wave length or results are bad
# array = np.array([[f'Wave {i}']+[np.sin(x + i*2*np.pi/2)*(i+1) for x in np.linspace(-1000, 1000, 10051)] for i in range(2)])


part_names = array.T[0]
floats = np.array(array.T[1:], dtype='float32')

# Normalize quantities to be between 0 - 1
min_ = floats.min(0)
max_ = floats.max(0)
normalize   = lambda values: (values - min_) / (max_ - min_)
unnormalize = lambda normalized: normalized * (max_ - min_) + min_

tensor = torch.from_numpy(normalize(floats))


def batch_data(items, seq_len, batch_size):
    num_batches = (len(items)-1) // (seq_len * batch_size)
    keep = num_batches*seq_len*batch_size
    discard = len(items) - keep - 1
    if discard: print(f"Discarding last {discard} items")
    features = items[:keep]   .view(batch_size, num_batches, seq_len, -1).transpose(0,1).transpose(1,2).to(device)
    targets  = items[1:keep+1].view(batch_size, num_batches, seq_len, -1).transpose(0,1).transpose(1,2).to(device)
    return [*zip(features, targets)]

# split = int(tensor.shape[0] * .8)
# train = batch_data(tensor[:split], seq_len, batch_size)
# valid = batch_data(tensor[split:], seq_len, batch_size)
train = valid = batch_data(tensor, seq_len, batch_size)


class RNN(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, dropout=.3)
        self.fc = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(.3)

    def forward(self, x, hidden):
        y, hidden = self.lstm(x, hidden)
        y = y.view(-1, hidden_dim)
        y = self.dropout(y)
        y = self.fc(y)
        return y.view(x.shape), hidden


model = RNN(tensor.shape[-1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

try:
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        hidden = None
        for x, target in train:
            if hidden:
                hidden = tuple(h.detach() for h in hidden)

            y, hidden = model(x, hidden)
            loss = criterion(y, target)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            hidden = None
            for x, target in valid:
                y, hidden = model(x, hidden)
                loss = criterion(y, target)
                valid_loss += loss.item()

        print(f"Epoc {epoch}",
              f"train_loss={round(train_loss*1e5/len(train))}",
              f"valid_loss={round(valid_loss*1e5/len(valid))}")

except KeyboardInterrupt as e:
    pass


print('predicting...')
input = tensor[predict_input]
sequence = batch_data(input, len(input)-1, 1)
model.eval()
with torch.no_grad():
    x, _ = sequence[0]
    y, hidden = model(x, None)
    y = y[-1:, :, :]
    predict = [y[0,0,:]]
    for i in range(predict_steps):
        y, hidden = model(y, hidden)
        predict.append(y[0,0,:])

actual = tensor[predict_input.start:, :]
actual = unnormalize(actual.numpy())
predict = torch.stack(predict)
predict = unnormalize(predict.cpu().numpy())

for p in range(actual.shape[1]):
    plt.title(part_names[p])
    act = actual[:, p]
    plt.plot(range(len(act)), act)
    pred = predict[:, p]
    plt.plot(range(len(x), len(x) + len(pred)), pred, linestyle='--')
    plt.show()

# Show all curves on single chart
# import matplotlib.colors as mcolors
# colors = iter(mcolors.BASE_COLORS)
# for p in range(tensor.shape[1]):
#     color = next(colors)
#     actual = tensor[predict_input.start:,p]
#     plt.plot(range(len(actual)), actual, label=parts[p], color=color)
#     pred = [qty[p] for qty in predict]
#     plt.plot(range(len(x), len(x) + len(pred)), pred, color=color, linestyle='--')
# 
# plt.legend()
# plt.show()