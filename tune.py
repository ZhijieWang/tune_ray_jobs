import ray
from ray import tune
import torch
import torch.nn as nn
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import max_error
from sklearn.preprocessing import MinMaxScaler
import time
import os
ray.init()
flight_data = sns.load_dataset("flights")
all_data = flight_data['passengers'].values.astype(float)
test_data_size = 12
scaler = MinMaxScaler()
all_data_normalized = scaler.fit_transform(all_data .reshape(-1, 1))
train_data_normalized  = all_data_normalized[:-test_data_size]
test_data_normalized = all_data_normalized[-test_data_size:]
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)


all_data = flight_data['passengers'].values.astype(float)
test_data_size = 12
scaler = MinMaxScaler()
all_data_normalized = scaler.fit_transform(all_data .reshape(-1, 1))
train_data_normalized  = all_data_normalized[:-test_data_size]
test_data_normalized = all_data_normalized[-test_data_size:]

train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
train_window = 12
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def train(model, optimizer, loss_function, epochs):
    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
def test(model, test_input, truth):
    model.eval()
    for i in range(train_window):
        seq = torch.FloatTensor(test_input[-train_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
            test_input.append(model(seq).item())
    return max_error(test_input[train_window:],test_data_normalized)
def orchestrate(config):  
    epochs = 50
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    train(model, optimizer, loss_function, epochs)
    test_inputs = train_data_normalized[-train_window:].tolist()
    
    error = test(model, test_inputs, test_data_normalized)
    tune.report(error = error)
    # This saves the model to the trial directory
    torch.save(model.state_dict(), "./model.pth")
    print(error)


search_space = {
    "lr":  tune.grid_search([0.1,0.01,0.001,0.0001]),
}

analysis = tune.run(orchestrate, config=search_space)
print("Best config: ", analysis.get_best_config(
    metric="error", mode="min"))

best_trial = analysis.get_best_trial("error", "min", "last")
print(best_trial)

df = analysis.results_df
print(df.head())
best_model = LSTM() 

logdir = best_trial.logdir
state_dict = torch.load(os.path.join(logdir, "model.pth"))
best_model.load_state_dict(state_dict)

best_model.eval()
fut_pred = 12
test_inputs = train_data_normalized[-train_window:].tolist()
for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        best_model.hidden = (torch.zeros(1, 1, best_model.hidden_layer_size),
                        torch.zeros(1, 1, best_model.hidden_layer_size))
        test_inputs.append(best_model(seq).item())

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))

print(actual_predictions)

