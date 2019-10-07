import torch
import torch.nn as nn

from forecast import multi_steps_forecasting, plot_multisteps_forecasting
from gru_model import GRUModel
from my_dataloader import MyDataset
from trainer import fit

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)

batch_size = 100
n_iters = 12000
window_size = 60

train_dataset = MyDataset(window_size,'data/monthly-beer-production-in-austr.csv','Monthly beer production')
test_dataset = MyDataset(window_size,'data/monthly-beer-production-in-austr.csv','Monthly beer production')

num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

input_dim = 1
hidden_dim = 256
layer_dim = 1  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 1

model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)

if torch.cuda.is_available():
    model.cuda()

criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters())

seq_dim = window_size

fit(num_epochs, optimizer, test_loader, train_loader, model, criterion, input_dim, seq_dim)
model.load_state_dict(torch.load("gru_weight.pt"))
model = model.eval()

n_th_sample = 18
steps = 10

sample, true_prediction = train_dataset[n_th_sample]
sample1, _ = train_dataset[n_th_sample + steps]
scale = train_dataset.get_scale()
multi_step_prediction_results = multi_steps_forecasting( model, steps, sample)
plot_multisteps_forecasting(scale,multi_step_prediction_results, sample1)
