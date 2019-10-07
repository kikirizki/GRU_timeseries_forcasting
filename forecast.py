import torch
import matplotlib.pyplot as plt
import numpy as np
def multi_steps_forecasting(model, steps, input_data):
    input_data = input_data.unsqueeze(0).unsqueeze(-1).cuda()
    result = []
    for i in range(steps):
        o = model.forward(input_data)
        input_data = torch.cat((input_data[:, 1:, :], o.unsqueeze(0).unsqueeze(-1)), 1)
        result.append(o.squeeze().detach().cpu().numpy())
    return np.array(result)


def plot_multisteps_forecasting(scale,predictions, real_data):
    print(predictions)
    real_data = real_data.squeeze().squeeze(-1).detach().cpu().numpy()
    real_data_size = len(real_data)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(scale * real_data, 'b',label="Data sebenarnya")
    ax.plot(np.arange(real_data_size - 10, real_data_size + len(predictions) - 10), scale * predictions, 'r--',label="Prediksi")
    ax.legend()
    plt.show()
