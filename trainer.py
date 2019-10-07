import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


def fit(num_epochs,  optimizer, test_loader, train_loader, model, criterion, input_dim, seq_dim):
    iter = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images.view(-1, seq_dim, input_dim))
                labels = Variable(labels.view(-1, 1).cuda())

            else:
                images = Variable(images.view(-1, seq_dim, input_dim))
                labels = Variable(labels.view(-1, 1))
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            # outputs.size() --> 100, 10
            images = images.to(torch.float).cuda()
            labels = labels.to(torch.float)

            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss

            loss = criterion(labels, outputs)
            if torch.cuda.is_available():
                loss.cuda()

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            # loss_list.append(loss.item())
            iter += 1

            if iter % 500 == 0:

                total = 0
                for images, labels in test_loader:

                    if torch.cuda.is_available():
                        images = Variable(images.view(-1, seq_dim, input_dim))
                        images = images.to(torch.float).cuda()
                        labels = labels.to(torch.float).cuda()
                    else:
                        images = Variable(images.view(-1, seq_dim, input_dim))

                    outputs = model(images)

                # Print Loss
                print('Iteration: {}. Loss: {}.'.format(iter, loss.item()))
    torch.save(model.state_dict(),"gru_weight.pt")
