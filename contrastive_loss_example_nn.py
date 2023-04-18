import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import imageio as im

# define the network with two hidden layers
class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


# define the contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label):
        # calculate pairwise distance
        distance = torch.nn.functional.pairwise_distance(output[0], output[1])
        
        # calculate contrastive loss
        loss_contrastive = torch.mean((label) * torch.pow(distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        
        return loss_contrastive


# set random seed for reproducibility
torch.manual_seed(0)

# define the toy dataset
mean1 = np.array([1, 1, 1, 1, 1])
mean2 = np.array([-1, -1, -1, -1, -1])
cov = np.eye(5)

data1 = np.random.multivariate_normal(mean1, cov, 100)
data2 = np.random.multivariate_normal(mean2, cov, 100)
data = np.concatenate([data1, data2], axis=0)

# convert numpy array to tensor
data_tensor = torch.from_numpy(data).float()

# define the labels
label = np.zeros(200)
label[:100] = 1
label = torch.from_numpy(label).float()

# store figures and make gif
image_sequence = []

# define the network and contrastive loss
net = Net(input_size=5, hidden_size=10)
criterion = ContrastiveLoss()

# define the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# training loop
epochs = 3000
batch_size = 128

for epoch in range(epochs):
    # randomly sample two batches from the data
    batch_indices = torch.randint(low=0, high=len(data), size=(2, batch_size))
    x1 = data_tensor[batch_indices[0]]
    x2 = data_tensor[batch_indices[1]]
    labels = (label[batch_indices[0]] == label[batch_indices[1]]).long()

    # forward pass
    output1 = net(x1)
    output2 = net(x2)
    output = torch.stack([output1, output2], dim=0)

    # calculate loss and update parameters
    optimizer.zero_grad()
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    # plot the 2-D embeddings every 100 epochs
    if epoch % 100 == 0:
        embed1 = net(data_tensor).detach().numpy()
        sns.scatterplot(x=embed1[:, 0], y=embed1[:, 1], hue=label, legend="full")
        plt.title(f"Epoch {epoch}")
        #plt.show()
        plt.savefig('epoch_' + str(epoch + 1) + '.jpg')
        plt.close()
        # append the current plot to the image sequence
        current_image = im.imread('epoch_' + str(epoch + 1) + '.jpg')
        image_sequence.append(current_image)

# create a GIF of the image sequence
im.mimsave('animation.gif', image_sequence, fps=5)