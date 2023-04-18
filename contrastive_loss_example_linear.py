import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio

# Set random seed for reproducibility
torch.manual_seed(0)

# Generate toy dataset
#mean1 = np.array([0, 0, 0, 0, 0])
#cov1 = np.eye(5)
#mean2 = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
#cov2 = np.eye(5)


mean1 = np.array([0, 0, 0])
cov1 = np.eye(3)
mean2 = np.array([1, 1, 1])
cov2 = np.eye(3)

data1 = np.random.multivariate_normal(mean1, cov1, 50)
data2 = np.random.multivariate_normal(mean2, cov2, 50)
data = np.vstack([data1, data2])
labels = np.hstack([np.zeros(50), np.ones(50)])
data = torch.from_numpy(data).float()
labels = torch.from_numpy(labels).long()

# Define linear projection model
class LinearProjection(nn.Module):
    def __init__(self):
        super(LinearProjection, self).__init__()
        #self.fc = nn.Linear(5, 2)
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        x = self.fc(x)
        return x

# Define contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, label1, label2):
        # Calculate the Euclidean distance between the pairs of feature vectors
        distance = torch.pow(x1 - x2, 2).sum(dim=1)
        distance = torch.sqrt(distance)

        # Create the mask
        mask = torch.eq(label1, label2).float()
                    
        # Calculate the contrastive loss
        loss_contrastive = (mask) * torch.pow(distance, 2) + (1-mask) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss_contrastive = 0.5 * loss_contrastive.mean()

        return loss_contrastive


# Define training parameters
learning_rate = 0.01
num_epochs = 100

# Instantiate model and loss function
model = LinearProjection()
criterion = ContrastiveLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# store figures and make gif
image_sequence = []

# Train the model
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output[:50], output[50:], labels[:50], labels[50:])
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # Project data onto 2D space and visualize
    with torch.no_grad():
        outputs = model(data)
        #plt.scatter(outputs[:50, 0], outputs[:50, 1], color='r')
        #plt.scatter(outputs[50:, 0], outputs[50:, 1], color='b')
        plt.title('Epoch %d' % (epoch + 1))
        sns.scatterplot(x=outputs[:, 0], y=outputs[:, 1], hue=labels, legend="full")
        plt.savefig('epoch_' + str(epoch + 1) + '.jpg')
        plt.close()

    # append the current plot to the image sequence
    current_image = imageio.imread('epoch_' + str(epoch + 1) + '.jpg')
    image_sequence.append(current_image)

# create a GIF of the image sequence
imageio.mimsave('animation.gif', image_sequence, fps=5)