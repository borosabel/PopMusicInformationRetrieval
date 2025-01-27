import torch.nn as nn
import torch.nn.functional as F
import torch as th

class GunshotDetectionCNN(nn.Module):
    def __init__(self, num_frames):
        super(GunshotDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=(3, 7))
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1))

        dummy_input = th.zeros(1, 3, 80, num_frames)
        dummy_output = self.pool2(F.relu(self.conv2(self.pool1(F.relu(self.conv1(dummy_input))))))
        output_size = dummy_output.view(-1).shape[0]

        self.fc1 = nn.Linear(output_size, 256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

# class GunshotDetectionCNN(nn.Module):
#     def __init__(self, num_frames, n_mels):
#         super(GunshotDetectionCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=(3, 7))
#         self.pool1 = nn.MaxPool2d(kernel_size=(3, 1))
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=(3, 3))
#         self.pool2 = nn.MaxPool2d(kernel_size=(3, 1))
#         self.dropout = nn.Dropout(0.5)
#         self.sigmoid = nn.Sigmoid()
#
#         # Update dummy_input with new dimensions
#         dummy_input = th.zeros(1, 3, n_mels, num_frames)
#         dummy_output = self.pool2(F.relu(self.conv2(self.pool1(F.relu(self.conv1(dummy_input))))))
#         output_size = dummy_output.view(-1).shape[0]
#
#         self.fc1 = nn.Linear(output_size, 256)
#         self.fc2 = nn.Linear(256, 1)
#
#     def forward(self, x):
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.sigmoid(self.fc2(x))
#         return x

# model = GunshotDetectionCNN(num_frames=utils.NUM_FRAMES)
# model = GunshotDetectionCNN(num_frames=utils.NUM_FRAMES, n_mels=utils.N_MELS)