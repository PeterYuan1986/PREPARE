import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary


class Net(nn.Module):
    def __init__(self, in_model, out_model,  dropout_prob=0.1):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_model, 512),
            # Layer normalization after Linear layer
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),       # Dropout after ReLU
            nn.Linear(512, 1028),
            nn.ReLU(),
            nn.Linear(1028, 2056),
            nn.ReLU(),
            nn.Dropout(dropout_prob),       # Dropout after ReLU
            nn.Linear(2056, 1028),
            nn.ReLU(),
            nn.Dropout(dropout_prob),       # Dropout after ReLU
            nn.Linear(1028, 512),
            nn.ReLU(),
            nn.Linear(512, out_model),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        # Apply sigmoid to each of the 343 outputs
        # x = self.softmax(x)
        # x = torch.round(x)
        # Sum along the feature dimension
        return x


class TransformerBinaryClassifier(nn.Module):
    def __init__(self, input_dim=169, output_dim=384, d_model=128, nhead=4, num_layers=5):
        super(TransformerBinaryClassifier, self).__init__()

        # Expand the input tensor from [batch_size, 169] to [batch_size, 169, d_model] with an embedding layer
        # Each element in the sequence goes from [batch_size, 169, 1] to [batch_size, 169, d_model]
        self.embedding = nn.Linear(1, d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        # Linear layer to transform to output shape [batch_size, 384]
        self.fc_out = nn.Linear(input_dim * d_model, output_dim)

        # Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch_size, 169]

        # Reshape input to [batch_size, 169, 1] to embed each sequence element separately
        x = x.unsqueeze(-1)  # Shape: [batch_size, 169, 1]

        # Expand the input to shape [batch_size, 169, d_model]
        x = self.embedding(x)  # Shape: [batch_size, 169, d_model]

        # Transformer expects input shape [sequence_length, batch_size, d_model]
        x = x.permute(1, 0, 2)  # Reshape to [169, batch_size, d_model]

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # Shape: [169, batch_size, d_model]

        # Flatten and pass through the fully connected layer
        # Reshape to [batch_size, 169 * d_model]
        x = x.permute(1, 0, 2).reshape(x.size(1), -1)

        # Output layer to get [batch_size, 384]
        x = self.fc_out(x)  # Shape: [batch_size, 384]
        # Apply sigmoid for binary output
        x = self.sigmoid(x)
        # x = torch.round(x)
        # Sum along the feature dimension
        return torch.sum(x, dim=1)


# # Initialize your model
# model = TransformerBinaryClassifier()

# # Define the input shape [batch_size, 169]
# batch_size = 32
# input_shape = (batch_size, 169)

# # Print the model summary
# summary(model, input_size=input_shape)
