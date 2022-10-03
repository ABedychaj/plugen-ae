import torch
from torch import nn


class AE_single_layer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=512
        )
        self.encoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["hidden_dim"]
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=kwargs["hidden_dim"], out_features=512
        )
        self.decoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.tanh(activation)
        code = self.encoder_output_layer(activation)
        code = torch.tanh(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.tanh(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.tanh(activation)
        return reconstructed, code


class AE_multiple_layers(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer_1 = nn.Linear(
            in_features=kwargs["input_shape"], out_features=512
        )
        self.encoder_hidden_layer_2 = nn.Linear(
            in_features=512, out_features=512
        )
        self.encoder_hidden_layer_3 = nn.Linear(
            in_features=512, out_features=512
        )
        self.encoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["hidden_dim"]
        )
        self.decoder_hidden_layer_1 = nn.Linear(
            in_features=kwargs["hidden_dim"], out_features=512
        )
        self.decoder_hidden_layer_2 = nn.Linear(
            in_features=512, out_features=512
        )
        self.decoder_hidden_layer_3 = nn.Linear(
            in_features=512, out_features=512
        )
        self.decoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer_1(features)
        activation = torch.relu(activation)
        activation = self.encoder_hidden_layer_2(activation)
        activation = torch.relu(activation)
        activation = self.encoder_hidden_layer_3(activation)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer_1(code)
        activation = torch.relu(activation)
        activation = self.decoder_hidden_layer_2(activation)
        activation = torch.relu(activation)
        activation = self.decoder_hidden_layer_3(activation)
        activation = torch.relu(activation)
        reconstructed = self.decoder_output_layer(activation)
        return reconstructed, code