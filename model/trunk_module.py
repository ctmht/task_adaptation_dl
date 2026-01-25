from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrunkModule(nn.Module):
	_activation_class = {
		'relu': nn.ReLU,
		'sigmoid': nn.Sigmoid,
		'tanh': nn.Tanh
	}
	
	def __init__(
		self,
		n_features: int,
		hidden_dims: list[int] | None = None,
		activation: Literal["relu", "sigmoid", "tanh"] = "relu",
		dropout: float = 0.0,
	):
		"""
		Initialize a trunk module capable of performing Monte Carlo Dropout
		
		Args:
			n_features (int): number of input neurons, required
			hidden_dims (list[int]): hidden layer sizes. If None or empty,
				this will correspond to a linear regression with activation
			activation (Literal["relu", "sigmoid", "tanh"]): the nonlinearity
				which will be used on all layers in the network besides the
				output heads
			dropout (float): probability of randomly dropping a neuron's
				activations in the forward pass
		"""
		super(TrunkModule, self).__init__()
		
		self._mcdropout: bool = False
		
		self.dropout = dropout
		
		self.activation_class = self._activation_class[activation]
		
		hidden_dims = [] if hidden_dims is None else hidden_dims
		hidden_dims.insert(0, n_features)
		
		self.linears = nn.ModuleList([
			nn.Sequential(
				nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]),
				self.activation_class()
			)
			
			for idx in range(len(hidden_dims) - 1)
		])
		
		self.mu_head = nn.Sequential(
			nn.Linear(hidden_dims[-1], 1),
			nn.Identity()   # TODO: change as necessary
		)
		
		
		self.sigma_head = nn.Sequential(
			nn.Linear(hidden_dims[-1], 1),
			nn.Softplus()   # TODO: change as necessary
		)
		
		self.init_weights()
	
	
	def init_weights(
		self
	) -> None:
		...
	
	
	def forward(
		self,
		data: torch.Tensor
	):
		"""
		Forward pass of the model
		
		Args:
			data (torch.Tensor): the input data to the model, which is expected
				to have tensor shape (... , n_features)
		"""
		out = data
		
		for idx, layseq in enumerate(self.linears):
			out = layseq(out)
			
			out = F.dropout(
				out,
				p = self.dropout,
				training = self.training or self._mcdropout,
				inplace = False
			)
		
		mus = self.mu_head(out)
		sigmas = self.sigma_head(out)
		
		out = torch.cat([mus, sigmas], dim = -1)
		
		return out
		
	
	
	def train(
		self,
		mode: bool = True
	) -> None:
		"""
		Docstring for train
		
		:param mode: Description
		:type mode: bool
		"""
		self._mcdropout = False
		super(TrunkModule, self).train(mode = mode)
	
	
	def eval_mcdropout(
		self
	) -> None:
		"""
		Docstring for eval_mcdropout
		"""
		self._mcdropout = True
		super(TrunkModule, self).eval()



if __name__ == '__main__':
	BATCH_SIZE = 10
	NUM_PREDICTORS = 50
	N_FEATURES = 9
	
	model = TrunkModule(
		n_features = N_FEATURES,
		hidden_dims = [32, 16],
		activation = "tanh",
		dropout = 0.3
	)
	
	dummy_in = torch.rand(BATCH_SIZE, NUM_PREDICTORS, N_FEATURES)
	dummy_out = model(dummy_in)
	
	print(dummy_in.shape)
	print(dummy_out.shape)