from typing import Literal, Any

import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import tqdm

import matplotlib.pyplot as plt

from model.trunk_module import TrunkModule
from model.score_losses import *


NAME_TO_SCORE = {
    "gaussian_kernel": GaussianKernelScore,
    "gaussian_nll": NLL,
    "gaussian_se": SquaredError,
    "gaussian_crps": NormalCRPS,
}
RANDOM_SEED = 42


def _forwardpass_over_data(
	model,
	input_data,
	output_data,
	training: bool = False,
	**hyperparameters: dict[str, Any],
):
	"""
	
	"""
	# Generally used hyperparameters, and definition of the scores (here as metrics)
	batch_size = hyperparameters.get('batch_size', 64)
	num_epochs = hyperparameters.get('num_epochs', 10)
	
	l2reg_strength = hyperparameters.get('l2reg_strength', 0.1) # TODO: use
	
	reduction = hyperparameters.get('reduction', 'mean')
	ensemble = hyperparameters.get('ensemble', False)
	gaussker_gamma = hyperparameters.get('gaussker_gamma', 1.0)
	score_metric_objs = {
		name: val(
			reduction = reduction,
			ensemble = ensemble,
			loss_gk_gamma = gaussker_gamma
		) for name, val in NAME_TO_SCORE.items()
	}
	metrics = {
		name: [] for name in NAME_TO_SCORE.keys()
	}
	
	if not training:
		# Test hyperparameters
		use_mcdropout = hyperparameters.get('use_mcdropout', False)
		num_epochs = 1 # Force one epoch
		
		if use_mcdropout:
			model.eval_mcdropout()
		else:
			model.eval()
	else:
		# Training hyperparameters
		lr = hyperparameters.get('lr', 1e-4)
		
		# Score used as loss
		loss = hyperparameters['loss']
		score_loss = NAME_TO_SCORE[loss](
			reduction = reduction,
			ensemble = ensemble,
			loss_gk_gamma = gaussker_gamma
		)
		
		early_stopping = hyperparameters.get('early_stopping', False)
		if early_stopping:
			val_input_data = hyperparameters['val_input_data']
			val_output_data = hyperparameters['val_output_data']
		
		model.train()
		optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
	
	# Setup model device
	device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
	device = torch.device(device_name)
	model = model.to(device)
	
	# Convert to PyTorch tensors
	X_tensor = torch.from_numpy(input_data.copy().astype(np.float32))
	y_tensor = torch.from_numpy(output_data.copy().astype(np.float32))
	
	num_samples = len(X_tensor)
	indices = np.arange(num_samples)
	
	# Iterate through epochs
	epoch_losses = []
	for epoch in range(num_epochs):
		if training:
			np.random.shuffle(indices)
		
		scores = []
		metrics_perbatch = {
			name: [] for name in NAME_TO_SCORE.keys()
		}
		
		if training:
			print()
		
		# Iterate through batches
		for batch_start in tqdm.tqdm(
			range(0, num_samples, batch_size),
			desc = f"Epoch {epoch+1}" if training else "Testing"
		):
			# Prepare batch
			batch_end = min(batch_start + batch_size, num_samples)
			batch_indices = indices[batch_start:batch_end]
			
			batch_X = X_tensor[batch_indices].to(device)
			batch_y = y_tensor[batch_indices].to(device)
			
			if training:
				optimizer.zero_grad()
			
			# Forward pass
			pred_y = model(batch_X)
			
			if training:
				# Backward pass
				score = score_loss(pred_y, batch_y)
				
				score.backward()
				optimizer.step()
				
				scores.append(score.item())
			
			# Metrics
			# TODO: also add UQ metrics
			for name, score_metric in score_metric_objs.items():
				metric = score_metric(
					pred_y,
					batch_y
				)
				metrics_perbatch[name].append(metric.item())
		
		# Average metrics over all batches in the epoch and save in metrics
		for name in metrics.keys():
			metrics[name].append(np.mean(metrics_perbatch[name]))
		
		if training:
			epoch_score = np.mean(scores)
			epoch_losses.append(epoch_score)
			print(f"\tLoss {loss}: {epoch_score:.6f}")
		
		case = "Training" if training else "Val/Test"
		print(f"\t{case} Metrics:", *[f"{key} {metrics[key][-1]:.6f}" for key in metrics], sep = ', ')
		
		if training:
			if early_stopping:
				test_model(
					model,
					val_input_data,
					val_output_data,
					**hyperparameters
				)


def train_model(
	model,
	input_data,
	output_data,
	**hyperparameters: dict[str, Any]
):
	"""
	
	"""
	return _forwardpass_over_data(
		model,
		input_data,
		output_data,
		training = True,
		**hyperparameters
	)
	
	
def test_model(
	model,
	input_data,
	output_data,
	**hyperparameters: dict[str, Any]
):
	"""
	
	"""
	return _forwardpass_over_data(
		model,
		input_data,
		output_data,
		training = False,
		**hyperparameters
	)






if __name__ == '__main__':
	import os
	TRAIN_PATH = os.path.abspath('./airlines_train.h5')
	VAL_PATH = os.path.abspath('./airlines_val.h5')
	
	X_train = pd.read_hdf(TRAIN_PATH, mode='r', key='X')
	y_train = pd.read_hdf(TRAIN_PATH, mode='r', key='y')
	X_val = pd.read_hdf(VAL_PATH, mode='r', key='X')
	y_val = pd.read_hdf(VAL_PATH, mode='r', key='y')
	
	# TODO: UniqueCarrier, Origin, Dest need one-hot
	X_train = X_train.drop(["DayofMonth", "UniqueCarrier", "Origin", "Dest"], axis="columns").values
	X_val = X_val.drop(["DayofMonth", "UniqueCarrier", "Origin", "Dest"], axis="columns").values
	
	y_train = y_train.values
	y_val = y_val.values
	
	print(X_val)
	print(X_train.shape, X_val.shape)
	print(y_val)
	print(y_train.shape, y_val.shape)
	
	ACTIVATION = 'relu'
	HIDDEN_DIMS = [512, 512, 64]
	DROPOUT = 0.3
	
	model = TrunkModule(
		n_features = 11,
		hidden_dims = HIDDEN_DIMS,
		activation = ACTIVATION,
		dropout = DROPOUT,
	)
	model.train()
	
	print('Model parameters count:', sum(p.numel() for p in model.parameters() if p.requires_grad))
	
	train_model(
		model,
		X_train,
		y_train,
		loss = 'gaussian_kernel',
		# Hyperparameters
		loss_gk_gamma = 0.5,
		lr = 1e-3,
		batch_size = 64,
		num_epochs = 5,
		l2reg_strength = 0.0, # not implemented,
		early_stopping = True,
		val_input_data = X_val,
		val_output_data = y_val,
	)
	
	model.eval_mcdropout()
