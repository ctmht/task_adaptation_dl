from typing import Literal, Any

import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import tqdm

import matplotlib.pyplot as plt


from data.processing.process_data import load_and_preprocess_data
from model.trunk_module import TrunkModule
from model.score_losses import *


NAME_TO_SCORE = {
	'gaussian_kernel': GaussianKernelScore,
	'gaussian_nll': NLL,
	'gaussian_se': SquaredError,
	'gaussian_crps': NormalCRPS
}


def train_model(
	model: TrunkModule,
	X_train,
	y_train,
	loss: Literal['gaussian_kernel', 'gaussian_nll', 'gaussian_se', 'gaussian_crps'],
	**hyperparameters
) -> Any: # TODO: update as needed
	"""
	
	"""
	# Training hyperparameters
	lr = hyperparameters.get('lr', 1e-3)
	l2reg_strength = hyperparameters.get('l2reg_strength', 0.1) # TODO: use
	num_epochs = hyperparameters.get('num_epochs', 10)
	batch_size = hyperparameters.get('batch_size', 128)
	
	loss_reduction = hyperparameters.get('loss_reduction', 'mean')
	loss_ensemble = hyperparameters.get('loss_ensemble', False)
	loss_gk_gamma = hyperparameters.get('loss_gk_gamma', 1.0)
	score_loss = NAME_TO_SCORE[loss](
		reduction = loss_reduction,
		ensemble = loss_ensemble,
		loss_gk_gamma = loss_gk_gamma
	)
	
	# Setup training
	device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
	device = torch.device(device_name)
	model = model.to(device)
	print(device_name)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
	
	# Convert to PyTorch tensors
	X_train_pt = torch.tensor(X_train, dtype = torch.float32)
	y_train_pt = torch.tensor(y_train, dtype = torch.float32)
	
	num_samples = len(X_train_pt)
	indices = np.arange(num_samples)
	
	epoch_scores = []
	for epoch in range(num_epochs):
		# Randomize order of training data
		np.random.shuffle(indices)
		
		scores = []
		for batch_start in tqdm.tqdm(range(0, num_samples, batch_size), desc = f"Epoch {epoch+1}"):
			batch_end = min(batch_start + batch_size, num_samples)
			batch_indices = indices[batch_start:batch_end]
			
			# Prepare batch
			batch_X = X_train_pt[batch_indices].to(device)
			batch_y = y_train_pt[batch_indices].to(device)
			
			# Forward pass
			optimizer.zero_grad()
			pred_y = model(batch_X)
			
			score = score_loss(pred_y, batch_y)
			
			score.backward()
			optimizer.step()
			
			scores.append(score.item())
		
		epoch_score = np.mean(scores)
		epoch_scores.append(epoch_score)
	
	plt.plot(epoch_scores)
	plt.show()
	
	return epoch_scores


if __name__ == '__main__':
	import os
	CSV_PATH = os.path.abspath('./CASP.csv')
	TEST_SIZE = 0.20
	
	ACTIVATION = 'relu'
	HIDDEN_DIMS = [32]
	DROPOUT = 0.0
	
	X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler = load_and_preprocess_data(
		csv_path = CSV_PATH,
    	test_size = TEST_SIZE,
    	random_state = 42,
    	log_features = ('F7', 'F5')
	)
	
	model = TrunkModule(
		n_features = 9,
		hidden_dims = HIDDEN_DIMS,
		activation = ACTIVATION,
		dropout = DROPOUT,
	)
	
	train_model(
		model,
		X_train_scaled,
		y_train_scaled,
		loss = 'gaussian_kernel',
		# Hyperparameters
		loss_gk_gamma = 0.5, # only does something for loss == 'gaussian_kernel'
		lr = 1e-3,
		batch_size = 64,
		num_epochs = 5,
		l2reg_strength = 0.0, # not implemented
	)