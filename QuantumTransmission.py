import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from joblib import Parallel, delayed
import multiprocessing
import mpmath as mp
from tqdm import tqdm

# Physical constants (SI)
h = 1.054571817e-34
e = 1.602176634e-19
m = 0.0665 * (9.1093837139e-31)

# Number of x points
nx = 500
# Potential region size
X = 10
# x0 - potential destribution region linspace
x0 = np.linspace(0, X, nx)

def expk(ENERGY, i, x, x0, POTENTIAL_DISTRIBUTION):
	kk = 1e-9 / h * np.sqrt(2 * m * e * (ENERGY - POTENTIAL_DISTRIBUTION[i] + 0j), dtype=np.complex128)
	if i == 0:
		return np.exp(1j * kk * (x - x0[0]))
	else:
		return np.exp(1j * kk * (x - x0[i - 1]))

def S_matrix(ENERGY, i, x0, POTENTIAL_DISTRIBUTION):
	# Create 2Kx2K matrix by blocks
	S = np.zeros((2, 2), dtype=np.complex128)

	# Precalculation
	ki = 1e-9 / h * np.sqrt(2 * m * e * (ENERGY - POTENTIAL_DISTRIBUTION[i] + 0j), dtype=np.complex128)
	kip1 = 1e-9 / h * np.sqrt(2 * m * e * (ENERGY - POTENTIAL_DISTRIBUTION[i + 1] + 0j), dtype=np.complex128)
	expki = expk(ENERGY, i, x0[i], x0, POTENTIAL_DISTRIBUTION)
	Di = 1 / (ki + kip1)
	tildeDi = Di
	Fi = kip1 - ki
	tildeFi = -Fi

	# Top-left block
	S[0:1, 0:1] = 2 * (Di * ki * expki)
	
	# Top-right block
	S[0:1, 1:2] = Di * Fi
	
	# Bottom-left block
	S[1:2, 0:1] = (expki * tildeDi * tildeFi * expki)
	
	# Bottom-right block
	S[1:2, 1:2] = 2 * (expki * tildeDi * kip1)

	return S

def Transmission_S(ENERGY, PARAM):
	# Create 2Kx2K matrix by blocks
	S = np.zeros((2, 2), dtype=np.complex128)
	Stemp = S

	# x0 - potential destribution region linspace
	# x0 = np.linspace(0, X, nx) 

	# POTENTIAL_DISTRIBUTION
	POTENTIAL_DISTRIBUTION = np.linspace(0, X, nx + 1)
	POTENTIAL_DISTRIBUTION[0:nx] = x0**2 * (x0 - X)**2 * (x0 - X/3)**2 * (x0 - 2*X/3)**2	# Formula for potential distribution spatial profile
	POTENTIAL_DISTRIBUTION[0] = 0															# Left end constant potential level
	POTENTIAL_DISTRIBUTION[-1] = 0															# Right end constant potential level
	POTENTIAL_DISTRIBUTION = POTENTIAL_DISTRIBUTION / max(POTENTIAL_DISTRIBUTION) * PARAM		# Normalization of potential profile

	S = S_matrix(ENERGY, 0, x0, POTENTIAL_DISTRIBUTION)
	
	for i in np.linspace(1, x0.shape[0] - 1, x0.shape[0] - 1, dtype = 'int'):
		Stemp = np.zeros((2, 2), dtype=np.complex128)
		Sip1 = S_matrix(ENERGY, i, x0, POTENTIAL_DISTRIBUTION)
		D = 1 / (1 - Sip1[1:2, 0:1] * S[0:1, 1:2])

		# Top-left block
		Stemp[0:1, 0:1] = Sip1[0:1, 0:1] * (1 + S[0:1, 1:2] * D * Sip1[1:2, 0:1]) * S[0:1, 0:1]

		# Top-right block
		Stemp[0:1, 1:2] = Sip1[0:1, 1:2] + Sip1[0:1, 0:1] * S[0:1, 1:2] * D * Sip1[1:2, 1:2]

		# Bottom-left block
		Stemp[1:2, 0:1] = S[1:2, 0:1] + S[1:2, 1:2] * D * Sip1[1:2, 0:1] * S[0:1, 0:1]

		# Bottom-right block
		Stemp[1:2, 1:2] = S[1:2, 1:2] * D * Sip1[1:2, 1:2]

		S = Stemp

	return np.abs(S[0, 0])**2

def plot_transmission_density(energy_range, param_range, 
							 num_energy_points=100, num_param_points=100,
							 log_scale=False, n_jobs=-1, backend='loky'):
		
	# Create meshgrid for ENERGY and D4
	energy = np.linspace(energy_range[0], energy_range[1], num_energy_points)
	param = np.linspace(param_range[0], param_range[1], num_param_points)
	ENERGY_mesh, PARAM_mesh = np.meshgrid(energy, param)
	
	# Flatten the mesh for parallel processing
	energy_flat = ENERGY_mesh.flatten()
	param_flat = PARAM_mesh.flatten()
	
	# Calculate number of CPU cores
	num_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
	print(f"Using {num_cores} CPU cores for parallel computation...")
	
	# Parallel computation of transmission values
	transmission_flat = Parallel(n_jobs=n_jobs, backend=backend)(
		delayed(Transmission_S)(e, p) 
		for e, p in tqdm(zip(energy_flat, param_flat), total=len(energy_flat), desc="Computing transmission") # zip(energy_flat, param_flat)
	)
	
	# Reshape back to 2D
	transmission = np.array(transmission_flat).reshape(ENERGY_mesh.shape)
	
	# Create the plot
	fig, ax = plt.subplots(figsize=(10, 8))
	
	# Choose normalization
	if log_scale:
		# Handle potential zeros or negative values for log scale
		transmission_plot = np.maximum(transmission, 1e-10)  # Avoid log(0)
		norm = LogNorm()
		im = ax.pcolormesh(ENERGY_mesh, PARAM_mesh, transmission_plot, 
						  norm=norm, cmap='viridis', shading='auto')
	else:
		im = ax.pcolormesh(ENERGY_mesh, PARAM_mesh, transmission, 
						  cmap='viridis', shading='auto')
	
	# Add colorbar
	cbar = plt.colorbar(im, ax=ax)
	cbar.set_label('Transmission', fontsize=12)
	
	# Labels and title
	ax.set_xlabel('Energy', fontsize=12)
	ax.set_ylabel('Parameter', fontsize=12)
	ax.set_title(f'Transmission Density Plot\n'
				f'Grid: {num_energy_points}×{num_param_points} = {num_energy_points * num_param_points} points | '
				f'Cores: {num_cores}', 
				fontsize=12)
	
	plt.tight_layout()
	plt.show()
	
	# Optional: Print computation time info
	print(f"Computed {num_energy_points * num_param_points} points in parallel using {num_cores} cores")
	
	return fig, ax, transmission

# Define ranges
energy_min, energy_max = 0.001, 15	# Energy range
param_min, param_max = 5, 15		# Parameters range

# Create the plot
fig, ax, transmission_data = plot_transmission_density(
	(energy_min, energy_max),
	(param_min, param_max),
	num_energy_points = 500,
	num_param_points = 50,
	log_scale=True  # Set to True if transmission values span many orders of magnitude
)