import matplotlib.pyplot as plt
import numpy as np
import json
from time import perf_counter
import glob

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = 1j * X @ Z

def kron(*args):
    assert len(args) >= 1
    X = args[0]
    for Y in args[1:]:
        X = np.kron(X, Y)
    return X


def get_marg_probs(file):
	"""
	Get marginal probabilities for the 3 measured qubits
	from the measurement probabilities
	"""
	data = json.loads(open(file).read())

	target_qubits = data["additionalMetadata"]["action"]["results"][0]["targets"] 
	meas_probs = data["measurementProbabilities"]
	target_qubits = np.sort(target_qubits)

	prob_distr = np.zeros(8)
	for state, prob in meas_probs.items():

		# check what state the 3 target qubits are in
		target_qubits_state = ""
		for qubit in target_qubits:
			target_qubits_state += state[qubit]

		prob_distr[int(target_qubits_state, 2)] += prob

	return prob_distr


def get_std(prob_distr, obs):
	"""
	Compute standard deviation from a given probability distribution
	by reproducing the experiment
	"""

	repeats = 500
	shots = 1000	

	l_distr = []
	for i in range(repeats):

		distr = np.zeros(len(prob_distr))
		for _ in range(shots):
			distr[np.random.choice(len(distr), p=prob_distr)] += 1/shots

		if obs == "XZX":
			diag = np.array([1,-1,-1,1,-1,1,1,-1]) # diagonal of ZZZ
		elif obs == "IXX":
			diag = np.array([1,-1,-1,1,1,-1,-1,1]) # diagonal of IZZ
		elif obs == "XXI":
			diag = np.array([1,1,-1,-1,-1,-1,1,1]) # diagonal of ZZI
		
		l_distr.append(np.sum(diag * distr))

	return np.std(l_distr)


def get_energy_from_data(date):
	"""Given a date, compute energy from data from that day"""

	layers = [0, 1, 2]
	observables = ["XZX", "IXX", "XXI"] # terms in Hamiltonian
	E = np.zeros(len(layers))

	for layer in layers:
		for obs in observables:
			vec = np.zeros(8)
			list_files = glob.glob(f"./braket/results/{date}/*shots_{layer}layers_{obs}*")

			for file in list_files:
				vec = np.sqrt(get_marg_probs(file))

				print("\n")
				print(file)
				if obs == "XZX":
					print(f"{vec.conj().T @ kron(Z,Z,Z) @ vec:.3f}")
					E[layer] += vec.conj().T @ kron(Z,Z,Z) @ vec 
				elif obs == "IXX":
					print(f"{vec.conj().T @ kron(I,Z,Z) @ vec:.3f}")
					E[layer] -= 0.5*vec.conj().T @ kron(I,Z,Z) @ vec
				elif obs == "XXI":
					print(f"{vec.conj().T @ kron(Z,Z,I) @ vec:.3f}")
					E[layer] -= 0.5*vec.conj().T @ kron(Z,Z,I) @ vec

	return E / np.array([1,2,4])

def energy_all_layers():
	"""Make a figure to display QC energies"""

	layers = [0, 1, 2]
	sim_data_noiseless = [0, -0.916, -1.159] # p = 0
	sim_data_noisy = [ 0, -0.269, -0.270] # p = 0.105
	exp_data = [-0.055, -0.2895, -0.24975]

	fig, ax = plt.subplots()
	plt.plot(layers, sim_data_noiseless, linestyle = "--", label = "Simulation, noiseless", \
			color="black")
	plt.plot(layers, sim_data_noisy, linestyle = "--", label = "Simulation, depolarizing p = 0.105", \
			color="orangered")
	plt.errorbar(layers, exp_data, yerr = 0.03, marker = "d", linewidth = 0, elinewidth = 1, capsize = 2, \
			color="firebrick", label = "Measurement data")

	plt.legend(fancybox=False, edgecolor="black")
	plt.xlabel("Layers")
	plt.ylabel("Energy (a.u.)")
	plt.title("Energy measurement with IonQ QC (1000 shots)")
	plt.grid(axis="y")
	plt.xlim(0-0.05, 2+0.05)
	plt.xticks(range(3))
	ax.spines["right"].set_visible(False)
	ax.spines["top"].set_visible(False)
	fig.set_size_inches(7, 5)
	# plt.savefig("./jupyter_figs/qc_energies.pdf", format="pdf", bbox_inches="tight")
	# plt.savefig("./jupyter_figs/qc_energies.jpg", format="jpg", bbox_inches="tight", dpi=300)

	plt.show()

def impure_qubit_noiseless():
	"""Make a figure of results of algorithmic cooling in a noiseless environment"""

	fig, ax = plt.subplots()
	plt.plot(range(4), [0.0, -0.9162658773652752, -1.1493137447006063, -1.2109748932859374], marker="o", color="black", \
				label="3 pure layers", markersize=5, linestyle="--")
	plt.plot(range(4), [0.0, -0.9162658773652752, -1.1493137447006063, -1.1164715889462105], marker="o", color="darkcyan", \
				label = "After 1st layer", markersize=5)
	plt.plot(range(4), [0.0, -0.9162658773652752, -1.1493137447006063, -1.133242942350201], marker="o", color="darkorange", \
				label = "After 2nd layer", markersize=5)


	plt.legend(fancybox=False, edgecolor="black")
	plt.xlabel("Layers")
	plt.ylabel("Energy (a.u.)")
	plt.title("Energy with cooled qubit (noiseless)")
	plt.grid(axis="y")
	plt.xlim(0-0.05, 3+0.05)
	plt.ylim(-1.25, 0.02)
	plt.xticks(range(4))
	ax.spines["right"].set_visible(False)
	ax.spines["top"].set_visible(False)
	fig.set_size_inches(7, 5)
	# plt.savefig("./jupyter_figs/cooling_noiseless.pdf", format="pdf", bbox_inches="tight")

	plt.show()

def impure_qubit_noisy():
	"""Make a figure of results of algorithmic cooling in a noisy environment"""

	fig, ax = plt.subplots()
	plt.plot(range(4), [0.0, -0.5221226016593634, -0.5723475964537521, -0.5731807137739467], marker="x", color="black", \
				label="3 pure layers", markersize=5, linestyle="--")
	plt.plot(range(4), [0.0, -0.5221226016593634, -0.5723475964537521, -0.5531077093761892], marker="x", color="darkcyan", \
				label = "After 1st layer", markersize=5)
	plt.plot(range(4), [0.0, -0.5221226016593634, -0.5723475964537521, -0.5070984509952712], marker="x", color="darkorange", \
				label = "After 2nd layer", markersize=5)

	plt.legend(fancybox=False, edgecolor="black")
	plt.xlabel("Layers")
	plt.ylabel("Energy (a.u.)")
	plt.title("Energy with cooled qubit (noisy)")
	plt.grid(axis="y")
	plt.xlim(0-0.05, 3+0.05)
	plt.ylim(-0.6, 0.02)
	plt.xticks(range(4))
	ax.spines["right"].set_visible(False)
	ax.spines["top"].set_visible(False)
	fig.set_size_inches(7, 5)
	# plt.savefig("./jupyter_figs/cooling_noisy.pdf", format="pdf", bbox_inches="tight")

	plt.show()

if __name__ == "__main__":
	# energy_all_layers()
	# impure_qubit_noiseless()
	# impure_qubit_noisy()
	print(get_energy_from_data("2022-08-05"))
