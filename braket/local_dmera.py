import matplotlib.pyplot as plt
import numpy as np
import sys 
from time import perf_counter

from braket.circuits import Gate, Noise
from braket.devices import LocalSimulator

sys.path.append('.')
import native_gate_decomp as ngd
import quantum_circuits as qc

kron = ngd.kron
X = ngd.X
Z = ngd.Z
I = ngd.I


def run_simulation(layers, shots, type, aligned1="left", aligned2="right", observable="XZX"):
	"""
	Run DMERA circuit on local machine
	type = ["state_vector", "density_matrix"] to choose between resp. noiseless and noisy situation
	"""

	circ = qc.create_MERA_circuit(layers, aligned1, aligned2, observable)

	# noiseless simulation
	if type == "state_vector":
		device = LocalSimulator()

	# depolarizing noise simulation
	else:
		device = LocalSimulator("braket_dm")
		noise_depol = Noise.Depolarizing(probability = 0.105)
		circ.apply_gate_noise(noise_depol, target_gates = [Gate.XX, Gate.CSwap, Gate.CNot, Gate.Swap, Gate.X])

	
	task = device.run(circ, shots)
	results = task.result()

	return results.result_types[0].value

def compute_energy(layers, shots, type):
	"""
	Computes energies for all available layers with all alignments
	and outputs list of energy per layer
	Local simulation of calc_energy() from remote_dmera.py
	"""
	E = np.zeros(layers+1)
	for layer in range(layers+1):
		if layer == 0:
			aligned1_opts = ["n/a"]
			aligned2_opts = ["n/a"]
		elif layer == 1:
			aligned1_opts = ["left", "right"]
			aligned2_opts = ["n/a"]
		elif layer == 2:
			aligned1_opts = ["left", "right"]
			aligned2_opts = ["left", "right"]

		for aligned1 in aligned1_opts:
			for aligned2 in aligned2_opts:		

				for observable in ["XZX", "XXI", "IXX"]:
					result = run_simulation(layer, shots, type, aligned1, aligned2, observable)

					if observable == "XZX":
						E[layer] += result
					elif observable in ["XXI", "IXX"]:
						E[layer] -= 0.5*result

	return E / np.array([1,2,4]) # account for left-right alignments




