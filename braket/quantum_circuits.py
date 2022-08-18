import numpy as np
import sys 

from braket.circuits import Circuit, Observable, Gate
from braket.circuits.instruction import Instruction

sys.path.append('.')
import native_gate_decomp as ngd

kron = ngd.kron
pi = np.pi

theta1 = pi/12
theta2 = -pi/6

# Pauli gates
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = 1j * X @ Z

def W_gate(qubit0, qubit1, theta):
	return 	Instruction(Gate.XX(-pi/2), (qubit0, qubit1)),\
			Instruction(Gate.Rz(theta), qubit0),\
			Instruction(Gate.Rz(theta - pi/2), qubit1),\
			Instruction(Gate.XX(pi/2), (qubit0, qubit1))

def U_gate(qubit0, qubit1, theta):
	return 	Instruction(Gate.XX(-pi/2), (qubit0, qubit1)),\
			Instruction(Gate.Rz(theta), qubit1),\
			Instruction(Gate.Rz(theta), qubit0),\
			Instruction(Gate.XX(pi/2), (qubit0, qubit1))


def create_MERA_circuit(layers, aligned1, aligned2, observable):
	"""
	Create MERA for given number of layers and add Hamiltonian as observable

	Input:
	layers - number of layers, options: [0,1,2] (3 layers includes cooling)
	aligned1 - left or right centering of 3 qubits after 1st layer, options: ["left", "right"] 
	aligned2 - left or right centering of 3 qubits after 2nd layer, options: ["left", "right"] 
	observable - observable to measure at end, options: ["XZX", "IXX", "XXI"]

	Output:
	List of instructions which can be sent to (simulated) QC
	"""
	
	circ = Circuit()


	if layers == 0:
		instructions = [Instruction(Gate.Rz(pi), 0), Instruction(Gate.Rz(-pi), 0), \
						Instruction(Gate.Rz(pi), 1), Instruction(Gate.Rz(-pi), 1), \
						Instruction(Gate.Rz(pi), 2), Instruction(Gate.Rz(-pi), 2)]
		target_qubits = [0,1,2]

	if layers >= 1:
		instructions = W_gate(0,1, theta1) + W_gate(2,3, theta1) + W_gate(4,5, theta1) \
					 + U_gate(1,2, theta2) + U_gate(3,4, theta2)
		if aligned1 == "left":
			target_qubits = [1,2,3]
		elif aligned1 == "right":
			target_qubits = [2,3,4]

	if layers >= 2:
		left, center, right = target_qubits
		instructions += W_gate(left,6, theta1) + W_gate(center,7, theta1) + W_gate(right,8, theta1) \
					+ U_gate(6,center, theta2) + U_gate(7,right, theta2)
		
		if aligned2 == "left":
			target_qubits = [6,center,7]
		elif aligned2 == "right":
			target_qubits = [center,7,right]

	if layers == 3:

		# COOLING
		# First layer
		instructions = W_gate(0,1, theta1) + W_gate(2,3, theta1) + W_gate(4,5, theta1) \
					 + U_gate(1,2, theta2) + U_gate(3,4, theta2)

		# Cooling
		instructions += (Instruction(Gate.CNot(), (5,4)), Instruction(Gate.X(), 4), Instruction(Gate.CSwap(), (4,0,5)), \
					Instruction(Gate.Swap(), (0,4)), Instruction(Gate.X(), 0))

		# Second layer
		instructions += W_gate(1,0, theta1) + W_gate(2,6, theta1) + W_gate(3,7, theta1) \
					 + U_gate(1,2, theta2) + U_gate(6,3, theta2)

		# Third layer
		instructions += W_gate(1,8, theta1) + W_gate(2,9, theta1) + W_gate(6,10, theta1) \
					 + U_gate(8,2, theta2) + U_gate(9,6, theta2)
		
		target_qubits = [8,2,9]

	if layers not in [0,1,2,3]:
		raise ValueError("Invalid number of layers")

	# create circuit from instructions 
	for instr in instructions:
		circ.add_instruction(instr)
	
	if observable == "XZX":
		prod_observable = Observable.X() @ Observable.Z() @ Observable.X()
	elif observable == "IXX":
		prod_observable = Observable.I() @ Observable.X() @ Observable.X()
	elif observable == "XXI":
		prod_observable = Observable.X() @ Observable.X() @ Observable.I()
	else:
		raise ValueError("No valid observable defined")

	circ.expectation(observable = prod_observable, target = target_qubits)
	return circ
