import sys 
import datetime
import os.path

import boto3
from braket.aws import AwsDevice, AwsQuantumTask
from braket.circuits import Circuit
from braket.circuits.gate import Gate
from braket.circuits.instruction import Instruction
from braket.circuits.unitary_calculation import calculate_unitary

sys.path.append('.')
import native_gate_decomp as ngd
import quantum_circuits as qc

s3_folder = ("", "") # FILL IN OWN FOLDER

# AWS DEVICES

# State vector machine, name = SV1 
# device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

# IonQ computer, name = IonQ Device
# device = AwsDevice("arn:aws:braket:::device/qpu/ionq/ionQdevice")

# Rigetti Aspen 10, name = Aspen-10
# device = AwsDevice("arn:aws:braket:::device/qpu/rigetti/Aspen-10")


def run_dmera(layers, shots, device, aligned1="left", aligned2="left", observable="XZX"):
	"""
	Run DMERA on an AWS machine and return the task ID
	layers = [0,1,2]
	device = ["sv1", "ionq", "aspen10"]
	"""

	if device.lower() == "sv1":
		device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
	elif device.lower() == "ionq":
		device = AwsDevice("arn:aws:braket:::device/qpu/ionq/ionQdevice")
	elif device.lower() == "aspen10":
		device = AwsDevice("arn:aws:braket:::device/qpu/rigetti/Aspen-10")
	else:
		raise ValueError("Expected a device from [\"sv1\", \"ionq\", \"aspen10\"]")

	circ = qc.create_MERA_circuit(layers, aligned1, aligned2, observable)

	task = device.run(circ, s3_folder, shots)
	task_id = task.id
	print("Task ID:", task_id)
	return task_id


def store_ids(all_ids):
	"""Store all the task IDs in one .txt file"""

	date = datetime.date.today()

	if not os.path.exists(f"./braket/results/{date}"):
		os.makedirs(f"./braket/results/{date}")

	filename = f"./braket/results/{date}/task_id.txt"
	with open(filename, "w") as f:
		for task_id in all_ids:
			f.write(task_id+"\n")
	print(f"Task IDs stored in {filename}")


def get_layer_nrs(len_instr):
	"""
	Get number of layers from amount of instructions
	Probably improve this at one point
	"""
	if len_instr == 6:
		return "0"
	elif len_instr == 20:
		return "1"
	elif len_instr == 40:
		return "2"
	else:
		raise ValueError("Length of instructions and layer do not match up")

def get_alignment(layer, results):
	if layer == "0":
		aligned1 = "none"
		aligned2 = "none"

	elif layer == "1":
		
		if results.result_types[0].type.targets == [1,2,3]:
			aligned1 = "left"
		else:
			aligned1 = "right"
		aligned2 = "none"

	elif layer == "2":
		# check if first gate of 2nd layer is for left or right qubit
		if results.additional_metadata.action.instructions[21].target == 1:
			aligned1 = "left"
			if results.result_types[0].type.targets == [6,2,7]:
				aligned2 = "left"
			else:
				aligned2 = "right"

		else:
			aligned1 = "right"
			if results.result_types[0].type.targets == [6,3,7]:
				aligned2 = "left"
			else:
				aligned2 = "right"
	
	return aligned1, aligned2


def download_json(task_id, shots, layers, observable, aligned1, aligned2, date):
	"""
	Download .json file from the given task and store it
	in a folder with date of experiment
	"""

	quantum_task = task_id.split("/")[-1]

	bucket_name = "" # FILL IN OWN BUCKET
	key_name = f"" # FILL IN OWN KEY_NAME
	file_name = f"./braket/results/{date}/{shots}shots_{layers}layers_{observable.upper()}_{aligned1}_{aligned2}.json"	

	if os.path.isfile(file_name):
		print("File already exists")
	else:
		s3 = boto3.client("s3")
		s3.download_file(bucket_name, key_name, file_name)
		print("Download successful")


def check_task(task_id):
	"""Check status of task and give results if task is finished"""

	try:
		load_task = AwsQuantumTask(arn=task_id)
	except:
		print("Enter a correct task ID")

	status = load_task.state()
	# print("Status:", load_task.state())

	# terminal_states are ['COMPLETED', 'FAILED', 'CANCELLED']
	if load_task.state() == "COMPLETED":
		results = load_task.result()
		print("\n")
		print("Energy density is", round(results.result_types[0].value, 6))

		shots = results.task_metadata.shots
		layer = get_layer_nrs(len(results.additional_metadata.action.instructions))
		date = results.task_metadata.endedAt[:10]
		observable = "".join(results.result_types[0].type.observable)
		aligned1, aligned2 = get_alignment(layer, results)

		print(f"{shots}_{layer}layers_{observable}_{aligned1}_{aligned2}")
		
		download_json(task_id, shots, layer, observable, aligned1, aligned2, date)
		return round(results.result_types[0].value, 6)

	elif load_task.state() in ["FAILED", "CANCELLED"]:
		print("Task is terminated but not completed")

	else:
		print("Task is not finished yet, try again later")


def check_all_tasks(date):
	"""Given a date from experiment, run check_task() on all tasks"""

	task_id_file = f"./braket/results/{date}/task_id.txt"

	with open(task_id_file) as f:
		all_ids = [line.strip("\n") for line in f]

	for layer, task_id in enumerate(all_ids):
		check_task(task_id)


def calc_energy(shots, device, layers):
	"""Calculate the energy for all layers"""

	all_ids = []

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
					task_id = run_dmera(layer, shots, device, aligned1, aligned2, observable)
					all_ids.append(task_id)

	store_ids(all_ids)

	print("Task successfully sent!")


# calc_energy() to send a task to the QC to compute energies for all layers
# check_all_tasks() to check the tasks of a certain date






