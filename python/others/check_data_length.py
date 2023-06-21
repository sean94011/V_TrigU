import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("case", help="enter the case you want to process", type=str)
parser.add_argument("scenario", help="enter the scenario you want to process", type=str)
args = parser.parse_args()


current_case = args.case
current_scenario = args.scenario

data_path = os.path.join('./data',current_case,current_scenario,'recording.npy')

data = np.load(data_path)

print(data.shape[0])