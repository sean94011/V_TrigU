import os
import datetime
import time
import shutil

folder_name = 'test2'
folder_name = f'{folder_name}_{datetime.datetime.now().strftime("%m-%d-%Y--%H-%M-%S")}_{time.time_ns()}'
folder_name_full = os.path.join('./collected_data',folder_name)
parameters_dir = os.path.join(folder_name_full,'parameters')
data_dir = os.path.join(folder_name_full,'data_queue')
# if folder_name not in os.listdir('./measurements'):
#     os.mkdir(folder_name_full)
# os.mkdir(parameters_dir)
# os.mkdir(data_dir)
shutil.copytree('./parameters',parameters_dir)
shutil.copytree('./data_queue',data_dir)

