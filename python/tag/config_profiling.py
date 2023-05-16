# ! Run in base project directory: `python -m python.tag.config_profiling`
from ..isens_vtrigU import isens_vtrigU
import time

radar = isens_vtrigU()

avg_time = 0
trials = 10

for i in range(trials):
    start_time = time.time()
    radar.scan_setup()
    end_time = time.time()
    avg_time += end_time - start_time

avg_time /= trials

print("Time taken to setup scan: {} seconds".format(end_time - start_time))