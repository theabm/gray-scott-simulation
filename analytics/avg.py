import asyncio
import numpy as np
import os
import time

import dask.array as da
from doreisa.head_node import init
from doreisa.window_api import ArrayDefinition, run_simulation

def simulation_callback(
        V: list[da.Array], 
        U: list[da.Array], 
        timestep: int):
    Vavg = V[0].mean().compute()
    Uavg = U[0].mean().compute()
    print(f"Average at timestep {timestep}: V={Vavg}, U={Uavg}", flush=True)
          

init()
print("Analytics Initialized", flush=True)
run_simulation(
    simulation_callback,
    [
        ArrayDefinition("V", window_size=1),
        ArrayDefinition("U", window_size=1)
    ],
    max_iterations=10,
)