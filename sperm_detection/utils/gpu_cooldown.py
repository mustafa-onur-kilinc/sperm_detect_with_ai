"""
Script by Özgün Zeki BOZKURT
"""

import time
import subprocess


def check_gpu_temperature():
    try:
        # Execute nvidia-smi to get GPU temperature
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
        )
        temperature = int(result.stdout.decode("utf-8").split("\n")[0])
        return temperature
    except Exception as e:
        print(f"Error checking GPU temperature: {e}")
        return None


def cool_down_if_needed():
    temperature = check_gpu_temperature()
    if temperature is not None and temperature > 85:
        """
        print(
            f"GPU temperature is {temperature}°C, pausing for 30 seconds to cool down."
        )
        """
        time.sleep(30)
