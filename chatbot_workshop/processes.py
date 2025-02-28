import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from attr import dataclass
from dotenv import load_dotenv

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    from google.colab import userdata

    os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

ProcessTypes = Literal["chainlit", "tunnel"]


@dataclass
class Process:
    process: subprocess.Popen
    log_file: Path


_running_processes: dict[ProcessTypes, Process] = dict()


def show_logs(process_type: ProcessTypes):
    global _running_processes
    if process_type in _running_processes:
        running_process = _running_processes[process_type]
        print(running_process.log_file.read_text())
    else:
        print(f"{process_type} is not running")


def restart(process_type: ProcessTypes):
    global _running_processes
    if process_type in _running_processes:
        running_process = _running_processes[process_type]
        proccess = running_process.process
        proccess.kill()
        proccess.wait()
        _running_processes.pop(process_type)
    if process_type == "chainlit":
        command_line = [
            "chainlit",
            "run",
            "app.py",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
    elif process_type == "tunnel":
        lt_executable = str(shutil.which("lt"))
        command_line = [lt_executable, "--port", "8000"]
    (_, output_file) = tempfile.mkstemp(suffix=".log")
    out_writer = open(output_file, "wb")
    process = subprocess.Popen(
        command_line, stdout=out_writer, stderr=subprocess.STDOUT
    )
    _running_processes[process_type] = Process(process, Path(output_file))


def assert_all_running():
    global _running_processes
    for process_type in _running_processes:
        running_process = _running_processes[process_type]
        assert running_process.process.poll() is None
