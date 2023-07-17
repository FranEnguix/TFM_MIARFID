import time
import os
import Config

from pathlib import Path
from Agents.LauncherAgent import LauncherAgent

n0 = LauncherAgent(f"my_launcher_agent@{Config.xmpp_server}", "abcdefg", Config.web_port, 11000)
f1 = n0.start(auto_register=True)
f1.result()


while n0.is_alive():
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        n0.stop()
        break

if Config.export_logs_at_end_of_execution:
    export_logs_root = Path(Config.export_logs_root_path)
    if not export_logs_root.exists():
        export_logs_root.mkdir(parents=True, exist_ok=True)
    i = 0
    experiment_folder = None
    last_experiment_folder = None
    while experiment_folder is None:
        exp = Path(f"{Config.export_logs_root_path}/{Config.export_logs_folder_prefix}{i}")
        if not exp.exists():
            if Config.export_logs_append_to_last_log_folder_instead_of_create:
                experiment_folder = last_experiment_folder
            if not Config.export_logs_append_to_last_log_folder_instead_of_create or experiment_folder is None:
                experiment_folder = exp
                experiment_folder.mkdir(parents=True, exist_ok=True)
        last_experiment_folder = exp
        i += 1
    for folder in Config.logs_folders:
        for f in os.listdir(f"{Config.logs_root_folder}/{folder}"):
            file = Path(f"{Config.logs_root_folder}/{folder}/{f}")
            if file.is_file():
                Path(f"{experiment_folder.absolute()}/{Config.logs_root_folder}/{folder}/").mkdir(parents=True, exist_ok=True)
                if not Path(f"{experiment_folder.absolute()}/{Config.logs_root_folder}/{folder}/{f}").exists():
                    file.rename(f"{experiment_folder.absolute()}/{Config.logs_root_folder}/{folder}/{f}")
                else:
                    print(f"ERROR: File {experiment_folder.absolute()}/{Config.logs_root_folder}/{folder}/{f} exists")
    print(f"All logs moved to: {experiment_folder.absolute()}")

