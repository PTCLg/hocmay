# run_all.py

import subprocess

# Chạy create_dataset.py
print("Running create_dataset.py...")
subprocess.run(["python", "create_dataset.py"])

# Chạy train_model.py
print("Running train_model.py...")
subprocess.run(["python", "train_model.py"])

# Chạy app.py
print("Running app.py...")
subprocess.run(["python", "app.py"])
