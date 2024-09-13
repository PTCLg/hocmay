# run_all.py

import subprocess

# Cài đặt các thư viện từ requirements.txt
print("Installing dependencies from requirements.txt...")
subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

# Chạy create_dataset.py
print("Running create_dataset.py...")
subprocess.run(["python", "create_dataset.py"], check=True)

# Chạy train_model.py
print("Running train_model.py...")
subprocess.run(["python", "train_model.py"], check=True)

# Chạy app.py
print("Running app.py...")
subprocess.run(["python", "app.py"], check=True)
