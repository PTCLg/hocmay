# run_all.py

import subprocess
import sys
import os

def run_script(script_name):
    """Run a script and handle encoding issues."""
    try:
        # Set environment variable for UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        # Run the script
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True, env=env)

        # Print the output
        print(f"Output of {script_name}:")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}:")
        print(e.stderr)
 
def main():
    scripts = [
        'create_dataset.py',
        'train_model.py',
        'app.py'
    ]

    for script in scripts:
        print(f"Running {script}...")
        run_script(script)
        print(f"Finished running {script}\n")

if __name__ == '__main__':
    main()
