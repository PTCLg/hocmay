import sys

REQUIRED_PYTHON_VERSION = (3, 8)
current_version = sys.version_info

def check_python_version():
    if current_version >= REQUIRED_PYTHON_VERSION:
        print(f"Python version is {current_version.major}.{current_version.minor} or greater.")
        print("Python version is compatible.")
    else:
        print(f"Python version is {current_version.major}.{current_version.minor}.")
        print("Error: Python version is not compatible. Please upgrade to Python 3.8 or higher.")

if __name__ == "__main__":
    check_python_version()
