import os

def print_directory_structure(root_dir, indent=''):
    if not os.path.exists(root_dir):
        print(f"Thư mục {root_dir} không tồn tại.")
        return

    items = os.listdir(root_dir)
    for item in items:
        path = os.path.join(root_dir, item)
        if os.path.isdir(path):
            print(f'{indent}├── {item}/')
            print_directory_structure(path, indent + '│   ')
        else:
            print(f'{indent}└── {item}')

# Đảm bảo rằng bạn cung cấp đúng đường dẫn
root_directory = 'E:\HocMay\shoe_classification_project'
print(f'{root_directory}')
print_directory_structure(root_directory)
