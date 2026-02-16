import sys

if len(sys.argv) != 3:
    print("Usage: python append_files.py <file1.txt> <file2.txt>")
    sys.exit(1)

file1 = sys.argv[1]
file2 = sys.argv[2]

try:
    with open(file2, 'r') as f2:
        data = f2.read()
except FileNotFoundError:
    print(f"Error: {file2} not found.")
    sys.exit(1)

try:
    with open(file1, 'a') as f1:
        f1.write(data)
    print(f"Contents of {file2} appended to {file1}.")
except FileNotFoundError:
    print(f"Error: {file1} not found.")
    sys.exit(1)

