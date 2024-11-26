def func(x):
    x = x + 1
    return x
print(func(5))

# @ parse_python_files(".")
# (['def func(x):', 'x = x + 1', 'return x'],
# ['x = x + 1', 'return x', 'print(func(5))'])

