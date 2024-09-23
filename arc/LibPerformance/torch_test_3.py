import time
import torch

# Number of operations
n_operations = 1000

# Test using normal Python variables
start_time = time.time()

a = 3.14
b = 2.71
for _ in range(n_operations):
    result = a * b + b - a

normal_var_time = time.time() - start_time

# Test using Torch tensors
start_time = time.time()

a_tensor = torch.tensor(3.14)
b_tensor = torch.tensor(2.71)
for _ in range(n_operations):
    result_tensor = a_tensor * b_tensor + b_tensor - a_tensor

torch_tensor_time = time.time() - start_time

print(normal_var_time)
print(torch_tensor_time)
