import torch

dtypes = [torch.bool, torch.int32, torch.int64, torch.float32, torch.float64]

print(f"{'Dtype A':<15} | {'Dtype B':<15} | {'Result Type':<15}")
print("-" * 51)

for dtype_a in dtypes:
    for dtype_b in dtypes:
        tensor_a = torch.empty(1, dtype=dtype_a)
        tensor_b = torch.empty(1, dtype=dtype_b)
        res_type = torch.result_type(tensor_a, tensor_b)
        
        name_a = str(dtype_a).replace("torch.", "")
        name_b = str(dtype_b).replace("torch.", "")
        name_res = str(res_type).replace("torch.", "")
        
        print(f"{name_a:<15} | {name_b:<15} | {name_res:<15}")

print("-" * 51)

expr_dtype = (torch.ones(1, dtype=torch.int64) + 1.5).dtype
func_dtype = torch.result_type(torch.ones(1, dtype=torch.int64), torch.tensor(1.5))

print(f"Expr (Tensor+1.5): {str(expr_dtype).replace('torch.', ''):<8} vs  result_type() Tensor: {str(func_dtype).replace('torch.', '')}")
