import torch

## 2D x 2D matmul - assume batch size is squeezed out for inference
mat1 = torch.rand((4, 8))
mat2 = torch.rand((8, 8))
block = mat2.shape[0] // 4
output = torch.zeros(mat1.shape[0], mat2.shape[0])
torch.set_printoptions(profile="full")
print(mat1)
print(mat2)
for i in range(4):
    print(mat2[i * block : (i + 1) * block, :])
    output[:, i * block : (i + 1) * block] = torch.einsum(
        "jk,lk->jl",
        mat1, 
        mat2[i * block : (i + 1) * block, :]
    )
print(torch.einsum("jk,lk->jl", mat1, mat2))
print(output)