import torch
from torch import Tensor
import torch.nn.functional as f


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    
    #
    print("Calculations for attention:\n")
    tmp = query @ key.T
    print(tmp)

    scale = key.size(-1) ** 0.5
    print(scale)

    softmax = f.softmax(tmp / scale, dim=-1)
    print(softmax)

    attention = softmax @ value
    print(attention)

    return attention


if __name__ == "__main__":

    # three random inputs of dim=4
    x = [
        [1, 0, 1, 0], # Input 1
        [0, 2, 0, 2], # Input 2
    ]
    x = torch.tensor(x, dtype=torch.float32)


    # Randomly assigned weigts for k, q and v
    w_key = [
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 0],
        [1, 1, 0]
    ]
    w_query = [
        [1, 0, 1],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 1]
    ]
    w_value = [
        [0, 2, 0],
        [0, 3, 0],
        [1, 0, 3],
        [1, 1, 0]
    ]
    w_key = torch.tensor(w_key, dtype=torch.float32)
    w_query = torch.tensor(w_query, dtype=torch.float32)
    w_value = torch.tensor(w_value, dtype=torch.float32)

    # Define key, value and query
    keys = x @ w_key
    queries = x @ w_query
    values = x @ w_value

    print(keys)
    print(queries)
    print(values)

    # get scaled dot product attention
    attention = scaled_dot_product_attention(queries, keys, values)