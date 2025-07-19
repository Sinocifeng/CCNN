import torch

def attribute_co_extent_similarity(extents: torch.Tensor, epsilon=1e-8) -> torch.Tensor:

    co_occurence_matrix = torch.matmul(extents.transpose(-1, -2), extents)
    occurence_vector = extents.sum(dim=-2)

    if extents.dim() == 3: 
        occurence_matrix = occurence_vector.expand(co_occurence_matrix.size(-1), *occurence_vector.size()).transpose(1, 0)
        appearance_matrix = occurence_matrix.transpose(-1, -2) + occurence_matrix + epsilon
    else:
        appearance_matrix = occurence_vector.expand(co_occurence_matrix.size()).transpose(-1, -2) + occurence_vector + epsilon


    co_occurence_similarity_matrix = 2 * co_occurence_matrix.float() / appearance_matrix
    co_occurence_similarity_matrix += (appearance_matrix <= epsilon).float() 

    return co_occurence_similarity_matrix