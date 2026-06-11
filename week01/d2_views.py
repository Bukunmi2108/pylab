import torch

t = torch.arange(0, 100, 2)

storage_start = t.data_ptr()
storage_end = storage_start + (t.nelement() * t.element_size())

slice_ptr = t[1:].data_ptr()
fancy_ptr = t[torch.tensor([1])].data_ptr()

assert storage_start <= slice_ptr < storage_end
assert not (storage_start <= fancy_ptr < storage_end)

print(f"Conclusion 1: Basic slice pointer ({slice_ptr}) lies inside t's storage [{storage_start}, {storage_end}), while fancy-indexed pointer ({fancy_ptr}) does not.")


slice_view = t[1:2]
slice_view[0] = -999
slice_mutated = (t[1].item() == -999)

t[1] = 2

fancy_copy = t[torch.tensor([1])]
fancy_copy[0] = -999
fancy_mutated = (t[1].item() == -999)

assert slice_mutated == True
assert fancy_mutated == False

print(f"Conclusion 2: Mutating the basic slice modified the original tensor t, whereas mutating the fancy-indexed result left t unchanged.")
