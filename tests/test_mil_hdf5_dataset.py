from src.data.mil_hdf5_dataset import MILHDF5Dataset


HDF5_PATH = "PATH_TO_YOUR_HDF5_FILE"


dataset = MILHDF5Dataset(HDF5_PATH)

print("Number of slides:", len(dataset))

embeddings, coords, label, slide_id, case_id = dataset[0]

print("Slide ID:", slide_id)
print("Case ID:", case_id)
print("Label:", label.item())
print("Embeddings shape:", embeddings.shape)
print("Coordinates shape:", coords.shape)