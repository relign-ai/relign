from datasets import Dataset

def remove_null_columns(ds: Dataset):
    null_columns = []
    for k, v in ds.features.items():
        if v.dtype == "null":
            null_columns.append(k)
    return ds.remove_columns(null_columns)

