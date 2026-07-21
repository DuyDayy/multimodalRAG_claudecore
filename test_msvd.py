from datasets import load_dataset
try:
    ds = load_dataset("VLM2Vec/MSVD", streaming=True)
    print(next(iter(ds['train'])))
except Exception as e:
    print(f"Error: {e}")
