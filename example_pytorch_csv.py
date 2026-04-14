import pandas as pd
from src.cddd.pytorch_inference import PyTorchInferenceModel

def main():
    # 1. Create a dummy CSV file with a few SMILES strings
    print("Creating dummy CSV file 'dummy_input.csv'...")
    dummy_data = {
        "smiles": [
            "C1CCCCC1",  # Cyclohexane
            "CCO",       # Ethanol
            "c1ccccc1",  # Benzene
            "CC(=O)O"    # Acetic acid
        ]
    }
    df_input = pd.DataFrame(dummy_data)
    df_input.to_csv("dummy_input.csv", index=False)
    
    # 2. Read the CSV file
    print("Reading 'dummy_input.csv'...")
    df = pd.read_csv("dummy_input.csv")
    smiles_list = df["smiles"].tolist()

    # Note: If you want to use the RDKit preprocessing (like removing salts), 
    # you can import and apply `preprocess_smiles` from `cddd.preprocessing` here.

    # 3. Load the PyTorch Inference Model
    # Assuming the converted 'cddd_model.pt' is inside the 'default_model' directory
    print("Loading PyTorch CDDD model...")
    model = PyTorchInferenceModel(model_dir='default_model')

    # 4. Extract continuous data-driven descriptors (embeddings)
    print(f"Extracting descriptors for {len(smiles_list)} SMILES strings...")
    embeddings = model.seq_to_emb(smiles_list)
    print(f"Extraction complete. Descriptor shape: {embeddings.shape}")

    # 5. Save the results to a new CSV
    # Name the columns cddd_1 to cddd_512
    col_names = [f"cddd_{i+1}" for i in range(embeddings.shape[1])]
    df_embeddings = pd.DataFrame(embeddings, columns=col_names)
    
    df_output = pd.concat([df, df_embeddings], axis=1)
    df_output.to_csv("dummy_output.csv", index=False)
    print("Successfully saved all descriptors to 'dummy_output.csv'.")

if __name__ == "__main__":
    main()