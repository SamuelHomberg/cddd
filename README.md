# Continuous and Data-Driven Descriptors (CDDD)

A PyTorch port of the original implementation of the Paper "Learning Continuous and Data-Driven Molecular
Descriptors by Translating Equivalent Chemical Representations" by Robin Winter, Floriane Montanari, Frank Noe and Djork-Arne Clevert.<sup>1</sup>

<img src="example/model.png" width="75%" height="75%">

## First Setup (installation and converting TF weights)

```bash
git clone https://github.com/SamuelHomberg/cddd.git
cd cddd
uv venv # install uv first if necessary
uv sync --extra tensorflow # tf is only required to convert the original TF weights to PyTorch
```

Download the default_model.zip file under https://drive.google.com/open?id=1oyknOulq_j0w9kzOKKIHdTLo5HphT99h and extract it to the `cddd` folder.

```bash
[ -d "default_model" ] || { echo "Error: default_model not found"; exit 1; }
uv run convert_tf_to_pt.py
```

## Usage 

The pretrained model can be run entirely in PyTorch using the new inference class. 

```python 
from cddd.pytorch_inference import PyTorchInferenceModel 

# Create an instance of the PyTorch inference class 
inference_model = PyTorchInferenceModel(model_dir='default_model') 

# Encode all SMILES into the continuous embedding (molecular descriptor): 
smiles_list = ["C1CCCCC1", "CCO"] 
smiles_embedding = inference_model.seq_to_emb(smiles_list) 

# Decode embeddings back to SMILES strings: 
decoded_smiles_list = inference_model.emb_to_seq(smiles_embedding) 
```

## Extracting Molecular Descripotrs
Run the script run_cddd.py to extract molecular descripotrs of your provided SMILES:
```bash
cddd --input smiles.smi --output descriptors.csv  --smiles_header smiles
```
Supported input: 
  * .csv-file with one SMILES per row
  * .smi-file with one SMILES per row

For .csv: Specify the header of the SMILES column with the flag --smiles_header (default: smiles)

## Differences to the original implementation

This port was created to make the embeddings / molecular descriptors usable in a more up to date setup. The ported code is therefore localized to the `src/cddd` folder and installable with the `uv` package manager. The `example` folder has not been updated to use the ported code. 

### References
[1] R. Winter, F. Montanari, F. Noe and D. Clevert, Chem. Sci, 2019, https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc04175j#!divAbstract
