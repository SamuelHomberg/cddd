import os
import json
import argparse
import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from cddd.pytorch_inference import CDDDEncoder, CDDDDecoder, PyTorchInferenceModel
import cddd.run_cddd_pytorch as run_app


def test_cddd_encoder():
    """Test the PyTorch CDDDEncoder class for correct outputs and shapes."""
    vocab_size = 10
    char_emb_size = 8
    cell_sizes = [16, 16]
    emb_size = 12
    encoder = CDDDEncoder(vocab_size, char_emb_size, cell_sizes, emb_size)
    
    # Create dummy input batch (sorted lengths for pack_padded_sequence)
    x = torch.tensor([[1, 3, 4, 2, 0], [1, 5, 2, 0, 0]])
    lengths = torch.tensor([4, 3])
    
    # Test forward pass with lengths parameter
    out = encoder(x, lengths)
    assert out.shape == (2, emb_size), "Encoder output shape mismatch with lengths"
    
    # Test forward pass without lengths parameter
    out_no_lengths = encoder(x)
    assert out_no_lengths.shape == (2, emb_size), "Encoder output shape mismatch without lengths"


def test_cddd_decoder():
    """Test the PyTorch CDDDDecoder class for correct outputs and shapes."""
    vocab_size = 10
    char_emb_size = 8
    cell_sizes = [16, 16]
    emb_size = 12
    decoder = CDDDDecoder(vocab_size, char_emb_size, cell_sizes, emb_size)
    
    z = torch.randn(2, emb_size)
    max_len = 5
    start_token = 1
    
    out = decoder(z, max_len, start_token)
    assert out.shape == (2, max_len, 1), "Decoder output sequence shape mismatch"


@pytest.fixture
def dummy_model_dir(tmp_path):
    """Fixture that generates a dummy PyTorch model directory."""
    model_dir = tmp_path / "default_model"
    model_dir.mkdir()
    
    hparams = {
        "cell_size": [16],
        "char_embedding_size": 8,
        "emb_size": 12,
        "reverse_decoding": False
    }
    (model_dir / "hparams.json").write_text(json.dumps(hparams))
    
    # Map integer index to character
    vocab = {0: '<unk>', 1: '<s>', 2: '</s>', 3: 'C', 4: 'O', 5: 'N'}
    np.save(model_dir / "indices_char.npy", np.array(vocab))
    
    encoder = CDDDEncoder(len(vocab), 8, [16], 12)
    decoder = CDDDDecoder(len(vocab), 8, [16], 12)
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict()
    }, model_dir / "cddd_model.pt")
    
    return str(model_dir)


def test_pytorch_inference_model_seq_to_emb(dummy_model_dir):
    """Test PyTorchInferenceModel SMILES to embedding functionality."""
    with patch('cddd.pytorch_inference.DEFAULT_DATA_DIR', dummy_model_dir):
        model = PyTorchInferenceModel(dummy_model_dir, use_gpu=False, batch_size=2)
        smiles = ["C", "CCO", "N"]
        emb = model.seq_to_emb(smiles)
        
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (3, 12), "Embedding shape mismatch"


def test_pytorch_inference_model_emb_to_seq(dummy_model_dir):
    """Test PyTorchInferenceModel embedding to SMILES decoding functionality."""
    with patch('cddd.pytorch_inference.DEFAULT_DATA_DIR', dummy_model_dir):
        model = PyTorchInferenceModel(dummy_model_dir, use_gpu=False, batch_size=2)
        embs = np.random.randn(3, 12).astype(np.float32)
        seqs = model.emb_to_seq(embs)
        
        assert isinstance(seqs, list)
        assert len(seqs) == 3
        assert isinstance(seqs[0], str)


@pytest.fixture
def dummy_inputs(tmp_path):
    """Fixture that generates dummy .csv and .smi files to test the app."""
    csv_in = tmp_path / "input.csv"
    smi_in = tmp_path / "input.smi"
    csv_out = tmp_path / "output.csv"
    
    pd.DataFrame({"smiles": ["C", "CCO"]}).to_csv(csv_in, index=False)
    smi_in.write_text("C\nCCO\n")
    
    return str(csv_in), str(smi_in), str(csv_out)


def test_run_cddd_pytorch_read_input(dummy_inputs):
    """Test the file-reading extraction module."""
    csv_in, smi_in, _ = dummy_inputs
    
    with patch('cddd.run_cddd_pytorch.FLAGS', argparse.Namespace(smiles_header="smiles")):
        df_csv = run_app.read_input(csv_in)
        assert len(df_csv) == 2
        assert "smiles" in df_csv.columns
        
        df_smi = run_app.read_input(smi_in)
        assert len(df_smi) == 2
        assert "smiles" in df_smi.columns


@pytest.mark.parametrize("preprocess_flag", [
    "--no-preprocess",
    "--preprocess"
])
@patch('cddd.run_cddd_pytorch.preprocess_smiles', side_effect=lambda x: x)
@patch('cddd.run_cddd_pytorch.PyTorchInferenceModel')
def test_run_cddd_pytorch_main(mock_model_cls, mock_preprocess, preprocess_flag, dummy_inputs, dummy_model_dir):
    """Test the CLI entry point covering the application script."""
    csv_in, _, csv_out = dummy_inputs
    
    instance = mock_model_cls.return_value
    instance.seq_to_emb.return_value = np.zeros((2, 12))  # batch size of 2, embedding dim 12
    
    test_args = [
        "run_cddd_pytorch.py", "--input", csv_in, "--output", csv_out,
        "--smiles_header", "smiles", "--model_dir", dummy_model_dir, preprocess_flag
    ]
    
    with patch('sys.argv', test_args):
        run_app.main()
        
        instance.seq_to_emb.assert_called_once_with(["C", "CCO"])
        assert os.path.exists(csv_out)
        out_df = pd.read_csv(csv_out)
        assert "cddd_1" in out_df.columns and "cddd_12" in out_df.columns