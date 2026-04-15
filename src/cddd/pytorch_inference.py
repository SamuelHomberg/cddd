"""
PyTorch implementation for CDDD Inference.

To use this file, you must first convert the pretrained TensorFlow weights 
into a PyTorch format. You can do this by running the included `convert_tf_to_pt` 
function in an environment where TensorFlow 1.x is installed.

Example usage for conversion (requires TensorFlow):
>>> from cddd.pytorch_inference import convert_tf_to_pt
>>> convert_tf_to_pt('/path/to/default_model')

Example usage for inference (pure PyTorch):
>>> from cddd.pytorch_inference import PyTorchInferenceModel
>>> model = PyTorchInferenceModel('/path/to/default_model')
>>> emb = model.seq_to_emb(["C1CCCCC1", "CCO"])
"""
import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
from importlib import resources

DEFAULT_DATA_DIR = resources.files("cddd").joinpath("data")


REGEX_SML = r'Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps]'

class CDDDEncoder(nn.Module):
    def __init__(self, vocab_size, char_embedding_size, cell_sizes, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, char_embedding_size)
        
        self.grus = nn.ModuleList()
        in_size = char_embedding_size
        for size in cell_sizes:
            self.grus.append(nn.GRU(in_size, size, batch_first=True))
            in_size = size
            
        self.fc_bottleneck = nn.Linear(sum(cell_sizes), embedding_size)
        
    def forward(self, x, lengths=None):
        x = self.embedding(x)
        
        states = []
        for gru in self.grus:
            if lengths is not None:
                packed_x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=True
                )
                packed_out, h = gru(packed_x)
                x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            else:
                x, h = gru(x)
            states.append(h.squeeze(0))
            
        concat_state = torch.cat(states, dim=1)
        emb = self.fc_bottleneck(concat_state)
        return torch.tanh(emb)

class CDDDDecoder(nn.Module):
    def __init__(self, vocab_size, char_embedding_size, cell_sizes, embedding_size):
        super().__init__()
        self.cell_sizes = cell_sizes
        self.embedding = nn.Embedding(vocab_size, char_embedding_size)
        self.fc_init = nn.Linear(embedding_size, sum(cell_sizes))
        
        self.grus = nn.ModuleList()
        in_size = char_embedding_size
        for size in cell_sizes:
            self.grus.append(nn.GRUCell(in_size, size))
            in_size = size
            
        self.fc_out = nn.Linear(cell_sizes[-1], vocab_size, bias=False)

    def forward(self, z, max_len, start_token):
        batch_size = z.size(0)
        device = z.device
        
        init_states = self.fc_init(z)
        h = list(torch.split(init_states, self.cell_sizes, dim=1))
        
        dec_in = torch.full((batch_size,), start_token, dtype=torch.long, device=device)
        
        outputs = []
        for _ in range(max_len):
            x = self.embedding(dec_in)
            for i, gru in enumerate(self.grus):
                h[i] = gru(x, h[i])
                x = h[i]
            
            logits = self.fc_out(x)
            dec_in = logits.argmax(dim=-1)
            outputs.append(dec_in.unsqueeze(1))
            
        return torch.stack(outputs, dim=1)

class PyTorchInferenceModel:
    def __init__(self, model_dir, use_gpu=True, batch_size=256, max_len=1000):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.max_len = max_len
        
        hparams_path = os.path.join(model_dir, 'hparams.json')
        if os.path.exists(hparams_path):
            with open(hparams_path, 'r') as f:
                hparams = json.load(f)
                if isinstance(hparams, str):
                    hparams = json.loads(hparams)
        else:
            hparams = {}
            
        cell_sizes = hparams.get('cell_size', [128])
        char_embedding_size = hparams.get('char_embedding_size', 32)
        embedding_size = hparams.get('emb_size', 128)
        
        vocab_path = hparams.get('encode_vocabulary_file', os.path.join(DEFAULT_DATA_DIR, "indices_char.npy"))
        if not os.path.exists(vocab_path):
            vocab_path = os.path.join(DEFAULT_DATA_DIR, "indices_char.npy")
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)
            original_dict = np.load(vocab_path, allow_pickle=True).item()
        self.encode_vocabulary = {v: k for k, v in original_dict.items()}
        self.decode_vocabulary_reverse = {v: k for k, v in self.encode_vocabulary.items()}
        vocab_size = len(self.encode_vocabulary)
        
        self.encoder = CDDDEncoder(vocab_size, char_embedding_size, cell_sizes, embedding_size).to(self.device)
        
        decoder_cell_sizes = cell_sizes[::-1] if hparams.get('reverse_decoding', False) else cell_sizes
        self.decoder = CDDDDecoder(vocab_size, char_embedding_size, decoder_cell_sizes, embedding_size).to(self.device)
        
        self.encoder.eval()
        self.decoder.eval()
        
        pt_model_path = os.path.join(model_dir, "cddd_model.pt")
        if os.path.exists(pt_model_path):
            state_dict = torch.load(pt_model_path, map_location=self.device)
            self.encoder.load_state_dict(state_dict['encoder'])
            self.decoder.load_state_dict(state_dict['decoder'])
        else:
            print(f"PyTorch model weights not found at {pt_model_path}.")
            print("If you only have TensorFlow weights, run `convert_tf_to_pt` first.")

    def _seq_to_idx(self, seq):
        char_list = re.findall(REGEX_SML, seq)
        idx_list = [self.encode_vocabulary.get(c, self.encode_vocabulary.get('<unk>', 0)) for c in char_list]
        idx_list = [self.encode_vocabulary['<s>']] + idx_list + [self.encode_vocabulary['</s>']]
        return idx_list

    def seq_to_emb(self, seq_list):
        if isinstance(seq_list, str):
            seq_list = [seq_list]
            
        embeddings = []
        with torch.no_grad():
            for ndx in range(0, len(seq_list), self.batch_size):
                batch_seqs = seq_list[ndx:ndx + self.batch_size]
                batch_idxs = [self._seq_to_idx(seq) for seq in batch_seqs]
                
                lengths = torch.tensor([len(s) for s in batch_idxs], dtype=torch.long)
                sorted_lengths, sort_idx = lengths.sort(descending=True)
                
                max_len = sorted_lengths[0].item()
                padded_batch = np.full((len(batch_seqs), max_len), self.encode_vocabulary['</s>'], dtype=np.int64)
                for i, idx in enumerate(sort_idx.numpy()):
                    s = batch_idxs[idx]
                    padded_batch[i, :len(s)] = s
                    
                input_tensor = torch.tensor(padded_batch).to(self.device)
                emb = self.encoder(input_tensor, sorted_lengths)
                
                unsort_idx = torch.argsort(sort_idx)
                emb = emb[unsort_idx]
                
                embeddings.append(emb.cpu().numpy())
                
        return np.concatenate(embeddings, axis=0)

    def emb_to_seq(self, embeddings):
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, 0)
            
        seq_list = []
        start_token = self.encode_vocabulary['<s>']
        
        with torch.no_grad():
            for ndx in range(0, len(embeddings), self.batch_size):
                batch_emb = torch.tensor(embeddings[ndx:ndx + self.batch_size], dtype=torch.float32).to(self.device)
                out_tokens = self.decoder(batch_emb, self.max_len, start_token)
                
                for seq in out_tokens.squeeze(-1).cpu().numpy():
                    chars = []
                    for idx in seq:
                        if idx == self.encode_vocabulary['</s>']:
                            break
                        if idx not in [self.encode_vocabulary['<s>'], -1]:
                            chars.append(self.decode_vocabulary_reverse[idx])
                    seq_list.append(''.join(chars))
        
        if len(seq_list) == 1:
            return seq_list[0]
        return seq_list

def convert_tf_to_pt(model_dir, pt_model_path=None):
    """
    Extracts weights from the pre-trained TF 1.x checkpoint and saves them
    as a PyTorch state_dict format file.
    """
    import tensorflow as tf
    
    if pt_model_path is None:
        pt_model_path = os.path.join(model_dir, "cddd_model.pt")
        
    # Load hparams and vocab to initialize PyTorch structure
    hparams_path = os.path.join(model_dir, 'hparams.json')
    with open(hparams_path, 'r') as f:
        hparams = json.load(f)
        if isinstance(hparams, str):
            hparams = json.loads(hparams)
        
    cell_sizes = hparams.get('cell_size', [128])
    char_embedding_size = hparams.get('char_embedding_size', 32)
    embedding_size = hparams.get('emb_size', 128)
    
    vocab_path = hparams.get('encode_vocabulary_file', os.path.join(DEFAULT_DATA_DIR, "indices_char.npy"))
    if not os.path.exists(vocab_path):
        vocab_path = os.path.join(DEFAULT_DATA_DIR, "indices_char.npy")
        
    # TODO: re-save .npy file to get rid of VisibleDeprecationWarning
    vocab = np.load(vocab_path, allow_pickle=True).item()
    vocab_size = len(vocab)
    
    encoder = CDDDEncoder(vocab_size, char_embedding_size, cell_sizes, embedding_size)
    
    decoder_cell_sizes = cell_sizes[::-1] if hparams.get('reverse_decoding', False) else cell_sizes
    decoder = CDDDDecoder(vocab_size, char_embedding_size, decoder_cell_sizes, embedding_size)
    
    tf_ckpt_path = os.path.join(model_dir, 'model.ckpt')
    reader = tf.train.load_checkpoint(tf_ckpt_path)
    
    enc_state_dict = encoder.state_dict()
    dec_state_dict = decoder.state_dict()
    
    # char embeddings
    enc_state_dict['embedding.weight'] = torch.tensor(reader.get_tensor('char_embedding'))
    dec_state_dict['embedding.weight'] = enc_state_dict['embedding.weight']
    
    # encoder grus
    for i, size in enumerate(cell_sizes):
        gates_k = reader.get_tensor(f'Encoder/rnn/multi_rnn_cell/cell_{i}/gru_cell/gates/kernel')
        gates_b = reader.get_tensor(f'Encoder/rnn/multi_rnn_cell/cell_{i}/gru_cell/gates/bias')
        cand_k = reader.get_tensor(f'Encoder/rnn/multi_rnn_cell/cell_{i}/gru_cell/candidate/kernel')
        cand_b = reader.get_tensor(f'Encoder/rnn/multi_rnn_cell/cell_{i}/gru_cell/candidate/bias')
        
        in_dim = gates_k.shape[0] - size
        w_ih_gates = gates_k[:in_dim, :].T
        w_hh_gates = gates_k[in_dim:, :].T
        b_ih_gates = gates_b.T
        b_hh_gates = np.zeros_like(b_ih_gates)
        
        w_ih_cand = cand_k[:in_dim, :].T
        w_hh_cand = cand_k[in_dim:, :].T
        b_ih_cand = cand_b.T
        b_hh_cand = np.zeros_like(b_ih_cand)
        
        enc_state_dict[f'grus.{i}.weight_ih_l0'] = torch.tensor(np.concatenate([w_ih_gates, w_ih_cand], axis=0))
        enc_state_dict[f'grus.{i}.weight_hh_l0'] = torch.tensor(np.concatenate([w_hh_gates, w_hh_cand], axis=0))
        enc_state_dict[f'grus.{i}.bias_ih_l0'] = torch.tensor(np.concatenate([b_ih_gates, b_ih_cand], axis=0))
        enc_state_dict[f'grus.{i}.bias_hh_l0'] = torch.tensor(np.concatenate([b_hh_gates, b_hh_cand], axis=0))

    # encoder bottleneck
    enc_state_dict['fc_bottleneck.weight'] = torch.tensor(reader.get_tensor('Encoder/dense/kernel').T)
    enc_state_dict['fc_bottleneck.bias'] = torch.tensor(reader.get_tensor('Encoder/dense/bias'))
    
    # decoder init
    dec_state_dict['fc_init.weight'] = torch.tensor(reader.get_tensor('Decoder/dense/kernel').T)
    dec_state_dict['fc_init.bias'] = torch.tensor(reader.get_tensor('Decoder/dense/bias'))
    
    # decoder grus
    for i, size in enumerate(decoder_cell_sizes):
        gates_k = reader.get_tensor(f'Decoder/decoder/multi_rnn_cell/cell_{i}/gru_cell/gates/kernel')
        gates_b = reader.get_tensor(f'Decoder/decoder/multi_rnn_cell/cell_{i}/gru_cell/gates/bias')
        cand_k = reader.get_tensor(f'Decoder/decoder/multi_rnn_cell/cell_{i}/gru_cell/candidate/kernel')
        cand_b = reader.get_tensor(f'Decoder/decoder/multi_rnn_cell/cell_{i}/gru_cell/candidate/bias')
        
        in_dim = gates_k.shape[0] - size
        w_ih_gates = gates_k[:in_dim, :].T
        w_hh_gates = gates_k[in_dim:, :].T
        b_ih_gates = gates_b.T
        b_hh_gates = np.zeros_like(b_ih_gates)
        w_ih_cand = cand_k[:in_dim, :].T
        w_hh_cand = cand_k[in_dim:, :].T
        b_ih_cand = cand_b.T
        b_hh_cand = np.zeros_like(b_ih_cand)
        
        dec_state_dict[f'grus.{i}.weight_ih'] = torch.tensor(np.concatenate([w_ih_gates, w_ih_cand], axis=0))
        dec_state_dict[f'grus.{i}.weight_hh'] = torch.tensor(np.concatenate([w_hh_gates, w_hh_cand], axis=0))
        dec_state_dict[f'grus.{i}.bias_ih'] = torch.tensor(np.concatenate([b_ih_gates, b_ih_cand], axis=0))
        dec_state_dict[f'grus.{i}.bias_hh'] = torch.tensor(np.concatenate([b_hh_gates, b_hh_cand], axis=0))
        
    # decoder projection
    dec_state_dict['fc_out.weight'] = torch.tensor(reader.get_tensor('Decoder/decoder/dense/kernel').T)
    
    encoder.load_state_dict(enc_state_dict)
    decoder.load_state_dict(dec_state_dict)
    
    torch.save({'encoder': enc_state_dict, 'decoder': dec_state_dict}, pt_model_path)
    print(f"Successfully converted and saved to {pt_model_path}")
