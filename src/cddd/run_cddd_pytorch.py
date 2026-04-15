"""Module to extract continuous data-driven descriptors for a file of SMILES using PyTorch."""
import os
import sys
import argparse
import pandas as pd
import torch
from importlib import resources
from cddd.pytorch_inference import PyTorchInferenceModel
from cddd.preprocessing import preprocess_smiles

DEFAULT_DATA_DIR = resources.files("cddd").joinpath("data")

_default_model_dir = 'default_model'
FLAGS = None

def add_arguments(parser):
    """Helper function to fill the parser object.

    Args:
        parser: Parser object
    Returns:
        None
    """
    parser.add_argument('-i',
                        '--input',
                        help='input file. Either .smi or .csv file.',
                        type=str)
    parser.add_argument('-o',
                        '--output',
                        help='output .csv file with a descriptor for each SMILES per row.',
                        type=str)
    parser.add_argument('--smiles_header',
                        help='if .csv, specify the name of the SMILES column header here.',
                        default="smiles",
                        type=str)
    parser.add_argument('--preprocess', dest='preprocess', action='store_true')
    parser.add_argument('--no-preprocess', dest='preprocess', action='store_false')
    parser.set_defaults(preprocess=True)
    parser.add_argument('--model_dir', default=_default_model_dir, type=str)
    parser.add_argument('--use_gpu', dest='gpu', action='store_true')
    parser.set_defaults(gpu=False)
    parser.add_argument('--device', default="2", type=str)
    parser.add_argument('--cpu_threads', default=5, type=int)
    parser.add_argument('--batch_size', default=512, type=int)

def read_input(file):
    """Function that reads the provided file into a pandas dataframe.
    Args:
        file: File to read.
    Returns:
        pandas dataframe
    Raises:
        ValueError: If file is not a .smi or .csv file.
    """
    if file.endswith('.csv'):
        sml_df = pd.read_csv(file)
    elif file.endswith('.smi'):
        sml_df = pd.read_table(file, header=None).rename({0: FLAGS.smiles_header, 1: "EXTREG"},
                                                         axis=1)
    else:
        raise ValueError("use .csv or .smi format...")
    return sml_df

def main():
    """Main function that extracts the continuous data-driven descriptors for a file of SMILES."""
    global FLAGS
    PARSER = argparse.ArgumentParser()
    add_arguments(PARSER)
    FLAGS, UNPARSED = PARSER.parse_known_args()
    print(FLAGS)

    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.device)
        
    torch.set_num_threads(FLAGS.cpu_threads)

    model_dir = FLAGS.model_dir
    file = FLAGS.input
    df = read_input(file)
    
    if FLAGS.preprocess:
        print("start preprocessing SMILES...")
        df["new_smiles"] = df[FLAGS.smiles_header].map(preprocess_smiles)
        sml_list = df[~df.new_smiles.isna()].new_smiles.tolist()
        print("finished preprocessing SMILES!")
    else:
        sml_list = df[FLAGS.smiles_header].tolist()
        
    print("start calculating descriptors...")
    infer_model = PyTorchInferenceModel(model_dir=model_dir,
                                        use_gpu=FLAGS.gpu,
                                        batch_size=FLAGS.batch_size)
    descriptors = infer_model.seq_to_emb(sml_list)
    print("finished calculating descriptors! %d out of %d input SMILES could be interpreted"
          %(len(sml_list), len(df)))
          
    # Create cddd_1 to cddd_N dynamically based on embedding feature size
    col_names = ["cddd_" + str(i+1) for i in range(descriptors.shape[1])]
    index = df[~df.new_smiles.isna()].index if FLAGS.preprocess else df.index
    df = df.join(pd.DataFrame(descriptors, index=index, columns=col_names))
                                  
    print("writing descriptors to file...")
    df.to_csv(FLAGS.output)

if __name__ == "__main__":
    main()