from src.cddd.pytorch_inference import convert_tf_to_pt

def main():
    convert_tf_to_pt(model_dir='default_model', pt_model_path=None)

if __name__ == "__main__":
    main()
