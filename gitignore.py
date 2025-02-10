from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    # Use your HF pretrained model identifier or the local directory containing the config and weights.
    model_name_or_dir = "tests/output/tests_test_eval.py_TestEval_test_evaluation_pipeline/policy/cache/actor/hf_pretrained"
    print(f"Loading tokenizer from '{model_name_or_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)
    print("Tokenizer loaded successfully!")

    print(f"Loading model from '{model_name_or_dir}'...")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_dir)
    print("Model loaded successfully!")


if __name__ == "__main__":
    main()
