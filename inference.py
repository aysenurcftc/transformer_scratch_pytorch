import torch
from config import get_config, latest_weights_file_path
from train import get_model, get_ds
from translate import translate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


config = get_config()


_, _, tokenizer_src, tokenizer_tgt = get_ds(config)

# Load the model
model = get_model(
    config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
).to(device)

# Load the trained weights
model_filename = latest_weights_file_path(config)
state = torch.load(model_filename, map_location=device, weights_only=True)
model.load_state_dict(state["model_state_dict"])
model.eval()


def perform_translation(sentence: str):
    if not isinstance(sentence, str):
        raise TypeError("Input text must be a string.")
    if not sentence.strip():
        return "Input text cannot be empty."
    return translate(sentence)


# Example usage
input_sentence = "I like reading books."
translated_sentence = perform_translation(input_sentence)
print(f"Input: {input_sentence}\nTranslation: {translated_sentence}")
