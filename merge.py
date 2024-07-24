from pathlib import Path
import safetensors
from mistral_inference.transformer import Transformer


def merge(base_path, lora_path):
    lora_file = Path(lora_path)
    base_file = Path(base_path)
    if not (lora_file.is_file() and base_file.is_dir()):
        raise Exception(f"File paths incorrect: lora={lora_path}, base={base_path}")
    lora_dir = lora_file.parent.absolute()
    model = Transformer.from_folder(str(base_file.absolute()), device="cpu")
    model.load_lora(str(lora_file.absolute()))
    safetensors.torch.save_model(model, f"{str(lora_dir)}/merged.safetensors")

if __name__ == "__main__":
    base_path = "/mnt/lawbotica-disk-1/models/Mistral-7B-Instruct-v0.3"
    lora_path = "/mnt/lawbotica-disk-1/runs/test-2/checkpoints/checkpoint_000050/consolidated/lora.safetensors"
    merge(base_path, lora_path)
