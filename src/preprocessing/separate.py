import os
import subprocess


def separate_vocals(input_wav: str, out_dir: str = "separated") -> str:
    """
    Uses Demucs to separate vocals from accompaniment.
    Returns path to the separated vocals WAV file.
    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Call demucs via subprocess
    cmd = [
        "demucs",
        "--two-stems=vocals",
        input_wav,
        "--out",
        out_dir
    ]
    subprocess.run(cmd, check=True)

    # Demucs outputs in structure: out_dir/{model_name}/{input_filename}/vocals.wav
    # Find the generated vocals file
    base_name = os.path.splitext(os.path.basename(input_wav))[0]
    # Demucs uses the model folder "htdemucs" by default
    vocals_path = os.path.join(out_dir, "htdemucs", base_name, "vocals.wav")
    if not os.path.isfile(vocals_path):
        raise FileNotFoundError(f"Expected separated vocals at {vocals_path}")
    return vocals_path

# Example usage:
# vocals = separate_vocals("data/raw/preview.wav")
# print(f"Extracted vocals at: {vocals}")