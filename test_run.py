import torch
import numpy as np
from src.data.preprocess import AudioPreprocessor
from src.models.hybrid_model import HybridAudioClassifier


def main():
    print("1. Initializing Preprocessor...")
    preprocessor = AudioPreprocessor(duration=5.0)

    audio_path = "test_audio.wav"
    print(f"2. Processing Audio File: {audio_path}")

    try:
        spectrogram_tensor = preprocessor.process_file(audio_path)
    except Exception as e:
        print(f"Failed to process audio: {e}")
        return

    if spectrogram_tensor is None:
        print("Failed to process audio. Check the path.")
        return

    # Add batch dimension: [1, 1, n_mels, time_frames]
    batch_tensor = spectrogram_tensor.unsqueeze(0)
    print(f"3. Input Spectrogram Shape: {batch_tensor.shape}")

    print("4. Initializing Hybrid Model...")
    model = HybridAudioClassifier(num_genres=10)

    print("5. Running Forward Pass...")
    model.eval()
    with torch.no_grad():
        output = model(batch_tensor)

    print(f"6. Final Model Output Shape: {output.shape}")
    print("Expectation: [1, 10]")


if __name__ == "__main__":
    main()
