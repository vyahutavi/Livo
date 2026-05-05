"""Test full multimodal + speech integration."""
import sys
import os
import tempfile
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from model.multimodal import MultimodalLIVO, MultimodalConfig
from model.llm import Config
from model.speech_decoder import SpeechDecoderConfig, GriffinLimVocoder
from model.vision_encoder import VisionEncoderConfig
from model.audio_encoder import AudioEncoderConfig


def test_full_multimodal_speech():
    """Full multimodal + speech pipeline integration test."""
    config = MultimodalConfig(
        llm=Config(d_model=384, num_layers=2),
        vision=VisionEncoderConfig(d_model=384, num_layers=2),
        audio=AudioEncoderConfig(d_model=384, num_layers=2),
        speech=SpeechDecoderConfig(d_model=384, num_layers=2),
    )
    model = MultimodalLIVO(config)

    image = torch.randn(1, 3, 224, 224)
    mel_input = torch.randn(1, 80, 300)
    text_ids = torch.randint(0, 16384, (1, 20))

    # Forward: image + audio + text -> logits + loss + hidden_states
    out = model(text_ids=text_ids, image=image, audio=mel_input, labels=text_ids)
    assert out.logits is not None
    assert out.loss is not None

    # Speech: hidden_states -> mel spectrogram -> audio waveform
    speech = model.speak(out.hidden_states, max_audio_length=200)
    assert "mel_postnet" in speech

    # Vocoder: mel -> .wav (use temp dir so it works in CI)
    vocoder = GriffinLimVocoder(num_iterations=5)
    audio = vocoder.synthesize(speech["mel_postnet"])
    assert len(audio) > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "test_full.wav")
        vocoder.save_wav(audio, wav_path)
        assert os.path.exists(wav_path)

    # Parameter breakdown
    params = model.num_parameters
    assert params["total"] > 0

