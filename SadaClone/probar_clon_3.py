import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

print("Cargando SadaClone - prueba 3...")

model = Qwen3TTSModel.from_pretrained(
    r"C:\SadaClone\Qwen3-TTS-0.6B",
    device_map="cpu",
    dtype=torch.float32,
)

print("Modelo cargado.")
print("Generando clon estable...")

wavs, sr = model.generate_voice_clone(
    text=(
        "Hola. Esto es SadaClone. "
        "Ahora estoy probando una versión más natural de mi voz. "
        "Quiero hablar con claridad, manteniendo mi identidad y mi forma de expresarme."
    ),
    language="Spanish",
    ref_audio=r"C:\SadaClone\rosa.mp3",
    x_vector_only_mode=True,
    non_streaming_mode=True,
)

sf.write(
    r"C:\SadaClone\resultado_rosa_3.wav",
    wavs[0],
    sr
)

print("¡Prueba 3 terminada!")
print(r"Audio: C:\SadaClone\resultado_rosa_3.wav")