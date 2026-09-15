import sys
import os
sys.path.insert(0, os.path.abspath('src'))
import numpy as np
from anc.interface.processor import AudioProcessingPipeline

proc = AudioProcessingPipeline()
dummy_audio = np.random.randn(16000 * 2).astype(np.float32)

print("Testing DTLN Quantized...")
try:
    res = proc.process(dummy_audio, sample_rate=16000, model_name="dtln_quantized")
    print("Success")
except Exception as e:
    print("Error:", e)

print("Testing RNNoise...")
try:
    res = proc.process(dummy_audio, sample_rate=16000, model_name="rnnoise")
    print("Success")
except Exception as e:
    print("Error:", e)
    
print("Testing Conv-TasNet...")
try:
    res = proc.process(dummy_audio, sample_rate=16000, model_name="conv_tasnet")
    print("Success")
except Exception as e:
    print("Error:", e)
