import os
import sys
import torch
import torchaudio
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import boto3
import time

# --- Input/output config ---
input_bucket_name = sys.argv[1]
input_file_key = sys.argv[2]

output_bucket_name = os.environ['OUTPUT_BUCKET_NAME']
output_file_prefix = os.environ['OUTPUT_FILE_PREFIX']

model_artifact_bucket_name = os.environ['MODEL_BUCKET_NAME']
model_artifact_encoder_key = os.environ['MODEL_ENCODER_S3_KEY']
model_artifact_decoder_key = os.environ['MODEL_DECODER_S3_KEY']
model_artifact_proj_key = os.environ['MODEL_PROJ_S3_KEY']

s3_client = boto3.client('s3')

# --- Model config ---
suffix = "large-v3"
model_id = f"openai/whisper-{suffix}"

# --- Load processor ---
processor = WhisperProcessor.from_pretrained(model_id)

# --- Load models ---
# Fast Neuron model for quick text transcription (no accurate timestamps)
model = WhisperForConditionalGeneration.from_pretrained(model_id, torchscript=True)
# CPU model for accurate timestamps
cpu_model = WhisperForConditionalGeneration.from_pretrained(model_id, torchscript=True)

# --- Download audio from S3 ---
s3_client.download_file(input_bucket_name, input_file_key, input_file_key.split("/")[-1])
audio_path = input_file_key.split("/")[-1]

# --- Load audio ---
waveform, sample_rate = torchaudio.load(audio_path)

# Convert to mono and 16kHz
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)
if sample_rate != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

# --- Chunk audio ---
chunk_size = 30*16000  # 30 seconds
chunks = waveform.split(chunk_size, dim=1)

# --- Run inference ---
transcriptions_text = []
transcriptions_timestamps = []

start_time = time.time()

for chunk in chunks:
    inputs = processor(chunk.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    
    # 1️⃣ Fast transcription (Neuron)
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)
    text_chunk = processor.decode(predicted_ids[0], skip_special_tokens=True).strip()
    transcriptions_text.append(text_chunk)
    
    # 2️⃣ Accurate timestamps (CPU)
    with torch.no_grad():
        predicted_ids_cpu = cpu_model.generate(inputs.input_features, return_timestamps=True)
    text_chunk_cpu = processor.decode(predicted_ids_cpu[0], skip_special_tokens=True).strip()
    transcriptions_timestamps.append({
        "text": text_chunk_cpu,
        "timestamps": getattr(predicted_ids_cpu, "timestamps", None)  # may need adjustment based on output type
    })

print(f"Elapsed total: {time.time()-start_time:.2f}s")

# --- Combine transcriptions ---
full_text = " ".join(transcriptions_text)

# --- Save combined transcription ---
output_filename = audio_path + '.txt'
with open(output_filename, 'w') as f:
    f.write(full_text)

# Upload to S3
s3_client.put_object(
    Body=full_text,
    Bucket=output_bucket_name,
    Key=output_file_prefix + output_filename
)

print("Transcription complete. Full text uploaded to S3.")
