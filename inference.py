import os
import sys
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import boto3
import time
import re

# -----------------------------
# S3 setup
# -----------------------------
input_bucket_name = sys.argv[1]
input_file_key = sys.argv[2]
output_bucket_name = os.environ['OUTPUT_BUCKET_NAME']
output_file_prefix = os.environ['OUTPUT_FILE_PREFIX']

s3_client = boto3.client('s3')

# Download audio from S3
s3_client.download_file(input_bucket_name, input_file_key, input_file_key.split("/")[-1])
audio_path = input_file_key.split("/")[-1]

# -----------------------------
# Load audio
# -----------------------------
waveform, sample_rate = torchaudio.load(audio_path)
# mono & 16kHz
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)
if sample_rate != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

# -----------------------------
# Load Neuron Whisper model
# -----------------------------
suffix = "large-v3"
model_id = f"openai/whisper-{suffix}"

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id, torchscript=True)

# Set Neuron model parameters
model.model.decoder.max_length = 448  # adjust for longer audio chunks if needed
torch.set_num_threads(1)

# -----------------------------
# Chunking audio
# -----------------------------
chunk_size = 30 * 16000        # 30s
overlap = 5 * 16000            # 5s
chunks, start = [], 0
while start < waveform.shape[1]:
    end = min(start + chunk_size, waveform.shape[1])
    chunks.append(waveform[:, max(0, start - overlap):end])
    start += chunk_size

# -----------------------------
# Inference with token-based timestamps
# -----------------------------
all_sentences = []
current_time = 0.0  # seconds
leftover_text = ""
leftover_time = 0.0

t0 = time.time()
for chunk in chunks:
    inputs = processor(chunk.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)
    
    # Decode tokens into text
    transcription = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True).strip()
    chunk_duration = chunk.shape[1] / 16000

    if not transcription:
        # silent chunk
        all_sentences.append({
            "text": "[Music / Silence]",
            "start": current_time,
            "end": current_time + chunk_duration
        })
        current_time += chunk_duration
        continue

    # Prepend leftover text from previous chunk
    if leftover_text:
        transcription = leftover_text + " " + transcription
        word_start_time = leftover_time
        leftover_text = ""
        leftover_time = 0.0
    else:
        word_start_time = current_time

    # Token-based timestamp approximation
    tokens = processor.tokenizer.tokenize(transcription)
    token_duration = chunk_duration / len(tokens)
    sentence = {"text": "", "start": None, "end": None}

    for i, token in enumerate(tokens):
        start = word_start_time
        end = start + token_duration

        if sentence["start"] is None:
            sentence["start"] = start
        sentence["text"] += token.replace("Ġ", " ")  # Whisper tokenizer adds 'Ġ' for spaces
        sentence["end"] = end

        word_start_time = end

        # End sentence on punctuation
        if re.search(r'[.?!]$', token):
            all_sentences.append(sentence)
            sentence = {"text": "", "start": None, "end": None}

    # Handle leftover tokens not ending with punctuation
    if sentence["text"].strip():
        leftover_text = sentence["text"].strip()
        leftover_time = sentence["start"]

    current_time += chunk_duration

# Add leftover sentence
if leftover_text:
    all_sentences.append({
        "text": leftover_text,
        "start": leftover_time,
        "end": current_time
    })

print(f"Elapsed inference: {time.time() - t0:.2f}s")

# -----------------------------
# Save TXT
# -----------------------------
full_transcription = "\n".join(
    [f"[{s['start']:.2f}s - {s['end']:.2f}s] {s['text'].strip()}" for s in all_sentences]
)
output_filename = audio_path + '.txt'
with open(output_filename, 'w') as f:
    f.write(full_transcription)
s3_client.put_object(
    Body=full_transcription,
    Bucket=output_bucket_name,
    Key=output_file_prefix + output_filename
)

# -----------------------------
# Save SRT
# -----------------------------
def seconds_to_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def save_srt(sentences, output_path):
    lines = []
    buffer_sentence = None
    for idx, s in enumerate(sentences, start=1):
        duration = s['end'] - s['start']
        if duration < 0.5 and buffer_sentence:
            buffer_sentence['text'] += " " + s['text'].strip()
            buffer_sentence['end'] = s['end']
            continue
        else:
            if buffer_sentence:
                start_time = seconds_to_srt_time(buffer_sentence['start'])
                end_time = seconds_to_srt_time(buffer_sentence['end'])
                lines.append(f"{len(lines)//4 + 1}\n{start_time} --> {end_time}\n{buffer_sentence['text'].strip()}\n")
            buffer_sentence = s
    if buffer_sentence:
        start_time = seconds_to_srt_time(buffer_sentence['start'])
        end_time = seconds_to_srt_time(buffer_sentence['end'])
        lines.append(f"{len(lines)//4 + 1}\n{start_time} --> {end_time}\n{buffer_sentence['text'].strip()}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return output_path

srt_filename = audio_path.replace(".wav", ".srt")
save_srt(all_sentences, srt_filename)
s3_client.put_object(
    Body=open(srt_filename, "rb"),
    Bucket=output_bucket_name,
    Key=output_file_prefix + srt_filename
)

print(f"TXT + SRT uploaded to S3: {output_file_prefix + srt_filename}")
