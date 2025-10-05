import os
import sys
import types
import torch
import torchaudio
import torch.nn.functional as F
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutput
import boto3
import time
import re

# -----------------------------
# Environment & arguments
# -----------------------------
os.environ['NEURON_RT_NUM_CORES']='1'
input_bucket_name = sys.argv[1]
input_file_key = sys.argv[2]

output_bucket_name = os.environ['OUTPUT_BUCKET_NAME']
output_file_prefix = os.environ['OUTPUT_FILE_PREFIX']

model_artifact_bucket_name = os.environ['MODEL_BUCKET_NAME']
model_artifact_encoder_key = os.environ['MODEL_ENCODER_S3_KEY']
model_artifact_decoder_key = os.environ['MODEL_DECODER_S3_KEY']
model_artifact_proj_key = os.environ['MODEL_PROJ_S3_KEY']

s3_client = boto3.client('s3')

# -----------------------------
# Model setup
# -----------------------------
suffix="large-v3"
model_id=f"openai/whisper-{suffix}"
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id, torchscript=True)

batch_size=1
output_attentions=True
max_dec_len = 448
dim_enc=model.config.num_mel_bins
dim_dec=model.config.d_model
print(f'Dim enc: {dim_enc}; Dim dec: {dim_dec}')

# -----------------------------
# Neuron forward overrides
# -----------------------------
def enc_f(self, input_features, attention_mask, **kwargs):
    if hasattr(self, 'forward_neuron'):
        out = self.forward_neuron(input_features, attention_mask)
    else:
        out = self.forward_(input_features, attention_mask, return_dict=True)
    return BaseModelOutput(**out)

def dec_f(self, input_ids, attention_mask=None, encoder_hidden_states=None, **kwargs):
    inp = [input_ids, encoder_hidden_states]
    if inp[0].shape[1] > self.max_length:
        raise Exception(f"The decoded sequence is not supported. Max: {self.max_length}")
    pad_size = torch.as_tensor(self.max_length - inp[0].shape[1])
    inp[0] = F.pad(inp[0], (0, pad_size), "constant", processor.tokenizer.pad_token_id)

    if hasattr(self, 'forward_neuron'):
        out = self.forward_neuron(*inp)
    else:
        out = self.forward_(input_ids=inp[0], encoder_hidden_states=inp[1],
                            return_dict=True, use_cache=False, output_attentions=output_attentions)

    out['last_hidden_state'] = out['last_hidden_state'][:, :input_ids.shape[1], :]
    if out.get('attentions') is not None:
        out['attentions'] = torch.stack([torch.mean(o[:, :, :input_ids.shape[1], :input_ids.shape[1]], axis=2, keepdim=True)
                                         for o in out['attentions']])
    if out.get('cross_attentions') is not None:
        out['cross_attentions'] = torch.stack([torch.mean(o[:, :, :input_ids.shape[1], :], axis=2, keepdim=True)
                                               for o in out['cross_attentions']])
    return BaseModelOutputWithPastAndCrossAttentions(**out)

def proj_out_f(self, inp):
    pad_size = torch.as_tensor(self.max_length - inp.shape[1], device=inp.device)
    x = F.pad(inp, (0,0,0,pad_size), "constant", processor.tokenizer.pad_token_id)
    if hasattr(self, 'forward_neuron'):
        out = self.forward_neuron(x)
    else:
        out = self.forward_(x)
    return out[:, :inp.shape[1], :]

for m in [model.model.encoder, model.model.decoder, model.proj_out]:
    if not hasattr(m, 'forward_'):
        m.forward_ = m.forward

model.model.encoder.forward = types.MethodType(enc_f, model.model.encoder)
model.model.decoder.forward = types.MethodType(dec_f, model.model.decoder)
model.proj_out.forward = types.MethodType(proj_out_f, model.proj_out)
model.model.decoder.max_length = max_dec_len
model.proj_out.max_length = max_dec_len

# -----------------------------
# Load Neuron model artifacts from S3
# -----------------------------
for s3_key, attr in [(model_artifact_encoder_key, 'encoder'), 
                     (model_artifact_decoder_key, 'decoder'),
                     (model_artifact_proj_key, 'proj_out')]:
    local_file = s3_key.split('/')[-1]
    s3_client.download_file(model_artifact_bucket_name, s3_key, local_file)
    if not os.path.isfile(local_file):
        raise Exception(f"{attr} model artifact not found.")
    setattr(model.model if attr != 'proj_out' else model, f"{attr}.forward_neuron", torch.jit.load(local_file))

# -----------------------------
# Load input audio from S3
# -----------------------------
local_audio_file = input_file_key.split("/")[-1]
s3_client.download_file(input_bucket_name, input_file_key, local_audio_file)
waveform, sample_rate = torchaudio.load(local_audio_file)

# Mono & 16kHz
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)
if sample_rate != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

# -----------------------------
# Silence/Music detection
# -----------------------------
def detect_silence(waveform, sample_rate, frame_size=1024, hop_size=512, threshold=0.01):
    waveform = waveform.mean(dim=0)
    num_frames = (waveform.shape[0] - frame_size) // hop_size + 1
    segments, current_start, is_speech = [], 0, False
    for i in range(num_frames):
        start = i*hop_size
        frame = waveform[start:start+frame_size]
        energy = torch.sqrt(torch.mean(frame**2))
        speech_frame = energy > threshold
        if speech_frame != is_speech:
            segments.append({"start": current_start/sample_rate, "end": start/sample_rate, "speech": is_speech})
            current_start = start
            is_speech = speech_frame
    segments.append({"start": current_start/sample_rate, "end": waveform.shape[1]/sample_rate, "speech": is_speech})
    return segments

segments = detect_silence(waveform, 16000)

# -----------------------------
# Inference & timestamp generation
# -----------------------------
all_sentences = []

for seg in segments:
    seg_wave = waveform[:, int(seg['start']*16000):int(seg['end']*16000)]
    duration = seg['end'] - seg['start']
    
    if not seg['speech']:
        all_sentences.append({"text":"[Music / Silence]", "start":seg['start'], "end":seg['end']})
        continue
    
    inputs = processor(seg_wave.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True).strip()
    if not transcription: 
        all_sentences.append({"text":"[Unintelligible]", "start":seg['start'], "end":seg['end']})
        continue

    words = transcription.split()
    total_chars = sum(len(w) for w in words)
    char_time_ratio = duration / total_chars
    word_start_time = seg['start']

    sentence = {"text":"", "start":None, "end":None}
    for word in words:
        word_duration = len(word) * char_time_ratio
        start = word_start_time
        end = start + word_duration
        if sentence["start"] is None:
            sentence["start"] = start
        sentence["text"] += word + " "
        sentence["end"] = end
        word_start_time = end
        if re.search(r'[.?!]$', word):
            all_sentences.append(sentence)
            sentence = {"text":"", "start":None, "end":None}
    if sentence["text"].strip():
        all_sentences.append({"text":sentence["text"].strip(), "start":sentence["start"], "end":sentence["end"]})

# -----------------------------
# Save TXT
# -----------------------------
full_transcription = "\n".join([f"[{s['start']:.2f}s - {s['end']:.2f}s] {s['text'].strip()}" for s in all_sentences])
txt_filename = local_audio_file + ".txt"
with open(txt_filename, "w") as f:
    f.write(full_transcription)
s3_client.put_object(Body=full_transcription, Bucket=output_bucket_name, Key=output_file_prefix+txt_filename)

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
    for s in sentences:
        duration = s['end'] - s['start']
        if duration < 0.5 and buffer_sentence is not None:
            buffer_sentence['text'] += " " + s['text'].strip()
            buffer_sentence['end'] = s['end']
            continue
        else:
            if buffer_sentence:
                lines.append(f"{len(lines)//4+1}\n{seconds_to_srt_time(buffer_sentence['start'])} --> {seconds_to_srt_time(buffer_sentence['end'])}\n{buffer_sentence['text'].strip()}\n")
            buffer_sentence = s
    if buffer_sentence:
        lines.append(f"{len(lines)//4+1}\n{seconds_to_srt_time(buffer_sentence['start'])} --> {seconds_to_srt_time(buffer_sentence['end'])}\n{buffer_sentence['text'].strip()}\n")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return output_path

srt_filename = local_audio_file.replace(".wav",".srt")
save_srt(all_sentences, srt_filename)
s3_client.put_object(Body=open(srt_filename,"rb"), Bucket=output_bucket_name, Key=output_file_prefix+srt_filename)

print(f"TXT & SRT uploaded to S3 at {output_file_prefix}{txt_filename} & {output_file_prefix}{srt_filename}")
