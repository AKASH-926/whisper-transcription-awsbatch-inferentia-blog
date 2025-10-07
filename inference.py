import os
import sys
os.environ['NEURON_RT_NUM_CORES']='1'
import types
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import boto3

input_bucket_name = sys.argv[1]
input_file_key = sys.argv[2]

output_bucket_name = os.environ['OUTPUT_BUCKET_NAME']
output_file_prefix =  os.environ['OUTPUT_FILE_PREFIX']

model_artifact_bucket_name = os.environ['MODEL_BUCKET_NAME']
model_artifact_encoder_key = os.environ['MODEL_ENCODER_S3_KEY']
model_artifact_decoder_key = os.environ['MODEL_DECODER_S3_KEY']
model_artifact_proj_key = os.environ['MODEL_PROJ_S3_KEY']

s3_client = boto3.client('s3')

# please, start by selecting the desired model size
#suffix="tiny"
#suffix="small"
#suffix="medium"
suffix="large-v3"
model_id=f"openai/whisper-{suffix}"

# this will load the tokenizer + two copies of the model.
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id, torchscript=True)

# batch size refers to the number of files processed in parallel
# and should not exceed the number of cores on the Neuron device.
batch_size=1

# output_attentions is required if you want to return word timestamps
# if you don't need timestamps, just set this to False and get some better latency
output_attentions=True

# this is the maximum number of tokens the model will be able to decode
# for the sample #3 we selected above, this is enough. If you're planning to 
# process larger samples, you need to adjust it accordinly.
max_dec_len = 448
# num_mel_bins,d_model --> these parameters where copied from model.conf (found on HF repo)
# we need them to correctly generate dummy inputs during compilation
dim_enc=model.config.num_mel_bins
dim_dec=model.config.d_model
print(f'Dim enc: {dim_enc}; Dim dec: {dim_dec}')


import types
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions,BaseModelOutput

# Now we need to simplify both encoding & decoding forward methods to make them 
# compilable. Please notice that these methods overwrite the original ones, but
# keeps retro-compatibility. Also, we'll use use a new variable "forward_neuron"
# to invoke the model on inf2
def enc_f(self, input_features, attention_mask, **kwargs):
    if hasattr(self, 'forward_neuron'):
        out = self.forward_neuron(input_features, attention_mask)
    else:
        out = self.forward_(input_features, attention_mask, return_dict=True)
    return BaseModelOutput(**out)

def dec_f(self, input_ids, attention_mask=None, encoder_hidden_states=None, **kwargs):
    out = None        
    if not attention_mask is None and encoder_hidden_states is None:
        # this is a workaround to align the input parameters for NeuronSDK tracer
        # None values are not allowed during compilation
        encoder_hidden_states, attention_mask = attention_mask,encoder_hidden_states
    inp = [input_ids, encoder_hidden_states]
    
    # pad the input to max_dec_len
    if inp[0].shape[1] > self.max_length:
        raise Exception(f"The decoded sequence is not supported. Max: {self.max_length}")
    pad_size = torch.as_tensor(self.max_length - inp[0].shape[1])
    inp[0] = F.pad(inp[0], (0, pad_size), "constant", processor.tokenizer.pad_token_id)
    
    if hasattr(self, 'forward_neuron'):
        out = self.forward_neuron(*inp)
    else:
        # output_attentions is required if you want timestamps
        out = self.forward_(input_ids=inp[0], encoder_hidden_states=inp[1], return_dict=True, use_cache=False, output_attentions=output_attentions)
    # unpad the output
    out['last_hidden_state'] = out['last_hidden_state'][:, :input_ids.shape[1], :]
    # neuron compiler doesn't like tuples as values of dicts, so we stack them into tensors
    # also, we need to average axis=2 given we're not using cache (use_cache=False)
    # that way, to avoid an issue with the pipeline we change the shape from:
    #  bs,num selected,num_tokens,1500 --> bs,1,num_tokens,1500
    # I suspect there is a bug in the HF pipeline code that doesn't support use_cache=False for
    # word timestamps, that's why we need that.
    if not out.get('attentions') is None:
        out['attentions'] = torch.stack([torch.mean(o[:, :, :input_ids.shape[1], :input_ids.shape[1]], axis=2, keepdim=True) for o in out['attentions']])
    if not out.get('cross_attentions') is None:
        out['cross_attentions'] = torch.stack([torch.mean(o[:, :, :input_ids.shape[1], :], axis=2, keepdim=True) for o in out['cross_attentions']])
    return BaseModelOutputWithPastAndCrossAttentions(**out)

if not hasattr(model.model.encoder, 'forward_'): 
    model.model.encoder.forward_ = model.model.encoder.forward
if not hasattr(model.model.decoder, 'forward_'): 
    model.model.decoder.forward_ = model.model.decoder.forward
if not hasattr(model.proj_out, 'forward_'):
    model.proj_out.forward_ = model.proj_out.forward

def proj_out_f(self, inp):
    pad_size = torch.as_tensor(self.max_length - inp.shape[1], device=inp.device)
    # pad the input to max_dec_len
    if inp.shape[1] > self.max_length:
        raise Exception(f"The decoded sequence is not supported. Max: {self.max_length}")
    x = F.pad(inp, (0,0,0,pad_size), "constant", processor.tokenizer.pad_token_id)

    if hasattr(self, 'forward_neuron'):
        out = self.forward_neuron(x)
    else:
        out = self.forward_(x)
    # unpad the output before returning
    out = out[:, :inp.shape[1], :]
    return out

model.model.encoder.forward = types.MethodType(enc_f, model.model.encoder)
model.model.decoder.forward = types.MethodType(dec_f, model.model.decoder)
model.proj_out.forward = types.MethodType(proj_out_f, model.proj_out)

model.model.decoder.max_length = max_dec_len
model.proj_out.max_length = max_dec_len

# Trace Encoder
import os
import torch
import torch_neuronx

# download model artifacts from S3
s3_client.download_file(model_artifact_bucket_name, model_artifact_encoder_key, model_artifact_encoder_key.split("/")[-1])
model_encoder_filename=model_artifact_encoder_key.split("/")[-1]
if not os.path.isfile(model_encoder_filename):
    raise Exception("encoder model artifact not found.")
else:
    model.model.encoder.forward_neuron = torch.jit.load(model_encoder_filename)
    
# Trace Decoder
import torch
import torch_neuronx

s3_client.download_file(model_artifact_bucket_name, model_artifact_decoder_key, model_artifact_decoder_key.split("/")[-1])
model_decoder_filename=model_artifact_decoder_key.split("/")[-1]
if not os.path.isfile(model_decoder_filename):
    raise Exception("decoder model artifact not found.")
else:
    model.model.decoder.forward_neuron = torch.jit.load(model_decoder_filename)
    
# Trace Projection Output
import torch
import torch_neuronx

s3_client.download_file(model_artifact_bucket_name, model_artifact_proj_key, model_artifact_proj_key.split("/")[-1])
model_proj_filename=model_artifact_proj_key.split("/")[-1]
if not os.path.isfile(model_proj_filename):
    raise Exception("projection model artifact not found.")
else:
    model.proj_out.forward_neuron = torch.jit.load(model_proj_filename)

# Inference
import torchaudio
import ffmpeg
import numpy as np

# copy from s3
s3_client.download_file(input_bucket_name, input_file_key, input_file_key.split("/")[-1])
audio_path = input_file_key.split("/")[-1]

def load_audio_with_fallback(file_path):
    """
    Load audio from various formats including MP4, with fallback mechanisms.
    Supports: MP3, MP4, WAV, FLAC, OGG, M4A, WMA, AAC, and more.
    
    Returns:
        tuple: (waveform as torch.Tensor, sample_rate as int)
    """
    print(f"Loading audio file: {file_path}")
    
    # Method 1: Try torchaudio first (fastest if it works)
    try:
        # Try to set ffmpeg backend for better format support
        try:
            torchaudio.set_audio_backend("ffmpeg")
            print("Using torchaudio with ffmpeg backend")
        except:
            print("FFmpeg backend not available for torchaudio, using default backend")
        
        waveform, sample_rate = torchaudio.load(file_path)
        print(f"Successfully loaded with torchaudio: {sample_rate}Hz, {waveform.shape}")
        return waveform, sample_rate
    except Exception as e:
        print(f"torchaudio failed: {e}")
        print("Falling back to ffmpeg-python...")
    
    # Method 2: Use ffmpeg-python for broader format support (MP4, video files, etc.)
    try:
        # Probe the file to get info
        probe = ffmpeg.probe(file_path)
        audio_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        
        if audio_info is None:
            raise Exception("No audio stream found in file")
        
        original_sample_rate = int(audio_info['sample_rate'])
        print(f"File info: {audio_info['codec_name']} codec, {original_sample_rate}Hz")
        
        # Extract audio using ffmpeg and convert to 16kHz mono
        out, _ = (
            ffmpeg
            .input(file_path)
            .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar='16000')
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        
        # Convert bytes to numpy array then to torch tensor
        audio_np = np.frombuffer(out, np.float32)
        waveform = torch.from_numpy(audio_np).unsqueeze(0)  # Add channel dimension
        sample_rate = 16000
        
        print(f"Successfully loaded with ffmpeg: {sample_rate}Hz, {waveform.shape}")
        return waveform, sample_rate
        
    except Exception as e:
        print(f"ffmpeg-python failed: {e}")
        print("Falling back to librosa...")
    
    # Method 3: Final fallback to librosa (slowest but most compatible)
    try:
        import librosa
        audio_np, sample_rate = librosa.load(file_path, sr=None, mono=False)
        
        # Convert to torch tensor and ensure correct shape
        if audio_np.ndim == 1:
            waveform = torch.from_numpy(audio_np).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_np)
        
        print(f"Successfully loaded with librosa: {sample_rate}Hz, {waveform.shape}")
        return waveform, sample_rate
        
    except Exception as e:
        raise Exception(f"All audio loading methods failed. Last error: {e}")

# Load the audio file with fallback support for various formats
try:
    waveform, sample_rate = load_audio_with_fallback(audio_path)
except Exception as e:
    print(f"ERROR: Failed to load audio file: {e}")
    raise

# Ensure the audio is in the correct format (mono, 16kHz)
if waveform.shape[0] > 1:
    print(f"Converting from {waveform.shape[0]} channels to mono")
    waveform = torch.mean(waveform, dim=0, keepdim=True)
if sample_rate != 16000:
    print(f"Resampling from {sample_rate}Hz to 16000Hz")
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

# Smart chunking using VAD (Voice Activity Detection) with overlap
# This approach detects natural silence points to avoid cutting words
# GUARANTEES COMPLETE AUDIO COVERAGE:
# - Always starts at 0.0s and ends at full audio duration
# - 2 second overlap between chunks ensures no content is lost at boundaries
# - Validates for gaps and falls back to safe mode if needed
# - Conservative merging keeps content when in doubt
def create_vad_chunks(waveform, max_chunk_duration=29, overlap_duration=1.5, min_silence_duration=0.3):
    """
    Create audio chunks at natural silence points using energy-based VAD with overlap.
    
    Args:
        waveform: Audio tensor (1, num_samples)
        max_chunk_duration: Maximum chunk duration in seconds (use 29s to stay safely under 30s)
        overlap_duration: Overlap between consecutive chunks in seconds
        min_silence_duration: Minimum silence duration to consider as a split point
    
    Returns:
        List of tuples: [(start_sample, end_sample, start_time, end_time), ...]
    """
    sample_rate = 16000
    audio = waveform.squeeze().numpy()
    audio_length = len(audio)
    audio_duration = audio_length / sample_rate
    
    print(f"Total audio duration: {audio_duration:.2f}s")
    
    # If audio is short enough, return as single chunk
    if audio_duration <= max_chunk_duration:
        print("Audio fits in single chunk, no splitting needed")
        return [(0, audio_length, 0.0, audio_duration)]
    
    # Calculate energy in sliding windows
    window_size = int(0.02 * sample_rate)  # 20ms windows
    hop_size = int(0.01 * sample_rate)  # 10ms hop
    
    # Calculate RMS energy for each window
    energies = []
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i + window_size]
        energy = np.sqrt(np.mean(window ** 2))
        energies.append(energy)
    
    energies = np.array(energies)
    
    # Determine silence threshold (more conservative)
    # Use 10th percentile instead of 20th to be more selective
    if len(energies) > 0:
        threshold = np.percentile(energies, 10)  # Bottom 10% is considered silence
        # Use median as reference for adaptive threshold
        median_energy = np.median(energies)
        # Set threshold as percentage of median (more robust)
        threshold = max(threshold, median_energy * 0.1, 0.005)
    else:
        threshold = 0.01
    
    print(f"VAD threshold: {threshold:.4f}, Median energy: {np.median(energies):.4f}, Max energy: {np.max(energies):.4f}")
    
    # Find silence regions
    is_silence = energies < threshold
    min_silence_samples = int(min_silence_duration * sample_rate / hop_size)
    
    # Find continuous silence regions
    silence_regions = []
    in_silence = False
    silence_start = 0
    
    for i, silent in enumerate(is_silence):
        if silent and not in_silence:
            # Start of silence
            in_silence = True
            silence_start = i
        elif not silent and in_silence:
            # End of silence
            silence_duration = i - silence_start
            if silence_duration >= min_silence_samples:
                # Convert to sample indices (middle of silence region)
                sample_idx = (silence_start + i) // 2 * hop_size
                silence_regions.append(sample_idx)
            in_silence = False
    
    print(f"Found {len(silence_regions)} silence points")
    
    # Create chunks based on silence points with overlap
    chunks = []
    max_chunk_samples = int(max_chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    
    # CRITICAL: Always start at 0 to ensure no content is lost
    chunk_start = 0
    
    while chunk_start < audio_length:
        # Target end point
        target_end = chunk_start + max_chunk_samples
        
        if target_end >= audio_length - (0.5 * sample_rate):  # Within 0.5s of end
            # Last chunk - take everything remaining to ensure complete coverage
            chunks.append((
                chunk_start,
                audio_length,
                chunk_start / sample_rate,
                audio_length / sample_rate
            ))
            break
        
        # Find the best silence point near the target end
        # Look in a window around target_end (±3 seconds)
        search_window = int(3 * sample_rate)
        search_start = max(target_end - search_window, chunk_start + int(10 * sample_rate))  # Don't go too far back
        search_end = min(target_end + search_window, audio_length)
        
        # Find silence points in the search window
        valid_silence_points = [sp for sp in silence_regions 
                               if search_start <= sp <= search_end]
        
        if valid_silence_points:
            # Choose the silence point closest to target_end
            chunk_end = min(valid_silence_points, key=lambda x: abs(x - target_end))
        else:
            # No silence found, use target_end but ensure minimum progress
            chunk_end = target_end
            print(f"Warning: No silence point found near {target_end/sample_rate:.2f}s, forcing split")
        
        # Ensure chunk is not too short and we're making progress
        if chunk_end - chunk_start < int(5 * sample_rate):  # Minimum 5 seconds
            chunk_end = min(chunk_start + int(10 * sample_rate), audio_length)
        
        chunks.append((
            chunk_start,
            chunk_end,
            chunk_start / sample_rate,
            chunk_end / sample_rate
        ))
        
        # Next chunk starts with overlap
        # Move forward by (chunk_duration - overlap_duration) to ensure overlap
        step_samples = max_chunk_samples - overlap_samples
        chunk_start = chunk_start + step_samples
        
        # Ensure we're making progress (minimum 10 seconds forward)
        if chunks:
            last_start = chunks[-1][0]
            if chunk_start - last_start < int(10 * sample_rate):
                chunk_start = last_start + int(10 * sample_rate)
    
    # Validate and merge very short chunks
    validated_chunks = []
    for i, chunk in enumerate(chunks):
        start_sample, end_sample, start_time, end_time = chunk
        duration = end_time - start_time
        
        # If chunk is very short (< 1 second) and not the last chunk, merge with next
        if duration < 1.0 and i < len(chunks) - 1:
            print(f"Chunk {i+1} is too short ({duration:.2f}s), will merge with next")
            continue
        
        validated_chunks.append(chunk)
    
    return validated_chunks if validated_chunks else [(0, audio_length, 0.0, audio_duration)]

# Create chunks at natural silence points with overlap
# Using 2 seconds overlap for maximum safety - ensures no content is lost
chunk_info = create_vad_chunks(waveform, max_chunk_duration=28, overlap_duration=2.0, min_silence_duration=0.3)
print(f"\nCreated {len(chunk_info)} chunks at natural silence points with overlap:")

# CRITICAL: Validate complete audio coverage - NO GAPS ALLOWED
def validate_chunk_coverage(chunk_info, total_duration):
    """Ensure all audio time is covered by at least one chunk"""
    print("\n🔍 Validating chunk coverage...")
    
    if not chunk_info:
        raise Exception("ERROR: No chunks created!")
    
    # Check start coverage
    if chunk_info[0][2] > 0.1:  # start_time of first chunk
        print(f"⚠️  WARNING: First chunk starts at {chunk_info[0][2]:.2f}s, not 0.0s")
        print(f"   Adding initial chunk to cover 0.0s - {chunk_info[0][2]:.2f}s")
        return False
    
    # Check end coverage
    last_end = chunk_info[-1][3]
    if last_end < total_duration - 0.1:
        print(f"⚠️  WARNING: Last chunk ends at {last_end:.2f}s, but audio is {total_duration:.2f}s")
        print(f"   Missing {total_duration - last_end:.2f}s at the end!")
        return False
    
    # Check for gaps between chunks
    has_gaps = False
    for i in range(1, len(chunk_info)):
        prev_end = chunk_info[i-1][3]
        current_start = chunk_info[i][2]
        
        if current_start > prev_end + 0.01:  # Gap detected (allowing 0.01s tolerance)
            gap_duration = current_start - prev_end
            print(f"⚠️  GAP DETECTED between chunk {i} and {i+1}: {gap_duration:.2f}s")
            print(f"   Chunk {i} ends at {prev_end:.2f}s, Chunk {i+1} starts at {current_start:.2f}s")
            has_gaps = True
    
    if not has_gaps:
        print("✅ Coverage validation PASSED: All audio time is covered with overlaps")
        
        # Show coverage statistics
        total_overlap = 0
        for i in range(1, len(chunk_info)):
            overlap = chunk_info[i-1][3] - chunk_info[i][2]
            if overlap > 0:
                total_overlap += overlap
        
        print(f"   Total audio: {total_duration:.2f}s")
        print(f"   Total overlap: {total_overlap:.2f}s")
        print(f"   Coverage redundancy: {((sum([c[3]-c[2] for c in chunk_info]) / total_duration - 1) * 100):.1f}%")
        return True
    
    return not has_gaps

total_audio_duration = waveform.shape[1] / 16000
coverage_ok = validate_chunk_coverage(chunk_info, total_audio_duration)

if not coverage_ok:
    print("\n❌ Coverage validation FAILED - Gaps detected!")
    print("🔧 Switching to SAFE MODE: Creating overlapping fixed-duration chunks...")
    
    # Fallback: Create fixed overlapping chunks that guarantee coverage
    safe_chunks = []
    sample_rate = 16000
    chunk_duration = 28  # seconds
    overlap_duration = 2.0  # seconds
    step_duration = chunk_duration - overlap_duration
    
    audio_length = waveform.shape[1]
    total_duration = audio_length / sample_rate
    
    current_time = 0
    while current_time < total_duration:
        start_time = current_time
        end_time = min(current_time + chunk_duration, total_duration)
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        safe_chunks.append((start_sample, end_sample, start_time, end_time))
        
        if end_time >= total_duration:
            break
        
        current_time += step_duration
    
    chunk_info = safe_chunks
    print(f"✅ Created {len(chunk_info)} safe overlapping chunks")
    coverage_ok = validate_chunk_coverage(chunk_info, total_audio_duration)
    if not coverage_ok:
        raise Exception("CRITICAL: Even safe mode failed to provide complete coverage!")

print(f"\n📋 Final chunk layout:")
for i, (start_sample, end_sample, start_time, end_time) in enumerate(chunk_info):
    duration = end_time - start_time
    print(f"  Chunk {i+1}: {start_time:.2f}s - {end_time:.2f}s (duration: {duration:.2f}s)")
    if i > 0:
        prev_end = chunk_info[i-1][3]
        overlap = prev_end - start_time
        if overlap > 0:
            print(f"    ✓ Overlap with previous: {overlap:.2f}s")
        else:
            print(f"    ⚠️  No overlap! Gap: {abs(overlap):.2f}s")

# Extract actual audio chunks
chunks = []
for start_sample, end_sample, _, _ in chunk_info:
    chunk = waveform[:, start_sample:end_sample]
    chunks.append(chunk)

import time

# Check if the model has the right configuration for timestamps
# Get timestamp token IDs from the tokenizer
timestamp_ids = list(processor.tokenizer.timestamp_ids())
timestamp_begin = min(timestamp_ids) if timestamp_ids else 50364
print(f"Timestamp token IDs: {timestamp_ids[:10]}... (showing first 10)")
print(f"Timestamp begin token ID: {timestamp_begin}")
print(f"Model generation config: {model.generation_config}")

# Get special token IDs for silence/no-speech detection
nospeech_token_id = processor.tokenizer.encode("<|nospeech|>", add_special_tokens=False)
print(f"No speech token ID: {nospeech_token_id}")

# Ensure timestamps and nospeech tokens are not suppressed in generation
if hasattr(model.generation_config, 'suppress_tokens'):
    print(f"Suppressed tokens before: {model.generation_config.suppress_tokens}")
    # Remove timestamp tokens from suppression list if they're there
    if model.generation_config.suppress_tokens is not None:
        # Keep all non-timestamp tokens in suppression, but allow timestamps and nospeech
        # Note: We keep nospeech suppressed by default as Whisper rarely uses it effectively
        model.generation_config.suppress_tokens = [t for t in model.generation_config.suppress_tokens if t < timestamp_begin]
        print(f"Suppressed tokens after: {model.generation_config.suppress_tokens}")

t=time.time()

transcriptions = []

for chunk_idx, (chunk, chunk_data) in enumerate(zip(chunks, chunk_info)):
    start_sample, end_sample, start_time, end_time = chunk_data
    chunk_duration = end_time - start_time
    print(f"\n{'='*60}")
    print(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")
    print(f"  Time range: {start_time:.2f}s - {end_time:.2f}s (duration: {chunk_duration:.2f}s)")
    print(f"  Audio shape: {chunk.shape}, samples: {chunk.shape[1]}")
    
    # Check for silent chunks
    chunk_energy = torch.sqrt(torch.mean(chunk ** 2))
    print(f"  Chunk energy (RMS): {chunk_energy:.6f}")
    
    if chunk_energy < 0.001:
        print(f"  WARNING: Chunk appears to be silent or very quiet!")
    
    inputs = processor(chunk.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    print(f"  Input features shape: {inputs.input_features.shape}")
    
    with torch.no_grad():
        # Force timestamp generation by not suppressing timestamp tokens
        # and explicitly setting max_new_tokens
        # NOTE: task="translate" will translate to English, use task="transcribe" for original language
        predicted_ids = model.generate(
            inputs.input_features,
            return_timestamps=True,
            max_new_tokens=448,
            num_beams=1,
            task="translate",  # Change to "transcribe" if you want original language
            # Additional parameters to improve quality:
            temperature=0.0,  # Use greedy decoding for consistency
            compression_ratio_threshold=2.4,  # Detect repetitive output
            logprob_threshold=-1.0,  # Filter low-confidence outputs
            no_speech_threshold=0.6,  # Threshold for detecting silence
        )
    
    # Debug: print the actual token IDs
    print(f"  Generated {len(predicted_ids[0])} tokens")
    print(f"  Token IDs (first 50): {predicted_ids[0][:50].tolist()}")
    
    # Manually build transcription with timestamps
    # Convert token IDs to text, preserving timestamp tokens
    tokens = predicted_ids[0].tolist()
    transcription_parts = []
    word_count = 0
    timestamp_count = 0
    
    for token_id in tokens:
        if token_id in timestamp_ids:
            # Convert timestamp token ID to time in seconds
            # Whisper timestamp tokens start at timestamp_begin and each represents 0.02 second intervals
            # Add start_time to make timestamps relative to the full audio
            time_seconds = (token_id - timestamp_begin) * 0.02 + start_time
            transcription_parts.append(f"<|{time_seconds:.2f}|>")
            timestamp_count += 1
        elif nospeech_token_id and token_id in nospeech_token_id:
            # Detected silence/no-speech segment
            transcription_parts.append("<|nospeech|>")
        else:
            # Decode regular token
            token_text = processor.tokenizer.decode([token_id], skip_special_tokens=False)
            if token_text and token_text.strip():
                transcription_parts.append(token_text)
                # Rough word count
                if ' ' in token_text or len(token_text) > 2:
                    word_count += len(token_text.split())
    
    transcription = "".join(transcription_parts)
    
    # Clean up control tokens for display
    display_text = transcription
    for token in ['<|startoftranscript|>', '<|en|>', '<|transcribe|>', '<|translate|>', '<|endoftext|>', '<|nospeech|>']:
        display_text = display_text.replace(token, '')
    # Remove timestamps for display
    import re
    display_text = re.sub(r'<\|\d+\.\d+\|>', '', display_text).strip()
    
    print(f"  Output: {len(transcription)} chars, ~{word_count} words, {timestamp_count} timestamps")
    print(f"  Preview: {display_text[:150]}...")  # First 150 chars without timestamps
    
    if transcription and word_count > 0:  # Only add non-empty transcriptions with actual content
        transcriptions.append(transcription)
    else:
        print(f"  ⚠️  WARNING: Chunk {chunk_idx + 1} produced empty or no-content transcription!")
        print(f"  This might indicate silence, noise, or non-speech audio in this segment.")
        # Still append to maintain chunk order, but with a marker
        transcriptions.append("")

print(f"Elapsed inf2: {time.time()-t}")

# Merge overlapping transcriptions intelligently
def merge_overlapping_transcriptions(transcriptions, chunk_info, overlap_duration=2.0):
    """
    Merge transcriptions from overlapping chunks by removing duplicate content in overlap regions.
    Uses conservative approach - when in doubt, keep content rather than cut it.
    """
    import re
    
    if len(transcriptions) == 0:
        return ""
    
    if len(transcriptions) == 1:
        return transcriptions[0]
    
    print("\n🔗 Merging overlapping transcriptions...")
    
    # Keep the first chunk completely
    merged = transcriptions[0]
    
    for i in range(1, len(transcriptions)):
        current_trans = transcriptions[i]
        
        # Skip empty transcriptions
        if not current_trans or not current_trans.strip():
            print(f"  Chunk {i+1}: Skipping (empty)")
            continue
        
        prev_chunk_end_time = chunk_info[i-1][3]
        current_chunk_start_time = chunk_info[i][2]
        
        # Check if there's actual overlap
        if prev_chunk_end_time > current_chunk_start_time + 0.1:  # Has meaningful overlap
            overlap_time = prev_chunk_end_time - current_chunk_start_time
            print(f"  Chunk {i+1}: Overlap detected = {overlap_time:.2f}s")
            
            # Extract timestamps from current transcription
            pattern = r'<\|(\d+\.\d+)\|>'
            timestamps = [(m.start(), m.group(0), float(m.group(1))) for m in re.finditer(pattern, current_trans)]
            
            if timestamps:
                # Find where the overlap ends - look for first timestamp beyond 75% of overlap
                # Being conservative: only cut 75% of overlap to avoid accidentally losing content
                cutoff_time = current_chunk_start_time + (overlap_time * 0.75)
                
                cut_point = None
                for idx, ts_text, ts_value in timestamps:
                    if ts_value >= cutoff_time:
                        cut_point = idx
                        print(f"    Found cut point at timestamp {ts_value:.2f}s (cutoff: {cutoff_time:.2f}s)")
                        break
                
                if cut_point and cut_point > 50:  # Only cut if we're cutting meaningful amount
                    # Find the start of text content after this timestamp
                    # Look for the timestamp token start
                    current_trans = current_trans[cut_point:]
                    print(f"    Removed {cut_point} characters from overlap region")
                else:
                    print(f"    Conservative: Keeping full chunk (cut point too early or not found)")
            else:
                print(f"    No timestamps found, keeping full chunk for safety")
        else:
            print(f"  Chunk {i+1}: No overlap, appending full chunk")
        
        # Append current transcription with separator
        merged += " " + current_trans
    
    return merged

# Combine the transcriptions with smart merging
print("\n" + "="*60)
full_transcription = merge_overlapping_transcriptions(transcriptions, chunk_info, overlap_duration=2.0)

# Clean up extra spaces
full_transcription = " ".join(full_transcription.split())

print(f"\n✅ Final transcription length: {len(full_transcription)} characters")

# Count actual words (excluding timestamp tokens)
import re
text_only = re.sub(r'<\|[^|]+\|>', '', full_transcription)
for token in ['<|startoftranscript|>', '<|en|>', '<|transcribe|>', '<|translate|>', '<|endoftext|>', '<|nospeech|>']:
    text_only = text_only.replace(token, '')
word_count = len(text_only.split())
print(f"   Estimated word count: {word_count} words")

# Count timestamps
timestamp_count = len(re.findall(r'<\|\d+\.\d+\|>', full_transcription))
print(f"   Timestamp markers: {timestamp_count}")
#print("Full Transcription:", full_transcription)

# Save TXT file
output_filename = audio_path + '.txt'
file = open(output_filename, 'w')
file.write(full_transcription)
file.close()

# Upload TXT to S3
s3_client.put_object(Body=full_transcription, Bucket=output_bucket_name, Key=output_file_prefix + output_filename)

# Generate SRT subtitle file
import re

def generate_srt(transcription_text):
    """Convert timestamped transcription to SRT format"""
    srt_entries = []
    entry_number = 1
    
    # Remove control tokens
    text = transcription_text.replace('<|startoftranscript|>', '')
    text = text.replace('<|en|>', '')
    text = text.replace('<|transcribe|>', '')
    text = text.replace('<|endoftext|>', '')
    
    # Extract timestamps and text using regex
    # Pattern: <|time|> text <|time|>
    pattern = r'<\|(\d+\.\d+)\|>([^<]*?)(?=<\||\Z)'
    matches = re.findall(pattern, text)
    
    # Create SRT entries by pairing consecutive timestamps
    for i in range(len(matches) - 1):
        start_time = float(matches[i][0])
        text_content = matches[i][1].strip()
        
        # Skip empty text
        if not text_content:
            continue
            
        # Get end time from next timestamp
        end_time = float(matches[i + 1][0])
        
        # Convert seconds to SRT time format (HH:MM:SS,mmm)
        def seconds_to_srt_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
        start_time_str = seconds_to_srt_time(start_time)
        end_time_str = seconds_to_srt_time(end_time)
        
        # Create SRT entry
        srt_entry = f"{entry_number}\n{start_time_str} --> {end_time_str}\n{text_content}\n"
        srt_entries.append(srt_entry)
        entry_number += 1
    
    # Handle last segment (if there's text after the last timestamp)
    if matches and matches[-1][1].strip():
        start_time = float(matches[-1][0])
        text_content = matches[-1][1].strip()
        # Estimate end time as start + 3 seconds (or use audio duration if available)
        end_time = start_time + 3.0
        
        start_time_str = seconds_to_srt_time(start_time)
        end_time_str = seconds_to_srt_time(end_time)
        
        srt_entry = f"{entry_number}\n{start_time_str} --> {end_time_str}\n{text_content}\n"
        srt_entries.append(srt_entry)
    
    return "\n".join(srt_entries)

# Generate SRT content
srt_content = generate_srt(full_transcription)
# Count subtitle entries (each entry is separated by blank lines)
num_entries = len([line for line in srt_content.split('\n') if line.strip().isdigit()])
print(f"\nGenerated SRT with {num_entries} subtitle entries")

# Save SRT file
srt_filename = audio_path + '.srt'
with open(srt_filename, 'w', encoding='utf-8') as srt_file:
    srt_file.write(srt_content)

# Upload SRT to S3
s3_client.put_object(Body=srt_content.encode('utf-8'), Bucket=output_bucket_name, Key=output_file_prefix + srt_filename)
print(f"Uploaded SRT file to S3: {output_file_prefix + srt_filename}")
