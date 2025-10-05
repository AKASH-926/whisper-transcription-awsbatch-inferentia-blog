FROM public.ecr.aws/neuron/pytorch-inference-neuronx:1.13.1-neuronx-py310-sdk2.20.2-ubuntu20.04

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt requirements.txt
COPY inference.py inference.py

# Pin protobuf to 3.9.2 to match Neuron SDK
RUN sed -i '/protobuf/d' requirements.txt
RUN echo -e "\nprotobuf==3.9.2" >> requirements.txt


RUN pip install -U --no-cache-dir -r requirements.txt

# Exit container after the job is done
RUN sed -i '/prevent docker exit/ {n; s/./# &/;}' /usr/local/bin/dockerd-entrypoint.py

# Ensure the model is cached during the image build rather than during runtime
RUN python3 -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
    model_id='openai/whisper-large-v3'; \
    WhisperProcessor.from_pretrained(model_id); \
    WhisperForConditionalGeneration.from_pretrained(model_id, torchscript=True)"

# Stage script to create model artifacts
COPY export-model.py export-model.py

# For inference
CMD ["python3", "inference.py"]
