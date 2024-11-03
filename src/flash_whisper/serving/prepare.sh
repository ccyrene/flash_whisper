engine_dir=/workspace/assets/whisper_medium
n_mels=80
zero_pad=false

model_repo=/triton_models

wget -nc --directory-prefix=$model_repo/infer_bls/1 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
wget -nc --directory-prefix=$model_repo/whisper_medium/1 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz

TRITON_MAX_BATCH_SIZE=8
MAX_QUEUE_DELAY_MICROSECONDS=100
python3 fill_template.py -i $model_repo/whisper_medium/config.pbtxt engine_dir:${engine_dir},n_mels:$n_mels,zero_pad:$zero_pad,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
python3 fill_template.py -i $model_repo/infer_bls/config.pbtxt engine_dir:${engine_dir},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}

tritonserver --model-repository=$model_repo --log-verbose=1