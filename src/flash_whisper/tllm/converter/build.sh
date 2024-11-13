model_size=$1

openai_dir=../assets/$model_size
safetensors_dir=../assets/$model_size/safetensors
output_dir=../assets/$model_size/tllm

model_url=""

case $model_size in
  "tiny.en")
    model_url="https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt"
    ;;
  "tiny")
    model_url="https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
    ;;
  "base.en")
    model_url="https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt"
    ;;
  "base")
    model_url="https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"
    ;;
  "small.en")
    model_url="https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt"
    ;;
  "small")
    model_url="https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt"
    ;;
  "medium.en")
    model_url="https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt"
    ;;
  "medium")
    model_url="https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt"
    ;;
  "large-v1")
    model_url="https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt"
    ;;
  "large-v2")
    model_url="https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt"
    ;;
  "large-v3")
    model_url="https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
    ;;
  "large")
    model_url="https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
    ;;
  "large-v3-turbo"|"turbo")
    model_url="https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt"
    ;;
  *)
    echo "Unknown model size: $model_size"
    exit 1
    ;;
esac

wget -O $openai_dir/$model_size.pt $model_size
wget -O ../multilingual.tiktoken https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
wget -O ../mel_filters.npz https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
python3 convert_model_to_safetensors.py --model_dir=$openai_dir --model_name=$model_size --output_dir=$safetensors_dir

MAX_BATCH_SIZE=8
MAX_BEAM_WIDTH=4
INFERENCE_PRECISION=float16
MAX_SEQ_LENGTH=434

trtllm-build --checkpoint_dir ${safetensors_dir}/encoder \
                --output_dir ${output_dir}/encoder \
                --moe_plugin disable \
                --enable_xqa disable \
                --max_batch_size ${MAX_BATCH_SIZE} \
                --gemm_plugin disable \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --max_input_len 3000 \
                --max_seq_len 3000

trtllm-build --checkpoint_dir ${safetensors_dir}/decoder \
                --output_dir ${output_dir}/decoder \
                --moe_plugin disable \
                --enable_xqa disable \
                --max_beam_width ${MAX_BEAM_WIDTH} \
                --max_batch_size ${MAX_BATCH_SIZE} \
                --max_seq_len ${MAX_SEQ_LENGTH} \
                --max_input_len 14 \
                --max_encoder_input_len 3000 \
                --gemm_plugin ${INFERENCE_PRECISION} \
                --bert_attention_plugin ${INFERENCE_PRECISION} \
                --gpt_attention_plugin ${INFERENCE_PRECISION}