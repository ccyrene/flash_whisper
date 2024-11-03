assets=/workspace/assets
huggingface_dir=/workspace/assets/whisper-medium
safetensors_dir=/workspace/assets/safetensors
output_dir=/workspace/assets/whisper_medium

git clone https://huggingface.co/openai/whisper-medium $huggingface_dir

python3 convert_ckpt_hf_to_model.py --model_name=$huggingface_dir --output_dir=$assets --output_name=medium
python3 convert_model_to_safetensors.py --model_dir=$assets --model_name=medium --output_dir=$safetensors_dir

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