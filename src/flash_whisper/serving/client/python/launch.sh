num_task=16
dataset=datasets
python3 client.py \
    --server-addr 0.0.0.0 \
    --model-name infer_bls \
    --num-tasks $num_task \
    --text-prompt "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>" \
    --manifest-dir ./$dataset \
    --log-dir ./log_sherpa_multi_hans_whisper_large_ifb_$num_task \
    --compute-cer