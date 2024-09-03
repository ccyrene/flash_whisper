num_hidden_layers = 4
model_path = "/workspace/models/whisper-tiny/decoder_model_merged.onnx"
model_output = "/workspace/models/whisper-tiny/decoder_model_merged.plan"

num_feats = 384

min_batch = 1
opt_batch = 4
max_batch = 8

min_length = 1
opt_length = 224
max_length = 448

script = "trtexec" + " "
script += f"--onnx={model_path}" + " "

# minShapes
script += f"--minShapes=input_ids:{min_batch}x{min_length},encoder_hidden_states:{min_batch}x1500x{num_feats},"
for i in range(num_hidden_layers):
    for name in ["decoder.key", "decoder.value", "encoder.key", "encoder.value"]:
        script += f"past_key_values.{i}.{name}:{min_batch}x6x{min_length}x64,"

script += "use_cache_branch:1"
# script = script[:-1]
script + " "

# optShapes
script += f"--optShapes=input_ids:{opt_batch}x{opt_length},encoder_hidden_states:{opt_batch}x1500x{num_feats},"
for i in range(num_hidden_layers):
    for name in ["decoder.key", "decoder.value", "encoder.key", "encoder.value"]:
        script += f"past_key_values.{i}.{name}:{opt_batch}x6x{opt_length}x64,"

script += "use_cache_branch:1"
# script = script[:-1]
script + " "

# maxShapes
script += f"--maxShapes=input_ids:{max_batch}x{max_length},encoder_hidden_states:{max_batch}x1500x{num_feats},"
for i in range(num_hidden_layers):
    for name in ["decoder.key", "decoder.value", "encoder.key", "encoder.value"]:
        script += f"past_key_values.{i}.{name}:{max_batch}x6x{max_length}x64,"

script += "use_cache_branch:1"
# script = script[:-1]
script + " "

script += f"--saveEngine={model_output}"

with open('trt.txt', 'w') as file:
    # write variables using repr() function
    file.write(script)