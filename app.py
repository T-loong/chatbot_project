from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
import subprocess

app = Flask(__name__)

# 加载预处理后的数据
data = np.load("preprocessed_data.npz", allow_pickle=True)
num_encoder_tokens = data["num_encoder_tokens"]
num_decoder_tokens = data["num_decoder_tokens"]
input_token_index = data["input_token_index"].item()
target_token_index = data["target_token_index"].item()
max_encoder_seq_length = data["max_encoder_seq_length"]
max_decoder_seq_length = data["max_decoder_seq_length"]

# 加载模型
model = tf.keras.models.load_model("chatbot_model.h5")

# 构建推理模型
encoder_inputs = model.input[0]
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]
decoder_state_input_h = Input(shape=(256,), name='decoder_state_input_h')
decoder_state_input_c = Input(shape=(256,), name='decoder_state_input_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# 解码序列
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.0  # '\t' 表示开始字符

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = list(target_token_index.keys())[list(target_token_index.values()).index(sampled_token_index)]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        states_value = [h, c]

    return decoded_sentence

# 将用户输入转换为模型输入格式
def str_to_input_seq(input_str):
    input_seq = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    for t, char in enumerate(input_str):
        if char in input_token_index:
            input_seq[0, t, input_token_index[char]] = 1.0
    return input_seq

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train')
def train():
    subprocess.Popen(["tensorboard", "--logdir=logs/fit", "--host=127.0.0.1", "--port=6006"])
    return render_template('train.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    input_seq = str_to_input_seq(user_input)
    decoded_sentence = decode_sequence(input_seq)
    return jsonify(response=decoded_sentence)

if __name__ == '__main__':
    app.run(debug=True)