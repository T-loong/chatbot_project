import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Attention, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import datetime

# 加载预处理后的数据
data = np.load("preprocessed_data.npz", allow_pickle=True)
encoder_input_data = data["encoder_input_data"]
decoder_input_data = data["decoder_input_data"]
decoder_target_data = data["decoder_target_data"]
num_encoder_tokens = data["num_encoder_tokens"]
num_decoder_tokens = data["num_decoder_tokens"]

# 定义模型参数
latent_dim = 256
embedding_dim = 128

# 构建更复杂的模型
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_embedding = Embedding(num_encoder_tokens, embedding_dim)(encoder_inputs)
encoder_lstm = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_embedding = Embedding(num_decoder_tokens, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

attention = Attention()([decoder_outputs, encoder_outputs])
decoder_combined_context = Concatenate()([decoder_outputs, attention])

decoder_dense = Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_combined_context)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 编译模型
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# 设置TensorBoard回调
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 训练模型
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, 
          batch_size=64, epochs=100, validation_split=0.2, 
          callbacks=[tensorboard_callback])

# 保存模型
model.save("chatbot_model.h5")