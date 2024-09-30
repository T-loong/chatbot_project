import numpy as np
import json
import argparse

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['input_texts'], data['target_texts']

def preprocess_data(input_texts, target_texts, max_seq_length=None):
    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    decoder_input_data = np.zeros((len(target_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
    decoder_target_data = np.zeros((len(target_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

    return (encoder_input_data, decoder_input_data, decoder_target_data,
            num_encoder_tokens, num_decoder_tokens, input_token_index,
            target_token_index, max_encoder_seq_length, max_decoder_seq_length)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理聊天机器人数据")
    parser.add_argument("--input", required=True, help="输入数据文件路径")
    parser.add_argument("--output", default="preprocessed_data.npz", help="输出文件路径")
    parser.add_argument("--max_length", type=int, help="最大序列长度")
    args = parser.parse_args()

    input_texts, target_texts = load_data(args.input)
    preprocessed_data = preprocess_data(input_texts, target_texts, args.max_length)

    np.savez(args.output, *preprocessed_data, allow_pickle=True)
    print(f"预处理数据已保存到 {args.output}")

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
    input_characters = sorted(list(set("".join(input_texts))))
    target_characters = sorted(list(set("".join(target_texts))))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    if max_seq_length:
        max_encoder_seq_length = min(max_encoder_seq_length, max_seq_length)
        max_decoder_seq_length = min(max_decoder_seq_length, max_seq_length)

    # 构建词汇表
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
decoder_input_data = np.zeros((len(target_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
decoder_target_data = np.zeros((len(target_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

# 保存预处理后的数据
np.savez("preprocessed_data.npz", 
         encoder_input_data=encoder_input_data, 
         decoder_input_data=decoder_input_data, 
         decoder_target_data=decoder_target_data, 
         num_encoder_tokens=num_encoder_tokens, 
         num_decoder_tokens=num_decoder_tokens, 
         input_token_index=input_token_index, 
         target_token_index=target_token_index, 
         max_encoder_seq_length=max_encoder_seq_length, 
         max_decoder_seq_length=max_decoder_seq_length,
         allow_pickle=True)