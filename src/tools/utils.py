import numpy as np


def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


class Decoder:
    """Interface for sequence decoding"""
    def decode(self, predicted_seq, chars_list):
        raise NotImplementedError


class GreedyDecoder(Decoder):
    def decode(self, predicted_seq, chars_list):
        full_pred_labels = []
        labels = []
        # predicted_seq.shape = [batch, len(chars_list), len_seq]
        for i in range(predicted_seq.shape[0]):
            single_prediction = predicted_seq[i, :, :]
            predicted_labels = []
            for j in range(single_prediction.shape[1]):
                predicted_labels.append(np.argmax(single_prediction[:, j], axis=0))
            #print(predicted_labels)
            without_repeating = []
            current_char = predicted_labels[0]
            if current_char != len(chars_list) - 1:
                without_repeating.append(current_char)
            for c in predicted_labels:
                if (current_char == c) or (c == len(chars_list) - 1):
                    if c == len(chars_list) - 1:
                        current_char = c
                    continue
                without_repeating.append(c)
                current_char = c

            full_pred_labels.append(without_repeating)

        for i, label in enumerate(full_pred_labels):
            decoded_label = ''
            for j in label:
                decoded_label += chars_list[j]
            labels.append(decoded_label)

        return labels, full_pred_labels


def decode_function(predicted_seq, chars_list, decoder=GreedyDecoder):
    return decoder().decode(predicted_seq, chars_list)
