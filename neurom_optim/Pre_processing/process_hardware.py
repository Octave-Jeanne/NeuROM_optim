import torch

def get_precision(config):
    match config['hardware']['IntPrecision']:
        case 'double':
            int_precision = torch.int32

        case 'simple':
            int_precision = torch.int16

    match config['hardware']['FloatPrecision']:
        case 'double':
            float_precision = torch.float64

        case 'simple':
            float_precision = torch.float32

    return int_precision, float_precision


def get_device(config):
    if config['hardware']['BoolGPU']:
        return try_to_get_gpu() 
    
    else:
        return 'cpu'
    
def try_to_get_gpu():
    print('GPU requested')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print(f'No GPU available. Using {device}')
    else:
        print(f'GPU available. Using {device}')
    return device