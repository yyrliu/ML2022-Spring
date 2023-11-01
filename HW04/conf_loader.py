import copy
from functools import reduce
import operator

# Load the config dict
conf = {
    "comment": "test",
    "data_dir": "data/Dataset",
    "batch_size": [64, 128, 256],
    "valid_steps": 2000,
    "save_steps": 2000,
    "total_steps": 30000,
    "model": {
        "input_mels": 40,
        "d_model": [80, 100, 120],
        "conf": {
            "prenet": {
                "dropout": 0.1
            },
            "encoder": {
                "type": "comformer",
                "layers": 3,
                "submodules": {
                    "feedforward": {
                        "version": 1,
                        "dropout": 0.1,
                        "residual_connection": {
                            "module_factor": 0.5,
                            "input_factor": 1.0
                        }
                    },
                    "multiheadAttention": {
                        "nhead": 2,
                        "dropout": 0.1,
                        "residual_connection": {
                            "module_factor": 0.5,
                            "input_factor": 1.0
                        }
                    },
                    "conv": {
                        "version": 2,
                        "kernel_size": 31,
                        "dropout": 0.1,
                        "residual_connection": {
                            "module_factor": 1.0,
                            "input_factor": 1.0
                        }
                    }
                }
            },
            "pooling": {
                "type": "self_attention",
                "reducer": "mean",
            },
        }
    },
    "loss_fn": {
        "type": "amsoftmax",
        "conf": {
            "m": 0.35,
            "s": 30,
            "norm_affine": True,
            "feat_norm": True
        }
    },
    "optimizer": {
        "lr": 1e-3,
    },
    "lr_scheduler": {
        "learning_rate": [1e-3, 1.5e-3, 2e-3],
        "type": "transformer",
        "num_warmup_steps": 2000,
    }
}

def scan(config, keep_default=False):
    """
    Scan the config dict recursively and find all the options that are lists.
    input:
        config_dict: dict
        keep_default: bool
            If True, output the default value (first item in the list) to the list_pairs.
    output:
        list_pairs: list of tuples
            [(keys, values), ...]
            keys: list of nested keys in order
            values: list of values
    """
    return _scan(config, keep_first=keep_default)

def _scan(config, parent_keys=None, list_pairs=None, keep_first=False):

    keys = [] if parent_keys is None else parent_keys
    list_pairs = [] if list_pairs is None else list_pairs
    
    for key, value in config.items():
        if isinstance(value, dict):
            _scan(value, [*keys, key], list_pairs=list_pairs, keep_first=keep_first)
        if isinstance(value, list):
            if keep_first:
                list_pairs.append(([*keys, key], value))
            else:
                list_pairs.append(([*keys, key], value[1:]))

    return list_pairs

def _create_and_overwrite(config, keys, value):
    """
    Create a new config base on config where the value of the key is overwritten.
    """
    config = copy.deepcopy(config)
    _set_dict(config, keys, value)
    return config

def _generate(config):
    """
    Generate all the possible configs from the given config.
    """

    default_config = get_default(config)
    configs = [default_config]

    list_pairs = scan(config)

    for keys, value in list_pairs:
        for item in value:
            configs.append(_create_and_overwrite(default_config, keys, item))
    return configs

def get_default(config):
    """
    The first item of the list is used as the default value.
    """

    list_pairs = scan(config, keep_default=True)
    config = copy.deepcopy(config)

    for keys, value in list_pairs:
        _set_dict(config, keys, value[0])
    
    return config

def _set_dict(dict, keys, value):
    """
    Set the value in place for the given key in the dict.
    """
    last_key = keys[-1]
    try:
        subitem = reduce(operator.getitem, keys[:-1], dict)
        subitem[last_key] = value
    except KeyError:
        raise KeyError(f"Invalid key: path `{'>'.join(keys[:-1])}` does not exist in the dict")
    
if __name__ == "__main__":
    import pprint
    pprint.pprint(_generate(conf))
    print(len(_generate(conf)))
