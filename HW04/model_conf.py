encoder_default_transformer = {
    "type": "transformer",
    "layers": 2,
    "submodules": {
        "encoder_layer": {
            "nhead": 2,
            "dim_feedforward": 256,
        },
    }
}

encoder_default_conformer = {
    "type": "comformer",
    "layers": 2,
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
}

loss_fn_conf = {
    "m": 0.2,
    "s": 30,
    "norm_affine": True,
    "feat_norm": True
}

model_base_conf = {
    "input_mels": 40,
    "d_model": 80,
    "n_class": 600,
}

comformer_default_conf = {
    **model_base_conf,
    "prenet": {
        "dropout": 0.1
    },
    "encoder": encoder_default_conformer,
    "pooling": {
        "type": "self_attention",
        "reducer": "mean",
    },
}

model_default_conf = {
    **model_base_conf,
    "encoder": encoder_default_transformer,
    "pooling": {
        "type": "mean",
    },
    "pred_layer": 0,
}
