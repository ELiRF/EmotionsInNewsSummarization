generation_hyperparameters = {
    "facebook/bart-large-cnn": {
        "cnn_dailymail": {
            "length_penalty": 2.0,
            "max_length": 142,
            "min_length": 56,
            "no_repeat_ngram_size": 3,
            "num_beams": 4,
        },
        "xsum": {
            "length_penalty": 1.0,
            "max_length": 62,
            "min_length": 11,
            "no_repeat_ngram_size": 3,
            "num_beams": 6,
        },
    },
    "google/pegasus-cnn_dailymail": {
        "cnn_dailymail": {
            "length_penalty": 0.8,
            "max_length": 128,
            "min_length": 32,
            "no_repeat_ngram_size": None,
            "num_beams": 4,
        },
        "xsum": {
            "length_penalty": 0.6,
            "max_length": 64,
            "min_length": None,
            "no_repeat_ngram_size": None,
            "num_beams": 6,
        },
    },
    "t5-base": {
        "cnn_dailymail": {
            "length_penalty": 2.0,
            "max_length": 142,
            "min_length": 56,
            "no_repeat_ngram_size": 3,
            "num_beams": 4,
        },
        "xsum": {
            "length_penalty": 2.0,
            "max_length": 62,
            "min_length": 11,
            "no_repeat_ngram_size": 3,
            "num_beams": 6,
        },
    },
    "facebook/bart-large-xsum": {
        "cnn_dailymail": {
            "length_penalty": 2.0,
            "max_length": 142,
            "min_length": 56,
            "no_repeat_ngram_size": 3,
            "num_beams": 4,
        },
        "xsum": {
            "length_penalty": 1.0,
            "max_length": 62,
            "min_length": 11,
            "no_repeat_ngram_size": 3,
            "num_beams": 6,
        },
    },
    "google/pegasus-xsum": {
        "cnn_dailymail": {
            "length_penalty": 0.8,
            "max_length": 128,
            "min_length": 32,
            "no_repeat_ngram_size": None,
            "num_beams": 4,
        },
        "xsum": {
            "length_penalty": 0.6,
            "max_length": 64,
            "min_length": None,
            "no_repeat_ngram_size": None,
            "num_beams": 6,
        },
    },
    "./checkpoints/bart-JES-cnn_dailymail": {
        "cnn_dailymail": {
            "length_penalty": 2.0,
            "max_length": 142 + 34,
            "min_length": 56 + 34,
            "no_repeat_ngram_size": 3,
            "num_beams": 4,
        },
        "xsum": {
            "length_penalty": 1.0,
            "max_length": 62 + 17,
            "min_length": 11 + 17,
            "no_repeat_ngram_size": 3,
            "num_beams": 6,
        },
    },
    "./checkpoints/pegasus-JES-cnn_dailymail": {
        "cnn_dailymail": {
            "length_penalty": 0.8,
            "max_length": 128 + 34,
            "min_length": 32 + 34,
            "no_repeat_ngram_size": None,
            "num_beams": 4,
        },
        "xsum": {
            "length_penalty": 0.6,
            "max_length": 64 + 17,
            "min_length": 17,
            "no_repeat_ngram_size": None,
            "num_beams": 6,
        },
    },
    "./checkpoints/bart-JES-xsum": {
        "cnn_dailymail": {
            "length_penalty": 2.0,
            "max_length": 142 + 34,
            "min_length": 56 + 34,
            "no_repeat_ngram_size": 3,
            "num_beams": 4,
        },
        "xsum": {
            "length_penalty": 1.0,
            "max_length": 62 + 17,
            "min_length": 11 + 17,
            "no_repeat_ngram_size": 3,
            "num_beams": 6,
        },
    },
    "./checkpoints/pegasus-JES-xsum": {
        "cnn_dailymail": {
            "length_penalty": 0.8,
            "max_length": 128 + 34,
            "min_length": 32 + 34,
            "no_repeat_ngram_size": None,
            "num_beams": 4,
        },
        "xsum": {
            "length_penalty": 0.6,
            "max_length": 64 + 17,
            "min_length": 17,
            "no_repeat_ngram_size": None,
            "num_beams": 6,
        },
    },
}
