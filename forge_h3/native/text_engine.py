"""MiniMax H3 prompt encoding, ported from ComfyUI comfy/text_encoders/minimax.py.

The H3 presentation is not chat-templated: token ids are the raw prompt text with no special tokens, and the
conditioning is the unnormalized hidden state after layer 50 (the checkpoint is truncated there and has no final
norm). Keyframe images ("<Picture 1>: " + a vision block before the prompt) come with image-to-video.
"""

from backend.args import dynamic_args
from backend.text_processing import emphasis
from backend.text_processing._comfy import EMBEDDINGS, INF, SDClipModel, SDTokenizer

PAD = 151643


class MiniMaxH3TextEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = SDClipModel(text_encoder, layer="last", special_tokens={"pad": PAD}, layer_norm_hidden_state=False,
                                        enable_attention_masks=False, return_attention_masks=False)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False,
                                     pad_to_max_length=False, max_length=INF, min_length=1, pad_token=PAD)

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def tokenize(self, texts: str | list[str]) -> EMBEDDINGS | list[EMBEDDINGS]:
        return self.tokenizer.tokenizer(texts, add_special_tokens=False)["input_ids"]

    def __call__(self, texts: list[str]) -> list:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs = []
        cache = {}
        for line in texts:
            if line not in cache:
                tokens = self.tokenizer.tokenize_with_weights(line, disable_weights=True) if line else [[(PAD, 1.0)]]
                if len(tokens) != 1:
                    raise ValueError("[MiniMax H3] the prompt is longer than the text encoder accepts")
                cache[line] = self.text_encoder.encode_token_weights(tokens)[0]
            zs.extend(cache[line])  # (L, D) per prompt; the prompt parser stacks the batch
        return zs
