"""MiniMax H3 prompt encoding, ported from ComfyUI comfy/text_encoders/minimax.py.

The H3 presentation is not chat-templated: token ids are the raw prompt text with no special tokens, and the
conditioning is the unnormalized hidden state after layer 50 (the checkpoint is truncated there and has no final
norm). With keyframes (image-to-video) each image comes first as "<Picture i>: " + a vision block, then the prompt;
Ref2VA adds its reference videos after the pictures ("<Video k>: ", then a "<T.T seconds>" label and a two-frame
vision block per pair of frames sampled at 2 FPS, a kept soundtrack being the "<Audio j>: " right before), then
"<Audio j>: " for the audio clips. Every text segment is tokenized on its own, as ComfyUI does.
"""

import torch
from backend.args import dynamic_args
from backend.text_processing import emphasis
from backend.text_processing._comfy import EMBEDDINGS, INF, SDClipModel, SDTokenizer

PAD = 151643
VISION_START = 151652
VISION_END = 151653


class VisionSpanClipModel(SDClipModel):
    # remembers where the vision embeddings of the last prompt landed, (start, length) per image
    image_spans: list[tuple[int, int]] = []

    def process_tokens(self, tokens, device):
        embeds, attention_mask, num_tokens, embeds_info = super().process_tokens(tokens, device)
        self.image_spans = [(e["index"], e["size"]) for e in embeds_info if e["type"] == "image"]
        return embeds, attention_mask, num_tokens, embeds_info


def vision_spans(image_spans) -> list[tuple[int, int]]:
    """Token ranges [start, stop) of the whole vision blocks, flanking <|vision_start|> / <|vision_end|> included;
    the DiT gives them the video modality tag (0) instead of the text one (1)."""
    return [(max(0, start - 1), start + size + 1) for start, size in image_spans]


class MiniMaxH3TextEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = VisionSpanClipModel(text_encoder, layer="last", special_tokens={"pad": PAD}, layer_norm_hidden_state=False,
                                                enable_attention_masks=False, return_attention_masks=False)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False,
                                     pad_to_max_length=False, max_length=INF, min_length=1, pad_token=PAD)
        # vision block ranges of the last encoded prompt; the keyframes come first, so the prompt and the negative
        # prompt share them
        self.vision_spans: list[tuple[int, int]] = []

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def tokenize(self, texts: str | list[str]) -> EMBEDDINGS | list[EMBEDDINGS]:
        return self.tokenizer.tokenizer(texts, add_special_tokens=False)["input_ids"]

    def __call__(self, texts: list[str], images: list = (), audios: int = 0, videos: list = ()) -> list:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs = []
        cache = {}
        self.vision_spans = []
        for line in texts:
            if line not in cache:
                cache[line] = self.text_encoder.encode_token_weights(self.tokens(line, images, audios, videos))[0]
                self.vision_spans = vision_spans(self.text_encoder.image_spans)
            zs.extend(cache[line])  # (L, D) per prompt; the prompt parser stacks the batch
        return zs

    def tokens(self, text: str, images=(), audios: int = 0, videos=()) -> list[list[tuple]]:
        """videos: {"frames": [T, H, W, 3] in [0, 1] sampled at 2 FPS, "timestamps": seconds per frame,
        "soundtrack": bool} each, in <Video k> order."""
        entries = []

        def add_text(segment):
            if not segment:
                return
            batches = self.tokenizer.tokenize_with_weights(segment, disable_weights=True)
            if len(batches) != 1:
                raise ValueError("[MiniMax H3] the prompt is longer than the text encoder accepts")
            entries.extend(batches[0])

        def add_vision(data, video_block=False):
            embed = {"type": "image", "data": data, "original_type": "image"}
            if video_block:
                embed["minimax_video_block"] = True
            entries.extend([(VISION_START, 1.0), (embed, 1.0), (VISION_END, 1.0)])

        for i, image in enumerate(images):
            add_text(f"<Picture {i + 1}>: ")
            add_vision(image)
        # Ref2VA audio never enters Qwen: only its label
        sounds = 0
        for k, video in enumerate(videos):
            if video["soundtrack"]:
                sounds += 1
                add_text(f"<Audio {sounds}>: ")
            frames, timestamps = video["frames"], list(video["timestamps"])
            if frames.shape[0] % 2:  # repeat-pad to the temporal patch of 2
                frames = torch.cat([frames, frames[-1:]])
                timestamps.append(timestamps[-1])
            add_text(f"<Video {k + 1}>: ")
            for i in range(0, frames.shape[0], 2):
                add_text("<%.1f seconds>" % ((timestamps[i] + timestamps[i + 1]) / 2.0))
                add_vision(frames[i:i + 2], video_block=True)
        for _ in range(audios):
            sounds += 1
            add_text(f"<Audio {sounds}>: ")
        add_text(text)
        return [entries or [(PAD, 1.0)]]
