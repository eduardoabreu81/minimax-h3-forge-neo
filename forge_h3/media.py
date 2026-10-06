"""Atomic local exports using Forge's FFmpeg or imageio's bundled executable."""

import os
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL.PngImagePlugin import PngInfo

from .contracts import FPS, SAMPLE_RATE, H3Error

# front left/right plus the center (a mono file's only channel), renormalized so nothing clips
MIX_TO_STEREO = "pan=stereo|FL<FL+FC|FR<FR+FC"


def find_ffmpeg(configured=""):
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        raise H3Error("H3 FFmpeg executable does not exist. Correct it in Settings.")
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        raise H3Error("FFmpeg is missing. Install FFmpeg or imageio-ffmpeg, or set H3 FFmpeg executable in Settings.") from None
    return executable


def read_audio(path, ffmpeg=""):
    """The sound of an audio or video file as stereo float32 [2, samples] at H3's 32 kHz in [-1, 1]; FFmpeg resamples.
    Mono goes to both channels at full level (plain -ac 2 lowers it by 3 dB) and surround folds into the front pair."""
    command = [find_ffmpeg(ffmpeg), "-v", "error", "-nostdin", "-i", str(path), "-vn", "-af", MIX_TO_STEREO,
               "-ar", str(SAMPLE_RATE), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    name = Path(path).name
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise H3Error(f"FFmpeg could not start to read {name}: {exc}") from None
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise H3Error(f"FFmpeg could not read the audio of {name}: {detail[-1] if detail else 'unknown error'}")
    samples = np.frombuffer(result.stdout, dtype="<f4")
    if not samples.size:
        raise H3Error(f"{name} has no audio.")
    return np.clip(samples.reshape(-1, 2).T, -1.0, 1.0).copy()


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    seconds: float
    has_audio: bool


def probe_video(path, ffmpeg="") -> VideoInfo:
    """Size (as FFmpeg decodes it, phone rotation applied), duration and sound of a video, from FFmpeg's own report;
    imageio's bundled FFmpeg comes without ffprobe."""
    name = Path(path).name
    try:
        result = subprocess.run([find_ffmpeg(ffmpeg), "-hide_banner", "-nostdin", "-i", str(path)],
                                capture_output=True, check=False)
    except OSError as exc:
        raise H3Error(f"FFmpeg could not start to read {name}: {exc}") from None
    report = result.stderr.decode("utf-8", "replace")
    size = re.search(r"Stream #\S+.*?: Video: .*?(\d{2,5})x(\d{2,5})", report)
    duration = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", report)
    if size is None or duration is None:
        raise H3Error(f"FFmpeg found no video in {name}.")
    width, height = int(size.group(1)), int(size.group(2))
    rotation = re.search(r"rotation of (-?\d+(?:\.\d+)?) degrees", report)
    if rotation and round(abs(float(rotation.group(1)))) % 180 == 90:
        width, height = height, width
    hours, minutes, seconds = duration.groups()
    return VideoInfo(width, height, int(hours) * 3600 + int(minutes) * 60 + float(seconds),
                     re.search(r"Stream #\S+.*?: Audio:", report) is not None)


def read_video(path, width, height, max_frames, ffmpeg=""):
    """The frames of a video at H3's 24 FPS, scaled to width x height, as uint8 [frames, height, width, 3] (at most
    max_frames from the start)."""
    command = [find_ffmpeg(ffmpeg), "-v", "error", "-nostdin", "-i", str(path), "-an",
               "-vf", f"fps={FPS},scale={width}:{height}:flags=lanczos", "-frames:v", str(int(max_frames)),
               "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    name = Path(path).name
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise H3Error(f"FFmpeg could not start to read {name}: {exc}") from None
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise H3Error(f"FFmpeg could not read the frames of {name}: {detail[-1] if detail else 'unknown error'}")
    frame = width * height * 3
    count = len(result.stdout) // frame
    return np.frombuffer(result.stdout[:count * frame], dtype=np.uint8).reshape(count, height, width, 3)


def _write_wave(audio, path, samples):
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.ndim != 2 or audio.shape[0] not in (1, 2) or not audio.shape[1]:
        raise H3Error("H3 audio must have shape [channels, samples] with one or two channels.")
    if not np.isfinite(audio).all():
        raise H3Error("H3 audio must contain finite samples.")
    aligned = np.zeros((audio.shape[0], samples), dtype=np.float32)
    count = min(samples, audio.shape[1])
    aligned[:, :count] = audio[:, :count]
    pcm = (np.clip(aligned, -1, 1).T * 32767).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(pcm.shape[1])
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(pcm.tobytes())


def export_video(frames, audio, output, *, ffmpeg="", infotext="", cancelled=lambda: False):
    if not frames:
        raise H3Error("The H3 backend returned no frames.")
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    executable = find_ffmpeg(ffmpeg)
    with tempfile.TemporaryDirectory(prefix=".h3-export-", dir=target.parent) as scratch:
        scratch = Path(scratch)
        size = frames[0].size
        for index, frame in enumerate(frames):
            if cancelled():
                from .contracts import GenerationCancelled
                raise GenerationCancelled("H3 export cancelled.")
            if frame.size != size or size[0] % 2 or size[1] % 2:
                raise H3Error("H3 frames must have equal, even dimensions.")
            frame.convert("RGB").save(scratch / f"{index:06d}.png")
        encoded = scratch / "result.mp4"
        command = [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-framerate", str(FPS), "-i", str(scratch / "%06d.png")]
        if audio is not None:
            _write_wave(audio, scratch / "audio.wav", round(len(frames) / FPS * SAMPLE_RATE))
            command += ["-i", str(scratch / "audio.wav"), "-map", "0:v:0", "-map", "1:a:0",
                        "-c:a", "aac", "-b:a", "192k"]
        else:
            command += ["-an"]
        command += ["-frames:v", str(len(frames)), "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-metadata", "comment=" + infotext, str(encoded)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=600,
                                    check=False,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise H3Error(f"H3 video export failed: {exc}") from exc
        if result.returncode or not encoded.is_file():
            raise H3Error("H3 video export failed: " + result.stderr[-2000:])
        if cancelled():
            from .contracts import GenerationCancelled
            raise GenerationCancelled("H3 export cancelled.")
        os.replace(encoded, target)
    return str(target)


def export_still(image, output, infotext):
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    info = PngInfo()
    info.add_text("parameters", infotext)
    with tempfile.TemporaryDirectory(prefix=".h3-image-", dir=target.parent) as scratch:
        candidate = Path(scratch) / "result.png"
        image.save(candidate, format="PNG", pnginfo=info)
        os.replace(candidate, target)
    return str(target)
