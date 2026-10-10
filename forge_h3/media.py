"""Atomic local exports using Forge's FFmpeg or imageio's bundled executable.

Video frames go to FFmpeg as raw RGB through its standard input: writing them as PNG files first is slower than
sampling a whole clip.
"""

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

from .contracts import FPS, SAMPLE_RATE, GenerationCancelled, H3Error

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


def delay_audio(audio, samples: int):
    """Stereo audio [2, n] starting samples later, silence before it."""
    audio = np.asarray(audio, dtype=np.float32)
    return np.concatenate([np.zeros((audio.shape[0], samples), dtype=np.float32), audio], axis=1)


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    seconds: float
    has_audio: bool


def _report(path, ffmpeg=""):
    """FFmpeg's own description of a file (imageio's bundled FFmpeg comes without ffprobe)."""
    try:
        result = subprocess.run([find_ffmpeg(ffmpeg), "-hide_banner", "-nostdin", "-i", str(path)],
                                capture_output=True, check=False)
    except OSError as exc:
        raise H3Error(f"FFmpeg could not start to read {Path(path).name}: {exc}") from None
    return result.stderr.decode("utf-8", "replace")


def _video_size(report):
    # the first real video stream: an MP3's cover art is a one-picture "(attached pic)" stream
    for line in report.splitlines():
        size = re.search(r"Stream #\S+.*?: Video: .*?(\d{2,5})x(\d{2,5})", line)
        if size and "(attached pic)" not in line:
            return int(size.group(1)), int(size.group(2))
    return None


def media_kind(path, ffmpeg="") -> str:
    """"video" for a file with moving pictures, "audio" for sound alone (cover art aside)."""
    report = _report(path, ffmpeg)
    if _video_size(report):
        return "video"
    if re.search(r"Stream #\S+.*?: Audio:", report):
        return "audio"
    raise H3Error(f"FFmpeg found no video or audio in {Path(path).name}.")


def probe_video(path, ffmpeg="") -> VideoInfo:
    """Size (as FFmpeg decodes it, phone rotation applied), duration and sound of a video."""
    name = Path(path).name
    report = _report(path, ffmpeg)
    size = _video_size(report)
    duration = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", report)
    if size is None or duration is None:
        raise H3Error(f"FFmpeg found no video in {name}.")
    width, height = size
    rotation = re.search(r"rotation of (-?\d+(?:\.\d+)?) degrees", report)
    if rotation and round(abs(float(rotation.group(1)))) % 180 == 90:
        width, height = height, width
    hours, minutes, seconds = duration.groups()
    return VideoInfo(width, height, int(hours) * 3600 + int(minutes) * 60 + float(seconds),
                     re.search(r"Stream #\S+.*?: Audio:", report) is not None)


def read_video(path, width, height, max_frames, ffmpeg="", cover=False):
    """The frames of a video at H3's 24 FPS, scaled to width x height, as uint8 [frames, height, width, 3] (at most
    max_frames from the start); cover keeps the aspect ratio and crops the center instead of stretching."""
    scale = f"scale={width}:{height}:flags=lanczos"
    if cover:
        scale = f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,crop={width}:{height}"
    command = [find_ffmpeg(ffmpeg), "-v", "error", "-nostdin", "-i", str(path), "-an",
               "-vf", f"fps={FPS},{scale}", "-frames:v", str(int(max_frames)),
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


def frame_array(frames) -> np.ndarray:
    """[T, H, W, 3] uint8 from PIL images or such an array; H and W even, as yuv420p needs."""
    if isinstance(frames, np.ndarray):
        array = frames
    else:
        frames = list(frames)
        if len({frame.size for frame in frames}) > 1:
            raise H3Error("H3 frames must have equal, even dimensions.")
        array = np.stack([np.asarray(frame.convert("RGB")) for frame in frames]) if frames else np.zeros((0, 2, 2, 3), np.uint8)
    if array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise H3Error("H3 frames must be RGB images.")
    if not len(array):
        raise H3Error("The H3 backend returned no frames.")
    if array.shape[1] % 2 or array.shape[2] % 2:
        raise H3Error("H3 frames must have equal, even dimensions.")
    return np.ascontiguousarray(array)


def export_video(frames, audio, output, *, ffmpeg="", infotext="", cancelled=lambda: False, transform=None):
    """frames: [T, H, W, 3] uint8 or PIL images; audio: [channels, samples] at 32 kHz, cut or padded to the clip's
    length, or None; transform: a function applied to each frame on its way to FFmpeg (an upscaler), which must give
    every frame the same even size."""
    array = frame_array(frames)
    count = len(array)
    first = np.ascontiguousarray(transform(array[0])) if transform else array[0]
    height, width = first.shape[:2]
    if first.shape[-1] != 3 or height % 2 or width % 2:
        raise H3Error("H3 frames must have equal, even dimensions.")
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    executable = find_ffmpeg(ffmpeg)
    with tempfile.TemporaryDirectory(prefix=".h3-export-", dir=target.parent) as scratch:
        scratch = Path(scratch)
        encoded = scratch / "result.mp4"
        command = [executable, "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-framerate", str(FPS),
                   "-i", "pipe:0"]
        if audio is not None:
            _write_wave(audio, scratch / "audio.wav", round(count / FPS * SAMPLE_RATE))
            command += ["-i", str(scratch / "audio.wav"), "-map", "0:v:0", "-map", "1:a:0",
                        "-c:a", "aac", "-b:a", "192k"]
        else:
            command += ["-an"]
        command += ["-frames:v", str(count), "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-metadata", "comment=" + infotext, str(encoded)]
        errors = scratch / "ffmpeg.log"
        try:
            with open(errors, "wb") as log:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                try:
                    for n, frame in enumerate(array):
                        if cancelled():
                            raise GenerationCancelled("H3 export cancelled.")
                        if transform and n:
                            frame = np.ascontiguousarray(transform(frame))
                            if frame.shape != first.shape:
                                raise H3Error("H3 frames must have equal, even dimensions.")
                        process.stdin.write((first if n == 0 else frame).tobytes())
                    process.stdin.close()
                    process.wait(timeout=600)
                except BaseException:
                    process.kill()
                    process.wait()
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                    raise
        except BrokenPipeError:
            pass  # FFmpeg stopped reading: its own error is in the log
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise H3Error(f"H3 video export failed: {exc}") from exc
        if process.returncode or not encoded.is_file():
            message = errors.read_text(encoding="utf-8", errors="replace")[-2000:] if errors.is_file() else ""
            raise H3Error("H3 video export failed: " + message)
        if cancelled():
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
