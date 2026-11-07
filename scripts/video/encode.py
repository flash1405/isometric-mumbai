"""Frames -> H.264 MP4 for X/Twitter (1080p30, yuv420p, faststart).
  uv run --with imageio-ffmpeg python scripts/video/encode.py out/video/frames out/video/isometric-mumbai.mp4"""
import subprocess, sys
import imageio_ffmpeg
src, dst = sys.argv[1], sys.argv[2]
ff = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ff, "-y", "-framerate", "30", "-i", f"{src}/frame_%05d.jpg", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-vf", "scale=1920:1080:flags=lanczos", dst], check=True)
print("->", dst)
