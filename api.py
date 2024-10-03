# api.py

import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
import platform

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Video Generation API", description="Generate videos from images and audio.", version="1.0")

# Define the base directory paths
BASE_DIR = Path(__file__).parent.resolve()
SCRIPT_PATH = BASE_DIR / "scripts" / "inference.py"
OUTPUT_DIR = BASE_DIR / "api_output"

# Ensure the output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def run_inference(
    ref_img_path: str,
    ref_audio_path: str,
    output_video_path: str,
    settings_face_expand_ratio: float,
    setting_steps: int,
    setting_cfg: float,
    settings_seed: int,
    settings_fps: int,
    settings_motion_pose_scale: float,
    settings_motion_face_scale: float,
    settings_motion_lip_scale: float,
    settings_n_motion_frames: int,
    settings_n_sample_frames: int,
):
    """
    Executes the inference script with the provided parameters.
    """
    # Determine the Python command based on the operating system
    if platform.system() == "Windows":
        python_cmd = "python"
    else:
        python_cmd = "python3"

    command = [
        python_cmd,
        str(SCRIPT_PATH),
        "--source_image", ref_img_path,
        "--driving_audio", ref_audio_path,
        "--output", output_video_path,
        "--setting_steps", str(setting_steps),
        "--setting_cfg", str(setting_cfg),
        "--settings_seed", str(settings_seed),
        "--settings_fps", str(settings_fps),
        "--settings_motion_pose_scale", str(settings_motion_pose_scale),
        "--settings_motion_face_scale", str(settings_motion_face_scale),
        "--settings_motion_lip_scale", str(settings_motion_lip_scale),
        "--settings_n_motion_frames", str(settings_n_motion_frames),
        "--settings_n_sample_frames", str(settings_n_sample_frames)
    ]

    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise HTTPException(status_code=500, detail=f"Error during video generation: {e.stderr}")


@app.post("/generate_video")
async def generate_video(
    ref_img: UploadFile = File(..., description="Reference image file"),
    ref_audio: UploadFile = File(..., description="Reference audio file"),
    settings_face_expand_ratio: Optional[float] = Form(1.2),
    setting_steps: Optional[int] = Form(40),
    setting_cfg: Optional[float] = Form(3.5),
    settings_seed: Optional[int] = Form(42),
    settings_fps: Optional[int] = Form(25),
    settings_motion_pose_scale: Optional[float] = Form(1.1),
    settings_motion_face_scale: Optional[float] = Form(1.1),
    settings_motion_lip_scale: Optional[float] = Form(1.1),
    settings_n_motion_frames: Optional[int] = Form(2),
    settings_n_sample_frames: Optional[int] = Form(16),
):
    """
    Endpoint to generate a video from a reference image and audio.
    """
    # Create a temporary directory to store uploaded files and output
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Save the uploaded reference image
        ref_img_path = tmpdir_path / ref_img.filename
        with ref_img_path.open("wb") as f:
            shutil.copyfileobj(ref_img.file, f)
        
        # Save the uploaded reference audio
        ref_audio_path = tmpdir_path / ref_audio.filename
        with ref_audio_path.open("wb") as f:
            shutil.copyfileobj(ref_audio.file, f)
        
        # Define the output video path
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_video_filename = f"{timestamp}.mp4"
        output_video_path = OUTPUT_DIR / output_video_filename
        
        # Run the inference script
        run_inference(
            ref_img_path=str(ref_img_path),
            ref_audio_path=str(ref_audio_path),
            output_video_path=str(output_video_path),
            settings_face_expand_ratio=settings_face_expand_ratio,
            setting_steps=setting_steps,
            setting_cfg=setting_cfg,
            settings_seed=settings_seed,
            settings_fps=settings_fps,
            settings_motion_pose_scale=settings_motion_pose_scale,
            settings_motion_face_scale=settings_motion_face_scale,
            settings_motion_lip_scale=settings_motion_lip_scale,
            settings_n_motion_frames=settings_n_motion_frames,
            settings_n_sample_frames=settings_n_sample_frames
        )
        
        # Check if the output video was created
        if not output_video_path.exists():
            raise HTTPException(status_code=500, detail="Video generation failed.")
        
        # Return the video file as a response
        return FileResponse(
            path=output_video_path,
            filename=output_video_filename,
            media_type="video/mp4"
        )

if __name__ == "__main__":
    import uvicorn

    # Determine the host and port, you can also make these configurable via environment variables or arguments
    host = "0.0.0.0"
    port = 8000

    # Start the Uvicorn server
    uvicorn.run("api:app", host=host, port=port, reload=True)
