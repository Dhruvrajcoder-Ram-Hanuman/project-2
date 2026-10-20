"""
Alias module for camera execution (provides standard English spelling 'camera.py').
Runs the ANPR live camera stream.
"""
from camara import run_live_camera

if __name__ == "__main__":
    run_live_camera(camera_index=0, cam_id="CAM_004")
