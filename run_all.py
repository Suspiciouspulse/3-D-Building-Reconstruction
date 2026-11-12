"""Run the whole pipeline: footprints and shadows -> checks and height fixes -> outputs.

All assumptions (scale, sun elevation, storey height, roof pitch) are in common.py.
Hand-drawn outlines are in manual_footprints.json.
"""
import build_scene
import segment
import verify

if __name__ == "__main__":
    segment.main()
    verify.main()
    build_scene.main()
