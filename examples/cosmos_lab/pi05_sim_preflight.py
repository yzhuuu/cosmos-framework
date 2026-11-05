"""Check EGL and native LIBERO rendering under the actual gateway child environment."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from cosmos_framework.posttrain.roles.env.sim_gateway import SimGatewayBody


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--load-only", action="store_true")
    args = parser.parse_args()
    root = Path(os.environ["COSMOS_LAB_ROOT"])
    python = str(root / "runtime/libero/venv/bin/python")
    # The constructor starts IPC immediately. Exercise only its environment builder here.
    gateway = object.__new__(SimGatewayBody)
    gateway._sim_gpu = "0"
    gateway._sim_ld = os.environ.get("SIM_LD_LIBRARY_PATH", "")
    child = gateway._child_env()
    subprocess.run(
        [python, "-c", "from OpenGL import EGL; assert EGL.eglQueryString; print('gateway EGL library load passed')"],
        env=child,
        check=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sim-environment-admission.json").write_text(
        json.dumps(
            {key: child.get(key) for key in ["LD_LIBRARY_PATH", "__EGL_VENDOR_LIBRARY_FILENAMES", "MUJOCO_GL"]},
            indent=2,
        )
        + "\n"
    )
    if not args.load_only:
        subprocess.run(
            [
                python,
                "-m",
                "cosmos_framework.posttrain.embodied.sims.libero.preflight",
                "--train-manifest",
                str(root / "runtime/libero/manifests/train.json"),
                "--eval-manifest",
                str(root / "runtime/libero/manifests/eval.json"),
                "--execution-profile",
                "rlinf_pi05_spatial_v2",
                "--output",
                str(args.output / "native-sim-preflight.json"),
            ],
            env=child,
            check=True,
        )


if __name__ == "__main__":
    main()
