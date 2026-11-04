"""Relay localhost HTTP to a Slurm job's localhost without exposing cluster ports."""

import argparse
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", type=int)
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.relay("GET")

        def do_POST(self) -> None:
            self.relay("POST")

        def relay(self, method: str) -> None:
            if self.path not in {"/generate", "/info", "/models"}:
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            if length > 2 * 1024**2:
                self.send_error(413)
                return
            body = self.rfile.read(length)
            command = [
                "srun",
                f"--jobid={args.job_id}",
                "--overlap",
                "--ntasks=1",
                "--cpus-per-task=1",
                "--gres=none",
                "curl",
                "--silent",
                "--show-error",
                "--max-time",
                "600",
                "--write-out",
                "\n%{http_code}",
                "-X",
                method,
                "-H",
                "Content-Type: application/json",
                "http://127.0.0.1:8000" + self.path,
            ]
            if method == "POST":
                command.extend(["--data-binary", "@-"])
            try:
                result = subprocess.run(command, input=body, capture_output=True, check=True, timeout=650)
                payload, status = result.stdout.rsplit(b"\n", 1)
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (subprocess.SubprocessError, ValueError) as error:
                self.send_error(502, str(error))

    ThreadingHTTPServer(("127.0.0.1", 8001), Handler).serve_forever()


if __name__ == "__main__":
    main()
