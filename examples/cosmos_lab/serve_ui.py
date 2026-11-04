"""Local video playground for the native Cosmos Ray /generate endpoint."""

import json
import os
import uuid
from pathlib import Path

import gradio as gr
import httpx

ROOT = Path(os.environ["COSMOS_LAB_ROOT"])
SERVER = "http://127.0.0.1:8001"
TEMPLATE = json.loads((ROOT / "data/baseline-robot-relocation/requests.jsonl").read_text().splitlines()[1])
RUN_DIR = Path(os.environ["COSMOS_SERVE_RUN_DIR"])


def generate(mode: str, prompt: str, image: str | None, seed: int, frames: int):
    """Submit an explicit video request and expose its saved request/response."""
    request = {
        **TEMPLATE,
        "name": f"interactive_{uuid.uuid4().hex[:12]}",
        "prompt": prompt,
        "seed": int(seed),
        "num_frames": int(frames),
        "model_mode": mode,
    }
    if mode == "image2video":
        request["vision_path"] = image or TEMPLATE["vision_path"]
    else:
        request.pop("vision_path", None)
    receipt = RUN_DIR / "ui-requests" / request["name"]
    receipt.mkdir(parents=True)
    (receipt / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    yield None, request, {"status": "generating", "message": "生成中，请等待。"}
    try:
        response = httpx.post(f"{SERVER}/generate", json=request, timeout=600)
        response.raise_for_status()
        result = response.json()
        (receipt / "response.json").write_text(json.dumps(result, indent=2) + "\n")
        if result.get("status") != "success":
            raise RuntimeError(result)
        videos = [p for item in result["outputs"] for p in item["files"] if p.endswith("vision.mp4")]
        assert len(videos) == 1, videos
        video = (RUN_DIR / "generated" / videos[0]).resolve()
        assert video.is_relative_to((RUN_DIR / "generated").resolve()) and video.is_file()
        yield str(video), request, result
    except Exception as error:
        (receipt / "error.txt").write_text(str(error))
        raise gr.Error(str(error)) from error


def main() -> None:
    """Expose only the run output and upload directory through localhost."""
    with gr.Blocks(title="Cosmos Nano 视频实验") as app:
        gr.Markdown("# Cosmos3-Nano 原始模型\n输入文字，或加一张首帧图片。描述直接提交，关闭自动改写。")
        mode = gr.Radio(["text2video", "image2video"], value="image2video", label="输入方式")
        prompt = gr.Textbox(value=TEMPLATE["prompt"], lines=10, label="文字描述（支持完整 JSON prompt）")
        initial = gr.Image(value=TEMPLATE["vision_path"], type="filepath", label="首帧图片（I2V）")
        seed = gr.Number(value=2026, precision=0, label="随机种子")
        frames = gr.Dropdown([97, 193], value=193, label="帧数（24 FPS，约 4／8 秒）")
        button = gr.Button("生成视频", variant="primary")
        output = gr.Video(label="生成结果")
        with gr.Accordion("本次输入和运行记录", open=False):
            request = gr.JSON(label="实际请求")
            response = gr.JSON(label="状态和结果")
        button.click(generate, [mode, prompt, initial, seed, frames], [output, request, response], concurrency_limit=1)
    app.queue(max_size=4).launch(server_name="127.0.0.1", server_port=7861, allowed_paths=[str(RUN_DIR)], share=False)


if __name__ == "__main__":
    main()
