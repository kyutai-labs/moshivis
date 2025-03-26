# pylint: disable=protected-access,no-member
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Start Pytorch backend"""

import asyncio
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple

import aiohttp
import fire
import numpy as np
import sentencepiece
import sphn
import torch
from aiohttp import web
from huggingface_hub import hf_hub_download
from kyuteye.config.enums import ImageEncoder
from kyuteye.config.kyuteye_config import KyuteyeConfig
from kyuteye.models.loaders import get_moshi_vis
from kyuteye.modules.image_transforms import get_minimal_transforms
from moshi.models.loaders import get_mimi
from torchvision.io import ImageReadMode, decode_image

if TYPE_CHECKING:
    from kyuteye.models.image_projection import ImageProjection
    from kyuteye.models.moshivis import MoshiVisGen
    from moshi.models import MimiModel


def colorize(text: str, color: str) -> str:
    """Add colors to log"""
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def make_log(level: str, msg: str) -> str:
    """Create log"""
    if level == "warning":
        prefix = colorize("[Warn]", "1;31")
    elif level == "info":
        prefix = colorize("[Info]", "1;34")
    elif level == "error":
        prefix = colorize("[Err ]", "1;31")
    else:
        raise ValueError(f"Unknown level {level}")
    return prefix + " " + msg


def log(level: str, msg: str) -> None:
    """Log with colors"""
    print(make_log(level, msg))


def seed_all(seed: int) -> None:
    """Seed"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


@dataclass
class ServerState:
    """Main server state"""

    mimi: "MimiModel"
    text_tokenizer: sentencepiece.SentencePieceProcessor
    moshi_vis: "MoshiVisGen"
    image_encoder_model: "ImageProjection"
    image_size: int
    xa_start: int
    lock: asyncio.Lock
    dtype: torch.dtype
    display_gating: bool

    def __init__(
        self,
        mimi: "MimiModel",
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        moshi_vis: "MoshiVisGen",
        image_encoder_model: "ImageProjection",
        device: str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
        max_msg_size: int = 0,
        image_size: int = 448,
        xa_start: int = 0,
    ):
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.moshi_vis = moshi_vis
        self.image_encoder_model = image_encoder_model
        self.embeddings: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None
        self.max_msg_size = max_msg_size
        self.image_size = image_size
        self.xa_start = xa_start
        self.display_gating = True
        self.device = device
        self.dtype = dtype
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lock = asyncio.Lock()

        self.mimi.streaming_forever(1)
        self.moshi_vis.streaming_forever(1)

    def warmup(self) -> None:
        """Warmup the models"""
        for _ in range(4):
            chunk = torch.zeros(
                1, 1, self.frame_size, dtype=torch.float32, device=self.device
            )
            codes = self.mimi.encode(chunk)
            ca_src = self.image_encoder_model(
                torch.zeros(1, 3, 224, 224, device=self.device)
            )["cross_attention_src"]
            for c in range(codes.shape[-1]):
                tokens, _ = self.moshi_vis.step(codes[:, :, c : c + 1], ca_src=ca_src)
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])
        torch.cuda.synchronize()
        self.mimi.reset_streaming()
        self.moshi_vis.reset_streaming()

    async def handle_chat(self, request: Any) -> Any:
        """start conversation"""
        ws = web.WebSocketResponse(max_msg_size=self.max_msg_size)
        await ws.prepare(request)
        close = False

        async def recv_loop() -> None:
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        log("error", f"{ws.exception()}")
                        break
                    if message.type == aiohttp.WSMsgType.CLOSED:
                        log("info", "closed received")
                        break
                    if message.type != aiohttp.WSMsgType.BINARY:
                        log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        log("warning", "empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        opus_reader.append_bytes(payload)
                    elif kind == 10:
                        log("info", f"received user rating {message[1]}")
                    else:
                        log("warning", f"unknown message kind {kind}")
            except Exception as e:
                print("Exception raised:", e)
            finally:
                close = True
                log("info", "connection closed")

        async def opus_loop() -> None:
            all_pcm_data = None

            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    be = time.time()
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size :]
                    chunk = torch.from_numpy(chunk)
                    chunk = chunk.to(device=self.device)[None, None]
                    codes = self.mimi.encode(chunk)
                    for c in range(codes.shape[-1]):
                        tokens, gate_weight = self.moshi_vis.step(
                            codes[:, :, c : c + 1],
                            ca_src=(
                                self.embeddings
                                if self.moshi_vis.get_streaming_attribute("offset", 0)
                                >= self.xa_start
                                else None
                            ),
                        )
                        if tokens is None:
                            continue
                        assert (
                            tokens.shape[1]
                            == self.moshi_vis.num_audio_codebooks_out + 1
                        )
                        main_pcm = self.mimi.decode(tokens[:, 1:])
                        main_pcm = main_pcm.cpu()
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                            _text = _text.replace("▁", " ")
                            text_color = round(
                                max(min((gate_weight - 0.005) / 0.016, 1.0), 0.0) * 10
                            )
                            msg = (
                                b"\x07"
                                + text_color.to_bytes(1, "big")
                                + bytes(_text, encoding="utf8")
                            )
                            log("info", f"text token '{_text}'")
                            await ws.send_bytes(msg)
                    log("info", f"frame handled in {1000 * (time.time() - be):.1f}ms")

        async def send_loop() -> None:
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        log("info", "accepted connection")
        close = False
        query_parameters = request.rel_url.query
        self.moshi_vis.update_gen_kwargs(
            temp_text=(
                float(aux)
                if (aux := query_parameters.get("text_temperature", None)) is not None
                else None
            ),
            temp=(
                float(aux)
                if (aux := query_parameters.get("audio_temperature", None)) is not None
                else None
            ),
            top_k_text=(
                int(aux)
                if (aux := query_parameters.get("text_topk", None)) is not None
                else None
            ),
            top_k=(
                int(aux)
                if (aux := query_parameters.get("audio_topk", None)) is not None
                else None
            ),
        )
        self.image_size = (
            int(aux)
            if (aux := query_parameters.get("image_resolution", None)) is not None
            else self.image_size
        )
        if (aux := query_parameters.get("xa_start", None)) is not None:
            self.xa_start = int(aux)

        async with self.lock:
            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)  # type: ignore
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)  # type: ignore
            self.mimi.reset_streaming()
            self.moshi_vis.reset_streaming()

            await self.extract_image(ws)
            # Send the handshake.
            await ws.send_bytes(b"\x00")
            await asyncio.gather(opus_loop(), recv_loop(), send_loop())
        log("info", "done with connection")
        return ws

    async def extract_image(self, ws: web.WebSocketResponse) -> None:
        """Embed imageat the beginning of the stream"""
        first_message = await ws.receive()
        first_message = first_message.data
        try:
            kind = first_message[0]
        except Exception as e:
            raise RuntimeError(f"Error in message: {first_message}") from e
        if kind != 8:  # image
            raise RuntimeError(f"unknown message kind {kind}")
        payload = first_message[1:]
        image_tensor = decode_image(
            torch.frombuffer(payload, dtype=torch.uint8), mode=ImageReadMode.RGB
        )
        image_tensor = get_minimal_transforms(self.image_size)(image_tensor)
        image_tensor = self.image_encoder_model.to_tensor_and_normalize(image_tensor)
        log("info", f"Loaded image tensor with shape {image_tensor.shape}")
        if self.image_encoder_model.encoder_type == ImageEncoder.PIXTRAL:
            image_tensor = [image_tensor.to(self.device)]
        else:
            image_tensor = image_tensor[None, ...].to(self.device)
        k, v = self.moshi_vis.precompte_ca_kv(
            self.image_encoder_model(image_tensor)["cross_attention_src"]
        )
        self.embeddings = (k.to(self.dtype), v.to(self.dtype))


def start_server(
    kyuteye_config_path: str,
    host: str = "localhost",
    port: int = 8998,
    static: Optional[str] = None,
    device: str = "cuda",
    dtype: Literal["float32", "bfloat16"] = "bfloat16",
    ssl: bool = True,
    ssl_cert_dir: Optional[str] = None,
) -> None:
    """Start server

    :param kyuteye_config: Config of the model to load
    :param host: Host to start the server on
    :param port: Port to start the server on
    :param static: Path to the built client source. If None, defaults to the local client
    :param device: Device of execution
    :param dtype: Dtype of execution
    :param ssl: Whether to launch on https or http protocol
    :param max_img_size: Max image size (in MB) that can be
    sent via aiohttp; If 0, no limit is set. Note that input images
    are resized in any case before being sent to the encoder
    """
    assert kyuteye_config_path is not None
    root_dir = Path(__file__).parents[2]
    static_path: None | str = None
    if static is None:
        static_path = str(root_dir / "client" / "dist")
    else:
        static_path = static

    static_path = os.path.abspath(static_path)
    assert static_path is not None and os.path.exists(static_path)
    seed_all(42)
    setup_tunnel = None
    tunnel_token = ""
    kyuteye_config = KyuteyeConfig.from_yml(kyuteye_config_path)
    # Load main model components
    log("info", "loading mimi")
    if kyuteye_config.hf_repo is None:
        assert os.path.exists(kyuteye_config.mimi_codec)
        mimi_weight = kyuteye_config.mimi_codec
    else:
        mimi_weight = hf_hub_download(kyuteye_config.hf_repo, kyuteye_config.mimi_codec)
    mimi = get_mimi(mimi_weight, device)
    log("info", "mimi loaded")

    if kyuteye_config.hf_repo is None:
        assert os.path.exists(kyuteye_config.text_tokenizer)
        text_tokenizer_path = kyuteye_config.text_tokenizer
    else:
        text_tokenizer_path = hf_hub_download(
            kyuteye_config.hf_repo, kyuteye_config.text_tokenizer
        )
    text_tokenizer = sentencepiece.SentencePieceProcessor(text_tokenizer_path)  # type: ignore

    log("info", "loading moshi + vision")
    if kyuteye_config.hf_repo is None:
        assert os.path.exists(kyuteye_config.model)
        moshi_weight = kyuteye_config.model
        if not moshi_weight.endswith("_pt.safetensors"):
            assert moshi_weight.endswith(".safetensors")
            moshi_weight = moshi_weight.replace(".safetensors", "_pt.safetensors")
            print(f"Will load from {moshi_weight}")
    else:
        moshi_weight = hf_hub_download(kyuteye_config.hf_repo, kyuteye_config.model)
    torch_dtype = getattr(torch, dtype)
    moshi_vis, image_embedder = get_moshi_vis(
        kyuteye_config, moshi_weight, device, torch_dtype
    )
    log("info", "moshi + vision loaded")

    state = ServerState(
        mimi=mimi,
        text_tokenizer=text_tokenizer,
        moshi_vis=moshi_vis,
        image_encoder_model=image_embedder,
        device=device,
        dtype=torch_dtype,
        xa_start=kyuteye_config.xa_start,
    )
    log("info", "warming up the model")
    state.warmup()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)

    async def handle_root(_):  # type: ignore
        return web.FileResponse(os.path.join(static_path, "index.html"))

    log("info", f"serving static content from {static_path}")
    app.router.add_get("/", handle_root)
    app.router.add_static("/", path=static_path, follow_symlinks=True, name="static")
    protocol = "http"
    ssl_context = None
    if ssl:
        import ssl as ssl_module

        ssl_context = ssl_module.create_default_context(ssl_module.Purpose.CLIENT_AUTH)
        ssl_cert_dir = ssl_cert_dir or str(root_dir)
        ssl_context.load_cert_chain(
            certfile=os.path.join(ssl_cert_dir, "cert.pem"),
            keyfile=os.path.join(ssl_cert_dir, "key.pem"),
        )
        protocol = "https"

    log("info", f"Access the Web UI directly at {protocol}://{host}:{port}")
    if setup_tunnel is not None:
        tunnel = setup_tunnel("localhost", port, tunnel_token, None)
        log(
            "info",
            f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.",
        )
        log(
            "info",
            "Note that this tunnel goes through the US and you"
            " might experience high latency in Europe.",
        )
    with torch.no_grad():
        web.run_app(app, port=port, ssl_context=ssl_context)


def sanity_check() -> None:
    pass


def main() -> None:
    """main"""
    fire.Fire(start_server)
