# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Entrypoint for the web server."""

import argparse
import asyncio
import io
import multiprocessing
import multiprocessing.queues
import os
import queue
import sys
import tarfile
import time
import webbrowser
from enum import Enum
from pathlib import Path
from typing import Any

import aiohttp
import huggingface_hub
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import rustymimi
import sentencepiece
import sphn
from aiohttp import web
from jaxtyping import UInt32
from PIL import Image

from kyuteye_mlx import models, utils
from kyuteye_mlx.models.pixtral import PixtralWrapper
from kyuteye_mlx.models.siglip import SiglipWrapper
from kyuteye_mlx.quantize import quantize
from kyuteye_mlx.utils.loading import repeat_shared_weights, split_embedder_weights
from kyuteye_mlx.utils.profiling import PROFILING_ENABLED, profile

SAMPLE_RATE = 24000
FRAME_SIZE = 1920
CHANNELS = 1


class ModelInput(Enum):
    AUDIO = 0
    IMAGE = 1


class ModelOutput(Enum):
    AUDIO = 0
    TEXT = 1
    START = 2
    IMAGE_PROCESSED = 3


class ServerMediaInput(Enum):
    AUDIO = 1
    IMAGE = 8


ClientServerQueue = multiprocessing.queues.Queue[tuple[ModelInput, Any]]
ServerClientQueue = multiprocessing.queues.Queue[tuple[ModelOutput, Any]]


def colorize(text: str, color: str) -> str:
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def log(level: str, msg: str) -> None:
    if level == "warning":
        prefix = colorize("[Warn]", "1;31")
    elif level == "info":
        prefix = colorize("[Info]", "1;34")
    elif level == "error":
        prefix = colorize("[Err ]", "1;31")
    else:
        raise ValueError(f"Unknown level {level}")
    print(prefix + " " + msg)


def hf_hub_download(repo: str | None, path: str) -> str:
    if repo is None or repo == "":
        raise ValueError(f"the --hf-repo flag is required to retrieve {path}")
    return huggingface_hub.hf_hub_download(repo, path)


def full_warmup(
    audio_tokenizer: rustymimi.StreamTokenizer,
    client_to_server: ClientServerQueue,
    server_to_client: ServerClientQueue,
) -> None:
    for i in range(4):
        pcm_data = np.array([0.0] * 1920).astype(np.float32)
        audio_tokenizer.encode(pcm_data)
        while True:
            time.sleep(0.01)
            data = audio_tokenizer.get_encoded()
            if data is not None:
                break
        client_to_server.put_nowait((ModelInput.AUDIO, data))
        if i == 0:
            continue
        while True:
            kind, data = server_to_client.get()
            if kind == ModelOutput.AUDIO:
                audio_tokenizer.decode(data)
                break
        while True:
            time.sleep(0.01)
            data = audio_tokenizer.get_decoded()
            if data is not None:
                break


def get_model_file(args) -> str:
    model_file = args.moshi_weights
    if model_file is None:
        if args.quantized in (4, 8):
            model_file = hf_hub_download(args.hf_repo, f"model.q{args.quantized}.safetensors")
        elif args.quantized is not None:
            raise ValueError(f"Invalid quantized value: {args.quantized}")
        else:
            model_file = hf_hub_download(args.hf_repo, "model.safetensors")
    return model_file


def get_tokenizer(args) -> sentencepiece.SentencePieceProcessor:
    tokenizer_file = args.tokenizer
    if tokenizer_file is None:
        tokenizer_file = hf_hub_download(args.hf_repo, "tokenizer_spm_32k_3.model")
    log("info", f"[SERVER] loading text tokenizer {tokenizer_file}")
    return sentencepiece.SentencePieceProcessor(tokenizer_file)  # type: ignore


def get_embedder(args) -> SiglipWrapper | PixtralWrapper:
    model_file = get_model_file(args)
    if args.encoder == "pixtral":
        lm_config = models.config_pixtral()
    elif args.encoder == "siglip":
        lm_config = models.config_siglip()
    else:
        raise ValueError(f"Unknown encoder {args.encoder}")
    weights = mx.load(model_file)
    if lm_config.transformer.xa_shared:
        # for shared cross-attention, we have the weights only once in ckpt
        weights = repeat_shared_weights(weights, lm_config.transformer.num_layers)

    _, embedder_weights = split_embedder_weights(weights)

    if args.encoder == "siglip":
        embedder = SiglipWrapper()
    elif args.encoder == "pixtral":
        embedder = PixtralWrapper()
    else:
        raise ValueError(f"Unknown encoder {args.encoder}")

    if embedder_weights:
        log("info", "[SERVER] loading embedder weights")
        embedder_weights["model.embeddings.patch_embedding.weight"] = embedder_weights[
            "model.embeddings.patch_embedding.weight"
        ].transpose(0, 2, 3, 1)
        embedder.load_weights(list(embedder_weights.items()), strict=True)
        log("info", "[SERVER] embedder weights loaded")
    embedder.eval()
    log("info", "[SERVER] Embedder warmed up")
    return embedder


def get_model(args, load_weights: bool = True) -> models.LmGen:
    mx.random.seed(299792458)
    if args.encoder == "pixtral":
        lm_config = models.config_pixtral()
    elif args.encoder == "siglip":
        lm_config = models.config_siglip()
    else:
        raise ValueError(f"Unknown encoder {args.encoder}")
    lm_config.transformer.xa_start = args.xa_start

    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if args.quantized is not None:
        group_size = 32 if args.quantized == 4 else 64
        nn.quantize(model, bits=args.quantized, group_size=group_size)

    if load_weights:
        model_file = get_model_file(args)
        log("info", f"[SERVER] loading weights {model_file}")
        weights = mx.load(model_file)
        if lm_config.transformer.xa_shared:
            # for shared cross-attention, we have the weights only once in ckpt
            weights = repeat_shared_weights(weights, lm_config.transformer.num_layers)
        weights, _ = split_embedder_weights(weights)

        model.load_weights(list(weights.items()), strict=True)
        log("info", "[SERVER] weights loaded")

    model.warmup()
    log("info", "[SERVER] model warmed up")
    gen = models.LmGen(
        model=model,
        max_steps=args.steps + 5,
        text_sampler=utils.Sampler(temp=args.text_temperature, top_p=args.text_top_p),
        audio_sampler=utils.Sampler(temp=args.audio_temperature, top_p=args.audio_top_p),
        check=False,
    )

    return gen


def model_server(
    client_to_server: ClientServerQueue,
    server_to_client: ServerClientQueue,
    args: argparse.Namespace,
):
    gen = get_model(args)
    embedder = get_embedder(args)
    text_tokenizer = get_tokenizer(args)

    server_to_client.put((ModelOutput.START, "start"))
    log("info", "[SERVER] connected!")
    try:
        uploaded_image_embeddings = None
        gen.reset()

        for i in range(10000000000000):
            if i == 150:
                if PROFILING_ENABLED:
                    profile.print_stats()

            data_type, data = client_to_server.get()
            if data_type == ModelInput.AUDIO:
                handle_audio(
                    data,
                    gen,
                    uploaded_image_embeddings,
                    server_to_client,
                    text_tokenizer,
                )
            elif data_type == ModelInput.IMAGE:
                img = Image.open(io.BytesIO(data))
                # compute longer image size
                if args.encoder == "pixtral":
                    w, h = img.width, img.height
                    if w > h:
                        new_w = args.img_size
                        new_h = int(h * new_w / w)
                    else:
                        new_h = args.img_size
                        new_w = int(w * new_h / h)
                    img = img.resize((new_w, new_h), resample=3)
                else:
                    img = img.resize((args.img_size, args.img_size), resample=3)
                image = np.asarray(img)
                uploaded_image_embeddings = embedder(mx.array(image)).astype(mx.bfloat16)
                gen.reset()
                log("info", f"received image embeddings: {image.shape}")
                server_to_client.put_nowait((ModelOutput.IMAGE_PROCESSED, None))
    except KeyboardInterrupt:
        pass


def handle_audio(
    data: UInt32[np.ndarray, "tokens 1"],
    gen: models.LmGen,
    uploaded_image_embeddings: mx.array | None,
    server_to_client: ServerClientQueue,
    text_tokenizer: sentencepiece.SentencePieceProcessor,
):
    t_start = time.time()
    text_token, audio_tokens = predict_text_and_audio(gen, mx.array(data), uploaded_image_embeddings)
    elapsed_eval_seconds = time.time() - t_start
    elapsed_eval_milliseconds = elapsed_eval_seconds * 1000
    print(f"eval in {elapsed_eval_milliseconds} ms")
    text_token = text_token.item()
    if text_token not in (0, 3):
        _text = text_tokenizer.id_to_piece(text_token)  # type: ignore
        _text = _text.replace("▁", " ")
        server_to_client.put_nowait((ModelOutput.TEXT, _text))
    if audio_tokens is not None:
        audio_tokens = np.array(audio_tokens).astype(np.uint32)
        server_to_client.put_nowait((ModelOutput.AUDIO, audio_tokens))
    elapsed_seconds = time.time() - t_start
    elapsed_milliseconds = elapsed_seconds * 1000
    print(f"step in {elapsed_milliseconds} ms")


def predict_text_and_audio(
    gen: models.LmGen, data: UInt32[mx.array, "tokens 1"], uploaded_image_embeddings
) -> tuple[mx.array, mx.array | None]:
    data = data.transpose(1, 0)[:, :8]
    if gen.model.xa_cache is not None and not gen.model.xa_cache.is_set:
        text_token = gen.step(data, uploaded_image_embeddings)
    else:
        text_token = gen.step(data, None)
    text_token = text_token[0]
    audio_tokens = gen.last_audio_tokens()
    mx.eval((text_token, audio_tokens))
    return text_token, audio_tokens


def web_server(
    client_to_server: ClientServerQueue,
    server_to_client: ServerClientQueue,
    args: argparse.Namespace,
):
    mimi_file = args.mimi_weights
    if mimi_file is None:
        mimi_file = hf_hub_download(args.hf_repo, "tokenizer-e351c8d8-checkpoint125.safetensors")
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    text_queue = queue.Queue()
    audio_tokenizer = rustymimi.StreamTokenizer(mimi_file)  # type: ignore
    kind, start_message = server_to_client.get()
    if kind != ModelOutput.START:
        log(
            "error",
            f"[CLIENT] recieve {(kind, start_message)} at startup, this is unexpected.",
        )
    log("info", f"[CLIENT] received '{start_message}' from server, starting...")

    full_warmup(audio_tokenizer, client_to_server, server_to_client)

    log("info", "warmup done")

    async def send_loop() -> None:
        while True:
            await asyncio.sleep(0.001)
            try:
                pcm_data = input_queue.get(block=False)
                audio_tokenizer.encode(pcm_data)
            except queue.Empty:
                continue

    async def recv_loop() -> None:
        while True:
            data = audio_tokenizer.get_decoded()
            if data is None:
                await asyncio.sleep(0.001)
                continue
            output_queue.put_nowait(data)

    async def send_loop2() -> None:
        while True:
            data = audio_tokenizer.get_encoded()
            if data is None:
                await asyncio.sleep(0.001)
                continue
            client_to_server.put_nowait((ModelInput.AUDIO, data))

    async def recv_loop2() -> None:
        while True:
            try:
                kind, data = server_to_client.get(block=False)
                if kind == ModelOutput.AUDIO:
                    audio_tokenizer.decode(data)
                elif kind == ModelOutput.TEXT:
                    text_queue.put_nowait(data)
            except queue.Empty:
                await asyncio.sleep(0.001)
                continue

    lock = asyncio.Lock()

    async def handle_chat(request) -> None:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async def recv_loop() -> None:
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
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
                    if kind == ServerMediaInput.AUDIO.value:
                        payload = message[1:]
                        opus_reader.append_bytes(payload)
                    else:
                        log("warning", f"unknown message kind {kind}")
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
                while all_pcm_data.shape[-1] >= FRAME_SIZE:
                    chunk = all_pcm_data[:FRAME_SIZE]
                    all_pcm_data = all_pcm_data[FRAME_SIZE:]
                    input_queue.put_nowait(chunk)

        async def send_loop() -> None:
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)
                try:
                    _text = text_queue.get(block=False)
                    await ws.send_bytes(b"\x02" + bytes(_text, encoding="utf8"))
                except queue.Empty:
                    continue

        async def another_loop() -> None:
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                try:
                    pcm_data = output_queue.get(block=False)
                    assert pcm_data.shape == (1920,), pcm_data.shape
                    opus_writer.append_pcm(pcm_data)
                except queue.Empty:
                    continue

        log("info", "accepted connection")
        close = False
        async with lock:
            opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
            opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
            # Send the handshake.
            encoded_image = await extract_image(ws)
            client_to_server.put_nowait((ModelInput.IMAGE, encoded_image))
            message_type, _ = server_to_client.get()
            if message_type != ModelOutput.IMAGE_PROCESSED:
                log("error", f"We recieved {message_type} instead of IMAGE_PROCESSED.")

            await ws.send_bytes(b"\x00")
            await asyncio.gather(opus_loop(), recv_loop(), send_loop(), another_loop())
        log("info", "done with connection")
        return ws

    async def extract_image(ws: web.WebSocketResponse) -> bytes:
        """Get the image at the start of the stream.

        The bytes returned are encoded (png, jpeg, etc...).
        """
        first_message = await ws.receive()
        first_message = first_message.data
        kind = first_message[0]
        if kind != ServerMediaInput.IMAGE.value:
            log("error", f"First messsage should be an image, got {kind}")
        return first_message[1:]

    async def go() -> None:
        app = web.Application()
        app.router.add_get("/api/chat", handle_chat)
        static_path: None | str = None
        if args.static is None:
            log("info", "retrieving the static content")
            dist_tgz = hf_hub_download("kyutai/moshi-artifacts", "vis_dist.tgz")
            dist_tgz = Path(dist_tgz)
            dist = dist_tgz.parent / "dist"
            if not dist.exists():
                with tarfile.open(dist_tgz, "r:gz") as tar:
                    tar.extractall(path=dist_tgz.parent)
            static_path = str(dist)
        elif args.static != "none":
            # When set to the "none" string, we don't serve any static content.
            static_path = args.static
        if static_path is not None:

            async def handle_root(_) -> web.FileResponse:
                return web.FileResponse(os.path.join(static_path, "index.html"))

            log("info", f"serving static content from {static_path}")
            app.router.add_get("/", handle_root)
            app.router.add_static("/", path=static_path, name="static")
        runner = web.AppRunner(app)
        await runner.setup()
        ssl_context = None
        protocol = "http"
        if args.ssl is not None:
            import ssl

            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            cert_file = os.path.join(args.ssl, "cert.pem")
            key_file = os.path.join(args.ssl, "key.pem")
            ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
            protocol = "https"
        site = web.TCPSite(runner, args.host, args.port, ssl_context=ssl_context)

        log("info", f"listening to {protocol}://{args.host}:{args.port}")

        if not args.no_browser:
            log("info", f"opening browser at {protocol}://{args.host}:{args.port}")
            webbrowser.open(f"{protocol}://{args.host}:{args.port}")

        await asyncio.gather(recv_loop(), send_loop(), recv_loop2(), send_loop2(), site.start())
        await runner.cleanup()

    try:
        asyncio.run(go())
    except KeyboardInterrupt:
        pass


def get_args_for_main() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--moshi-weights", type=str)
    parser.add_argument("--mimi-weights", type=str)
    parser.add_argument("-q", "--quantized", type=int, choices=[4, 8])
    parser.add_argument("--steps", default=4000, type=int)
    parser.add_argument("--hf-repo", type=str, default="kyutai/moshika-vis-mlx")
    parser.add_argument("--static", type=str)
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--encoder", type=str, default="siglip")
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8008, type=int)
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        ),
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--text-temperature", type=float, default=0.45)
    parser.add_argument("--audio-temperature", type=float, default=0.7)
    parser.add_argument("--text-top-p", type=float, default=0.95)
    parser.add_argument("--audio-top-p", type=float, default=0.95)
    parser.add_argument("--xa-start", type=int, default=16)

    args = parser.parse_args()
    if args.moshi_weights is not None:
        args.quantized = (
            args.quantized or 8 if "q8" in args.moshi_weights else 4 if "q4" in args.moshi_weights else None
        )
    return args


def main() -> None:
    args = get_args_for_main()

    client_to_server: ClientServerQueue = multiprocessing.Queue()
    server_to_client: ServerClientQueue = multiprocessing.Queue()

    # Create two processes
    subprocess_args = client_to_server, server_to_client, args
    p1 = multiprocessing.Process(target=web_server, args=subprocess_args)
    p2 = multiprocessing.Process(target=model_server, args=subprocess_args)

    # Start the processes
    p1.start()
    p2.start()

    try:
        while p1.is_alive() and p2.is_alive():
            time.sleep(0.001)
    except KeyboardInterrupt:
        log("warning", "Interrupting, exiting connection.")
        p1.terminate()
        p2.terminate()

    # Wait for both processes to finish
    p1.join()
    p2.join()
    log("info", "All done!")


def sanity_check() -> None:
    """A small sanity check to make sure all packages can be imported."""


if __name__ == "__main__":
    main()
