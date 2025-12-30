import image
import video

from video import Video

import asyncio
import json
import time
import torch
import gc
import multiprocessing
import signal
import PIL.Image
import PIL.ImageFilter


async def server():
    server = await asyncio.start_server(instance, "0.0.0.0", 9148)
    print("[kiroshi] Server started at 0.0.0.0:9148")

    async with server:
        await server.serve_forever()


async def instance(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    message = await read(reader)
    message = json.loads(message)

    print(f"[kiroshi] Received: {message}")

    match message["task"]:
        case "generate_image":
            recipe = image.Recipe.from_dict(message)

            await run(image.generate, writer, message, recipe)

        case "detail_faces":
            recipe = image.Recipe.from_dict(message)
            detail = image.Detail.from_dict(message["detail"])
            input = await read_image(reader, writer, recipe.size)

            await run(image.detail_faces, writer, message, recipe, detail, input)

        case "detail_hands":
            recipe = image.Recipe.from_dict(message)
            detail = image.Detail.from_dict(message["detail"])
            input = await read_image(reader, writer, recipe.size)

            await run(image.detail_hands, writer, message, recipe, detail, input)

        case "upscale":
            upscaler = image.Upscaler.from_dict(message["upscaler"])
            size = image.Size.from_dict(message["size"])
            input = await read_image(reader, writer, size)

            await run(image.upscale, writer, message, upscaler, input)

        case "generate_video":
            recipe = video.Recipe.from_dict(message)
            first_frame = await read_image(reader, writer, recipe.size)
            last_frame = None

            if await read(reader) == bytes([1]):
                last_frame = await read_image(reader, writer, recipe.size)

            await run(video.generate, writer, message, recipe, first_frame, last_frame)

        case task:
            print(f"unknown task: {task}")
            pass


# TODO: LRU cache
cache = {}


async def read_image(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    size: image.Size,
) -> image.Image:
    h = await read(reader)
    h = int.from_bytes(h, "big", signed=True)

    print(f"[kiroshi] Reading image: {h}")
    result = cache.get(h)

    if result is None:
        print("[kiroshi] Image is not cached")
        await send(writer, bytes([0]))

        rgba = await read(reader)

        start = time.time()
        rgb = PIL.Image.frombytes("RGBA", (size.width, size.height), rgba).convert(
            "RGB"
        )
        print(f"Converted to RGB: {time.time() - start}")

        cache[h] = [rgb, None]
        return image.Image(rgb, h)
    else:
        print("[kiroshi] Image is cached")
        await send(writer, bytes([1]))

        return image.Image(result[0], h)


async def run(f, writer, message, *args):
    global cache

    should_cache = message.get("cache", True)

    h = hash((f, tuple(args)))
    result = cache.get(h)

    if result is None:
        loop = asyncio.get_running_loop()
        preview_after = message.get("preview_after", 1.0)

        def on_progress(ratio, preview):
            if writer.is_closing():
                raise Interrupt()

            if ratio <= preview_after:
                return

            preview = preview.filter(PIL.ImageFilter.GaussianBlur)
            preview.putalpha(255)

            async def send_progress():
                await send_json(
                    writer,
                    {
                        "id": h,
                        "width": preview.width,
                        "height": preview.height,
                        "progress": ratio,
                        "is_final": False,
                    },
                )

                await send(writer, preview.tobytes())

            asyncio.run_coroutine_threadsafe(send_progress(), loop)

        start = time.time()
        result = await asyncio.to_thread(lambda: f(*args, on_progress=on_progress))
        print(f"Generated: {time.time() - start}s")

        if not isinstance(result, tuple):
            result = (result, {})

        image, metadata = result
        print(f"Metadata: {metadata}")

        result = [image, metadata]

        if should_cache:
            cache[h] = result

    output, metadata = result

    if isinstance(output, PIL.Image.Image):
        image = output

        start = time.time()
        image = image.copy()
        image.putalpha(255)
        print(f"Added alpha layer: {time.time() - start}s")

        start = time.time()
        await send_json(
            writer,
            {
                "id": h,
                "width": image.width,
                "height": image.height,
                "progress": 1.0,
                "is_final": True,
                **metadata,
            },
        )
        await send(writer, image.tobytes())
        print(f"Sent: {time.time() - start}s")

    elif isinstance(output, Video):
        video = output

        await send_json(
            writer,
            {
                "id": h,
                "width": video.width,
                "height": video.height,
                "framerate": video.framerate,
                "frames": [frame.hash for frame in video.frames],
                **metadata,
            },
        )

        start = time.time()

        for frame in video.frames:
            if should_cache:
                cache[frame.hash] = [frame.raw, {}]

            image = frame.raw
            image = image.copy()
            image.putalpha(255)

            await send(writer, image.tobytes())

        print(f"Sent: {time.time() - start}s")

    writer.close()
    await writer.wait_closed()

    gc.collect()
    torch.cuda.empty_cache()


async def read(reader: asyncio.StreamReader) -> bytes:
    n = await reader.readexactly(8)
    n = int.from_bytes(n, "big", signed=False)

    return await reader.readexactly(n)


async def send_json(writer: asyncio.StreamWriter, data={}):
    data = json.dumps(data).encode("utf-8")
    size = len(data)

    writer.write(int.to_bytes(size, 8, "big", signed=False))
    writer.write(data)
    await writer.drain()


async def send(writer: asyncio.StreamWriter, data):
    size = len(data)

    writer.write(int.to_bytes(size, 8, "big", signed=False))
    writer.write(data)
    await writer.drain()


class Interrupt(Exception):
    pass


if __name__ == "__main__":

    def terminate(signum, frame):
        print("[kiroshi] Exiting...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)

    try:
        asyncio.run(server())
    except KeyboardInterrupt:
        for process in multiprocessing.active_children():
            process.kill()
