import text_to_image

import asyncio
import json
import time
import torch
import gc
import multiprocessing
import signal
from PIL import ImageFilter


async def server():
    server = await asyncio.start_server(instance, '0.0.0.0', 9149)
    print("[kiroshi] Server started at 0.0.0.0:9149")

    async with server:
        await server.serve_forever()


async def instance(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    size = await reader.readexactly(8)
    size = int.from_bytes(size, "big", signed=False)

    message = await reader.readexactly(size)
    message = json.loads(message)

    print(f"[kiroshi] Received: {message}")

    model = message['model']
    prompt = message['prompt']
    negative_prompt = message['negative_prompt']
    size = message['size']
    quality = message['quality']
    steps = message.get('steps')
    seed = message.get('seed')
    loras = message.get('loras') or []
    sampler = message.get('sampler') or 'euler_a'
    preview_after = message.get('preview_after')
    face_detail = message.get('face_detail')
    hand_detail = message.get('hand_detail')
    cpu_offload = message.get('cpu_offload') or False

    if preview_after is None:
        preview_after = 1.0

    if not face_detail is None:
        face_detail = text_to_image.Detail.from_dict(face_detail)

    if not hand_detail is None:
        hand_detail = text_to_image.Detail.from_dict(hand_detail)

    if loras:
        loras = [text_to_image.Lora.from_dict(lora) for lora in loras]

    match quality:
        case 'low':
            quality = text_to_image.Quality.LOW
        case 'normal':
            quality = text_to_image.Quality.NORMAL
        case 'high':
            quality = text_to_image.Quality.HIGH
        case 'ultra':
            quality = text_to_image.Quality.ULTRA
        case 'insane':
            quality = text_to_image.Quality.INSANE

    match sampler:
        case 'euler_a':
            sampler = text_to_image.Sampler.EULER_A

        case 'dpm++_sde_karras':
            sampler = text_to_image.Sampler.DPM_SDE_KARRAS

        case 'dpm++_2m_karras':
            sampler = text_to_image.Sampler.DPM_2M_KARRAS

        case 'dpm++_2m_sde_karras':
            sampler = text_to_image.Sampler.DPM_2M_SDE_KARRAS

    loop = asyncio.get_running_loop()

    class Interrupt(Exception):
        pass

    def on_progress(ratio, preview):
        if writer.is_closing():
            raise Interrupt()

        if ratio <= preview_after:
            return

        preview = preview.filter(ImageFilter.GaussianBlur)
        preview.putalpha(255)

        async def send_progress():
            await send_json(
                writer, {
                    'width': preview.width,
                    'height': preview.height,
                    'progress': ratio,
                    'is_final': False
                })

            await send(writer, preview.tobytes())

        asyncio.run_coroutine_threadsafe(send_progress(), loop)

    def generate():
        parameters = text_to_image.Parameters(model=model,
                                              prompt=prompt,
                                              width=size['width'],
                                              height=size['height'],
                                              quality=quality,
                                              steps=steps,
                                              seed=seed,
                                              negative_prompt=negative_prompt,
                                              loras=loras,
                                              sampler=sampler)

        return text_to_image.generate(parameters=parameters,
                                      face_detail=face_detail,
                                      hand_detail=hand_detail,
                                      on_progress=on_progress,
                                      cpu_offload=cpu_offload)

    start = time.time()
    generation = await asyncio.to_thread(generate)
    print(f"Generated: {time.time() - start}s")

    start = time.time()
    generation.image.putalpha(255)
    print(f"Added alpha layer: {time.time() - start}s")

    start = time.time()
    await send_json(
        writer, {
            'width': generation.image.width,
            'height': generation.image.height,
            'faces': generation.faces,
            'hands': generation.hands,
            'progress': 1.0,
            'is_final': True,
        })
    await send(writer, generation.image.tobytes())
    print(f"Sent: {time.time() - start}s")

    writer.close()
    await writer.wait_closed()

    gc.collect()
    torch.cuda.empty_cache()


async def send_json(writer: asyncio.StreamWriter, data={}):
    data = json.dumps(data).encode('utf-8')
    size = len(data)

    writer.write(int.to_bytes(size, 8, "big", signed=False))
    writer.write(data)
    await writer.drain()


async def send(writer: asyncio.StreamWriter, data):
    size = len(data)

    writer.write(int.to_bytes(size, 8, "big", signed=False))
    writer.write(data)
    await writer.drain()


if __name__ == '__main__':

    def terminate(signum, frame):
        print("[kiroshi] Exiting...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)

    try:
        asyncio.run(server())
    except KeyboardInterrupt:
        for process in multiprocessing.active_children():
            process.kill()
