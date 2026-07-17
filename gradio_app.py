import logging
import os
import tempfile
import time
import zipfile

import gradio as gr
import numpy as np
import rembg
import torch
import xatlas
from PIL import Image
from functools import partial

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
from tsr.bake_texture import bake_texture

import argparse


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(8192)
model.to(device)

rembg_session = rembg.new_session()


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background, foreground_ratio):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image


def export_baked_zip(raw_mesh, scene_code, texture_resolution):
    # El bake consulta el color del modelo por posicion de vertice, por lo
    # que la malla debe conservar su orientacion original (sin rotar).
    bake_output = bake_texture(raw_mesh, model, scene_code, texture_resolution)

    tmp_dir = tempfile.mkdtemp()
    obj_path = os.path.join(tmp_dir, "mesh.obj")
    mtl_path = os.path.join(tmp_dir, "mesh.mtl")
    tex_path = os.path.join(tmp_dir, "texture.png")

    xatlas.export(
        obj_path,
        raw_mesh.vertices[bake_output["vmapping"]],
        bake_output["indices"],
        bake_output["uvs"],
        raw_mesh.vertex_normals[bake_output["vmapping"]],
    )
    Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(
        Image.FLIP_TOP_BOTTOM
    ).save(tex_path)

    # mtl + referencia en el obj para que Blender/visores carguen la textura solos
    with open(mtl_path, "w") as f:
        f.write("newmtl baked\nmap_Kd texture.png\n")
    with open(obj_path, "r") as f:
        obj_data = f.read()
    with open(obj_path, "w") as f:
        f.write("mtllib mesh.mtl\nusemtl baked\n" + obj_data)

    zip_path = os.path.join(tmp_dir, "mesh_textured.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in ("mesh.obj", "mesh.mtl", "texture.png"):
            zf.write(os.path.join(tmp_dir, name), name)
    return zip_path


def generate(image, mc_resolution, do_bake_texture, texture_resolution, formats=["obj", "glb"]):
    t0 = time.time()
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
    raw_mesh = mesh.copy()
    mesh = to_gradio_3d_orientation(mesh)
    rv = []
    for format in formats:
        mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
        mesh.export(mesh_path.name)
        rv.append(mesh_path.name)

    zip_path = None
    bake_msg = ""
    if do_bake_texture:
        try:
            zip_path = export_baked_zip(raw_mesh, scene_codes[0], int(texture_resolution))
            bake_msg = f" + textura UV {int(texture_resolution)}px"
        except Exception as e:
            bake_msg = f" — bake de textura falló: {e}"

    elapsed = time.time() - t0
    status = (
        f"Listo en {elapsed:.1f} s — {len(mesh.faces):,} caras, "
        f"{len(mesh.vertices):,} vértices (grilla {int(mc_resolution)}³){bake_msg}"
    )
    return rv[0], rv[1], zip_path, status


def run_example(image_pil):
    preprocessed = preprocess(image_pil, False, 0.9)
    mesh_name_obj, mesh_name_glb, _, _ = generate(preprocessed, 256, False, 2048, ["obj", "glb"])
    return preprocessed, mesh_name_obj, mesh_name_glb


with gr.Blocks(title="TripoSR") as interface:
    gr.Markdown(
        """
    # TripoSR — imagen a 3D, 100 % local
    [TripoSR](https://github.com/VAST-AI-Research/TripoSR) (Tripo AI × Stability AI) reconstruye un modelo 3D a partir de una sola imagen.

    **Consejos de calidad:**
    1. La imagen manda: objeto único, centrado, bien iluminado y sin sombras duras.
    2. Si el resultado no convence, prueba con otro **Foreground Ratio** (0.75–0.90).
    3. Sube la **Resolución Marching Cubes** para más detalle geométrico (más lento; sobre ~320 el retorno disminuye).
    4. Activa **Bake de textura UV** para obtener un OBJ con textura nítida (mucho mejor que el color por vértice). Descarga el ZIP y arrastra el `.obj` a Blender: la textura carga sola.
    5. Desactiva "Quitar fondo" solo si tu imagen ya es RGBA con fondo transparente y el objeto ocupa >70 % del encuadre.
    """
    )
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Imagen de entrada",
                    image_mode="RGBA",
                    sources="upload",
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(label="Imagen procesada", interactive=False)
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Quitar fondo",
                        value=True,
                        info="Recorta el objeto automáticamente y rellena el fondo con gris neutro. "
                        "Desactívalo solo si tu imagen ya es RGBA con fondo transparente.",
                    )
                    foreground_ratio = gr.Slider(
                        label="Foreground Ratio",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                        info="Cuánto del encuadre ocupa el objeto antes de inferir. No es 'más = mejor': "
                        "si el resultado no convence, prueba valores entre 0.75 y 0.90 — puede cambiar "
                        "mucho la reconstrucción de las zonas no visibles.",
                    )
                    mc_resolution = gr.Slider(
                        label="Resolución Marching Cubes",
                        minimum=32,
                        maximum=512,
                        value=256,
                        step=32,
                        info="Densidad de la grilla 3D de la que se extrae la malla: más alto = más "
                        "polígonos y detalle geométrico, pero más tiempo (crece al cubo). 256 para "
                        "iterar; 320–384 para la versión final. Sobre ~320 el retorno disminuye: el "
                        "detalle real lo limita el modelo, no la grilla.",
                    )
                    do_bake_texture = gr.Checkbox(
                        label="Bake de textura UV",
                        value=False,
                        info="Hornea la textura en un atlas UV en vez de color por vértice: resultado "
                        "mucho más nítido. Entrega un ZIP (OBJ + MTL + PNG); arrastra el .obj a "
                        "Blender y la textura carga sola. Agrega unos segundos por corrida.",
                    )
                    texture_resolution = gr.Slider(
                        label="Resolución de textura",
                        minimum=512,
                        maximum=4096,
                        value=2048,
                        step=512,
                        info="Tamaño en píxeles del atlas UV (solo aplica con bake activo). 2048 es el "
                        "equilibrio; 4096 solo para primeros planos — pesa 4× más y la nitidez extra "
                        "depende de la foto de entrada.",
                    )
            with gr.Row():
                submit = gr.Button("Generar", elem_id="generate", variant="primary")
            status_text = gr.Textbox(label="Estado", interactive=False)
        with gr.Column():
            with gr.Tab("OBJ"):
                output_model_obj = gr.Model3D(
                    label="Modelo (OBJ, color por vértice)",
                    interactive=False,
                )
                gr.Markdown("Nota: el visor lo muestra volteado; el archivo descargado es correcto.")
            with gr.Tab("GLB"):
                output_model_glb = gr.Model3D(
                    label="Modelo (GLB, color por vértice)",
                    interactive=False,
                )
                gr.Markdown("Nota: el visor lo muestra más oscuro; el archivo descargado es correcto.")
            output_baked_zip = gr.File(
                label="Malla texturizada (ZIP: OBJ + MTL + PNG) — solo con bake activo",
                interactive=False,
            )
    with gr.Row(variant="panel"):
        gr.Examples(
            examples=[
                "examples/hamburger.png",
                "examples/poly_fox.png",
                "examples/robot.png",
                "examples/teapot.png",
                "examples/tiger_girl.png",
                "examples/horse.png",
                "examples/flamingo.png",
                "examples/unicorn.png",
                "examples/chair.png",
                "examples/iso_house.png",
                "examples/marble.png",
                "examples/police_woman.png",
                "examples/captured.jpeg",
            ],
            inputs=[input_image],
            outputs=[processed_image, output_model_obj, output_model_glb],
            cache_examples=False,
            fn=partial(run_example),
            label="Ejemplos",
            examples_per_page=20,
        )
    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background, foreground_ratio],
        outputs=[processed_image],
    ).success(
        fn=generate,
        inputs=[processed_image, mc_resolution, do_bake_texture, texture_resolution],
        outputs=[output_model_obj, output_model_glb, output_baked_zip, status_text],
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str, default=None, help='Username for authentication')
    parser.add_argument('--password', type=str, default=None, help='Password for authentication')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server listener on')
    parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
    parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site")
    parser.add_argument("--queuesize", type=int, default=1, help="launch gradio queue max_size")
    args = parser.parse_args()
    interface.queue(max_size=args.queuesize)
    interface.launch(
        auth=(args.username, args.password) if (args.username and args.password) else None,
        share=args.share,
        server_name="0.0.0.0" if args.listen else None,
        server_port=args.port
    )
