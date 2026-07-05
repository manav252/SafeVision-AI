from __future__ import annotations

import base64
import io
from hashlib import md5

import numpy as np
from PIL import Image
from streamlit_drawable_canvas import CanvasResult
from streamlit_drawable_canvas import _component_func, _data_url_to_image, _resize_img


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def st_canvas(
    fill_color: str = "#eee",
    stroke_width: int = 20,
    stroke_color: str = "black",
    background_color: str = "",
    background_image: Image.Image | None = None,
    update_streamlit: bool = True,
    height: int = 400,
    width: int = 600,
    drawing_mode: str = "freedraw",
    initial_drawing: dict | None = None,
    display_toolbar: bool = True,
    point_display_radius: int = 3,
    key=None,
) -> CanvasResult:
    background_image_url = None
    if background_image:
        background_image = _resize_img(background_image, height, width).convert("RGB")
        background_image_url = _image_to_data_url(background_image)
        background_color = ""

    initial_drawing = {"version": "4.4.0"} if initial_drawing is None else dict(initial_drawing)
    initial_drawing["background"] = background_color

    component_key = key
    if background_image_url and key:
        component_key = f"{key}_{md5(background_image.tobytes()).hexdigest()[:10]}"

    component_value = _component_func(
        fillColor=fill_color,
        strokeWidth=stroke_width,
        strokeColor=stroke_color,
        backgroundColor=background_color,
        backgroundImageURL=background_image_url,
        realtimeUpdateStreamlit=update_streamlit and (drawing_mode != "polygon"),
        canvasHeight=height,
        canvasWidth=width,
        drawingMode=drawing_mode,
        initialDrawing=initial_drawing,
        displayToolbar=display_toolbar,
        displayRadius=point_display_radius,
        key=component_key,
        default=None,
    )
    if component_value is None:
        return CanvasResult()

    return CanvasResult(
        np.asarray(_data_url_to_image(component_value["data"])),
        component_value["raw"],
    )
