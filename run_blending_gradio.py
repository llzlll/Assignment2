import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
import cv2


def initialize_polygon():
    """
    Initializes the polygon state.
    """
    return {"points": [], "closed": False}


def reset_polygon_and_image(img_original):
    """
    Reset polygon state whenever a new foreground image is uploaded.
    """
    if img_original is None:
        return None, initialize_polygon()
    return img_original, initialize_polygon()


def set_background_image(img):
    """
    Keep a clean copy of the background image in state and display it.
    """
    return img, img


def draw_polygon_on_image(img_original, polygon_state):
    """
    Render the current polygon state on top of the original image.
    """
    if img_original is None:
        return None

    img_with_poly = img_original.copy()
    draw = ImageDraw.Draw(img_with_poly)

    points = polygon_state["points"]

    if len(points) > 1:
        draw.line(points, fill="red", width=2)

    if polygon_state["closed"] and len(points) > 2:
        draw.line([points[-1], points[0]], fill="red", width=2)

    for x, y in points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="blue")

    return img_with_poly


def add_point(img_original, polygon_state, evt: gr.SelectData):
    """
    Add a point to the polygon based on user click event.
    """
    if img_original is None:
        return None, polygon_state

    if polygon_state["closed"]:
        return draw_polygon_on_image(img_original, polygon_state), polygon_state

    x, y = evt.index
    polygon_state["points"].append((int(x), int(y)))

    return draw_polygon_on_image(img_original, polygon_state), polygon_state


def close_polygon(img_original, polygon_state):
    """
    Close the polygon if there are at least three points.
    """
    if img_original is None:
        return None, polygon_state

    if (not polygon_state["closed"]) and len(polygon_state["points"]) > 2:
        polygon_state["closed"] = True

    return draw_polygon_on_image(img_original, polygon_state), polygon_state


def update_background(background_image_original, polygon_state, dx, dy):
    """
    Draw the shifted polygon on the background image for preview.
    """
    if background_image_original is None:
        return None

    if not polygon_state["closed"] or len(polygon_state["points"]) < 3:
        return background_image_original

    img_with_poly = background_image_original.copy()
    draw = ImageDraw.Draw(img_with_poly)
    shifted_points = [(x + int(dx), y + int(dy)) for x, y in polygon_state["points"]]
    draw.polygon(shifted_points, outline="red")

    return img_with_poly


def create_mask_from_points(points, img_h, img_w):
    """
    Create a binary mask from polygon points.

    0 indicates outside the polygon.
    255 indicates inside the polygon.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    points = np.asarray(points, dtype=np.int32)

    if len(points) >= 3:
        cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)

    return mask


def compute_clipped_regions(src_mask, bg_h, bg_w, dx, dy):
    """
    Compute aligned source and destination bounding boxes after translation,
    while clipping to valid image boundaries.
    """
    ys, xs = np.where(src_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    src_y0, src_y1 = ys.min(), ys.max() + 1
    src_x0, src_x1 = xs.min(), xs.max() + 1

    dst_y0 = src_y0 + int(dy)
    dst_y1 = src_y1 + int(dy)
    dst_x0 = src_x0 + int(dx)
    dst_x1 = src_x1 + int(dx)

    clip_dst_y0 = max(0, dst_y0)
    clip_dst_y1 = min(bg_h, dst_y1)
    clip_dst_x0 = max(0, dst_x0)
    clip_dst_x1 = min(bg_w, dst_x1)

    if clip_dst_y0 >= clip_dst_y1 or clip_dst_x0 >= clip_dst_x1:
        return None

    shift_y0 = clip_dst_y0 - dst_y0
    shift_y1 = shift_y0 + (clip_dst_y1 - clip_dst_y0)
    shift_x0 = clip_dst_x0 - dst_x0
    shift_x1 = shift_x0 + (clip_dst_x1 - clip_dst_x0)

    clip_src_y0 = src_y0 + shift_y0
    clip_src_y1 = src_y0 + shift_y1
    clip_src_x0 = src_x0 + shift_x0
    clip_src_x1 = src_x0 + shift_x1

    return {
        "src": (clip_src_y0, clip_src_y1, clip_src_x0, clip_src_x1),
        "dst": (clip_dst_y0, clip_dst_y1, clip_dst_x0, clip_dst_x1),
    }


def cal_laplacian_loss(source_patch, source_mask, blended_patch):
    """
    Compute Laplacian loss for Poisson image editing inside the masked region.
    """
    device = source_patch.device
    channels = source_patch.shape[1]

    laplacian_kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 3, 3)

    laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)

    src_lap = F.conv2d(source_patch, laplacian_kernel, padding=1, groups=channels)
    blend_lap = F.conv2d(blended_patch, laplacian_kernel, padding=1, groups=channels)

    mask_3ch = source_mask.expand(-1, channels, -1, -1)
    denom = mask_3ch.sum().clamp_min(1.0)

    return (((blend_lap - src_lap) ** 2) * mask_3ch).sum() / denom


def poisson_blend_patch(source_patch, source_mask, target_patch, steps=1200, lr=1e-2):
    """
    Optimize a target patch so that its Laplacian inside the mask matches the source patch,
    while pixels outside the mask remain equal to the target background.
    """
    mask_3ch = source_mask.expand(-1, source_patch.shape[1], -1, -1)
    blended = target_patch.clone().detach()
    blended = blended.requires_grad_(True)

    optimizer = torch.optim.Adam([blended], lr=lr)

    for step in range(steps):
        # Enforce background values outside the mask during optimization
        blended_for_loss = blended * mask_3ch + target_patch * (1.0 - mask_3ch)

        loss = cal_laplacian_loss(source_patch, source_mask, blended_for_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            blended.data = blended.data * mask_3ch + target_patch * (1.0 - mask_3ch)
            blended.data.clamp_(0.0, 1.0)

        if step == int(steps * 2 / 3):
            optimizer.param_groups[0]["lr"] *= 0.1

    with torch.no_grad():
        blended = blended * mask_3ch + target_patch * (1.0 - mask_3ch)
        blended = blended.clamp(0.0, 1.0)

    return blended


def blending(foreground_image_original, background_image_original, dx, dy, polygon_state):
    """
    Blend the selected foreground polygon region onto the background using Poisson blending.
    """
    if not polygon_state["closed"] or background_image_original is None or foreground_image_original is None:
        return background_image_original

    foreground_np = np.array(foreground_image_original.convert("RGB"))
    background_np = np.array(background_image_original.convert("RGB"))

    foreground_polygon_points = np.array(polygon_state["points"], dtype=np.int32)
    if len(foreground_polygon_points) < 3:
        return background_image_original

    foreground_mask = create_mask_from_points(
        foreground_polygon_points,
        foreground_np.shape[0],
        foreground_np.shape[1],
    )

    regions = compute_clipped_regions(
        foreground_mask,
        background_np.shape[0],
        background_np.shape[1],
        dx,
        dy,
    )

    if regions is None:
        return background_image_original

    src_y0, src_y1, src_x0, src_x1 = regions["src"]
    dst_y0, dst_y1, dst_x0, dst_x1 = regions["dst"]

    source_patch_np = foreground_np[src_y0:src_y1, src_x0:src_x1]
    source_mask_np = foreground_mask[src_y0:src_y1, src_x0:src_x1]
    target_patch_np = background_np[dst_y0:dst_y1, dst_x0:dst_x1].copy()

    if source_patch_np.size == 0 or target_patch_np.size == 0:
        return background_image_original

    device = "cuda" if torch.cuda.is_available() else "cpu"

    source_patch = (
        torch.from_numpy(source_patch_np)
        .to(device)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        / 255.0
    )
    target_patch = (
        torch.from_numpy(target_patch_np)
        .to(device)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        / 255.0
    )
    source_mask = (
        torch.from_numpy((source_mask_np > 0).astype(np.float32))
        .to(device)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    blended_patch = poisson_blend_patch(
        source_patch=source_patch,
        source_mask=source_mask,
        target_patch=target_patch,
        steps=1200,
        lr=1e-2,
    )

    blended_patch_np = (
        blended_patch.detach()
        .cpu()
        .squeeze(0)
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    ).astype(np.uint8)

    result = background_np.copy()
    mask_bool = source_mask_np > 0
    result_region = result[dst_y0:dst_y1, dst_x0:dst_x1]
    result_region[mask_bool] = blended_patch_np[mask_bool]
    result[dst_y0:dst_y1, dst_x0:dst_x1] = result_region

    return Image.fromarray(result)


def close_polygon_and_reset_offsets(img_original, polygon_state, dx, dy, background_image_original):
    """
    Close the polygon, reset both offsets to zero, and refresh the preview.
    """
    img_with_poly, updated_polygon_state = close_polygon(img_original, polygon_state)
    updated_background = update_background(background_image_original, updated_polygon_state, 0, 0)

    return (
        img_with_poly,
        updated_polygon_state,
        updated_background,
        gr.update(value=0),
        gr.update(value=0),
    )


with gr.Blocks(
    title="Poisson Image Blending",
    css="""
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .gr-button {
        font-size: 1em;
        padding: 0.75em 1.5em;
        border-radius: 8px;
        background-color: #6200ee;
        color: #ffffff;
        border: none;
    }
    .gr-button:hover {
        background-color: #3700b3;
    }
    .gr-slider input[type=range] {
        accent-color: #03dac6;
    }
    .gr-text, .gr-markdown {
        font-size: 1.1em;
    }
    .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
        color: #bb86fc;
    }
    .gr-input, .gr-output {
        background-color: #2c2c2c;
        border: 1px solid #3c3c3c;
    }
"""
) as demo:
    polygon_state = gr.State(initialize_polygon())
    background_image_original = gr.State(value=None)

    gr.Markdown("<h1 style='text-align: center;'>Poisson Image Blending</h1>")
    gr.Markdown(
        "<p style='text-align: center; font-size: 1.2em;'>Blend a selected area from a foreground image onto a background image with adjustable positions.</p>"
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Foreground Image")
            foreground_image_original = gr.Image(label="", type="pil", interactive=True, height=300)
            gr.Markdown(
                "<p style='font-size: 0.9em;'>Upload the foreground image where the polygon will be selected.</p>"
            )

            gr.Markdown("### Foreground Image with Polygon")
            foreground_image_with_polygon = gr.Image(label="", type="pil", interactive=True, height=300)
            gr.Markdown(
                "<p style='font-size: 0.9em;'>Click on the image to define the polygon area. After selecting at least three points, click <strong>Close Polygon</strong>.</p>"
            )
            close_polygon_button = gr.Button("Close Polygon")

        with gr.Column():
            gr.Markdown("### Background Image")
            background_image = gr.Image(label="", type="pil", interactive=True, height=300)
            gr.Markdown(
                "<p style='font-size: 0.9em;'>Upload the background image where the polygon will be placed.</p>"
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Background Image with Polygon Overlay")
            background_image_with_polygon = gr.Image(label="", type="pil", height=500)
            gr.Markdown(
                "<p style='font-size: 0.9em;'>Adjust the position of the polygon using the sliders below.</p>"
            )

        with gr.Column():
            gr.Markdown("### Blended Image")
            output_image = gr.Image(label="", type="pil", height=500)

    with gr.Row():
        with gr.Column():
            dx = gr.Slider(label="Horizontal Offset", minimum=-500, maximum=500, step=1, value=0)
        with gr.Column():
            dy = gr.Slider(label="Vertical Offset", minimum=-500, maximum=500, step=1, value=0)
        blend_button = gr.Button("Blend Images")

    foreground_image_original.change(
        fn=reset_polygon_and_image,
        inputs=foreground_image_original,
        outputs=[foreground_image_with_polygon, polygon_state],
    )

    foreground_image_with_polygon.select(
        add_point,
        inputs=[foreground_image_original, polygon_state],
        outputs=[foreground_image_with_polygon, polygon_state],
    )

    close_polygon_button.click(
        fn=close_polygon_and_reset_offsets,
        inputs=[foreground_image_original, polygon_state, dx, dy, background_image_original],
        outputs=[foreground_image_with_polygon, polygon_state, background_image_with_polygon, dx, dy],
    )

    background_image.change(
        fn=set_background_image,
        inputs=background_image,
        outputs=[background_image_original, background_image_with_polygon],
    )

    dx.change(
        fn=update_background,
        inputs=[background_image_original, polygon_state, dx, dy],
        outputs=background_image_with_polygon,
    )

    dy.change(
        fn=update_background,
        inputs=[background_image_original, polygon_state, dx, dy],
        outputs=background_image_with_polygon,
    )

    blend_button.click(
        fn=blending,
        inputs=[foreground_image_original, background_image_original, dx, dy, polygon_state],
        outputs=output_image,
    )

demo.launch()
