"""Frozen renderer extracted without behavioral changes."""
from PIL import Image, ImageDraw

def _fit_size(source_size, maximum_size):
    source_width, source_height = source_size
    max_width, max_height = maximum_size
    scale = min(max_width / source_width, max_height / source_height)
    return max(1, round(source_width * scale)), max(1, round(source_height * scale))

def render_donor_patch(crop, size):
    width, height = size
    patch = Image.new("RGB", size, (128, 128, 128))
    maximum = max(1, int(width * 0.90)), max(1, int(height * 0.90))
    resized = crop.resize(_fit_size(crop.size, maximum), Image.Resampling.LANCZOS)
    patch.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return patch

def _gray_slot(image, slot):
    ImageDraw.Draw(image).rectangle(
        (slot[0], slot[1], slot[2] - 1, slot[3] - 1), fill=(128, 128, 128)
    )

def _marked_canvas(original, canvas_size, target_box):
    canvas = Image.new("RGB", canvas_size, (128, 128, 128))
    canvas.paste(original, (0, 0))
    ImageDraw.Draw(canvas).rectangle(
        (target_box[0], target_box[1], target_box[2] - 1, target_box[3] - 1),
        outline=(220, 20, 60),
        width=4,
    )
    return canvas

def render_target_donor_pair(original, canvas_size, target_box, r, p, s, donor_crop, condition):
    """Render one T or N-near condition using an identical donor patch."""

    if condition not in {"T", "N-near"}:
        raise ValueError(f"unsupported pair condition: {condition}")
    image = _marked_canvas(original, canvas_size, target_box)
    for slot in (r, p, s):
        _gray_slot(image, slot)
    patch = render_donor_patch(donor_crop, (r[2] - r[0], r[3] - r[1]))
    destination = r if condition == "T" else p
    image.paste(patch, (destination[0], destination[1]))
    return image, patch

def render_baseline(original, canvas_size, target_box, r, p, s, condition):
    if condition not in {"O", "G"}:
        raise ValueError(f"unsupported baseline condition: {condition}")
    image = _marked_canvas(original, canvas_size, target_box)
    _gray_slot(image, p)
    _gray_slot(image, s)
    if condition == "G":
        _gray_slot(image, r)
    return image

def render_donor_check(canvas_size, r, patch):
    image = Image.new("RGB", canvas_size, (128, 128, 128))
    image.paste(patch, (r[0], r[1]))
    return image
