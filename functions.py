from statistics import multimode
from typing import Union, Tuple
from enum import Enum

import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import KMeans


def open_img_to_array(
        file_path,
        new_size: Union[tuple, int, None] = None
        ) -> np.ndarray:
    
    # img_ = Image.open(file_path).convert(color_mode).convert("RGB")  # ensure 3 color channels
    img_ = Image.open(file_path).convert("RGB")
    img = ImageOps.exif_transpose(img_)
    img_.close()

    # if new_size is not None:
    #     new_ratio = new_size[0] / new_size[1]
    #     img_ratio = img.size[0] / img.size[1]
    #     if new_ratio < 1 and img_ratio > 1:
    #         new_size = (new_size[1], new_size[0])
    #     elif new_ratio > 1 and img_ratio < 1:
    #         new_size = (new_size[1], new_size[0])
    #     img = img.resize(new_size)
    
    if type(new_size) is int:  # if single dimension supplied treat as width
        old_w, old_h = img.size
        ratio = old_w / old_h
        new_w = int(new_size)
        new_h = int(new_w / ratio)
        img = img.resize((new_w, new_h))

    elif type(new_size) is tuple:
        assert len(new_size) == 2, \
            "incorrect number of arguments for size (width only or (width, height))"
        img = img.resize(new_size)  # type: ignore

    img_array = np.array(img)
    img.close()
    return img_array.astype("uint8")

def rgb_to_linrgb(img_array: np.ndarray) -> np.ndarray:
    """
    converts from encoded RGB to linear RGB values
    """

    img_array = img_array.astype(float)
    return (img_array / 255.0)**2.2

def rgb_to_sat(img_array: np.ndarray) -> np.ndarray:
    """
    derives saturation value (of HSV format) from encoded RGB
    returns value between [0, 1]
    """

    rgb_lin = rgb_to_linrgb(img_array)
    if len(rgb_lin.shape) == 3 and rgb_lin.shape[2] == 3:
        r = rgb_lin[:, :, 0]
        g = rgb_lin[:, :, 1]
        b = rgb_lin[:, :, 2]

    # elif len(rgb_lin.shape) == 1:
    else:
        r = rgb_lin[0]
        g = rgb_lin[1]
        b = rgb_lin[2]

    max_ = np.maximum(np.maximum(r, g), b)
    min_ = np.minimum(np.minimum(r, g), b)
    sat = (max_ - min_) / min_
    return sat

def rgb_to_grey(img_array: np.ndarray) -> np.ndarray:
    """
    converts image array or single pixel array to greyscale using perceived lightness
    follows: https://stackoverflow.com/users/10315269/myndex
    """

    if len(img_array.shape) == 1 and img_array.shape[0] == 3:
        single_color = True
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
        single_color = False
    else:
        msg = f"Supplied image with shape {img_array.shape}, should be (w, h, c) or (c,)"
        raise ValueError(msg)

    # linearise gamma-encoded sRGB values
    rgb_lin = rgb_to_linrgb(img_array)

    # find luminance
    coeffs = (0.2126, 0.7152, 0.0722)  # R, G, B
    for channel, coeff in enumerate(coeffs):
        if single_color:
            rgb_lin[channel] = coeff * rgb_lin[channel]
        else:
            rgb_lin[:, :, channel] = coeff * rgb_lin[:, :, channel]
    
    if single_color:
        lum = np.sum(rgb_lin)
    else:
        lum = np.sum(rgb_lin, axis=2)

    # find perceived lightness
    perc_lightness = np.clip(lum**0.4545 * 255.0, 0.0, 255.0)

    if single_color:
        grey = perc_lightness
    else:
        grey = np.stack([perc_lightness] * 3, axis=2)

    return grey.astype("uint8")

def normalize(img_array: np.ndarray) -> np.ndarray:
    """normalize all channels of an image"""
    max_ = np.max(img_array, axis=(0, 1))
    min_ = np.min(img_array, axis=(0, 1))
    normalized = np.round((img_array - min_) / (max_ - min_) * 255.0)

    return normalized.astype("uint8")

def find_closest_palette_color(oldpixel: np.ndarray, num_colors: int = 2):
    """quantizes a pixel value to the closest candidate via rounding"""
    newpixel = oldpixel / 255  # norm to [0, 1]
    newpixel = np.round(newpixel * (num_colors - 1)) / (num_colors - 1)  # quantize
    
    return np.round(newpixel * 255)

def fs_dither(img_array: np.ndarray, num_colors: int = 2):
    height = img_array.shape[0]
    width = img_array.shape[1]

    img_array = img_array.astype("float")

    for y in range(height - 1):
        for x in range(width - 1):
            oldpixel = img_array[y, x].copy()
            newpixel = find_closest_palette_color(oldpixel, num_colors)
            img_array[y, x] = newpixel
            quant_error = oldpixel - newpixel
            # spread quantization error
            img_array[y    , x + 1] = img_array[y    , x + 1] + quant_error * 7/16
            img_array[y + 1, x - 1] = img_array[y + 1, x - 1] + quant_error * 3/16
            img_array[y + 1, x    ] = img_array[y + 1, x    ] + quant_error * 5/16
            img_array[y + 1, x + 1] = img_array[y + 1, x + 1] + quant_error * 1/16
    # quantize last row and column
    # TODO: improve final quantization
    img_array[-1, :] = find_closest_palette_color(img_array[1, :])
    img_array[:, -1] = find_closest_palette_color(img_array[:, 1])

    return img_array.astype("uint8")

def expand_pixels(img_array: np.ndarray, expansion_coeff: int):
    """expands single pixels to squares of multiple pixels"""
    if expansion_coeff == 1:
        
        return img_array
    
    else:
        ec = expansion_coeff
        old_shape = img_array.shape
        old = img_array.astype(float)
        new_shape = (ec*old_shape[0], ec*old_shape[1], old_shape[2])
        new = np.zeros(new_shape)

        for y in range(old_shape[1]):
            y_new = ec * y
            for x in range(old_shape[0]):
                color = old[x, y]
                x_new = ec * x
                new[x_new:x_new + ec, y_new:y_new + ec] = color
        
        return new.astype("uint8")

def sub_bw_colors(
        img_array: np.ndarray,
        dark: tuple[int, ...],  # Union[tuple, None]
        light: tuple[int, ...]  # Union[tuple, None]
):
    """substitute greyscale value with custom b&w palette"""
    # if dark is None:
    #     dark = (0, 0, 0)
    # if light is None:
    #     light = (255, 255, 255)

    for c_channel in range(3):  # RGB channels
        cl = light[c_channel]
        cd = dark[c_channel]
        scale = (cl - cd) / 255.0
        img_channel = img_array[:, :, c_channel]
        img_channel = img_channel * scale + cd
        img_channel = np.round(img_channel)
        img_array[:, :, c_channel] = img_channel
    
    return img_array.astype("uint8")

def find_prominent_colors(
        img_array: np.ndarray,
        n_colors: int = 10
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    """finds prominent colors in image and returns list of most-to-least common"""
    
    flat_img_array = flatten_img(img_array).astype(float)
    n_pixels = flat_img_array.shape[0]
    if n_pixels >= 2000:
        random_idxs = np.random.choice(n_pixels, size=2000, replace=False)
    elif n_pixels >= 1000:
        random_idxs = np.random.choice(n_pixels, size=1000, replace=False)
    elif n_pixels >= 500:
        random_idxs = np.random.choice(n_pixels, size=500, replace=False)
    else:
        random_idxs = np.arange(0, n_pixels + 1, 1)

    x = flat_img_array[random_idxs]
    kmeans = KMeans(n_clusters=n_colors, n_init="auto").fit(x)
    colors = kmeans.cluster_centers_.astype("uint8")
    labels = kmeans.predict(x)
    # find most common color (cluster with most elements)
    color_idx, counts = np.unique(labels, return_counts=True)  # get colors and counts
    color_idx = tuple(color_idx.tolist())
    counts = tuple(counts.tolist())
    sorted_colors = sorted(
        zip(color_idx, counts), key=lambda item: item[1], reverse=True
        )  # sort by most counted
    colors = tuple([colors[color_idx] for color_idx, _ in sorted_colors])
    counts = tuple([int(counts) for _, counts in sorted_colors])

    return colors, counts

# def select_color_PL(
#         colors: tuple[np.ndarray, ...],
#         upper: int,
#         lower: int,
# ) -> Union[tuple[int, ...], None]:
#     """
#     selects first color that is within upper and lower bounds (inclusive) of percieved lightness
#     if none found returns None
#     """
#     for color in colors:
#         perc_lightness = rgb_to_grey(color)
#         if perc_lightness <= upper and perc_lightness >= lower:
#             return tuple(color.tolist())
#     else:
#         return None
    
# def auto_select_dark(
#         colors: tuple[np.ndarray, ...],
#         threshold: int = 127,
# ) -> Union[tuple[int, ...], None]:
#     """
#     finds dark color based on threshold
#     uses perceived lightness
#     """

#     return select_color_PL(colors, upper=threshold, lower=0)

# def auto_select_light(
#         colors: tuple[np.ndarray, ...],
#         threshold: int = 205
# ) -> Union[tuple[int, ...], None]:
#     """
#     finds light color based on threshold
#     uses perceived lightness
#     """
    
#     return select_color_PL(colors, upper=255, lower=threshold)

def auto_select_color(
        colors: tuple[np.ndarray, ...],
        counts: tuple[int, ...],
        total_counts: int,
) -> Union[tuple[int, ...], None]:
    
    score = 0
    best = None
    for color, count in zip(colors, counts):
        lightness = rgb_to_grey(color)
        if lightness < 0 or lightness > 180:
            continue
        else:
            sat = rgb_to_sat(color)
            popularity = count / total_counts
            s = 4*sat + popularity
            if s > score:
                best = color
                score = s
    
    if best is None:
        return best
    else:
        color = tuple(best.astype(int).tolist())
        return color
    
def flatten_img(img_array: np.ndarray):
    assert len(img_array.shape) == 3 and img_array.shape[-1] == 3, \
    "image array must have 3 colour channels"
    return img_array.reshape(-1, img_array.shape[-1])

def instagram(img: Image.Image, expand_coeff: int):
    img.convert("RGB")
    img = img.crop((0, 0, (img.width - expand_coeff), (img.height - expand_coeff)))
    white_bg = Image.new("RGB", (1080, 1080), color=(255,255,255))
    x = int((540 - np.ceil(img.size[0] / 2)))
    y = int((540 - np.ceil(img.size[1] / 2)))
    white_bg.paste(img, (x, y))
    return white_bg


if __name__ == "__main__":
    import os
    for dir, sub_dir, files in os.walk("original_images/"):
        for file in files:
            file_path = dir + file
            img = open_img_to_array(file_path, "RGB", (300,400))
            # flat = flatten_img(img)
            # step = int(flat.shape[0] / 100)
            # partial_pixels = flat[0::step]
            # print()
            # print(file)
            # print(partial_pixels.shape)
            print(file_path)
            colors = find_prominent_colors(img, n_colors=4)
            print(colors)
            # print(find_prominent_colors(partial_pixels, n_colors=3))
            # print(type(find_prominent_colors(partial_pixels, n_colors=3)))