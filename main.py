import argparse
import os
import warnings
from typing import Union

import numpy as np
from PIL import Image, ImageEnhance

from functions import (
    open_img_to_array, sub_bw_colors, auto_select_color, rgb_to_grey,
    normalize, fs_dither, expand_pixels, instagram, find_prominent_colors,
)

def black_white_flow(
        img_array: np.ndarray,
        num_colors: int,
) -> np.ndarray:
    
    img_array = rgb_to_grey(img_array)
    img_array = normalize(img_array)
    img_array = fs_dither(img_array, num_colors)

    return img_array
    
def custom_black_white_flow(
        img_array: np.ndarray,
        num_colors: int,
        auto_color: bool,
        custom_black: Union[tuple, None] = None,
        custom_white: Union[tuple, None] = None,
) -> np.ndarray:

    # auto color overrides custom black and white
    if auto_color:
        colors, counts = find_prominent_colors(img_array)
        dark = auto_select_color(colors, counts, img_array.size)
        if dark == None:
            dark = (0, 0, 0)
            msg = "No suitable custom color found"
            warnings.warn(msg)
        light = (255, 255, 255)
        img_array = black_white_flow(img_array, num_colors)
        img_array = sub_bw_colors(img_array, dark, light)
    else:
        if custom_black is None:
            custom_black = (0, 0, 0)
        if custom_white is None:
            custom_white = (255, 255, 255)
        img_array = sub_bw_colors(img_array, custom_black, custom_white)
        
    return img_array

def flow(
        img_array: np.ndarray,
        black_white: bool,
        auto_color: bool,
        custom_black: Union[tuple, None],
        custom_white: Union[tuple, None],
        nc: int,
        ec: int,
        insta: bool,
        save_path: str
):
    # if custom bw
    if black_white and \
    (auto_color or custom_black is not None or custom_white is not None):
        img_array = custom_black_white_flow(
            img_array,
            num_colors=nc,
            auto_color=auto_color,
            custom_black=custom_black,
            custom_white=custom_white
        )
    
    elif black_white:
        img_array = black_white_flow(
            img_array,
            num_colors=nc
        )
    
    else:  # RGB
        img_array = fs_dither(img_array, nc)

    img_array = expand_pixels(img_array, ec)
    img = Image.fromarray(img_array, "RGB")

    if insta:
        img = instagram(img, ec)

    img.save(save_path)
    

parser = argparse.ArgumentParser(
    description="Quantizes image with Floyd-Steinberg dithering"
)
parser.add_argument("SRC", 
                    help="source folder or file")
parser.add_argument("DEST", 
                    help="destination folder")
parser.add_argument("-wi", "--width", type=int, default=300,
                    help="dithered image width")
parser.add_argument("-he", "--height", type=int,
                    help=
                    "dithered image height, if not specified derived from original aspect ratio and width")
parser.add_argument("-nc", "--num-colors", type=int, default=8,
                    help="number of colors per channel")
parser.add_argument("-ec", "--expand-coeff", type=int, default=2,
                    help="expansion coefficient, final size = (ec*w, ec*h)")
parser.add_argument("-bw", "--black-white", action="store_true",
                    help="converts image to black & white if stated")
parser.add_argument("-ac", "--auto-color", action="store_true",
                    help="extracts colors and replaces black in BW image with most common saturated image")
parser.add_argument("-cb", "--custom-black", nargs="+",
                    help=
                    "specify custom black for BW image (format: R G B) or 'auto' automatically selects prominent dark colour\n")
parser.add_argument("-cw", "--custom-white", nargs="+",
                    help=
                    "specify custom white for BW image (format: R G B) or 'auto' automatically selects prominent light colour\n")
parser.add_argument("-i", "--instagram", action="store_true",
                    help="adds a 1080x1080 white background")
parser.add_argument("-pn", "--preserve-names", action="store_true",
                    help="preserve original file names")

if __name__ == "__main__":
    args = parser.parse_args()

    # paths
    src = args.SRC
    dest = args.DEST

    if not (os.path.isdir(src) or os.path.isfile(src)):
        msg = "Specified source directory/file path is invalid."
        raise ValueError(msg)

    if not os.path.isdir(dest):
        msg = "Specified destination directory path is invalid."
        raise ValueError(msg)
    
    # size
    if args.height is None:
        if args.width <= 0:
            msg = f"Specified width invalid: {args.width} <= 0. Width must be postive."
            raise ValueError(msg)
        
        size = args.width

    else:
        if args.width <= 0:
            msg = f"Specified height invalid: {args.height} <= 0. Width must be postive."
            raise ValueError(msg)
        
        size = (args.width, args.height)
    
    # num colors and expansion coeff
    nc = args.num_colors
    if nc <= 1:
        msg = f"Specified number of colors invalid: {nc} <= 1. Image must have 2+ colors"
        raise ValueError(msg)
    
    ec = args.expand_coeff
    if ec < 1:
        msg = f"Specified number of colors invalid: {ec} <= 1. Expansion coeff must be >= 1"
        raise ValueError(msg)

    # custom black and white
    custom_bw = {
        "custom_black": args.custom_black,
        "custom_white": args.custom_white
    }

    for key in custom_bw.keys():
        if custom_bw[key] is not None:
            num_values = len(custom_bw[key])
            if num_values != 3:
                msg = f"{num_values} value supplied for {key}, instead of 3 (R, G, B)."
                raise ValueError(msg)
            else:
                custom_bw[key] = tuple([int(c) for c in custom_bw[key]])
                for c in custom_bw[key]:
                    if c < 0 or c > 255:
                        msg = f"Invalid RGB value for {key}: {c} not in [0, 255]"
    
    custom_black = custom_bw["custom_black"]
    custom_white = custom_bw["custom_white"]

    # source is directory (multiple images)
    if os.path.isdir(src):
        for dir, sub_dir, files in os.walk(src):
            for idx, file in enumerate(files):

                img_path = os.path.join(dir, file)
                print("processing", file)
                img_array = open_img_to_array(img_path, size)

                if args.preserve_names:
                    filename = os.path.splitext(file)[0] + ".png"
                    new_img_path = os.path.join(dest, filename)
                else:
                    new_img_path = os.path.join(dest, str(idx) + ".png")

                flow(
                    img_array,
                    args.black_white,
                    args.auto_color,
                    custom_black=custom_black,
                    custom_white=custom_white,
                    nc=nc,
                    ec=ec,
                    insta=args.instagram,
                    save_path=new_img_path
                )

    # source is single file
    elif os.path.isfile(src):
        img_array = open_img_to_array(src, size)
        
        filename = os.path.split(src)[-1]
        filename = os.path.splitext(filename)[0] + ".png"
        new_img_path = os.path.join(dest, filename)

        flow(
            img_array,
            args.black_white,
            args.auto_color,
            custom_black=custom_black,
            custom_white=custom_white,
            nc=nc,
            ec=ec,
            insta=args.instagram,
            save_path=new_img_path
        )