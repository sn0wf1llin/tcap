__author__ = 'MA573RWARR10R'
import PIL
import skimage.draw
import random
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw, ImageOps
import numpy as np
from settings import *


def draw_inverting_ellipse(req_w, req_h, img):
    ellipse_center_pos_x = random.randint(0, req_w)
    ellipse_rv = random.choice(range(1, req_h))
    ellipse_center_pos_y = random.randint(0, req_h)
    ellipse_rh = random.choice(range(1, req_w / 2))
    rr, cc = skimage.draw.ellipse(ellipse_center_pos_x, ellipse_center_pos_y, ellipse_rh, ellipse_rv)

    for i in range(len(rr)):
        pixel_x, pixel_y = rr[i], cc[i]
        try:
            pixel = img.getpixel((pixel_x, pixel_y))
            npixel = (255 - pixel[0], 255 - pixel[1], 255 - pixel[2])
            img.putpixel((pixel_x, pixel_y), npixel)
        except TypeError:
            try:
                pixel = img.getpixel((int(pixel_x), int(pixel_y)))
                npixel = (255 - pixel[0], 255 - pixel[1], 255 - pixel[2])
                img.putpixel((pixel_x, pixel_y), npixel)
            except IndexError:
                pass

    return img


def draw_inverting_rect(req_w, req_h, img):
    rect_w = random.choice(range(req_w / 2, req_w))
    rect_h = random.randint(req_h / 2, req_h * 2)
    rect_top_left_coords = (random.choice(range(0, req_w)), random.randint(0, req_h))
    rect_bottom_right_coords = (rect_top_left_coords[0] + rect_w, rect_top_left_coords[1] + rect_h)

    rr, cc = skimage.draw.polygon(
        [rect_top_left_coords[1], rect_bottom_right_coords[1], rect_bottom_right_coords[1], rect_top_left_coords[1]],
        [rect_top_left_coords[0], rect_top_left_coords[0], rect_bottom_right_coords[0], rect_bottom_right_coords[0]])

    for i in range(len(rr)):
        pixel_x, pixel_y = rr[i], cc[i]
        try:
            pixel = img.getpixel((pixel_x, pixel_y))
            npixel = (255 - pixel[0], 255 - pixel[1], 255 - pixel[2])
            img.putpixel((pixel_x, pixel_y), npixel)
        except TypeError:
            try:
                pixel = img.getpixel((int(pixel_x), int(pixel_y)))
                npixel = (255 - pixel[0], 255 - pixel[1], 255 - pixel[2])
                img.putpixel((pixel_x, pixel_y), npixel)
            except IndexError:
                pass

    return img


def make_captcha(current_number, maked_digits, maked_name, font_name, background_color,
                 exclude_colors, figures, draw_center):
    count_of_chars = len(maked_digits)

    base_colors = {
        'light_gray': (192, 192, 192),
        'dark_gray': (63, 63, 63),
        'white': (255, 255, 255),
        'black': (0, 0, 0)
    }
    _background_color = base_colors[background_color]
    font_colors = base_colors.copy()
    font_colors.pop(background_color)

    if exclude_colors is not None:
        for color in exclude_colors:
            font_colors.pop(color)

    req_w, req_h = (200, 100)
    img = Image.new("RGBA", (req_w, req_h), _background_color)
    draw = ImageDraw.Draw(img)
    index = 0

    for digit in maked_digits:
        if not draw_center:
            char_eth_width = int(req_w / float(count_of_chars))
            _font_size = random.choice(range(40, 80))
            font = ImageFont.truetype(font_name, _font_size)
            _color = random.choice(font_colors.values())
            _p_x = random.randint(-3, 5) + char_eth_width * index
            _p_y = random.choice(range(0, req_h - _font_size))
            draw.text((_p_x, _p_y), str(digit), _color, font=font)
            index += 1
        else:
            char_eth_width = int(req_w / float(count_of_chars + 2))
            _font_size = random.choice(range(30, 50))
            font = ImageFont.truetype(font_name, _font_size)
            _color = 'black'
            _p_x = random.randint(-5, 1) + char_eth_width * index + 35
            _p_y = random.choice(range(20, 25))
            draw.text((_p_x, _p_y), str(digit), _color, font=font)
            index += 1

    if figures:
        figures_count = random.randint(2, 6)
        base_figures = ['rectangle', 'ellipse']

        for i in range(figures_count):
            chosen_figure = random.choice(base_figures)
            if chosen_figure == 'ellipse':
                img_new = draw_inverting_ellipse(req_w, req_h, img)

            elif chosen_figure == 'rectangle':
                img_new = draw_inverting_rect(req_w, req_h, img)

    else:
        img_new = img

    img_new.save(captcha_stored_dir + "\\" + str(current_number) + "_" + maked_name + ".png")


def generate_digits_for_captcha(captcha_len):
    maked = []
    maked_name = ''
    for i in range(0, captcha_len):
        generated_value = random.randint(0, 9)
        maked.append(generated_value)
        maked_name += str(generated_value)

    return maked, maked_name


def gen(count_first, count_second, captcha_length):
    for i in range(0, count_first):
        m, mn = generate_digits_for_captcha(captcha_length)
        make_captcha(i, m, mn, font_name="arial.ttf", background_color='black', exclude_colors=None, figures=True,
                     draw_center=False)

    for i in range(count_first, count_first + count_second):
        m, mn = generate_digits_for_captcha(captcha_length)
        make_captcha(i, m, mn, font_name="dotted.ttf", background_color='white',
                     exclude_colors=['light_gray', 'dark_gray'], figures=False, draw_center=True)


if __name__ == "__main__":
    gen(t_captcha_count, t_captcha_count, 6)
