__author__ = 'MA573RWARR10R'
import skimage.draw
import random
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw, ImageEnhance
import math
from settings import *
import scipy.misc


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


def ScaleRotateTranslate(image, angle, center=None, new_center=None, scale=None, expand=False):
    if center is None:
        return image.rotate(angle)
    angle = -angle / 180.0 * math.pi
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = scale
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)


def make_captcha(current_number, maked_digits, maked_name, font_name, font_list, background_color,
                 exclude_colors, figures, draw_center, path, blur=2):
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
    fixed = 35

    if font_name is None:
        font_name = random.choice(font_list)

    for digit in maked_digits:
        if not draw_center:
            char_eth_width = int(req_w / float(count_of_chars))
            _font_size = random.choice(range(40, 70))

            font = ImageFont.truetype(font_name, _font_size)
            _color = random.choice(font_colors.values())
            _p_x = random.randint(-3, 5) + char_eth_width * index
            _p_y = random.choice(range(0, req_h - _font_size))

            imtext = Image.new('RGBA', (char_eth_width, _font_size))
            textdraw = ImageDraw.Draw(imtext)
            tx, ty = random.randint(-1, 1), random.randint(-1, 1)
            textdraw.text((tx, ty), str(digit), _color, font=font)
            angle = random.randrange(-10, 10)
            imtext = imtext.rotate(angle, expand=1)
            en = ImageEnhance.Brightness(imtext)
            mask = en.enhance(0.2)
            img.paste(imtext, (_p_x, _p_y), mask)

            # draw.text((_p_x, _p_y), str(digit), _color, font=font)

            index += 1
        else:
            char_eth_width = int(req_w / float(count_of_chars + 2))
            _font_size = random.choice(range(80, 100))

            font = ImageFont.truetype(font_name, _font_size)
            _color = 'black'
            _p_x = -random.randint(1, 5) + char_eth_width * index + fixed
            _p_y = random.choice(range(10, 15))

            imtext = Image.new('RGBA', (char_eth_width, _font_size))
            textdraw = ImageDraw.Draw(imtext)
            tx, ty = 0, 0
            textdraw.text((tx, ty), str(digit), _color, font=font)
            angle = random.randrange(-5, 5)
            imtext = imtext.rotate(angle, expand=1)
            en = ImageEnhance.Brightness(imtext)
            mask = en.enhance(0.5)
            img.paste(imtext, (_p_x, _p_y), mask)
            # draw.text((_p_x, _p_y), str(digit), _color, font=font)

            index += 1
            fixed = fixed - random.randint(5, 10)

    if figures:
        figures_count = random.randint(5, 10)
        base_figures = ['rectangle', 'ellipse']

        for i in range(figures_count):
            chosen_figure = random.choice(base_figures)
            if chosen_figure == 'ellipse':
                img_new = draw_inverting_ellipse(req_w, req_h, img)

            elif chosen_figure == 'rectangle':
                img_new = draw_inverting_rect(req_w, req_h, img)

    else:
        img_new = img

    scipy.misc.imsave(path + str(current_number) + "_" + maked_name + ".png", img_new)

    # img_new.save(path + str(current_number) + "_" + maked_name + ".png")


def generate_digits_for_captcha(captcha_len):
    maked = []
    maked_name = ''
    for i in range(0, captcha_len):
        generated_value = random.randint(0, 9)
        maked.append(generated_value)
        maked_name += str(generated_value)

    return maked, maked_name


def gen(count_first, count_second, captcha_length, p="..\\..\\train_cap\\"):
    base_simple_fonts = ['arial.ttf', 'linsans.ttf']
    base_dotted_fonts = ['dotta1.ttf', 'dotta.ttf']
    # for i in range(0, count_first):
    #     m, mn = generate_digits_for_captcha(captcha_length)
    #
    #     make_captcha(i, m, mn, font_name=None, font_list=base_simple_fonts, background_color='black',
    #                  exclude_colors=None,
    #                  figures=True, draw_center=False, path=p)

    for i in range(count_first, count_first + count_second):
        m, mn = generate_digits_for_captcha(captcha_length)
        make_captcha(i, m, mn, font_name=None, font_list=base_dotted_fonts, background_color='white',
                     exclude_colors=['light_gray', 'dark_gray'], figures=False, draw_center=True, path=p)


# def gen(count, captcha_length, p):
#     base_dotted_fonts = ['dotta.ttf', 'dotta3.ttf', 'dotta4.ttf', 'dotta1.ttf']
#     base_simple_fonts = ['arial.ttf', 'linsans.ttf', 'Sanford.ttf']
#
#     for i in range(0, count):
#         captype = random.choice([1, 2])
#         print 'generated {0}'.format(i)
#
#         if captype == 1:
#             m, mn = generate_digits_for_captcha(captcha_length)
#
#             make_captcha(i, m, mn, font_name=None, font_list=base_simple_fonts, background_color='black',
#                          exclude_colors=None,
#                          figures=True, draw_center=False, path=p)
#         else:
#             m, mn = generate_digits_for_captcha(captcha_length)
#
#             make_captcha(i, m, mn, font_name=None, font_list=base_dotted_fonts, background_color='white',
#                          exclude_colors=['light_gray', 'dark_gray'], figures=False, draw_center=True, path=p)


if __name__ == "__main__":
    gen(t_captcha_count, t_captcha_count, 6)
    # gen(test_cap_count_need, 6, p=captcha_test_stored_dir)
