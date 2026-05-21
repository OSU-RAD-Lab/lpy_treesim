import noise
import numpy as np
from PIL import Image
import random
import cv2


def generate_noise(width, height, scale=100.0, octaves=6):
    """Generates simple procedural noise using Gaussian blurring of random values."""
    noise = np.random.rand(height, width).astype(np.float32)
    for i in range(octaves):
        s = int(scale / (2 ** i))
        if s < 1: break
        noise += cv2.GaussianBlur(np.random.rand(height, width).astype(np.float32), (0, 0), s)
    return (noise - noise.min()) / (noise.max() - noise.min())


def create_apple_bark(width, height):
    # 1. Base Wood Color (Grayish-brown typical for apple trees)
    base_color = np.array([90, 110, 130])  # BGR for a cool dark brown
    img = np.full((height, width, 3), base_color, dtype=np.uint8)

    # 2. Vertical Ridges (Stretched Perlin-style noise)
    # Stretch noise vertically by scaling the width of the noise source
    vertical_noise = generate_noise(width, height // 4, scale=20.0)
    vertical_noise = cv2.resize(vertical_noise, (width, height))

    # 3. Small Horizontal Lenticels (Characteristic of fruit tree bark)
    lenticel_noise = np.random.choice([0, 1], size=(height, width), p=[0.995, 0.005]).astype(np.float32)
    lenticel_noise = cv2.GaussianBlur(lenticel_noise, (5, 1), 0)  # Stretch horizontally

    # 4. Combine Textures
    # Darken areas for ridges and lighten for highlights
    bark_map = (vertical_noise * 0.7 + lenticel_noise * 0.3)

    # Apply to color channels with slight variation
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c] * (bark_map + 0.5), 0, 255)

    return img


def generate_apple_bark(width=1024, height=1024, scale=100.0, overall=255) ->Image:
    # Initialize image array
    bark_texture = np.zeros((height, width), dtype=np.float32)
    bark_sat = np.zeros((height, width), dtype=np.float32)
    bark_hue = np.zeros((height, width), dtype=np.float32)

    # Generate octaves of noise for layered texture
    for ix in range(width):
        for iy in range(height):
            # Base vertical grain (ridges)
            n = noise.pnoise2(ix / (scale * 0.2),
                              iy / scale,  # Stretched vertically
                              octaves=6,
                              persistence=0.5,
                              lacunarity=2.0,
                              repeatx=width,
                              repeaty=height,
                              base=random.randint(0, 100))

            # Add some "flaky" patches
            n2 = noise.pnoise2(ix / (scale * 0.5),
                               iy / (scale * 0.5),
                               octaves=4,
                               base=random.randint(0, 100))

            # Combine noises
            bark_texture[iy, ix] = n + (n2 * 0.2)

            sat = noise.pnoise2(ix / (scale * 0.5),
                                iy / (scale * 0.5),
                                octaves=2,
                                base=random.randint(100, 200))

            bark_sat[iy, ix] = sat

            hue = noise.pnoise2(ix / (scale * 0.25),
                                iy / (scale * 0.25),
                                octaves=8,
                                base=random.randint(25, 35))

            bark_hue[iy, ix] = hue
    # Normalize to 0-255
    bark_texture = 128 + (bark_texture - np.min(bark_texture)) / (np.max(bark_texture) - np.min(bark_texture)) * (overall)
    bark_sat = 150 + (bark_sat - np.min(bark_sat)) / (np.max(bark_sat) - np.min(bark_sat)) * 55
    bark_hue = 20 + (bark_hue - np.min(bark_hue)) / (np.max(bark_hue) - np.min(bark_hue)) * 10

    # Assuming your arrays are named l, s, and v
    # They should be uint8 (0-255) for standard Pillow handling
    v_img = Image.fromarray(bark_texture.astype(np.uint8), mode='L')
    s_img = Image.fromarray(bark_sat.astype(np.uint8), mode='L')
    h_img = Image.fromarray(bark_hue.astype(np.uint8), mode='L')

    # Merge into HSV then convert to RGB
    hsv_image = Image.merge("HSV", (h_img, s_img, v_img))
    rgb_image = hsv_image.convert("RGB")

    return rgb_image


def make_uv_texture(fname: str):
    img = np.zeros((1024, 512, 3))

    # Bleepin' opencv has width and height backwards
    vals_u = np.linspace(0.0, 255.0, img.shape[1])
    vals_v = np.linspace(0.0, 255.0, img.shape[0])
    for row in range(0, img.shape[0]):
        # And blue green red
        img[row, :, 2] = vals_u
    for col in range(0, img.shape[1]):
        # And blue green red
        img[:, col, 1] = vals_v
    cv2.imwrite(fname, img)


def make_texture_set(tree_type: str, dir_name: str) ->(list, list):
    """ Make a nested set of texture images for trunk through small branches"""
    pixs_per_meter = 256

    # in meters
    radii = [0.0025, 0.005, 0.01, 0.015, 0.02, 0.1, 0.2]
    tex_scale = [60, 80, 100, 120, 140, 160, 180]
    ret_name_radii_pair = []
    im_size_exp = 3
    for t_scl, radius in zip(tex_scale, radii):
        tex_name = f"{tree_type}_{radius}.png"
        # Our radius measurements are in meters
        n_pixs_circum = 2.0 * np.pi * radius * pixs_per_meter
        im_size = 2 ** im_size_exp
        while im_size < n_pixs_circum:
            im_size_exp += 1
            im_size = 2 ** im_size_exp
        img2 = create_apple_bark(width=im_size, height=im_size*2)
        img = generate_apple_bark(width=im_size, height=im_size*2, overall=t_scl)
        full_path_name = f"{dir_name}/{tex_name}"
        img.save(full_path_name)
        full_path_name = f"{dir_name}/cv_{tex_name}"
        cv2.imwrite(full_path_name, img2)
        ret_name_radii_pair.append(tex_name)

    return radii, ret_name_radii_pair


def main():
    make_uv_texture("/Users/grimmc/PycharmProjects/data/lpy_trees/uv.png")
    # Generate and save
    make_texture_set("apple", "/Users/cindygrimm/PycharmProjects/data/textures")


if __name__ == "__main__":
    main()
