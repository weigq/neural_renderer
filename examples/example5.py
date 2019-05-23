#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simple rendering example with specific background image
"""
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import neural_renderer as nr


# setup render
img_size = 256
renderer = nr.Renderer(camera_mode='look_at', image_size=img_size)
renderer.perspective = True
renderer.eye = [0, 0, -2.732]
renderer.near = 0.1
renderer.far = 100.0

# for brighter visualization
renderer.light_intensity_ambient = 1.
renderer.light_intensity_ambient = 1
renderer.light_intensity_directional = 0

# load obj with default texture image
verts, faces, textures = nr.load_obj('data/human.obj', load_texture=True)
verts[:, 1] *= -1.
verts = verts[None, :, :]
faces = faces[None, :, :]
textures = textures[None, :, :]

image_rendered, _, _ = renderer.render(verts, faces, textures)
image_1 = (image_rendered[0].cpu().numpy()).transpose((1, 2, 0))

# load obj with other texture
verts, faces, textures = nr.load_obj('data/human.obj', load_texture=True, texture_image='data/human2.jpg')
verts[:, 1] *= -1.
verts = verts[None, :, :]
faces = faces[None, :, :]
textures = textures[None, :, :]

image_rendered, _, _ = renderer.render(verts, faces, textures)
image_2 = (image_rendered[0].cpu().numpy()).transpose((1, 2, 0))

# render on other background image
renderer.background_image = 'data/background.jpg'
image_rendered, _, _ = renderer.render(verts, faces, textures)
image_3 = (image_rendered[0].cpu().numpy()).transpose((1, 2, 0))

# plot results
plt.figure()

plt.subplot(131)
plt.title('default texture')
plt.imshow(image_1)
plt.axis('off')
plt.subplot(132)
plt.title('specific texture')
plt.imshow(image_2)
plt.axis('off')
plt.subplot(133)
plt.title('specific background')
plt.imshow(image_3)
plt.axis('off')
plt.draw()
plt.savefig('data/result_example5.png', dpi=300)
