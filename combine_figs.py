import sys
from PIL import Image

dirname = 'hopper_imgs/'
images = [Image.open(dirname+x) for x in ['traj-000.jpg', 'traj-001.jpg', 'traj-002.jpg', 'traj-003.jpg', 'traj-004.jpg']]

width = images[0].size[0]
height = images[0].size[1]
blank_width = int(width*0.1)
blank_height = int(height*0.2)
total_width = width*5 + blank_width*4
total_height = height*3 + blank_height*2

new_im = Image.new('RGB', (total_width,  total_height))

x_offset = 0
y_offset = 0
for im in images:
  new_im.paste(im, (x_offset,y_offset))
  x_offset += width + blank_width

dirname = 'hopper_imgs_pred/'
images = [Image.open(dirname+x) for x in ['traj-000.jpg', 'traj-001.jpg', 'traj-002.jpg', 'traj-003.jpg', 'traj-004.jpg']]

x_offset = 0
y_offset = y_offset + height + blank_height
for im in images:
  new_im.paste(im, (x_offset,y_offset))
  x_offset += width + blank_width

dirname = 'hopper_imgs_pred_trans/'
images = [Image.open(dirname+x) for x in ['traj-000.jpg', 'traj-001.jpg', 'traj-002.jpg', 'traj-003.jpg', 'traj-004.jpg']]

x_offset = 0
y_offset = y_offset + height + blank_height
for im in images:
  new_im.paste(im, (x_offset,y_offset))
  x_offset += width + blank_width

new_im.save('results/combined_figs.jpg')