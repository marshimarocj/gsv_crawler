import os
import cPickle
import json
import math
from array import array
from collections import deque
import time
import itertools

import numpy as np
import StreetView
import cv2
import matplotlib.cm as cm
import plyfile
from geopy.distance import great_circle

import util


'''func
'''
def depth_map(file, outfile):
  with open(file) as f:
    data = cPickle.load(f)
    depthmap_indices = data['depthmap_indices']
    depthmap_planes = data['depthmap_planes']

  h = 256
  w = 512
  d = np.zeros((h, w))
  for i in range(h*w):
    x = i%w
    y = i/w
    idx = depthmap_indices[i]
    plane = depthmap_planes[idx]
    if idx == 0:
      d[y, x] = np.inf
    else:
      phi = x / float(w-1) * 2 * math.pi
      theta = y / float(h-1) * math.pi
      v = np.array([
        math.sin(theta)*math.sin(phi), 
        math.sin(theta)*math.cos(phi), 
        math.cos(theta)
      ])
      n = np.array([plane['nx'], plane['ny'], plane['nz']])
      t = plane['d'] / np.dot(n, v)
      d[y, x] = abs(t)

  np.save(outfile, d)


def point_cloud(file, imgfile, outfile, threshold=np.inf):
  with open(file) as f:
    data = cPickle.load(f)
    depthmap_indices = data['depthmap_indices']
    depthmap_planes = data['depthmap_planes']

  img = cv2.imread(imgfile)
  img_h, img_w, _ = img.shape

  h = 256
  w = 512
  d = np.zeros((h, w))

  points = []
  colors = []
  out = []
  for i in range(h*w):
    x = i%w
    y = i/w
    idx = depthmap_indices[i]
    plane = depthmap_planes[idx]
    if idx != 0:
      phi = x / float(w-1) * 2 * math.pi
      theta = y / float(h-1) * math.pi
      point = np.array([
        math.sin(theta)*math.sin(phi),
        math.sin(theta)*math.cos(phi),
        math.cos(theta),
      ])
      n = np.array([plane['nx'], plane['ny'], -plane['nz']])
      distance = -plane['d'] / np.dot(n, point)
      point *= distance
      point[0] = -point[0]
      point[1] = -point[1]
      img_x =  x * img_w / w
      img_y =  y * img_h / h
      color = img[img_y, img_x, :]

      data = point.tolist() + color.tolist()
      out.append(tuple(data))

  out = np.array(out, dtype=[
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('blue', 'u1'), ('green', 'u1'), ('red', 'u1')])

  el = plyfile.PlyElement.describe(out, 'vertex')
  plyfile.PlyData([el]).write(outfile)


def gen_bfs_tree(panoids, names, meta_dir):
  visited = set()
  q = deque()
  panoid = names[0].split('.')[0]
  q.append(panoid)
  visited.add(panoid)

  panoid2child = {panoid: []}
  root = panoid

  while len(q) > 0:
    panoid = q.popleft()

    file = os.path.join(meta_dir, panoid + '.pkl')
    with open(file) as f:
      data = cPickle.load(f)
      panomap = data['panomap']
      heading = float(data['pano_yaw_degree'])
      for d in panomap:
        neighbor_panoid = d['panoid']
        if neighbor_panoid in panoids and neighbor_panoid not in visited:
          q.append(neighbor_panoid)
          visited.add(neighbor_panoid)

          dx = d['x']
          dy = d['y']
          neighbor_file = os.path.join(meta_dir, neighbor_panoid + '.pkl')
          with open(neighbor_file) as f:
            data = cPickle.load(f)
            neighbor_heading = float(data['pano_yaw_degree'])
          dtheta = (heading - neighbor_heading) / 180.0 * math.pi

          panoid2child[panoid].append({
            'panoid': neighbor_panoid,
            'dx': dx,
            'dy': dy,
            'dtheta': dtheta,
          })
          panoid2child[neighbor_panoid] = [] 

  return root, panoid2child


def post_order_traverse(root, panoid2child, point_cloud_dir, postfix='.ply'):
  threshold_dist = 30
  out = []

  file = os.path.join(point_cloud_dir, root + postfix)
  model = plyfile.PlyData.read(file)
  data = model.elements[0].data

  dist = np.power(np.power(data['x'], 2) + np.power(data['y'], 2), 0.5)
  idx = np.where(dist <= threshold_dist)
  data = data[idx]
  out.append(data)

  children = panoid2child[root]
  cnt = 0
  for child in children:
    panoid = child['panoid']
    dx = child['dx']
    dy = child['dy']
    dtheta = child['dtheta']
    point_data = post_order_traverse(panoid, panoid2child, point_cloud_dir)

    _x = np.copy(point_data['x'])
    _y = np.copy(point_data['y'])
    point_data['x'] = _x*math.cos(dtheta) - _y*math.sin(dtheta)
    point_data['y'] = _x*math.sin(dtheta) + _y*math.cos(dtheta)
    point_data['x'] += dx
    point_data['y'] += dy
    out.append(point_data)
    cnt += 1

  out = np.concatenate(out, axis=0)
  return out


# http://geomalgorithms.com/a05-_intersect-1.html
def intersect_plane_to_lines(plane_n, plane_v, line_p0, line_p1):
  plane_n = np.expand_dims(plane_n, 0)
  plane_v = np.expand_dims(plane_v, 0)
  u = line_p1 - line_p0
  w = line_p0 - plane_v
  s = -np.sum(plane_n * w, axis=1) / np.sum(plane_n * u, axis=1)
  s = np.expand_dims(s, 1)
  intersect = plane_v + w + s*u

  return intersect


def perspective_proj(plane_n, plane_v, theta, phi):
  points = [
    np.sin(theta) * np.sin(phi), 
    np.sin(theta) * np.cos(phi),
    np.cos(theta)
  ]
  points = np.array(points).T
  origin = np.zeros(points.shape)
  intersect = intersect_plane_to_lines(plane_n, plane_v, origin, points)

  return intersect


def generate_plane(theta, phi): 
  plane_n = np.array([
    math.sin(theta) * math.sin(phi),
    math.sin(theta) * math.cos(phi),
    math.cos(theta),
  ])
  plane_x = np.array([
    math.sin(theta) * math.sin(phi+math.pi/2),
    math.sin(theta) * math.cos(phi+math.pi/2),
    math.cos(theta)
  ])
  plane_y = np.array([0, 0, math.cos(theta + math.pi/2)])

  return plane_n, plane_x, plane_y


def cylindrical_size_to_perspective_size(
    theta_range, phi_range, cylindrical_img_h, cylindrical_img_w):
  height_sphere_pixel = cylindrical_img_h / math.pi * theta_range
  width_sphere_pixel = cylindrical_img_w / 2.0 / math.pi * phi_range

  height_plane_range = 2 * math.tan(theta_range/2)
  width_plane_range = 2 * math.tan(phi_range/2)

  height_ratio = height_plane_range / theta_range
  width_ratio = width_plane_range / phi_range
  print width_plane_range / height_plane_range

  if height_ratio > width_ratio:
    height_plane_pixel = int(height_sphere_pixel)
    width_plane_pixel = int(height_plane_pixel / height_plane_range * width_plane_range)
    # width_plane_pixel = height_plane_pixel
  else:
    width_plane_pixel = int(width_sphere_pixel)
    height_plane_pixel = int(width_plane_pixel / width_plane_range * height_plane_range)
    # height_plane_pixel = width_plane_pixel
  # print width_plane_pixel / float(height_plane_pixel)

  return height_plane_pixel, width_plane_pixel


def cylindrical_img_to_perspective_imgs(cylindrical_file, perspective_prefix):
  cylindrical_img = cv2.imread(cylindrical_file)
  cylindrical_img_h, cylindrical_img_w, _ = cylindrical_img.shape

  vertical_ratio_range = (1./4, 3./4)
  horizontal_ratio_ranges = [(0./3, 1./3), (1./3, 2./3), (2./3, 1)]
  postfixs = ['l', 'm', 'r']

  for horizontal_ratio_range, postfix in zip(horizontal_ratio_ranges, postfixs):
    theta = math.pi * (vertical_ratio_range[0] + vertical_ratio_range[1]) / 2.
    phi = math.pi * (horizontal_ratio_range[0] + horizontal_ratio_range[1])
    theta_range = math.pi * (vertical_ratio_range[1] - vertical_ratio_range[0])
    phi_range = math.pi * 2 * (horizontal_ratio_range[1] - horizontal_ratio_range[0])
    perspective_x_max = math.tan(phi_range/2)
    perspective_y_max = math.tan(theta_range/2)

    perspective_img_h, perspective_img_w = cylindrical_size_to_perspective_size(
    theta_range, phi_range, cylindrical_img_h, cylindrical_img_w)

    perspective_img_h /= 2
    perspective_img_w /= 2

    plane_n, plane_x, plane_y = generate_plane(theta, phi)
    
    phi_step = 2 * math.pi / cylindrical_img_w
    theta_step = math.pi / cylindrical_img_h
    phi = np.arange(
      horizontal_ratio_range[0]*2*math.pi, horizontal_ratio_range[1]*2*math.pi, phi_step)
    x_num = phi.shape[0]
    theta = np.arange(
      vertical_ratio_range[0]*math.pi, vertical_ratio_range[1]*math.pi, theta_step)
    y_num = theta.shape[0]
    phi = np.repeat(phi, y_num)
    theta = np.repeat(np.expand_dims(theta, 0), x_num, axis=0)
    theta = theta.reshape(-1)

    intersect = perspective_proj(plane_n, plane_n, theta, phi)
    x = np.sum(intersect * np.expand_dims(plane_x, 0), axis=1)
    y = np.sum(intersect * np.expand_dims(plane_y, 0), axis=1)
    valid_idx = np.logical_and(
      np.abs(x) <= perspective_x_max, np.abs(y) <= perspective_y_max)

    num_valid = sum(valid_idx)
    valid_theta = theta[valid_idx]
    valid_phi = phi[valid_idx]
    valid_x = x[valid_idx]
    valid_y = y[valid_idx]

    perspective_img = np.zeros((perspective_img_h, perspective_img_w, 3), dtype=np.uint8)

    perspective_c = (valid_x / perspective_x_max + 1)/2 * perspective_img_w
    perspective_c = perspective_c.astype(int)
    perspective_r = (valid_y / perspective_y_max + 1)/2 * perspective_img_h
    perspective_r = perspective_r.astype(int)
    visited_perspective = set(
      ['%d %d'%(perspective_r[i], perspective_c[i]) for i in range(num_valid)])

    cylindrical_c = valid_phi / phi_step
    cylindrical_c = cylindrical_c.astype(int)
    cylindrical_r = valid_theta / theta_step
    cylindrical_r = cylindrical_r.astype(int)

    perspective_img[perspective_r, perspective_c, :] = cylindrical_img[cylindrical_r, cylindrical_c, :]

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r, c in itertools.product(range(perspective_img_h), range(perspective_img_w)):
      if '%d %d'%(r, c) not in visited_perspective:
        for n in neighbors:
          _r = r+n[0]
          _c = c+n[1]
          if '%d %d'%(_r, _c) in visited_perspective:
            perspective_img[r, c, :] = perspective_img[_r, _c, :]
            break

    outfile = perspective_prefix + '_' + postfix + '.jpg'
    cv2.imwrite(outfile, perspective_img)


'''expr
'''
def tst_panorama():
  # lat = 42.349839 
  # lng = -71.078256
  # outfile = '/tmp/boston.pkl'
  # lat = 50.449926
  # lng = 30.523876
  # outfile = '/tmp/maiden.pkl'
  # gpss = [(42.349346,-71.0794468), (42.3492757,-71.0794034), (42.349215,-71.0793713)]
  # gpss = [
  #   (42.3494824,-71.0795032), (42.3495447,-71.0794153), (42.3495822,-71.0795461), 
  #   (42.3494415,-71.0795594), (42.3493966,-71.0794861), (42.3496645,-71.079583),
  #   (42.3495788,-71.0793033)
  # ]
  outdir = '/tmp'
  # outdir = '/home/chenjia/hdd/data/human-rights/panorama/boston-cross/meta'
  # panoid = 'eNne6EEaQcat_A2rEaMd9w'
  # panoids = ['QbgkAdcojhzK89d4yRYrMA']
  # panoids = ['JACzW3o4qZbKBK9YrkdteA']
  panoids = ['xfQ4ExF2MhEbttsV1P7SQA']

  # for gps in gpss:
  #   lat = gps[0]
  #   lng = gps[1]
  #   meta_data = StreetView.GetPanoramaMetadata(lat=lat, lon=lng, radius=20)
  for panoid in panoids:
    meta_data = StreetView.GetPanoramaMetadata(panoid=panoid)
    # print meta_data.PanoId, meta_data.Lat, meta_data.Lon
    # print meta_data.TileWidth, meta_data.TileHeight
    # print len(meta_data.DepthMapIndices)
    # print max(meta_data.DepthMapIndices)
    # print min(meta_data.DepthMapIndices)
    # print len(meta_data.DepthMapPlanes)
    # print len(meta_data.PanoMapPanos)
    # print meta_data.PanoMapPanos
    # print len(meta_data.PanoMapIndices)
    # print meta_data.PanoMapIndices
    print meta_data.NumZoomLevels
    out = {
      'depthmap_indices': meta_data.DepthMapIndices,
      'depthmap_planes': meta_data.DepthMapPlanes,
      'panomap': meta_data.PanoMapPanos,
      'lat': meta_data.Lat,
      'lng': meta_data.Lon,
      'org_lat': meta_data.OriginalLat,
      'org_lng': meta_data.OriginalLon,
      'panoid': meta_data.PanoId,
      'pano_yaw_degree': meta_data.ProjectionPanoYawDeg,
      'tilt_yaw_degree': meta_data.ProjectionTiltYawDeg,
      'tilt_pitch_degree': meta_data.ProjectionTiltPitchDeg,
    }
    outfile = os.path.join(outdir, meta_data.PanoId + '.pkl')
    with open(outfile, 'w') as fout:
      cPickle.dump(out, fout)


def print_heading():
  meta_dir = '/home/chenjia/hdd/data/human-rights/panorama/boston-cross/meta'

  names = os.listdir(meta_dir)
  for name in names:
    meta_file = os.path.join(meta_dir, name)
    with open(meta_file) as f:
      data = cPickle.load(f)
      print name, data['pano_yaw_degree']


def bat_panorama():
  root_dir = '/home/chenjia/hdd/data/human-rights'
  gps_files = [
    # os.path.join(root_dir, 'streetview_high_resolution', 'boylston_hash.json'),
    os.path.join(root_dir, 'Ukraine', 'streetview', 'gps_hash.json'),
  ]
  out_dirs = [
    # os.path.join(root_dir, 'panorama', 'boston', 'meta'),
    os.path.join(root_dir, 'panorama', 'maiden', 'meta'),
  ]

  for p in range(1):
    gps_file = gps_files[p]
    out_dir = out_dirs[p]
    print p

    with open(gps_file) as f:
      data = json.load(f)
    for d in data:
      fields = d[0].split(',')
      lat = float(fields[0])
      lng = float(fields[1])
      meta_data = StreetView.GetPanoramaMetadata(lat=lat, lon=lng, radius=20)
      panoid = meta_data.PanoId
      outfile = os.path.join(out_dir, '%s.pkl'%panoid)
      # if not os.path.exists(outfile):
      out = {
        'depthmap_indices': meta_data.DepthMapIndices,
        'depthmap_planes': meta_data.DepthMapPlanes,
        'lat': meta_data.Lat,
        'lng': meta_data.Lon,
        'org_lat': meta_data.OriginalLat,
        'org_lng': meta_data.OriginalLon,
        'panoid': meta_data.PanoId,
        'panomap': meta_data.PanoMapPanos,
        'pano_yaw_degree': meta_data.ProjectionPanoYawDeg,
        'tilt_yaw_degree': meta_data.ProjectionTiltYawDeg,
        'tilt_pitch_degree': meta_data.ProjectionTiltPitchDeg,
      }
      with open(outfile, 'w') as fout:
        cPickle.dump(out, fout)


def get_tiles():
  # panoid = 'nHCSMi7kDUi-zz0CsCvpgA'
  # outfile = '/tmp/boston.jpg'
  # panoid = 'trxRLftaqQVIK7vBztK2Cw'
  # outfile = '/tmp/maiden.jpg'
  # panoids = ['RrLZXAUSFz_d_UoDRwpbeA', 'n15eCgQjRTyEraBZ3k6CRA', '5mqUuGNLlZzQzx4kd0-TvA']
  # panoids = ['eNne6EEaQcat_A2rEaMd9w']
  # panoids = ['lGjyvISj_UhdT9qnLVAC9g', 'mZOggA76N2J7WkViiWnekQ']
  panoids = [
    # 'Wa2urXHVDHLZSW9PHi6Lkg',
    # 'QbgkAdcojhzK89d4yRYrMA',
    # 'Lqb8T65BVl8FKbV-4lOHIw',
    # '7DnOjjrFJuLBWBUv7Smz1A',
    # 'k-9FHoPeq3eo7b3ys0eAZQ',
    # 'trv13Ky4ERWGdxYTLtQWbg',
    # 'gKCLVYFLZfTWerrf1JxAug',
    # '8-DTPwnD_QwV6EIidPqJCg',
    # 'E4YEiFEtNV68ct7Q6mEDVQ',
    # 'fmuLNcdlZ0phcFy6Vamikg',
    # 'pZsKbptkBJfvoVwjdIFb8Q',
    # 'nVb-KL7ny_jOpfnsB2WwJA',
    # '6GVbWHfZfHhNBFhsBf5SPw',
    # 'Dfyi1S6McWY8wDh2a4RxNg',
    # '9Ls25Ks2WmwVer4PekjF0Q',
    # 'H0ZYKyjvpK2Utqf74vGgIg',
    # 'H8Xk_q-Mz3GrdCJ5c1VVQQ',
    # 'uaBiu4Y_o3getJHb_SOWUQ',
    # 'B3rjd_yWMzCqUacICl1Hvw',
    # 'rGjIzASuTZ9vPN8I35UzOw',
    # '-jAHhl9NTB6JarPw0cETFw',
    # 'X47GsRmXnrUUY2XQWvm1bA',
    # 'vfNzppHHr8pTVacGuvz-LQ',
    # 'TWJDVExqlnHprH4QS_kZFA',
    # 'mlIfCaSidTeAx-jpFCoRyA',
    # 'N_t6XW9P87xCD4A4b4Vlzw',
    # 'HCzNg__-s81ua3relu1FjQ',
    # 'T-wHgw7KUMPTUlvE-tB7eg',
    # 'vnWiSZUGABdVI3_holGNvg',
    # 'sOVRmM2CU9EarTOPwli_VQ',
    # 'P_DC6O0h5O_MyInkiTBLKw',
    # 'JwkM7vqJNKR1vyPUCJRj7w',
    # 'I3aJAVeiTr0sIX3Bl_-QHA',
    # 'C4T2HFB7zdh4wEI_r5wL6w',
    # 'f14jAB3JliRml4Ng6VUz-g',
    # '_a3t4wc03g4AFNT1SHpUrQ',
    # 'mnNu4GkaLyP5GNo7lEjn4A',
    # '_XkcpmqaEjMojgJbU2bkdw',
    # '4hTdEIfrGLrLDQ5EiNRFIQ',
    # 'hsX3IG_YaeCfpsJK1GAepQ',
    # 'nJTmNqWbAgnco2jQTcDDzQ',
    # 'WDc1jUXRk35uVqgfKZAdOA',
    # '1EnY6Lw8Eqzw4HzkYp9dbQ',
    # 'NpBREZKMfFETSEB3S7_cpQ',
    # 'lZZC4HiZ8mcQe10y-AB5Sw',
    # 'BaMsWQXVZhl4SXTuo-L3LA',
    # '1wJNZ2Lqte39zLlWx3nwfA',
    # 'NY7UptLrQZQ_c6dtD7W31A',
    # 'lW8lsNBUrXsGTlfx3xBk2A',
    # '5WWL-CMhBw-wI-72RwK30A',
    # 'aOXqKK2tZau7a0V36DdUbA',
    # 'eMumvRXgWSV8g4cy0WdIbg',
    # '2w_GCS1QF6ycG2yax5-W7w',
    # 'bYOg6_qMH41qEi7MddEJJA',
    # 'w_Cubwc8WpSHBTKBN0Z1xw'

    # 'iBI3CnQSFAU71GeDwda3yQ',
    # 'JACzW3o4qZbKBK9YrkdteA',
    # 'ASmbaRxBFRXJYYWui_fhCA',
    # 'FzUSNUKF_gBxiqyrQ1Udkg',
    # 'ipCG73gGYP296FO1LXOZoA',
    # 'nr83S3rqB-USYkmngzTpIQ',
    # '6k4YKmuPi63Mh6OyGq8cbg',
    # 'FIThOsvwR8KFXoQ4G9_lWQ',
    # 'TCV8QPzj8Lqp4Cs5LpB2Gg',
    # 'IqtiSzTtFdssPpqnti7idg',
    # 'YIxTtdAKgWDw9Me2lXyRfQ',
    # 'zPEwXzBiIea_PvIuAbBZfQ',
    # 'DwmPrGeV0u-F69wLLP9HYQ',
    # 'WBLfHR65S4ZWTMXkR7tx7g',
    # 'OLP1Y4z_i0orUTIkEq5sPA',
    # 'WnBu4qqiWVnkNV-mj8ar9Q',
    # 'jzbt6vctMoyxDGcNw3ecYg',
    # '5-GY-GCI0ZhGllCgE17sLg',
    # 'WfS5wDieWDqUERDKgzfhdg',
    # 'j5-vVfMshzSgoirx-os_2A',
    # 'UNDQCLkC9kEmQg2uW2hgPw',
    # 'WKb6KT3U3wdXqzQIhVbzJg',
    # 'AIyH1lPQNo-pTAw5C_6KCw',
    # 'Vg8It8VOBQVdcVqGrgnmOA',
    # 'a_ttJ_8SXPqABm-SGxE3JA',
    # 'LXy69kgg7F8qaIiHvNPVew',
    # 'nD2h2Iv6IXvxMrbXqsBqvQ',
    # 'qywpyD6fkr5tFzAXbmqxzA',
    # 'zLbAQFXER0owWtGAHmPUdA',
    # 'duQYGH6p80oBOxChIYM_wQ',
    # 'CeZeJPzM7OOhOCSdgni81g',
    # 'dZ_0jUZt3DQnE7a_4dEXNA',
    # '59lyPVHJTOnjg3PsL56EgQ',
    # 'POwsOfCIcE5Tcmfaj0pANQ',
    # 'BTSJIPjc7MY-1TsymYxm3w',
    # 'K4nkKLwvsDqItimKQZc_QA',
    # 'lFKpivGiXZ4VC2F2rqjFwg',
    # 's988w_8LOIdmWe4WpDqeBQ',
    # 'YEERrfjVDDhFgppNMnBo7A',
    # 'eV54rfsj8IlZSjRywScH6w',
    # 'YUgMc0rD4SLg_azKDHLf8w',
    # 'FLYVCmykm5rEiXBofB-zHw',
    # 'RSd8wPUgS0ldS9RViimHrA',
    # 'aa1Bl-jwfXFsaaF8sOWYuw',
    # 'kGx3QxpKDApQDXRiPSj5Mg',
    # 'gvBBT12ZV6-P50Ja7MJ4-w',
    # 'UszPLJCW0MVcVFPsJdeyLg',
    # '1tKWpbY6HutkXp-r3FDKqQ',
    # 's4-09A3arb1zLEutrVeQ9g',
    # 'rRKtbOv5sQRiR55XsjJ-fw',
    # '4bGUNO0mzSjBz2RyD5uFrQ',
    # 'MPUk7sYC36SvG6FXfHwsnA',
    # 'OlD73GqD_XOuGAgksP6bXQ',
    # 'iTV2wd8igjwtiCyjW5hwdA',
    # 'AKwUXt5G1uQzq3A3vJX6gA',
    # 'DDEFnivjoHWW0k9QRgMZDw',
    # '3Amm_uN0wQv4pYLTBW_Prw',
    # 'j4Bpv_Vj_j1qJNMPp3FzUA',
    # 'WlYDLioFb6HxzWvabw3nWg',
    # 'sJTwnVIbmwxEhO1Gol9aNw',
    # 'RCzIXHo529psugPG6lqbdQ',
    # 'dWw6EjGsmxTnCPhUXQntkA',
    # 'IGsoB2-p4dyZJ_SpDp7TiA'

    'Nmx1NygPdg37OOZkaWmDLQ',
    'TDBzYIvHsAVSA9o3VhAeyQ',
    'xfQ4ExF2MhEbttsV1P7SQA',
    'b7tr7gXBt_8uuBfVfmbSBQ',
    'WlM-gi_33UIJHLgHWdZQTA',
    'PNZEXY6-gz3g5fi196u2cg',
    'eHZNZ9op4LD2FwLlvQ6vew',
    'lQ7Z-FEkCmpas5AuHNLWWA',
    'lygynNd-iFtIMKaoEwj2pg',
    '9DkI-HvYJUDODtfsaXsX-A',
    '7YDHTSGvs4at0tNBbxNpQw',
    'rZsDq4DGws6oW_U4OykN8Q',
    'JityuoVu67r70o4gzD3AJA',
    'YJ5MbSmo36FvR02B1yT4MA',
    'HjTuOqETyF2hGfjH4FMTYA',
    '20qWPGGz6SG4FVaDa6ckCw',
    'qrnY8YlIFlbWgWBrn4Ks5w',
    'Bd76tiGIlm9lCMhmu9eLWw',
    'rEjRD4klxr25Ik7K_8s9pg',
    '0JEfLzMeTrVZJDEoaoBcyw',
    'jLe6xFL31Qsc0i9dHTuK6A',
    'ktZT3pT-AbRGesnAfV8GEA',
    'bUATUwqrXZ4pl395AEtmZQ',
    'A6IQm9A__QUvyB3nOfx3GA',
    'lZbEGgKsy0STHRS7XTyuDw',
    'tdazeyMhrNckCx9pjbL-bQ',
    'T_BXTrp7BGW-w6BhAWrMzg',
    'Dgbws4RqjaYOh6I1JW52hQ',
    'P1AJubKTKYO7ZF4tBmYQIw',
    '9ybv19U4ysEaWUlYFpQAgw',
    'FK5RBObf8AzvuYQUdzJz6w',
    'WOiWyiWAlFoI8GbbTxJAGw',
    'xEe2Aoky5cUM4jBgnT0yjg',
    'b46ed7G-IAD2dVvGalW6GQ',
    'gkQNf4rdJTbTUPPS8jbU1A',
    'EIVn1rX-Uo_PIs2LMlRbmA',
    'KR-RLOgBxjaVFDp7uutKKA',
    '0Xkiu4Y6WLt3MdesYPQXXQ',
    'ovn9ACySJaOioFhKR67meA',
    'gmmloYMU9OCpx3H-GyF66g',
    'frCjiL72ThNwUDsWGNBnkQ',
    'dOCgvqJfYwKCmBXLnWssbA',
    'HTBgsUz-zstb1P4mZ620dA',
    '2N17hOW6yvfODWUizXWMnw',
    'EDW4zukG1OfFWamH_kD6Ow',
    'o9GOCR7yCU-o4OU8vYkqEg',
    'YfKJqGQKfrzE4PzZofTGvg',
    '8LQi4pmZMsAPDIfo5ewTwQ',
    'bQfgGrNU6HmwRLWD63CIvg',
    'L8jh66ZvzhDsAaybNTyF5w',
    'U6_8w2C5zefAHOSRY0GcTQ',
    'wMwir2gREWwzQW3C_Cc1pw',
    'yUWsoJDrvUzZ-wYqlVZPqA',
    'oPqeOsdsynHLMkF8wVfyIQ',
    'Pm9UbxYQPY2xi7IzR4_uWQ',
    'mIoDYV4V5lM38rip7Y8nUg',
    'zP9Rw4J9J0fdFf9Moglylw',
    'jYSWy-5rvf8v2iGEASmJhg',
    '2-wWF21KSlHDirSX_PcueQ',
    'nBrVxQdrELlTi_n_hsXLJQ',
    'qUJVRVZ-ATsDJ_oK20FHNQ',
    'PsNyBfxI1smf3otjac1fLg',
    '1EDTRn25lpgbMGyaWseDsg',
    'oHMHvUbiSX2t7qjz-0V5zA',
    '8ct7zBHlt837CQF6kcqLkQ',
    'L55nf24owf876QFkdmTXFg',
    'myqvdf6Wn0fUkz_EPU6cdg',
    'RSW1hD9XWKlqtcCTXr0pLw',
    'YcgD2PFKBZIsmJ-suHUctw',
    'HdVBt764HxvIWPovJuB5HQ',
    'vJ-NKrLKCKDHuSvrhICwdQ',
    'vpRGxnysoPIEYgh438v0wQ',
    'wUpsBnRxdN-xYYhTlkskZA',
    'BmlDtTYbALXaR61jyjlJuQ',
    'ztEwohPxZ1l-rfKEAnKOmA',
    'NqF1V5XqNHariGj70D1jeA',
    'G9_IqH8J4jOcdo0D6KofDw',
    '9JjLosHzqUeOJprtleJuPw',
    'vFc81t2wqV04LDrwwpfL4Q'
  ]
  # outdir = '/tmp'
  # outdir = '/home/chenjia/hdd/data/human-rights/panorama/boston-cross/img'
  # outdir = '/home/chenjia/hdd/data/human-rights/panorama/boston2015/img'
  # outdir = '/home/chenjia/hdd/data/human-rights/panorama/boston2007/img'
  outdir = '/home/chenjia/hdd/data/human-rights/panorama/boston2013/img'

  for panoid in panoids:
    img = np.zeros((512*4, 512*7, 3), dtype=np.uint8)

    for x in range(7):
      for y in range(4):
        data = StreetView.GetPanoramaTile(panoid, 3, x, y)
        data = array('B', data)
        data = np.array(data, dtype=np.byte)
        img[y*512:(y+1)*512, x*512:(x+1)*512, :] = \
          cv2.imdecode(data, cv2.IMREAD_COLOR)

    img = img[:1664, :3328, :]
    outfile = os.path.join(outdir, panoid + '.jpg')
    cv2.imwrite(outfile, img)


def bat_tiles():
  # # root_dir = '/home/chenjia/hdd/data/human-rights' # earth
  # meta_dirs = [
  #   # os.path.join(root_dir, 'panorama', 'boston', 'meta'),
  #   # os.path.join(root_dir, 'panorama', 'maiden', 'meta'),
  #   os.path.join(root_dir, 'panorama', 'boston-cross', 'meta'),
  # ]
  # out_dirs = [
  #   # os.path.join(root_dir, 'panorama', 'boston', 'img'),
  #   # os.path.join(root_dir, 'panorama', 'maiden', 'img'),
  #   os.path.join(root_dir, 'panorama', 'boston-cross', 'img'),
  # ]
  root_dir = '/home/jiac/data2/humanrights' # gpu9
  meta_dirs = [
    os.path.join(root_dir, 'panorama', 'boston', 'meta'),
  ]
  out_dirs = [
    os.path.join(root_dir, 'panorama', 'boston', 'img'),
  ]

  for meta_dir, out_dir in zip(meta_dirs, out_dirs):
    names = os.listdir(meta_dir)
    for name in names:
      file = os.path.join(meta_dir, name)
      name, _ = os.path.splitext(name)
      outfile = os.path.join(out_dir, name + '.jpg')
      if os.path.exists(outfile):
        continue

      with open(file) as f:
        data = cPickle.load(f)
        lat = data['lat']
        lng = data['lng']
        panoid = data['panoid']

      img = np.zeros((512*4, 512*7, 3), dtype=np.uint8)

      complete = True
      for x in range(7):
        for y in range(4):
          try:
            data = StreetView.GetPanoramaTile(panoid, 3, x, y)
          except:
            complete = False
          else:
            data = array('B', data)
            data = np.array(data, dtype=np.byte)
            img[y*512:(y+1)*512, x*512:(x+1)*512, :] = \
              cv2.imdecode(data, cv2.IMREAD_COLOR)

      if complete:
        img = img[:1664, :3328, :]
        cv2.imwrite(outfile, img)
        print outfile


def viz_depth_map():
  # file = '/tmp/boston.pkl'
  # outfile = '/tmp/boston.npy'
  file = '/tmp/maiden.pkl'
  outfile = '/tmp/maiden.npy'
  with open(file) as f:
    data = cPickle.load(f)

  depthmap_indices = data['depthmap_indices']
  depthmap_planes = data['depthmap_planes']

  h = 256
  w = 512
  d = np.zeros((h, w))
  for i in range(h*w):
    x = i%w
    y = i/w
    idx = depthmap_indices[i]
    plane = depthmap_planes[idx]
    if idx == 0:
      d[y, x] = np.inf
    else:
      phi = x / float(w-1) * 2 * math.pi
      theta = y / float(h-1) * math.pi
      v = np.array([
        math.sin(theta)*math.sin(phi), 
        math.sin(theta)*math.cos(phi), 
        math.cos(theta)
      ])
      n = np.array([plane['nx'], plane['ny'], plane['nz']])
      t = plane['d'] / np.dot(n, v)
      v = v*t
      d[y, x] = np.linalg.norm(v)

  np.save(outfile, d)


def bat_depth_map():
  root_dir = '/home/chenjia/hdd/data/human-rights'
  meta_dirs = [
    # os.path.join(root_dir, 'panorama', 'boston', 'meta'),
    os.path.join(root_dir, 'panorama', 'maiden', 'meta'),
  ]
  out_dirs = [
    # os.path.join(root_dir, 'panorama', 'boston', 'depth'),
    os.path.join(root_dir, 'panorama', 'maiden', 'depth'),
  ]

  for meta_dir, out_dir in zip(meta_dirs, out_dirs):
    names = os.listdir(meta_dir)
    for name in names:
      file = os.path.join(meta_dir, name)
      name, _ = os.path.splitext(name)
      outfile = os.path.join(out_dir, name + '.npy')

      depth_map(file, outfile)


def viz_depth_on_tiles():
  # imgfile = '/tmp/boston.jpg'
  # depth_file = '/tmp/boston.npy'
  # imgfile = '/tmp/maiden.jpg'
  # depth_file = '/tmp/maiden.npy'
  root_dir = '/home/chenjia/hdd/data/human-rights'
  img_dir = os.path.join(root_dir, 'panorama', 'boston', 'img')
  depth_dir = os.path.join(root_dir, 'panorama', 'boston', 'osm_align_depth')
  name = 'wHlmBmRXiVgcNiVLdwkvfQ'

  imgfile = os.path.join(img_dir, name + '.jpg')
  img = cv2.imread(imgfile)
  img = cv2.resize(img, (1024, 512))

  depth_file = os.path.join(depth_dir, name + '.npy')
  depth = np.load(depth_file)
  inf_idx = np.isinf(depth)
  finite_idx = np.logical_not(inf_idx)
  max_bound = np.percentile(depth[finite_idx], 95) + 50
  depth[inf_idx] = max_bound
  depth[finite_idx] = np.minimum(depth[finite_idx], max_bound)
  # depth = max_bound - depth
  # depth = depth / max_bound * 255
  # depth = np.asarray(depth, dtype=np.uint8)
  # print depth.shape
  # depth = np.repeat(np.expand_dims(depth, -1), 3, axis=2)
  # print depth.shape
  # depth = cv2.resize(depth, (1024, 512))
  depth = depth / max_bound
  depth = cm.jet(depth)
  depth = depth[:, :, :3]
  depth = np.asarray(depth*255, dtype=np.uint8)
  depth = cv2.resize(depth, (1024, 512))

  blend = 0.5 * img + 0.5 * depth
  blend = np.asarray(blend, dtype=np.uint8)
  outfile = os.path.join(depth_dir, name + '.jpg')
  cv2.imwrite(outfile, blend)


def bat_viz_depth_on_tiles():
  root_dir = '/home/chenjia/hdd/data/human-rights'
  meta_dirs = [
    # os.path.join(root_dir, 'panorama', 'boston', 'meta'),
    os.path.join(root_dir, 'panorama', 'maiden', 'meta'),
  ]
  img_dirs = [
    # os.path.join(root_dir, 'panorama', 'boston', 'img'),
    os.path.join(root_dir, 'panorama', 'maiden', 'img'),
  ]
  depth_dirs = [
    # os.path.join(root_dir, 'panorama', 'boston', 'depth'),
    os.path.join(root_dir, 'panorama', 'maiden', 'depth'),
  ]
  out_dirs = [
    # os.path.join(root_dir, 'panorama', 'boston', 'viz_depth'),
    os.path.join(root_dir, 'panorama', 'maiden', 'viz_depth'),
  ]

  for meta_dir, img_dir, depth_dir, out_dir in zip(meta_dirs, img_dirs, depth_dirs, out_dirs):
    names = os.listdir(meta_dir)
    for name in names:
      file = os.path.join(meta_dir, name)
      name, _ = os.path.splitext(name)
      imgfile = os.path.join(img_dir, name + '.jpg')
      depth_file = os.path.join(depth_dir, name + '.npy')
      outfile = os.path.join(out_dir, name + '.jpg')

      img = cv2.imread(imgfile)
      img = cv2.resize(img, (1024, 512))

      depth = np.load(depth_file)
      inf_idx = np.isinf(depth)
      finite_idx = np.logical_not(inf_idx)
      max_bound = np.percentile(depth[finite_idx], 95) + 50
      depth[inf_idx] = max_bound
      depth[finite_idx] = np.minimum(depth[finite_idx], max_bound)
      depth = depth / max_bound
      depth = cm.jet(depth)
      depth = depth[:, :, :3]
      depth = np.asarray(depth*255, dtype=np.uint8)
      depth = cv2.resize(depth, (1024, 512))

      blend = 0.5 * img + 0.5 * depth
      blend = np.asarray(blend, dtype=np.uint8)
      cv2.imwrite(outfile, blend)


def viz_plane_on_tiles():
  imgfile = '/tmp/boston.jpg'
  info_file = '/tmp/boston.pkl'

  with open(info_file) as f:
    data = cPickle.load(f)
    plane_idx = data['depthmap_indices']
    plane_idx = np.array(plane_idx, dtype=np.float32).reshape((256, 512))
    plane_idx = plane_idx / np.max(plane_idx)
    plane_idx = cm.prism(plane_idx)
    plane_idx = plane_idx[:, :, :3]
    plane_idx = np.asarray(plane_idx*255, dtype=np.uint8)
    plane_idx = cv2.resize(plane_idx, (1024, 512))

  img = cv2.imread(imgfile)
  img = cv2.resize(img, (1024, 512))

  blend = 0.5 * img + 0.5 * plane_idx
  cv2.imwrite('/tmp/tmp.jpg', blend)


def bat_viz_plane_on_tiles():
  root_dir = '/home/chenjia/hdd/data/human-rights'
  meta_dirs = [
    os.path.join(root_dir, 'panorama', 'boston', 'meta'),
    os.path.join(root_dir, 'panorama', 'maiden', 'meta'),
  ]
  img_dirs = [
    os.path.join(root_dir, 'panorama', 'boston', 'img'),
    os.path.join(root_dir, 'panorama', 'maiden', 'img'),
  ]
  out_dirs = [
    os.path.join(root_dir, 'panorama', 'boston', 'viz_plane'),
    os.path.join(root_dir, 'panorama', 'maiden', 'viz_plane'),
  ]

  for meta_dir, img_dir, out_dir in zip(meta_dirs, img_dirs, out_dirs):
    names = os.listdir(meta_dir)
    for name in names:
      file = os.path.join(meta_dir, name)
      name, _ = os.path.splitext(name)
      imgfile = os.path.join(img_dir, name + '.jpg')
      outfile = os.path.join(out_dir, name + '.jpg')

      img = cv2.imread(imgfile)
      img = cv2.resize(img, (1024, 512))

      with open(file) as f:
        data = cPickle.load(f)
        plane_idx = data['depthmap_indices']
        plane_idx = np.array(plane_idx, dtype=np.float32).reshape((256, 512))
        plane_idx = plane_idx / np.max(plane_idx)
        plane_idx = cm.prism(plane_idx)
        plane_idx = plane_idx[:, :, :3]
        plane_idx = np.asarray(plane_idx*255, dtype=np.uint8)
        plane_idx = cv2.resize(plane_idx, (1024, 512))

      blend = 0.5 * img + 0.5 * plane_idx
      cv2.imwrite(outfile, blend)


def color_point_cloud():
  # imgfile = '/tmp/boston.jpg'
  # info_file = '/tmp/boston.pkl'
  # outfile = '/tmp/boston.ply'
  # imgfile = '/tmp/maiden.jpg'
  # info_file = '/tmp/maiden.pkl'
  # outfile = '/tmp/maiden.ply'
  # panoids = ['RrLZXAUSFz_d_UoDRwpbeA', 'n15eCgQjRTyEraBZ3k6CRA', '5mqUuGNLlZzQzx4kd0-TvA']
  # panoids = ['eNne6EEaQcat_A2rEaMd9w']
  panoids = ['lGjyvISj_UhdT9qnLVAC9g', 'mZOggA76N2J7WkViiWnekQ']
  # meta_dir = '/tmp'
  # img_dir = '/tmp'
  # point_dir = '/tmp'
  meta_dir = '/home/chenjia/hdd/data/human-rights/panorama/boston-cross/meta'
  img_dir = '/home/chenjia/hdd/data/human-rights/panorama/boston-cross/img'
  point_dir = '/home/chenjia/hdd/data/human-rights/panorama/boston-cross/point_cloud'

  for panoid in panoids:
    info_file = os.path.join(meta_dir, panoid + '.pkl')
    imgfile = os.path.join(img_dir, panoid + '.jpg')
    outfile = os.path.join(point_dir, panoid + '.ply')

    img = cv2.imread(imgfile)
    img_h, img_w, _ = img.shape

    with open(info_file) as f:
      data = cPickle.load(f)
      depthmap_indices = data['depthmap_indices']
      depthmap_planes = data['depthmap_planes']

      h = 256
      w = 512
      d = np.zeros((h, w))

      out = []
      for i in range(h*w):
        x = i%w
        y = i/w
        idx = depthmap_indices[i]
        plane = depthmap_planes[idx]
        if idx != 0:
          phi = x / float(w-1) * 2 * math.pi
          theta = y / float(h-1) * math.pi
          point = np.array([
            math.sin(theta)*math.sin(phi), 
            math.sin(theta)*math.cos(phi), 
            math.cos(theta),
          ])
          n = np.array([plane['nx'], plane['ny'], -plane['nz']])
          distance = abs(plane['d'] / np.dot(n, point))
          point *= distance
          point[0] = -point[0]
          point[1] = -point[1]
          img_x =  x * img_w / w
          img_y =  y * img_h / h
          color = img[img_y, img_x, :]

          # data = point.tolist() + n.tolist() + [1] + color.tolist()
          data = point.tolist() + color.tolist()
          out.append(tuple(data))

      out = np.array(out, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        # ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        # ('intensity', 'f4'),
        ('blue', 'u1'), ('green', 'u1'), ('red', 'u1')])

      el = plyfile.PlyElement.describe(out, 'vertex')    
      plyfile.PlyData([el]).write(outfile)


def sphere_point_cloud():
  panoid = 'lGjyvISj_UhdT9qnLVAC9g'
  meta_dir = '/home/chenjia/hdd/data/human-rights/panorama/boston-cross/meta'
  img_dir = '/home/chenjia/hdd/data/human-rights/panorama/boston-cross/img'

  meta_file = os.path.join(meta_dir, panoid + '.pkl')
  img_file = os.path.join(img_dir, panoid + '.jpg')
  outfile = '/tmp/tmp.ply'

  img = cv2.imread(img_file)
  img_h, img_w, _ = img.shape
  r = img_w / math.pi / 2.0

  point_data = []
  for i, j in itertools.product(range(img_h), range(img_w)):
    phi = j / float(img_w) * 2 * math.pi
    theta = i / float(img_h) * math.pi
    point = np.array([
      math.sin(theta)*math.sin(phi),
      math.sin(theta)*math.cos(phi), 
      math.cos(theta),
    ])
    point = point*r
    color = img[i, j]

    data = point.tolist() + color.tolist()
    point_data.append(tuple(data))

  out = np.array(point_data, dtype=[
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('blue', 'u1'), ('green', 'u1'), ('red', 'u1')])

  el = plyfile.PlyElement.describe(out, 'vertex')    
  plyfile.PlyData([el]).write(outfile)


def sphere_proj_to_linear_proj_table():
  height_pixel = 1664
  width_pixel = 3228
  theta_range = math.pi/2
  phi_range = math.pi/3*2
  height = 2 * math.tan(theta_range/2)
  width = 2 * math.tan(phi_range/2)

  height_ratio = height / theta_range
  width_ratio = width / phi_range
  print width_ratio / height_ratio

  height_sphere = height_pixel / math.pi * theta_range
  width_sphere = width_pixel / 2.0 / math.pi * phi_range
  height_max = height / theta_range * height_sphere
  width_max = width / phi_range * width_sphere
  print height_max, width_max, width_max / height_max
  print height_sphere, width_sphere, width_sphere / height_sphere


# def perspective_project_img_to_patch():
#   panoid = 'lGjyvISj_UhdT9qnLVAC9g'
#   img_dir = '/home/chenjia/hdd/data/human-rights/panorama/boston-cross/img'

#   img_file = os.path.join(img_dir, panoid + '.jpg')
#   outfile = '/tmp/tmp.jpg'

#   img = cv2.imread(img_file)
#   img_h, img_w, _ = img.shape

#   vertical_ratio_range = [1./4, 3./4]
#   horizontal_ratio_range = [0./3, 1./3]
#   # horizontal_ratio_range = [1./3, 2./3]
#   # horizontal_ratio_range = [2./3, 1]
#   theta = math.pi * (vertical_ratio_range[0] + vertical_ratio_range[1]) / 2.
#   phi = math.pi * (horizontal_ratio_range[0] + horizontal_ratio_range[1])
#   plane_norm = [
#     math.sin(theta) * math.sin(phi),
#     math.sin(theta) * math.cos(phi),
#     math.cos(theta),
#   ]
#   plane_x = [
#     math.sin(theta)*math.sin(phi+math.pi/2), 
#     math.sin(theta)*math.cos(phi+math.pi/2), 
#     math.cos(theta)
#   ]
#   plane_y = [0, 0, math.cos(theta + math.pi/2)]

#   theta_range = math.pi * (vertical_ratio_range[1] - vertical_ratio_range[0])
#   phi_range = math.pi * 2 * (horizontal_ratio_range[1] - horizontal_ratio_range[0])

#   height_sphere_pixel = img_h / math.pi * theta_range
#   width_sphere_pixel = img_w / 2.0 / math.pi * phi_range

#   height_plane_range = 2 * math.tan(theta_range/2)
#   width_plane_range = 2 * math.tan(phi_range/2)

#   height_ratio = height_plane_range / theta_range
#   width_ratio = width_plane_range / phi_range

#   if height_ratio > width_ratio:
#     height_plane_pixel = int(height_sphere_pixel)
#     width_plane_pixel = int(height_plane_pixel * width_ratio / height_ratio * theta_range / phi_range)
#   else:
#     width_plane_pixel = int(width_sphere_pixel)
#     height_plane_pixel = int(width_plane_pixel * height_ratio / width_ratio * phi_range / theta_range)

#   width_plane_pixel /= 2
#   height_plane_pixel /= 2

#   plane_img = np.zeros((height_plane_pixel, width_plane_pixel, 3), dtype=np.uint8)

#   plane = vtk.vtkPlane()
#   plane.SetNormal(plane_norm)
#   plane.SetOrigin(plane_norm)
#   plane_x_max = math.tan(phi_range/2)
#   plane_y_max = math.tan(theta_range/2)

#   for i, j in itertools.product(
#       range(int(img_h*vertical_ratio_range[0]), int(img_h*vertical_ratio_range[1])), 
#       range(int(img_w*horizontal_ratio_range[0]), int(img_w*horizontal_ratio_range[1]))):
#     color = img[i, j]

#     phi = j / float(img_w) * 2 * math.pi
#     theta = i / float(img_h) * math.pi
#     point = np.array([
#       math.sin(theta) * math.sin(phi),
#       math.sin(theta) * math.cos(phi),
#       math.cos(theta),
#     ])
#     t = vtk.mutable(0.0)
#     intersect_point = [0, 0, 0]
#     plane.IntersectWithLine([0, 0, 0], point, t, intersect_point)
#     x = vtk.vtkMath.Dot(intersect_point, plane_x)
#     y = vtk.vtkMath.Dot(intersect_point, plane_y)
#     if abs(x) >= plane_x_max or abs(y) >= plane_y_max:
#       continue

#     plane_j = int((x / plane_x_max + 1)/2 * width_plane_pixel)
#     plane_i = int((y / plane_y_max + 1)/2 * height_plane_pixel)
#     if plane_j == width_plane_pixel or plane_i == height_plane_pixel:
#       continue
#     plane_img[plane_i, plane_j, :] = color

#   cv2.imwrite('/tmp/tmp.jpg', plane_img)


def perspective_project_img_to_patch():
  panoid = 'lGjyvISj_UhdT9qnLVAC9g'
  img_dir = '/home/chenjia/hdd/data/human-rights/panorama/boston-cross/img'

  img_file = os.path.join(img_dir, panoid + '.jpg')
  outfile = '/tmp/tmp.jpg'

  cylindrical_img = cv2.imread(img_file)
  cylindrical_img_h, cylindrical_img_w, _ = cylindrical_img.shape

  vertical_ratio_range = [1./4, 3./4] # 90
  # horizontal_ratio_range = [0./3, 1./3]
  # horizontal_ratio_range = [1./3, 2./3]
  # horizontal_ratio_range = [2./3, 1]
  horizontal_ratio_range = [1./6, 1./2] # 120
  theta = math.pi * (vertical_ratio_range[0] + vertical_ratio_range[1]) / 2.
  phi = math.pi * (horizontal_ratio_range[0] + horizontal_ratio_range[1])
  theta_range = math.pi * (vertical_ratio_range[1] - vertical_ratio_range[0])
  phi_range = math.pi * 2 * (horizontal_ratio_range[1] - horizontal_ratio_range[0])
  perspective_x_max = math.tan(phi_range/2)
  perspective_y_max = math.tan(theta_range/2)

  perspective_img_h, perspective_img_w = cylindrical_size_to_perspective_size(
    theta_range, phi_range, cylindrical_img_h, cylindrical_img_w)

  perspective_img_h /= 2
  perspective_img_w /= 2

  plane_n, plane_x, plane_y = generate_plane(theta, phi)
  
  phi_step = 2 * math.pi / cylindrical_img_w
  theta_step = math.pi / cylindrical_img_h
  phi = np.arange(
    horizontal_ratio_range[0]*2*math.pi, horizontal_ratio_range[1]*2*math.pi, phi_step)
  x_num = phi.shape[0]
  theta = np.arange(
    vertical_ratio_range[0]*math.pi, vertical_ratio_range[1]*math.pi, theta_step)
  y_num = theta.shape[0]
  phi = np.repeat(phi, y_num)
  theta = np.repeat(np.expand_dims(theta, 0), x_num, axis=0)
  theta = theta.reshape(-1)

  intersect = perspective_proj(plane_n, plane_n, theta, phi)
  x = np.sum(intersect * np.expand_dims(plane_x, 0), axis=1)
  y = np.sum(intersect * np.expand_dims(plane_y, 0), axis=1)
  valid_idx = np.logical_and(
    np.abs(x) <= perspective_x_max, np.abs(y) <= perspective_y_max)
  # np.savez_compressed('/tmp/transform.npz', 
  #   theta=theta, phi=phi, x=x, y=y, valid_idx=valid_idx)

  num_valid = sum(valid_idx)
  valid_theta = theta[valid_idx]
  valid_phi = phi[valid_idx]
  valid_x = x[valid_idx]
  valid_y = y[valid_idx]

  perspective_img = np.zeros((perspective_img_h, perspective_img_w, 3), dtype=np.uint8)

  perspective_c = (valid_x / perspective_x_max + 1)/2 * perspective_img_w
  perspective_c = perspective_c.astype(int)
  perspective_r = (valid_y / perspective_y_max + 1)/2 * perspective_img_h
  perspective_r = perspective_r.astype(int)
  visited_perspective = set(
    ['%d %d'%(perspective_r[i], perspective_c[i]) for i in range(num_valid)])

  cylindrical_c = valid_phi / phi_step
  cylindrical_c = cylindrical_c.astype(int)
  cylindrical_r = valid_theta / theta_step
  cylindrical_r = cylindrical_r.astype(int)

  perspective_img[perspective_r, perspective_c, :] = cylindrical_img[cylindrical_r, cylindrical_c, :]

  neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  for r, c in itertools.product(range(perspective_img_h), range(perspective_img_w)):
    if '%d %d'%(r, c) not in visited_perspective:
      for n in neighbors:
        _r = r+n[0]
        _c = c+n[1]
        if '%d %d'%(_r, _c) in visited_perspective:
          perspective_img[r, c, :] = perspective_img[_r, _c, :]
          break

  cv2.imwrite('/tmp/tmp.jpg', perspective_img)


def bat_cylindrical_img_to_perspective_imgs():
  root_dir = '/home/chenjia/hdd/data/human-rights' # earth
  img_dirs = [
    # os.path.join(root_dir, 'panorama', 'boston', 'img'),
    os.path.join(root_dir, 'panorama', 'boston-cross', 'img'),
    os.path.join(root_dir, 'panorama', 'maiden', 'img'),
    # os.path.join(root_dir, 'panorama', 'dallas', 'img'),
  ]
  out_dirs = [
    # os.path.join(root_dir, 'panorama', 'boston', 'perspective_img'),
    os.path.join(root_dir, 'panorama', 'boston-cross', 'perspective_img'),
    os.path.join(root_dir, 'panorama', 'maiden', 'perspective_img'),
    # os.path.join(root_dir, 'panorama', 'dallas', 'perspective_img'),
  ]

  for img_dir, out_dir in zip(img_dirs, out_dirs):
    if not os.path.exists(out_dir):
      os.mkdir(out_dir)

    names = os.listdir(img_dir)
    for name in names:
      _name, _ = os.path.splitext(name)
      cylindrical_file = os.path.join(img_dir, name)
      perspective_prefix = os.path.join(out_dir, _name)
      cylindrical_img_to_perspective_imgs(cylindrical_file, perspective_prefix)


def bat_color_point_cloud():
  root_dir = '/home/chenjia/hdd/data/human-rights' # earth
  # root_dir = '/home/jiac/data/humanrights' # aladdin1
  meta_dirs = [
    # os.path.join(root_dir, 'panorama', 'boston', 'meta'),
    # os.path.join(root_dir, 'panorama', 'boston-cross', 'meta'),
    os.path.join(root_dir, 'panorama', 'maiden', 'meta'),
    # os.path.join(root_dir, 'panorama', 'dallas', 'meta'),
  ]
  img_dirs = [
    # os.path.join(root_dir, 'panorama', 'boston', 'img'),
    # os.path.join(root_dir, 'panorama', 'boston-cross', 'img'),
    os.path.join(root_dir, 'panorama', 'maiden', 'img'),
    # os.path.join(root_dir, 'panorama', 'dallas', 'img'),
  ]
  out_dirs = [
    # os.path.join(root_dir, 'panorama', 'boston', 'point_cloud')
    # os.path.join(root_dir, 'panorama', 'boston-cross', 'point_cloud')
    os.path.join(root_dir, 'panorama', 'maiden', 'point_cloud')
    # os.path.join(root_dir, 'panorama', 'dallas', 'point_cloud')
  ]

  for meta_dir, img_dir, out_dir in zip(meta_dirs, img_dirs, out_dirs):
    names = os.listdir(meta_dir)
    for name in names:
      file = os.path.join(meta_dir, name)
      name, _ = os.path.splitext(name)
      imgfile = os.path.join(img_dir, name + '.jpg')
      outfile = os.path.join(out_dir, name + '.ply')

      point_cloud(file, imgfile, outfile)


def tst_merge():
  root_dir = '/home/chenjia/hdd/data/human-rights'
  # meta_dir = os.path.join(root_dir, 'panorama', 'boston', 'meta')
  # point_dir = os.path.join(root_dir, 'panorama', 'boston', 'point_cloud')
  meta_dir = os.path.join(root_dir, 'panorama', 'boston-cross', 'meta')
  point_dir = os.path.join(root_dir, 'panorama', 'boston-cross', 'point_cloud')
  # meta_dir = '/tmp'
  # point_dir = '/tmp'
  # anchor_panoid = 'Ak3BgdMb6oV0ijW09TdMmQ'
  # neighbor_panoid = 'wfIifqAm2uet22tZIMeGlQ'
  # anchor_panoid = 'RrLZXAUSFz_d_UoDRwpbeA'
  # neighbor_panoid = 'n15eCgQjRTyEraBZ3k6CRA'
  # neighbor_panoid = 'eNne6EEaQcat_A2rEaMd9w'
  anchor_panoid = 'lGjyvISj_UhdT9qnLVAC9g'
  neighbor_panoid = 'mZOggA76N2J7WkViiWnekQ'
  outfile = '/tmp/tmp.ply'

  meta_file = os.path.join(meta_dir, anchor_panoid + '.pkl')

  with open(meta_file) as f:
    data = cPickle.load(f)
    base_heading = float(data['pano_yaw_degree'])

  panomap = data['panomap']
  x = -1
  y = -1
  for d in panomap:
    if d['panoid'] == neighbor_panoid:
      x = d['x']
      y = d['y']
      break

  neighbor_meta_file = os.path.join(meta_dir, neighbor_panoid + '.pkl')
  with open(neighbor_meta_file) as f:
    data = cPickle.load(f)
    neighbor_heading = float(data['pano_yaw_degree'])
  diff_heading = (base_heading - neighbor_heading) / 180.0 * math.pi

  point_data = []

  base_point_file = os.path.join(point_dir, anchor_panoid + '.ply')
  base_model = plyfile.PlyData.read(base_point_file)
  data = base_model.elements[0].data
  point_data.append(data)
  # print np.std(data['x']), np.std(data['y']), np.std(data['z'])

  shift_point_file = os.path.join(point_dir, neighbor_panoid + '.ply')
  shift_model = plyfile.PlyData.read(shift_point_file)
  data = shift_model.elements[0].data
  _x = np.copy(data['x'])
  _y = np.copy(data['y'])
  data['x'] = _x*math.cos(diff_heading) - _y*math.sin(diff_heading)
  data['y'] = _x*math.sin(diff_heading) + _y*math.cos(diff_heading)
  # data['x'] = _x*math.cos(math.pi/2) - _y*math.sin(math.pi/2)
  # data['y'] = _x*math.sin(math.pi/2) + _y*math.cos(math.pi/2)
  # data['x'] -= x
  # data['y'] -= y
  data['x'] += x
  data['y'] += y
  point_data.append(data)

  data = np.concatenate(point_data, axis=0)
  el = plyfile.PlyElement.describe(data, 'vertex')
  plyfile.PlyData([el]).write(outfile)


def bat_merge():
  root_dir = '/home/chenjia/hdd/data/human-rights' # earth
  meta_dirs = [
    # os.path.join(root_dir, 'panorama', 'boston', 'meta'),
    # os.path.join(root_dir, 'panorama', 'boston-cross', 'meta'),
    os.path.join(root_dir, 'panorama', 'boston', 'meta'),
    # os.path.join(root_dir, 'panorama', 'maiden', 'meta'),
  ]
  point_cloud_dirs = [
    # os.path.join(root_dir, 'panorama', 'boston', 'point_cloud')
    # os.path.join(root_dir, 'panorama', 'boston-cross', 'point_cloud')
    # os.path.join(root_dir, 'panorama', 'maiden', 'point_cloud')
    # os.path.join(root_dir, 'panorama', 'boston', 'segment_point_cloud')
    # os.path.join(root_dir, 'panorama', 'maiden', 'segment_point_cloud')
    # os.path.join(root_dir, 'panorama', 'boston', 'osm_align_point_cloud')
    os.path.join(root_dir, 'panorama', 'boston', 'highresolution_segment_point_cloud')
  ]
  postfix = '.avg.0.05.ply'

  for meta_dir, point_cloud_dir in zip(meta_dirs, point_cloud_dirs):
    outfile = os.path.join(point_cloud_dir, 'merge.ply')

    names = os.listdir(meta_dir)
    panoids = [name.split('.')[0] for name in names]
    panoids = set(panoids)

    root, panoid2child = gen_bfs_tree(panoids, names, meta_dir)

    out = post_order_traverse(root, panoid2child, point_cloud_dir, postfix=postfix)
    el = plyfile.PlyElement.describe(out, 'vertex')
    plyfile.PlyData([el]).write(outfile)


def bat_merge_in_rectangle():
  top_left = (32.780097, -96.806323)
  lower_right = (32.779210, -96.804195)
  root_dir = '/home/jiac/data/humanrights' # aladdin1
  meta_dirs = [
    # os.path.join(root_dir, 'panorama', 'boston', 'meta'),
    # os.path.join(root_dir, 'panorama', 'maiden', 'meta'),
    os.path.join(root_dir, 'panorama', 'dallas', 'meta'),
  ]
  point_cloud_dirs = [
    # os.path.join(root_dir, 'panorama', 'boston', 'point_cloud'),
    # os.path.join(root_dir, 'panorama', 'maiden', 'point_cloud'),
    os.path.join(root_dir, 'panorama', 'dallas', 'point_cloud'),
  ]

  for meta_dir, point_cloud_dir in zip(meta_dirs, point_cloud_dirs):
    outfile = os.path.join(point_cloud_dir, 'merge.ply')

    names = os.listdir(meta_dir)
    panoids = [name.split('.')[0] for name in names]
    panoids = set(panoids)

    out = []

    visited = set()
    q = deque()
    panoid = names[0].split('.')[0]
    q.append((panoid, 0., 0.))
    visited.add(panoid)
    while len(q) > 0:
      panoid, x, y = q.popleft()

      point_file = os.path.join(point_cloud_dir, panoid + '.ply')
      model = plyfile.PlyData.read(point_file)
      point_data = model.elements[0].data
      point_data['x'] += x
      point_data['y'] += y

      file = os.path.join(meta_dir, panoid + '.pkl')
      with open(file) as f:
        data = cPickle.load(f)
        panomap = data['panomap']
        lat = float(data['lat'])
        lng = float(data['lng'])

      if lat >= lower_right[0] and lat <= top_left[0] and \
          lng >= top_left[1] and lng <= lower_right[1]:
        out.append(point_data)

      for d in panomap:
        neighbor_panoid = d['panoid']
        dx = d['x']
        dy = d['y']
        if neighbor_panoid in panoids and neighbor_panoid not in visited:
          q.append((neighbor_panoid, x + dx, y - dy))
          visited.add(neighbor_panoid)

    out = np.concatenate(out, axis=0)
    el = plyfile.PlyElement.describe(out, 'vertex')
    plyfile.PlyData([el]).write(outfile)


def tst_bfs_crawl():
  # root_dir = '/home/chenjia/hdd/data/human-rights' # earth
  # # root_dir = '/usr0/home/jiac/data/humanrights' # aladdin1
  # # out_dir = os.path.join(root_dir, 'panorama', 'dallas', 'meta')
  # # out_dir = os.path.join(root_dir, 'panorama', 'boston2007', 'meta')
  # out_dir = os.path.join(root_dir, 'panorama', 'boston2015', 'meta')
  # center_gps = (42.3498346,-71.0782973)
  # # panoid = 'IqtiSzTtFdssPpqnti7idg' # 2007
  # panoid = '8-DTPwnD_QwV6EIidPqJCg' # 2015
  # radius = 100

  root_dir = '/home/jiac/data2/humanrights' # gpu9
  # out_dir = os.path.join(root_dir, 'panorama', 'boston', 'meta')
  center_gps = (42.348846, -71.081813)
  radius = 800
  out_dirs = [
    os.path.join(root_dir, 'panorama', 'boston.2007.9', 'meta'),
    os.path.join(root_dir, 'panorama', 'boston.2009.8', 'meta'),
    os.path.join(root_dir, 'panorama', 'boston.2011.6', 'meta'),
    os.path.join(root_dir, 'panorama', 'boston.2011.7', 'meta'),
    os.path.join(root_dir, 'panorama', 'boston.2013.7', 'meta'),
    os.path.join(root_dir, 'panorama', 'boston.2013.9', 'meta'),
    os.path.join(root_dir, 'panorama', 'boston.2014.5', 'meta'),
    os.path.join(root_dir, 'panorama', 'boston.2014.6', 'meta'),
    os.path.join(root_dir, 'panorama', 'boston.2015.6', 'meta'),
  ]
  panoids = [
    'IqtiSzTtFdssPpqnti7idg',
    'FY5pXHpi2K9jqLkRgxwHZA',
    'ADPLf-bN0DGBJ9pG-LW9vg',
    'ABifs604RkWXy8h1hT_A5A',
    'frCjiL72ThNwUDsWGNBnkQ',
    'XpzRYsI-dwBrUHBLAWfi1w',
    'ABifs604RkWXy8h1hT_A5A',
    'HmYGZWec-qWSU-KxFBWsQg',
    '8-DTPwnD_QwV6EIidPqJCg',
  ]

  for panoid, out_dir in zip(panoids, out_dirs):
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
    util.bfs_crawl(center_gps, radius, out_dir, initial_panoid=panoid)


def download_tiles():
  root_dir = '/usr0/home/jiac/data/humanrights'
  meta_dir = os.path.join(root_dir, 'panorama', 'dallas', 'meta')
  img_dir = os.path.join(root_dir, 'panorama', 'dallas', 'img')
  names = os.listdir(meta_dir)

  for name in names:
    meta_file = os.path.join(meta_dir, name)
    with open(meta_file) as f:
      data = cPickle.load(f)
      panoid = data['panoid']

    name, ext = os.path.splitext(name)
    img_file = os.path.join(img_dir, name + '.jpg')

    img = np.zeros((512*4, 512*7, 3), dtype=np.uint8)

    for x in range(7):
      for y in range(4):
        data = StreetView.GetPanoramaTile(panoid, 3, x, y)
        data = array('B', data)
        data = np.array(data, dtype=np.byte)
        img[y*512:(y+1)*512, x*512:(x+1)*512, :] = \
          cv2.imdecode(data, cv2.IMREAD_COLOR)

    img = img[:1664, :3328, :]
    cv2.imwrite(img_file, img)


def crawled_gps():
  root_dir = '/usr0/home/jiac/data/humanrights'
  meta_dir = os.path.join(root_dir, 'panorama', 'dallas', 'meta')
  outfile = os.path.join(root_dir, 'panorama', 'dallas', 'gps.json')

  names = os.listdir(meta_dir)
  gpss = []
  for name in names:
    meta_file = os.path.join(meta_dir, name)
    with open(meta_file) as f:
      data = cPickle.load(f)
      gps = {
        'lat': data['lat'],
        'lng': data['lng'],
      }
      gpss.append(gps)

  with open(outfile, 'w') as fout:
    json.dump(gpss, fout)


def generate_lst_json():
  root_dir = '/home/chenjia/hdd/data/human-rights'
  img_dir = os.path.join(root_dir, 'panorama', 'maiden', 'img')
  out_file = os.path.join(root_dir, 'panorama', 'maiden', 'name.json')

  names = os.listdir(img_dir)
  out = []
  for name in names:
    name, _ = os.path.splitext(name)
    out.append(name)

  with open(out_file, 'w') as fout:
    json.dump(out, fout)


if __name__ == '__main__':
  # tst_panorama()
  # print_heading()
  # bat_panorama()
  # viz_depth_map()
  # bat_depth_map()
  # get_tiles()
  # bat_tiles()
  # viz_depth_on_tiles()
  # bat_viz_depth_on_tiles()
  # viz_plane_on_tiles()
  # bat_viz_plane_on_tiles()
  # color_point_cloud()
  # bat_color_point_cloud()
  # tst_merge()
  # bat_merge()
  # bat_merge_in_rectangle()
  # tst_bfs_crawl()
  # download_tiles()
  # crawled_gps()
  # sphere_proj_to_linear_proj_table()
  perspective_project_img_to_patch()
  # bat_cylindrical_img_to_perspective_imgs()
  # sphere_point_cloud()
  # generate_lst_json()
  # tst_bilinear_upsample_depthmap()
  # bat_bilinear_upsample_depthmap()
