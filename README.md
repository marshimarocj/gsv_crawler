# Google Street View crawler
Given the center GPS coordinate and radius of a circle area, it crawls the Google Street View data, including panorama image, depth map and the transformation panoramas

### StreetView.py
* GetPanoramaMetadata: parse the panaorama meta data given panoid or GPS coordinate
* PanoramaMetadata: parse panorama map and depth map

### panorama.py
* depth_map: decode the depth map to (x, y, d) representation
* point_cloud: generated point cloud given panorama image and depth map
* gen_bfs_tree: breath-first search on panorama graph (each node is one paranoma)
* post_order_traverse: merge paranoramas into one large point cloud
