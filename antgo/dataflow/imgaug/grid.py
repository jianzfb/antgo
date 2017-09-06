# encoding=utf-8
# @Time    : 17-6-8
# @File    : grid.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.core import *
from antgo.utils.bboxes import *
import copy


class Grid(Node):
  def __init__(self, inputs, grid=(), cell_size=0.5, obj_overlap=0.5):
    super(Grid, self).__init__(name=None, action=self.action, inputs=inputs)
    self._grid = grid
    self._cell_size = cell_size
    self._obj_overlap = obj_overlap

  def _grid_crop(self, image_size, grid, cell_ratio):
    width, height = image_size[1::-1]
    square_cell_size = np.minimum(width * cell_ratio, height * cell_ratio)
    grid_hori_num = grid[0]
    grid_vert_num = grid[1]

    hori_step = 0
    if grid_hori_num >= 2:
      hori_step = int(float(width - square_cell_size) / float(grid_hori_num - 1))
    vert_step = 0
    if grid_vert_num >= 2:
      vert_step = int(float(height - square_cell_size) / float(grid_vert_num - 1))

    crops = np.zeros((grid_hori_num * grid_vert_num, 4), np.int32)
    for grid_hori_index in range(grid_hori_num):
      for grid_vert_index in range(grid_vert_num):
        crop_index = grid_hori_index * grid_vert_num + grid_vert_index
        crops[crop_index, 0] = hori_step * grid_hori_index
        crops[crop_index, 1] = vert_step * grid_vert_index

        crops[crop_index, 2] = crops[crop_index, 0] + square_cell_size
        crops[crop_index, 3] = crops[crop_index, 1] + square_cell_size

    return crops

  def action(self, *args, **kwargs):
    assert(len(args) == 1)
    data, annotation = args[0] if len(args[0]) == 2 else (args[0], {})
    assert(type(data) == np.ndarray)
    crops = self._grid_crop(data.shape, self._grid, self._cell_size)

    patches = []
    for xy in np.split(crops, crops.shape[0], axis=0):
      patch_x0, patch_y0, patch_x1, patch_y1 = xy[0, :]
      # crop
      patch = data[int(patch_y0):int(patch_y1), int(patch_x0):int(patch_x1)]

      annotation_cpy = copy.deepcopy(annotation)
      if 'bbox' in annotation_cpy:
        bboxes = annotation_cpy['bbox']
        remained_bboxes, remained_bboxes_ind = \
          bboxes_filter_overlap(np.array((patch_x0,patch_y0,patch_x1,patch_y1)), bboxes, self._obj_overlap)

        remained_bboxes = bboxes_translate(np.array((patch_x0,patch_y0,patch_x1,patch_y1)), remained_bboxes)
        annotation_cpy['bbox'] = remained_bboxes

        if 'category_id' in annotation_cpy:
          annotation_cpy['category_id'] = annotation_cpy['category_id'][remained_bboxes_ind]
        if 'category' in annotation_cpy:
          annotation_cpy['category'] = [annotation_cpy['category'][i] for i in remained_bboxes_ind]
        if 'area' in annotation_cpy:
          annotation_cpy['area'] = annotation_cpy['area'][remained_bboxes_ind]

        if 'segmentation' in annotation_cpy:
          annotation_cpy['segmentation'] = \
            [annotation_cpy['segmentation'][i][int(patch_y0):int(patch_y1), int(patch_x0):int(patch_x1)]
             for i in remained_bboxes_ind]

      annotation_cpy['cell'] = xy[0, :]
      annotation_cpy['info'] = patch.shape
      annotation_cpy['image'] = data
      patches.append((patch, annotation_cpy))
    return patches


def sample_grid_integrate(cells, bboxes):
  '''
  :param cells: [[x0,y0,x1,y1],[],[],...]
  :param bboxes: grid_cells x bboxes_num x 4
  :return:
  '''
  # translate all bboxes in every grid cell
  cells_num = cells.shape[0]
  for cell_index in range(cells_num):
    cell_offset_x = cells[cell_index,0]
    cell_offset_y = cells[cell_index,1]
    bboxes[cell_index, :, :] = bboxes[cell_index, :, :] + \
                               np.array((cell_offset_x, cell_offset_y, cell_offset_x, cell_offset_y))

  return bboxes.reshape(-1, 4)