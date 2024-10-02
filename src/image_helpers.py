import vtk
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from vtk.util import numpy_support

NUMBER_OF_JOBS = 8

def read_image(filename):
  # if filename ends iwth .vti, use vtkXMLImageDataReader
  if filename.endswith('.vti'):
    reader = vtk.vtkXMLImageDataReader()
  # if filename ends iwth .mha, use vtkMetaImageReader
  elif filename.endswith('.mha'):
    reader = vtk.vtkMetaImageReader()
  # if filename ends iwth .nii, use vtkNIFTIImageReader
  elif filename.endswith('.nii') or filename.endswith('.nii.gz'):
    reader = vtk.vtkNIFTIImageReader()
  else:
    raise ValueError('Unknown file format')
  
  reader.SetFileName(filename)
  reader.Update()
  return reader.GetOutput()


def compute_volume_chunk(voxels, start, end, volvox):
  volume = 0
  for i in range(start, end):
    if voxels[i] > 0:
      volume += volvox
  return volume

def compute_volume(image):
  # get the spacing of the image
  spacing = image.GetSpacing()
  dims = image.GetDimensions()
  print("-- compute_volume: spacing:", spacing, "dims:", dims)
  
  # compute the volume of non-zero voxels
  volume = 0
  volvox = spacing[0] * spacing[1] * spacing[2]
  
  # Convert vtkUnsignedShortArray to NumPy array
  voxels = numpy_support.vtk_to_numpy(image.GetPointData().GetScalars())
  num_voxels = len(voxels)
  
  # Divide the work into 8 chunks
  chunk_size = num_voxels // NUMBER_OF_JOBS
  futures = []
  
  with ProcessPoolExecutor(max_workers=NUMBER_OF_JOBS) as executor:
    for i in range(NUMBER_OF_JOBS):
      start = i * chunk_size
      end = num_voxels if i == NUMBER_OF_JOBS - 1 else (i + 1) * chunk_size
      futures.append(executor.submit(compute_volume_chunk, voxels, start, end, volvox))
  
    for future in futures:
      volume += future.result()
  
  return volume

def compute_surface_area_chunk(voxels, dims, spacing, xRange, yRange, zRange, label):
  surface_cnt_XY = 0
  surface_cnt_XZ = 0
  surface_cnt_YZ = 0
  
  for z in range(zRange[0], zRange[1]):
    for y in range(yRange[0], yRange[1]):
      for x in range(xRange[0], xRange[1]):
        if voxels[x, y, z] == label:
          # Check the six neighbors
          if x == 0 or voxels[x - 1, y, z] != label:
            surface_cnt_YZ += 1
          if x == dims[0] - 1 or voxels[x + 1, y, z] != label:
            surface_cnt_YZ += 1
          if y == 0 or voxels[x, y - 1, z] != label:
            surface_cnt_XZ += 1
          if y == dims[1] - 1 or voxels[x, y + 1, z] != label:
            surface_cnt_XZ += 1
          if z == 0 or voxels[x, y, z - 1] != label:
            surface_cnt_XY += 1
          if z == dims[2] - 1 or voxels[x, y, z + 1] != label:
            surface_cnt_XY += 1

  dx, dy, dz = spacing
  area_XY = surface_cnt_XY * dx * dy
  area_XZ = surface_cnt_XZ * dx * dz
  area_YZ = surface_cnt_YZ * dy * dz

  return area_XY + area_XZ + area_YZ

def compute_surface_area_from_image(image, label):
  # Get the dimensions of the image
  dims = image.GetDimensions()
  
  # Convert vtkUnsignedShortArray to NumPy array
  voxels = numpy_support.vtk_to_numpy(image.GetPointData().GetScalars())
  voxels = voxels.reshape(dims, order='F')  # Reshape to 3D array with Fortran order
  
  futures = []

  xMid = (dims[0] + 1) // 2
  yMid = (dims[1] + 1) // 2
  zMid = (dims[2] + 1) // 2

  xRanges = [[0, xMid], [xMid, dims[0]]]
  yRanges = [[0, yMid], [yMid, dims[1]]]
  zRanges = [[0, zMid], [zMid, dims[2]]]

  spacing = image.GetSpacing()
  
  with ProcessPoolExecutor(max_workers=NUMBER_OF_JOBS) as executor:

    for xRange in xRanges:
      for yRange in yRanges:
        for zRange in zRanges:
          futures.append(executor.submit(compute_surface_area_chunk, voxels, dims, spacing, xRange, yRange, zRange, label))

    surface_area = sum(future.result() for future in futures)
  
  return surface_area

def threshold_image (image, lower, upper, valueIn, valueOut):
  # Create a vtkThreshold filter
  threshold = vtk.vtkImageThreshold()
  threshold.SetInputData(image)
  threshold.ThresholdBetween(lower, upper)
  threshold.SetInValue(valueIn)
  threshold.SetOutValue(valueOut)
  threshold.Update()
  
  return threshold.GetOutput()