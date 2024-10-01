import vtk
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from vtk.util import numpy_support

NUMBER_OF_JOBS = 4

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
  voxels = vtk.util.numpy_support.vtk_to_numpy(image.GetPointData().GetScalars())
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