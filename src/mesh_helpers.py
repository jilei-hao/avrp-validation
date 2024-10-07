import vtk
import numpy as np

def read_polydata(filename):
  # if filename ends iwth .vtk, use vtkPolyDataReader
  if filename.endswith('.vtk'):
    reader = vtk.vtkPolyDataReader()
  # if filename ends iwth .vtp, use vtkXMLPolyDataReader
  elif filename.endswith('.vtp'):
    reader = vtk.vtkXMLPolyDataReader()
  # if filename ends iwth .stl, use vtkSTLReader
  elif filename.endswith('.stl'):
    reader = vtk.vtkSTLReader()
  # if filename ends iwth .obj, use vtkOBJReader
  elif filename.endswith('.obj'):
    reader = vtk.vtkOBJReader()
  else:
    raise ValueError('Unknown file format')
  
  reader.SetFileName(filename)
  reader.Update()
  return reader.GetOutput()

def write_polydata(polydata, filename):
  # if filename ends iwth .vtk, use vtkPolyDataWriter
  if filename.endswith('.vtk'):
    writer = vtk.vtkPolyDataWriter()
  # if filename ends iwth .vtp, use vtkXMLPolyDataWriter
  elif filename.endswith('.vtp'):
    writer = vtk.vtkXMLPolyDataWriter()
  # if filename ends iwth .stl, use vtkSTLWriter
  elif filename.endswith('.stl'):
    writer = vtk.vtkSTLWriter()
    writer.SetFileTypeToBinary()
  # if filename ends iwth .obj, use vtkOBJWriter
  elif filename.endswith('.obj'):
    writer = vtk.vtkOBJWriter()
  else:
    raise ValueError('Unknown file format')
  
  writer.SetInputData(polydata)
  writer.SetFileName(filename)
  writer.Write()

def compute_mass_properties(polydata):
  # compute the volume
  mass = vtk.vtkMassProperties()
  mass.SetInputData(polydata)
  mass.Update()
  volume = mass.GetVolume()
  surface_area = mass.GetSurfaceArea()
  return volume, surface_area


def convert_to_triangle(polydata):
  # convert the polydata to triangles
  triangle = vtk.vtkTriangleFilter()
  triangle.SetInputData(polydata)
  triangle.Update()
  return triangle.GetOutput()

def marching_cubes(image, threshold):
  mc = vtk.vtkMarchingCubes()
  mc.SetInputData(image)
  mc.SetValue(0, threshold)
  mc.Update()
  return mc.GetOutput()

def center_of_mass(polydata):
  com = vtk.vtkCenterOfMass()
  com.SetInputData(polydata)
  com.SetUseScalarsAsWeights(False)
  com.Update()
  return com.GetCenter()

def construct_nifti_sform(m_dir, v_origin, v_spacing):
  # Set the NIFTI/RAS transform
  m_scale = np.diag(v_spacing)
  m_lps_to_ras = np.diag([1.0, 1.0, 1.0])
  m_lps_to_ras[0, 0] = -1
  m_lps_to_ras[1, 1] = -1
  m_ras_matrix = m_lps_to_ras @ m_dir @ m_scale

  # Compute the vector
  v_ras_offset = m_lps_to_ras @ v_origin

  # Create the larger matrix
  vcol = np.ones(4)
  vcol[:3] = v_ras_offset

  m_sform = np.eye(4)
  m_sform[:3, :3] = m_ras_matrix
  m_sform[:4, 3] = vcol
  return m_sform

def construct_vtk_to_nifti_transform(m_dir, v_origin, v_spacing):
  vox2nii = construct_nifti_sform(m_dir, v_origin, v_spacing)
  vtk2vox = np.eye(4)
  for i in range(3):
      vtk2vox[i, i] = 1.0 / v_spacing[i]
      vtk2vox[i, 3] = -v_origin[i] / v_spacing[i]

  return vox2nii @ vtk2vox

def print_methods(obj):
  print([method for method in dir(obj) if callable(getattr(obj, method))])

def get_vtk_to_nifti_transform(itk_img):
  dir = itk_img.GetDirection()

  m_dir = np.zeros((3, 3))
  for i in range(3):
      for j in range(3):
          m_dir[i, j] = dir(i, j)

  v_origin = np.array(itk_img.GetOrigin())
  v_spacing = np.array(itk_img.GetSpacing())

  vtk2nii = construct_vtk_to_nifti_transform(m_dir, v_origin, v_spacing)

  transform = vtk.vtkTransform()
  transform.SetMatrix(vtk2nii.flatten())
  transform.Update()

  return transform

def transform_mesh(mesh, transform):
  transform_filter = vtk.vtkTransformPolyDataFilter()
  transform_filter.SetInputData(mesh)
  transform_filter.SetTransform(transform)
  transform_filter.Update()
  return transform_filter.GetOutput()