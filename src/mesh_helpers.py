import vtk

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

def compute_volume(polydata):
  # compute the volume
  mass = vtk.vtkMassProperties()
  mass.SetInputData(polydata)
  mass.Update()
  volume = mass.GetVolume()
  return volume

def convert_to_triangle(polydata):
  # convert the polydata to triangles
  triangle = vtk.vtkTriangleFilter()
  triangle.SetInputData(polydata)
  triangle.Update()
  return triangle.GetOutput()