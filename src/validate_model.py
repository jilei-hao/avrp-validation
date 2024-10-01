import mesh_helpers as mh
import image_helpers as ih
import sys

def validate_model(fn_image, fn_mesh):
  # read the image
  image = ih.read_image(fn_image)
  # compute the volume of the image
  volume_image = ih.compute_volume(image)
  
  # read the mesh
  mesh = mh.read_polydata(fn_mesh)
  mesh = mh.convert_to_triangle(mesh)

  # compute the volume of the mesh
  volume_mesh = mh.compute_volume(mesh)

  print("Volume of image:", volume_image)
  print("Volume of mesh:", volume_mesh)

  
  # return the difference between the two volumes
  return volume_mesh - volume_image


def main():
  # get the image and mesh filenames from the command line
  fn_image = sys.argv[1]
  fn_mesh = sys.argv[2]
  
  # validate the model
  diff = validate_model(fn_image, fn_mesh)
  print('Volume difference:', diff)


if __name__ == '__main__':
  main()