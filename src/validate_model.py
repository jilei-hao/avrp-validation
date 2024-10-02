import mesh_helpers as mh
import image_helpers as ih
import sys
import time

def validate_model(fn_image, fn_mesh):
  # read the image
  image = ih.read_image(fn_image)
  
  # compute the volume of the image
  start_time = time.time()
  volume_image = ih.compute_volume(image)
  end_time = time.time()
  print(f"Time spent computing volume of image: {end_time - start_time} seconds")

  # compute the surface area of the image
  bimg = ih.threshold_image(image, 1, 999, 1, 0)
  start_time = time.time()
  surface_area_image = ih.compute_surface_area_from_image(bimg, 1)
  end_time = time.time()
  print(f"Time spent computing surface area of image: {end_time - start_time} seconds")

  
  # read the mesh
  mesh = mh.read_polydata(fn_mesh)
  mesh = mh.convert_to_triangle(mesh)

  # compute the volume of the mesh
  volume_mesh, surface_area_mesh = mh.compute_mass_properties(mesh)

  print("Volume of image:", volume_image)
  print("Surface area of image:", surface_area_image)
  print("Volume of mesh:", volume_mesh)
  print("Surface area of mesh:", surface_area_mesh)

  # return the difference between the two volumes
  return (volume_mesh - volume_image) // volume_image


def main():
  # get the image and mesh filenames from the command line
  fn_image = sys.argv[1]
  fn_mesh = sys.argv[2]
  
  # tolerance
  tolerance = {
    "vol_diff_pct": 1,
    "cent_of_mass_pct": 1
    }
  
  # validate the model
  diff = validate_model(fn_image, fn_mesh)

if __name__ == '__main__':
  main()