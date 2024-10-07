import image_helpers as ih
import mesh_helpers as mh

def main():
  # read the image
  image = ih.read_image('/Users/jileihao/data/avrp-data/bavcta005-baseline/srd.nii.gz')
  print(image)

  # read the mesh
  bimg = ih.threshold_image(image, 1, 999, 1.0, 0.0)
  mesh = mh.marching_cubes(bimg, 1.0)
  print(mesh)
  return 1

if __name__ == '__main__':
  main()