import mesh_helpers as mh
import image_helpers as ih
import sys
import time

# Define ANSI escape codes for colors
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"


def generate_simple_model(fn_image):
  # generate a simple model
  vtk_img = ih.read_image(fn_image)
  itk_img = ih.read_as_itk_image(fn_image)
  bimg = ih.threshold_image(vtk_img, 1, 999, 1.0, 0)
  mesh = mh.marching_cubes(bimg, 0.5)

  # get transform from vtk to nifti
  vtk2nii = mh.get_vtk_to_nifti_transform(itk_img)
  mesh = mh.transform_mesh(mesh, vtk2nii)

  return mesh, vtk_img

def print_failed(msg):
  print(f"{RED}{msg}{RESET}", file=sys.stderr)


def print_passed(msg):
  print(f"{GREEN}{msg}{RESET}")


def validate_result(title, result, tolerance, unit):
  passed = True
  if (abs(result) > tolerance):
    print_failed(f"-- Failed: ({title}) {result}{unit} exceeds tolerance of {tolerance}{unit}")
    passed = False
  else:
    print_passed(f"-- Passed: ({title}) {result}{unit} within tolerance of {tolerance}{unit}")

  return passed


def validate_i2m(image, mesh, tolerance):
  print("\n==== Validating image vs mesh ====\n")

  # compute the surface area of the image
  bimg = ih.threshold_image(image, 1, 999, 1, 0)

  start_time = time.time()
  surface_area_image = ih.compute_surface_area_from_image(bimg, 1)
  end_time = time.time()
  print(f"-- (Performance) image surface area: {round(end_time - start_time, 3)}s")

  start_time = time.time()
  volume_image = ih.compute_volume(image)
  end_time = time.time()
  print(f"-- (Performance) image volume: {round(end_time - start_time, 3)}s")

  # compute the surface area of the mesh
  volume_mesh, surface_area_mesh = mh.compute_mass_properties(mesh)

  # get diff
  volume_diff = volume_mesh - volume_image
  volume_diff_pct = round(volume_diff / volume_image * 100, 3)

  surface_area_diff = surface_area_mesh - surface_area_image
  surface_area_diff_pct = round(surface_area_diff / surface_area_image * 100, 3)

  # print the results
  print(f"-- Volume image: {volume_image}")
  print(f"-- Volume mesh: {volume_mesh}")
  print(f"-- Volume diff: {volume_diff} ({volume_diff_pct}%)")
  i2m_vol_diff_tolerance = tolerance["vol_diff_pct_i2m"]
  passed = validate_result("Volume i2m", volume_diff_pct, i2m_vol_diff_tolerance, "%")

  print(f"-- Surface area image: {surface_area_image}")
  print(f"-- Surface area mesh: {surface_area_mesh}")
  print(f"-- Surface area diff: {surface_area_diff} ({surface_area_diff_pct}%)")
  i2m_area_diff_tolerance = tolerance["area_diff_pct_i2m"]
  passed = validate_result("Surface Area i2m", surface_area_diff_pct, i2m_area_diff_tolerance, "%")

  return passed

def validate_m2m(mesh_gt, mesh_in, tolerance):
  print("\n==== Validating grount truth mesh vs mesh ====\n")

  # compute the volume of the mesh_gt
  volume_mesh_gt, surface_area_mesh_gt = mh.compute_mass_properties(mesh_gt)

  # compute the volume of the mesh_in
  volume_mesh_in, surface_area_mesh_in = mh.compute_mass_properties(mesh_in)

  # compute com
  com_gt = mh.center_of_mass(mesh_gt)
  com_in = mh.center_of_mass(mesh_in)

  # compute distance
  avg_dist = mh.compute_average_distance(mesh_gt, mesh_in)

  # get diff
  volume_diff = volume_mesh_in - volume_mesh_gt
  volume_diff_pct = round(volume_diff / volume_mesh_gt * 100, 3)

  surface_area_diff = surface_area_mesh_in - surface_area_mesh_gt
  surface_area_diff_pct = round(surface_area_diff / surface_area_mesh_gt * 100, 3)

  com_diff = [com_in[0] - com_gt[0], com_in[1] - com_gt[1], com_in[2] - com_gt[2]]
  com_diff_dist = (com_diff[0] ** 2 + com_diff[1] ** 2 + com_diff[2] ** 2) ** 0.5


  # print the results
  print(f"-- Volume ground truth mesh: {volume_mesh_gt}")
  print(f"-- Volume input mesh: {volume_mesh_in}")
  print(f"-- Volume diff: {volume_diff} ({volume_diff_pct}%)")
  m2m_vol_diff_tolerance = tolerance["vol_diff_pct_m2m"]
  passed = validate_result("Volume m2m", volume_diff_pct, m2m_vol_diff_tolerance, "%")

  print(f"-- Surface area ground truth mesh: {surface_area_mesh_gt}")
  print(f"-- Surface area input mesh: {surface_area_mesh_in}")
  print(f"-- Surface area diff: {surface_area_diff} ({surface_area_diff_pct}%)")
  m2m_area_diff_tolerance = tolerance["area_diff_pct_m2m"]
  passed = validate_result("Surface Area m2m", surface_area_diff_pct, m2m_area_diff_tolerance, "%")

  print(f"-- Center of mass ground truth mesh: {com_gt}")
  print(f"-- Center of mass input mesh: {com_in}")
  print(f"-- Center of mass diff: {com_diff_dist}")
  m2m_com_diff_tolerance = tolerance["com_diff_in_mm_m2m"]
  passed = validate_result("Center of Mass m2m", com_diff_dist, m2m_com_diff_tolerance, "mm")

  print(f"-- Average distance: {avg_dist}")
  m2m_avg_dist_tolerance = tolerance["avg_dist_in_mm_m2m"]
  passed = validate_result("Average Distance m2m", avg_dist, m2m_avg_dist_tolerance, "mm")

  return passed


def validate_model(fn_image, fn_mesh, tolerance):
  # generate a ground truth model
  gt_model, vtk_img = generate_simple_model(fn_image)
  mh.write_polydata(gt_model, "/Users/jileihao/data/avrp-data/bavcta005-baseline/output/gt.vtp")
  
  # read the mesh
  mesh = mh.read_polydata(fn_mesh)
  mesh = mh.convert_to_triangle(mesh)

  # compare image vs mesh
  validate_i2m(vtk_img, mesh, tolerance)

  # compare ground truth mesh vs mesh
  validate_m2m(gt_model, mesh, tolerance)


def main():
  # get the image and mesh filenames from the command line
  fn_image = sys.argv[1]
  fn_mesh = sys.argv[2]
  
  # tolerance
  tolerance = {
    "vol_diff_pct_i2m": 1,
    "area_diff_pct_i2m": 1,
    "vol_diff_pct_m2m": 1,
    "area_diff_pct_m2m": 1,
    "com_diff_in_mm_m2m": 2,
    "avg_dist_in_mm_m2m": 1,
    }
  
  # validate the model
  diff = validate_model(fn_image, fn_mesh, tolerance)

if __name__ == '__main__':
  main()