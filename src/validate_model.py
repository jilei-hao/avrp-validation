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

  # vtk2nii transform assumes identity direction matrix
  vtk_img.SetDirectionMatrix([1, 0, 0, 0, 1, 0, 0, 0, 1]) 

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


def validate_i2m_volume(image, volume_mesh, tolerance):
  # compute the volume of the image
  start_time = time.time()
  volume_image = ih.compute_volume(image)
  end_time = time.time()
  print(f"-- (Performance) image volume: {round(end_time - start_time, 3)}s")

  # get diff
  volume_diff = volume_mesh - volume_image
  volume_diff_pct = round(volume_diff / volume_image * 100, 3)

  # print the results
  print(f"-- Volume image: {volume_image}")
  print(f"-- Volume mesh: {volume_mesh}")
  print(f"-- Volume diff: {volume_diff} ({volume_diff_pct}%)")
  i2m_vol_diff_tolerance = tolerance["vol_diff_pct_i2m"]
  passed = validate_result("Volume i2m", volume_diff_pct, i2m_vol_diff_tolerance, "%")
  
  return passed, volume_image, volume_mesh, volume_diff, volume_diff_pct


def validate_i2m_surface_area(image, surface_area_mesh, tolerance):
  # compute the surface area of the image
  bimg = ih.threshold_image(image, 1, 999, 1, 0)

  start_time = time.time()
  surface_area_image = ih.compute_surface_area_from_image(bimg, 1)
  end_time = time.time()
  print(f"-- (Performance) image surface area: {round(end_time - start_time, 3)}s")

  # compute diff
  surface_area_diff = surface_area_mesh - surface_area_image
  surface_area_diff_pct = round(surface_area_diff / surface_area_image * 100, 3)

  print(f"-- Surface area image: {surface_area_image}")
  print(f"-- Surface area mesh: {surface_area_mesh}")
  print(f"-- Surface area diff: {surface_area_diff} ({surface_area_diff_pct}%)")
  i2m_area_diff_tolerance = tolerance["area_diff_pct_i2m"]
  passed = validate_result("Surface Area i2m", surface_area_diff_pct, i2m_area_diff_tolerance, "%")

  return passed, surface_area_image, surface_area_mesh, surface_area_diff, surface_area_diff_pct

def validate_i2m(image, mesh, tolerance):
  print("\n==== Validating image vs mesh ====\n")

  # compute the surface area of the mesh
  volume_mesh, surface_area_mesh = mh.compute_mass_properties(mesh)

  # validate volume
  passed_1, volume_image, volume_mesh, volume_diff, volume_diff_pct \
    = validate_i2m_volume(image, volume_mesh, tolerance)
  
  # validate surface area
  passed_2, surface_area_image, surface_area_mesh, surface_area_diff, surface_area_diff_pct \
    = validate_i2m_surface_area(image, surface_area_mesh, tolerance)

  results = {
    "volume_image_mm3": volume_image,
    "volume_mesh_mm3": volume_mesh,
    "volume_diff_mm3": volume_diff,
    "volume_diff_pct": volume_diff_pct,
    "surface_area_image_mm2": surface_area_image,
    "surface_area_mesh_mm2": surface_area_mesh,
    "surface_area_diff_mm2": surface_area_diff,
    "surface_area_diff_pct": surface_area_diff_pct
  }

  passed = passed_1 and passed_2

  return passed, results

def validate_m2m_volume(volume_gt, volume_in, tolerance):
  # get diff
  volume_diff = volume_in - volume_gt
  volume_diff_pct = round(volume_diff / volume_gt * 100, 3)

  # print the results
  print(f"-- Volume ground truth mesh: {volume_gt}")
  print(f"-- Volume input mesh: {volume_in}")
  print(f"-- Volume diff: {volume_diff} ({volume_diff_pct}%)")
  m2m_vol_diff_tolerance = tolerance["vol_diff_pct_m2m"]
  passed = validate_result("Volume m2m", volume_diff_pct, m2m_vol_diff_tolerance, "%")

  return passed, volume_diff, volume_diff_pct


def validate_m2m_surface_area(surface_area_gt, surface_area_in, tolerance):
  # get diff
  surface_area_diff = surface_area_in - surface_area_gt
  surface_area_diff_pct = round(surface_area_diff / surface_area_gt * 100, 3)

  # print the results
  print(f"-- Surface area ground truth mesh: {surface_area_gt}")
  print(f"-- Surface area input mesh: {surface_area_in}")
  print(f"-- Surface area diff: {surface_area_diff} ({surface_area_diff_pct}%)")
  m2m_area_diff_tolerance = tolerance["area_diff_pct_m2m"]
  passed = validate_result("Surface Area m2m", surface_area_diff_pct, m2m_area_diff_tolerance, "%")

  return passed, surface_area_diff, surface_area_diff_pct


def validate_m2m_com(mesh_gt, mesh_in, tolerance):
  # compute com
  com_gt = mh.center_of_mass(mesh_gt)
  com_in = mh.center_of_mass(mesh_in)

  com_diff = [com_in[0] - com_gt[0], com_in[1] - com_gt[1], com_in[2] - com_gt[2]]
  com_diff_dist = (com_diff[0] ** 2 + com_diff[1] ** 2 + com_diff[2] ** 2) ** 0.5

  # print the results
  print(f"-- Center of mass ground truth mesh: {com_gt}")
  print(f"-- Center of mass input mesh: {com_in}")
  print(f"-- Center of mass diff: {com_diff_dist}")
  m2m_com_diff_tolerance = tolerance["com_diff_in_mm_m2m"]

  passed = validate_result("Center of Mass m2m", com_diff_dist, m2m_com_diff_tolerance, "mm")

  return passed, com_gt, com_in, com_diff_dist

def validate_m2m_distance(mesh_gt, mesh_in, tolerance):
  # compute distance
  avg_dist = mh.compute_average_distance(mesh_gt, mesh_in)

  print(f"-- Average distance: {avg_dist}")
  m2m_avg_dist_tolerance = tolerance["avg_dist_in_mm_m2m"]
  passed = validate_result("Average Distance m2m", avg_dist, m2m_avg_dist_tolerance, "mm")

  return passed, avg_dist


def validate_m2m(mesh_gt, mesh_in, tolerance):
  print("\n==== Validating ground truth mesh vs mesh ====\n")

  # compute the volume of the meshes
  volume_mesh_gt, surface_area_mesh_gt = mh.compute_mass_properties(mesh_gt)
  volume_mesh_in, surface_area_mesh_in = mh.compute_mass_properties(mesh_in)

  # validate volume
  passed_1, volume_diff, volume_diff_pct = validate_m2m_volume(volume_mesh_gt, volume_mesh_in, tolerance)

  # validate surface area
  passed_2, surface_area_diff, surface_area_diff_pct = validate_m2m_surface_area(surface_area_mesh_gt, surface_area_mesh_in, tolerance)

  # validate center of mass
  passed_3, com_gt, com_in, com_diff_dist = validate_m2m_com(mesh_gt, mesh_in, tolerance)

  # validate distance
  passed_4, avg_dist = validate_m2m_distance(mesh_gt, mesh_in, tolerance)

  results = {
    "volume_gt_mm3": volume_mesh_gt,
    "volume_in_mm3": volume_mesh_in,
    "volume_diff_mm3": volume_diff,
    "volume_diff_pct": volume_diff_pct,
    "surface_area_gt_mm2": surface_area_mesh_gt,
    "surface_area_in_mm2": surface_area_mesh_in,
    "surface_area_diff_mm2": surface_area_diff,
    "surface_area_diff_pct": surface_area_diff_pct,
    "com_gt": com_gt,
    "com_in": com_in,
    "com_diff_mm": com_diff_dist,
    "avg_dist_mm": avg_dist
  }

  passed = passed_1 and passed_2 and passed_3 and passed_4
  
  return passed, results


def validate_model(fn_image, fn_mesh, tolerance):
  # generate a ground truth model
  gt_model, vtk_img = generate_simple_model(fn_image)
  # mh.write_polydata(gt_model, "/Users/jileihao/data/avrp-data/bavcta005-baseline/output/gt.vtp")
  
  # read the mesh
  mesh = mh.read_polydata(fn_mesh)
  mesh = mh.convert_to_triangle(mesh)

  print(f"-- Loaded mesh: {fn_mesh}")
  print(mesh.GetBounds())

  # compare image vs mesh
  passed_1, results_i2m = validate_i2m(vtk_img, mesh, tolerance)

  # compare ground truth mesh vs mesh
  passed_2, results_m2m = validate_m2m(gt_model, mesh, tolerance)

  # concatenate results
  results = {**results_i2m, **results_m2m}

  passed = passed_1 and passed_2

  return passed, results

def get_tolerance():
  return {
    "vol_diff_pct_i2m": 1,
    "area_diff_pct_i2m": 1,
    "vol_diff_pct_m2m": 1,
    "area_diff_pct_m2m": 1,
    "com_diff_in_mm_m2m": 2,
    "avg_dist_in_mm_m2m": 1,
  }


def main():
  # get the image and mesh filenames from the command line
  fn_image = sys.argv[1]
  fn_mesh = sys.argv[2]
  
  # tolerance
  tolerance = get_tolerance()
  
  # validate the model
  passed, results = validate_model(fn_image, fn_mesh, tolerance)

  if passed:
    print_passed("\n\n-- Model validation passed\n\n")
  else:
    print_failed("\n\n-- Model validation failed\n\n")

if __name__ == '__main__':
  main()