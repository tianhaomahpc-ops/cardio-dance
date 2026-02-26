#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "mfem.hpp"

namespace {

struct Box3 {
  double xmin = 0.0;
  double xmax = 0.0;
  double ymin = 0.0;
  double ymax = 0.0;
  double zmin = 0.0;
  double zmax = 0.0;
};

Box3 ComputeBoundingBox(const mfem::Mesh& mesh) {
  Box3 b;
  b.xmin = b.ymin = b.zmin = std::numeric_limits<double>::infinity();
  b.xmax = b.ymax = b.zmax = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < mesh.GetNV(); ++i) {
    const double* v = mesh.GetVertex(i);
    b.xmin = std::min(b.xmin, v[0]);
    b.xmax = std::max(b.xmax, v[0]);
    b.ymin = std::min(b.ymin, v[1]);
    b.ymax = std::max(b.ymax, v[1]);
    b.zmin = std::min(b.zmin, v[2]);
    b.zmax = std::max(b.zmax, v[2]);
  }
  return b;
}

}  // namespace

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);

  if (mfem::Mpi::WorldSize() != 1) {
    if (mfem::Mpi::WorldRank() == 0) {
      std::cerr << "generate_torso_box must run with -np 1" << std::endl;
    }
    return 1;
  }

  try {
    std::string heart_mesh_path = "benchmarks/niederer/niederer_benchmark.mesh";
    std::string out_mesh_path = "benchmarks/torso_box/torso_box.mesh";
    double padding_mm = 8.0;
    double h_mm = 1.0;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto need_value = [&](const char* key) -> std::string {
        if (i + 1 >= argc) {
          throw std::runtime_error(std::string("Missing value for ") + key);
        }
        return std::string(argv[++i]);
      };

      if (arg == "--heart-mesh") {
        heart_mesh_path = need_value("--heart-mesh");
      } else if (arg == "--out-mesh") {
        out_mesh_path = need_value("--out-mesh");
      } else if (arg == "--padding-mm") {
        padding_mm = std::stod(need_value("--padding-mm"));
      } else if (arg == "--h-mm") {
        h_mm = std::stod(need_value("--h-mm"));
      } else {
        throw std::runtime_error("Unknown option: " + arg);
      }
    }

    if (padding_mm <= 0.0 || h_mm <= 0.0) {
      throw std::runtime_error("padding-mm and h-mm must be > 0");
    }

    mfem::Mesh heart_mesh(heart_mesh_path.c_str(), 1, 1);
    const Box3 hb = ComputeBoundingBox(heart_mesh);

    const double xmin = hb.xmin - padding_mm;
    const double xmax = hb.xmax + padding_mm;
    const double ymin = hb.ymin - padding_mm;
    const double ymax = hb.ymax + padding_mm;
    const double zmin = hb.zmin - padding_mm;
    const double zmax = hb.zmax + padding_mm;

    const double lx = xmax - xmin;
    const double ly = ymax - ymin;
    const double lz = zmax - zmin;
    const int nx = std::max(4, static_cast<int>(std::ceil(lx / h_mm)));
    const int ny = std::max(4, static_cast<int>(std::ceil(ly / h_mm)));
    const int nz = std::max(4, static_cast<int>(std::ceil(lz / h_mm)));

    mfem::Mesh torso_mesh = mfem::Mesh::MakeCartesian3D(
        nx, ny, nz, mfem::Element::TETRAHEDRON, lx, ly, lz, false);
    for (int i = 0; i < torso_mesh.GetNV(); ++i) {
      double* v = torso_mesh.GetVertex(i);
      v[0] += xmin;
      v[1] += ymin;
      v[2] += zmin;
    }

    std::filesystem::path out_path(out_mesh_path);
    std::filesystem::create_directories(out_path.parent_path());
    std::ofstream out(out_path);
    if (!out) {
      throw std::runtime_error("Cannot write mesh file: " + out_mesh_path);
    }
    torso_mesh.Print(out);

    std::cout << "Generated torso box mesh: " << out_mesh_path << "\n"
              << "  nx,ny,nz = " << nx << "," << ny << "," << nz << "\n"
              << "  bbox(mm) = [" << xmin << "," << xmax << "] x [" << ymin << "," << ymax
              << "] x [" << zmin << "," << zmax << "]" << std::endl;
  } catch (const std::exception& ex) {
    std::cerr << "generate_torso_box failed: " << ex.what() << std::endl;
    return 2;
  }

  return 0;
}
