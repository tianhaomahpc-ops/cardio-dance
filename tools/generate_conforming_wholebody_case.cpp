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

bool InBox(const Box3& b, const double x, const double y, const double z) {
  const double tol = 1e-12;
  return x >= b.xmin - tol && x <= b.xmax + tol && y >= b.ymin - tol && y <= b.ymax + tol &&
         z >= b.zmin - tol && z <= b.zmax + tol;
}

}  // namespace

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);

  if (mfem::Mpi::WorldSize() != 1) {
    if (mfem::Mpi::WorldRank() == 0) {
      std::cerr << "generate_conforming_wholebody_case must run with -np 1" << std::endl;
    }
    return 1;
  }

  try {
    std::string heart_mesh_path = "benchmarks/niederer/niederer_benchmark.mesh";
    std::string out_mesh_path = "benchmarks/wholebody/wholebody_conforming.mesh";
    double padding_mm = 8.0;
    double h_mm = 1.0;
    int heart_attr = 1;
    int torso_attr = 2;

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
      } else if (arg == "--heart-attr") {
        heart_attr = std::stoi(need_value("--heart-attr"));
      } else if (arg == "--torso-attr") {
        torso_attr = std::stoi(need_value("--torso-attr"));
      } else {
        throw std::runtime_error("Unknown option: " + arg);
      }
    }

    if (padding_mm <= 0.0 || h_mm <= 0.0) {
      throw std::runtime_error("padding-mm and h-mm must be > 0");
    }
    if (heart_attr <= 0 || torso_attr <= 0 || heart_attr == torso_attr) {
      throw std::runtime_error("heart-attr and torso-attr must be positive and distinct");
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

    mfem::Mesh wholebody =
        mfem::Mesh::MakeCartesian3D(nx, ny, nz, mfem::Element::TETRAHEDRON, lx, ly, lz, false);
    for (int i = 0; i < wholebody.GetNV(); ++i) {
      double* v = wholebody.GetVertex(i);
      v[0] += xmin;
      v[1] += ymin;
      v[2] += zmin;
    }

    int heart_elem_count = 0;
    for (int e = 0; e < wholebody.GetNE(); ++e) {
      mfem::Array<int> vtx;
      wholebody.GetElementVertices(e, vtx);
      double cx = 0.0;
      double cy = 0.0;
      double cz = 0.0;
      for (int j = 0; j < vtx.Size(); ++j) {
        const double* v = wholebody.GetVertex(vtx[j]);
        cx += v[0];
        cy += v[1];
        cz += v[2];
      }
      cx /= static_cast<double>(vtx.Size());
      cy /= static_cast<double>(vtx.Size());
      cz /= static_cast<double>(vtx.Size());
      if (InBox(hb, cx, cy, cz)) {
        wholebody.SetAttribute(e, heart_attr);
        ++heart_elem_count;
      } else {
        wholebody.SetAttribute(e, torso_attr);
      }
    }
    wholebody.SetAttributes();
    if (heart_elem_count == 0) {
      throw std::runtime_error("No heart elements were tagged in wholebody mesh");
    }

    const std::filesystem::path out_path(out_mesh_path);
    std::filesystem::create_directories(out_path.parent_path());
    std::ofstream out(out_path);
    if (!out) {
      throw std::runtime_error("Cannot write mesh file: " + out_mesh_path);
    }
    wholebody.Print(out);

    std::cout << "Generated conforming wholebody mesh: " << out_mesh_path << "\n"
              << "  attrs: heart=" << heart_attr << ", torso=" << torso_attr << "\n"
              << "  elements: heart=" << heart_elem_count
              << ", total=" << wholebody.GetNE() << std::endl;
  } catch (const std::exception& ex) {
    std::cerr << "generate_conforming_wholebody_case failed: " << ex.what() << std::endl;
    return 2;
  }

  return 0;
}

