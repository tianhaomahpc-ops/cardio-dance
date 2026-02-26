#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "mfem.hpp"

namespace {

std::string NeedValue(int& i, int argc, char* argv[], const char* key) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("Missing value for ") + key);
  }
  return std::string(argv[++i]);
}

}  // namespace

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);

  if (mfem::Mpi::WorldSize() != 1) {
    if (mfem::Mpi::WorldRank() == 0) {
      std::cerr << "generate_mesh_attr_gf must run with -np 1" << std::endl;
    }
    return 1;
  }

  try {
    std::string mesh_path = "benchmarks/wholebody/wholebody_conforming.mesh";
    std::string out_gf_path = "benchmarks/wholebody/wholebody_attr.gf";

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--mesh") {
        mesh_path = NeedValue(i, argc, argv, "--mesh");
      } else if (arg == "--out-gf") {
        out_gf_path = NeedValue(i, argc, argv, "--out-gf");
      } else {
        throw std::runtime_error("Unknown option: " + arg);
      }
    }

    mfem::Mesh mesh(mesh_path.c_str(), 1, 1);
    const int dim = mesh.Dimension();
    const int ne = mesh.GetNE();
    if (ne <= 0) {
      throw std::runtime_error("Mesh has no elements: " + mesh_path);
    }

    mfem::L2_FECollection fec(0, dim);
    mfem::FiniteElementSpace fes(&mesh, &fec, 1);
    mfem::GridFunction attr_gf(&fes);
    attr_gf = 0.0;

    mfem::Array<int> vdofs;
    for (int e = 0; e < ne; ++e) {
      fes.GetElementVDofs(e, vdofs);
      const double attr = static_cast<double>(mesh.GetAttribute(e));
      for (int j = 0; j < vdofs.Size(); ++j) {
        const int dof = std::abs(vdofs[j]);
        attr_gf(dof) = attr;
      }
    }

    const std::filesystem::path out_path(out_gf_path);
    std::filesystem::create_directories(out_path.parent_path());
    std::ofstream out(out_path);
    if (!out) {
      throw std::runtime_error("Cannot write GridFunction: " + out_gf_path);
    }
    attr_gf.Save(out);

    std::cout << "Generated attribute GridFunction: " << out_gf_path << "\n"
              << "  mesh: " << mesh_path << "\n"
              << "  elements: " << ne << "\n"
              << "  scalar space: L2 order 0 (DG0)" << std::endl;
  } catch (const std::exception& ex) {
    std::cerr << "generate_mesh_attr_gf failed: " << ex.what() << std::endl;
    return 2;
  }

  return 0;
}
