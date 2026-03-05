#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "mfem.hpp"

namespace {

std::vector<std::string> SplitCsv(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (const char ch : s) {
    if (ch == ',') {
      if (!cur.empty()) {
        out.push_back(cur);
      }
      cur.clear();
      continue;
    }
    cur.push_back(ch);
  }
  if (!cur.empty()) {
    out.push_back(cur);
  }
  return out;
}

std::unordered_set<int> ParseAttrSet(const std::string& csv) {
  std::unordered_set<int> out;
  const auto tokens = SplitCsv(csv);
  for (const auto& t : tokens) {
    const int v = std::stoi(t);
    if (v <= 0) {
      throw std::runtime_error("heart attrs must be positive");
    }
    out.insert(v);
  }
  if (out.empty()) {
    throw std::runtime_error("heart attrs cannot be empty");
  }
  return out;
}

double ElementCentroidX(const mfem::Mesh& mesh, int e) {
  mfem::Array<int> vtx;
  mesh.GetElementVertices(e, vtx);
  if (vtx.Size() <= 0) {
    return 0.0;
  }
  double x = 0.0;
  for (int j = 0; j < vtx.Size(); ++j) {
    const double* v = mesh.GetVertex(vtx[j]);
    x += v[0];
  }
  return x / static_cast<double>(vtx.Size());
}

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
      std::cerr << "split_niederer_x_regions_av must run with -np 1" << std::endl;
    }
    return 1;
  }

  try {
    std::string in_mesh = "benchmarks/niederer/niederer_benchmark.mesh";
    std::string out_mesh = "benchmarks/niederer/niederer_purkinje_avdelay.mesh";
    std::string heart_attrs_csv = "1";

    // x segmentation:
    // atria    [x0, x1)
    // av-delay [x1, x2)
    // fibrosis [x2, x3)
    // ventricle[x3, x4]
    double x0 = 0.0;
    double x1 = 7.0;
    double x2 = 9.5;
    double x3 = 11.5;
    double x4 = 20.0;

    int attr_atria = 11;
    int attr_av_delay = 14;
    int attr_fibrosis = 13;
    int attr_ventricle = 12;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--in-mesh") {
        in_mesh = NeedValue(i, argc, argv, "--in-mesh");
      } else if (arg == "--out-mesh") {
        out_mesh = NeedValue(i, argc, argv, "--out-mesh");
      } else if (arg == "--heart-attrs") {
        heart_attrs_csv = NeedValue(i, argc, argv, "--heart-attrs");
      } else if (arg == "--x0") {
        x0 = std::stod(NeedValue(i, argc, argv, "--x0"));
      } else if (arg == "--x1") {
        x1 = std::stod(NeedValue(i, argc, argv, "--x1"));
      } else if (arg == "--x2") {
        x2 = std::stod(NeedValue(i, argc, argv, "--x2"));
      } else if (arg == "--x3") {
        x3 = std::stod(NeedValue(i, argc, argv, "--x3"));
      } else if (arg == "--x4") {
        x4 = std::stod(NeedValue(i, argc, argv, "--x4"));
      } else if (arg == "--atria-attr") {
        attr_atria = std::stoi(NeedValue(i, argc, argv, "--atria-attr"));
      } else if (arg == "--av-delay-attr") {
        attr_av_delay = std::stoi(NeedValue(i, argc, argv, "--av-delay-attr"));
      } else if (arg == "--fibrosis-attr") {
        attr_fibrosis = std::stoi(NeedValue(i, argc, argv, "--fibrosis-attr"));
      } else if (arg == "--ventricle-attr") {
        attr_ventricle = std::stoi(NeedValue(i, argc, argv, "--ventricle-attr"));
      } else {
        throw std::runtime_error("Unknown option: " + arg);
      }
    }

    if (!(x0 < x1 && x1 < x2 && x2 < x3 && x3 <= x4)) {
      throw std::runtime_error("Require x0 < x1 < x2 < x3 <= x4");
    }

    std::unordered_set<int> out_attrs = {attr_atria, attr_av_delay, attr_fibrosis, attr_ventricle};
    if (out_attrs.size() != 4 || attr_atria <= 0 || attr_av_delay <= 0 || attr_fibrosis <= 0 ||
        attr_ventricle <= 0) {
      throw std::runtime_error("Output attrs must be positive and distinct");
    }

    const auto heart_attrs = ParseAttrSet(heart_attrs_csv);

    mfem::Mesh mesh(in_mesh.c_str(), 1, 1);
    int n_atria = 0;
    int n_av_delay = 0;
    int n_fibrosis = 0;
    int n_ventricle = 0;
    int n_kept_heart = 0;
    int n_other = 0;

    for (int e = 0; e < mesh.GetNE(); ++e) {
      const int a = mesh.GetAttribute(e);
      if (heart_attrs.find(a) == heart_attrs.end()) {
        ++n_other;
        continue;
      }

      const double x = ElementCentroidX(mesh, e);
      if (x >= x0 && x < x1) {
        mesh.SetAttribute(e, attr_atria);
        ++n_atria;
      } else if (x >= x1 && x < x2) {
        mesh.SetAttribute(e, attr_av_delay);
        ++n_av_delay;
      } else if (x >= x2 && x < x3) {
        mesh.SetAttribute(e, attr_fibrosis);
        ++n_fibrosis;
      } else if (x >= x3 && x <= x4) {
        mesh.SetAttribute(e, attr_ventricle);
        ++n_ventricle;
      } else {
        ++n_kept_heart;
      }
    }

    mesh.SetAttributes();
    const std::filesystem::path out_path(out_mesh);
    std::filesystem::create_directories(out_path.parent_path());
    std::ofstream out(out_path);
    if (!out) {
      throw std::runtime_error("Cannot write output mesh: " + out_mesh);
    }
    mesh.Print(out);

    std::cout << "split_niederer_x_regions_av wrote: " << out_mesh << "\n"
              << "  heart attrs in: " << heart_attrs_csv << "\n"
              << "  x ranges: atria[" << x0 << "," << x1 << "), av_delay[" << x1 << "," << x2
              << "), fibrosis[" << x2 << "," << x3 << "), ventricle[" << x3 << "," << x4
              << "]\n"
              << "  counts: atria=" << n_atria << ", av_delay=" << n_av_delay
              << ", fibrosis=" << n_fibrosis << ", ventricle=" << n_ventricle
              << ", kept_heart=" << n_kept_heart << ", unchanged_other=" << n_other << std::endl;
  } catch (const std::exception& ex) {
    std::cerr << "split_niederer_x_regions_av failed: " << ex.what() << std::endl;
    return 2;
  }

  return 0;
}
