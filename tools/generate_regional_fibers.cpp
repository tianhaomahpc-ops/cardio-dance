#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mfem.hpp"

namespace {

constexpr double kEps = 1e-12;
constexpr double kPi = 3.14159265358979323846;

using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;

struct Mat3 {
  std::array<std::array<double, 3>, 3> m{};
};

Vec3 MakeVec3(const double x, const double y, const double z) { return {x, y, z}; }

Vec3 operator+(const Vec3& a, const Vec3& b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vec3 operator-(const Vec3& a, const Vec3& b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3 operator*(const Vec3& a, const double s) { return {a[0] * s, a[1] * s, a[2] * s}; }

Vec3 operator*(const double s, const Vec3& a) { return a * s; }

Vec3 operator/(const Vec3& a, const double s) {
  if (std::abs(s) < kEps) {
    return {0.0, 0.0, 0.0};
  }
  return {a[0] / s, a[1] / s, a[2] / s};
}

void AddInPlace(Vec3& a, const Vec3& b) {
  a[0] += b[0];
  a[1] += b[1];
  a[2] += b[2];
}

double Dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

Vec3 Cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

double Norm(const Vec3& a) { return std::sqrt(Dot(a, a)); }

Vec3 Normalize(const Vec3& a, const Vec3& fallback = {1.0, 0.0, 0.0}) {
  const double n = Norm(a);
  if (n <= kEps) {
    return fallback;
  }
  return a / n;
}

bool IsNonZero(const Vec3& a) { return Dot(a, a) > kEps; }

Vec3 OrthogonalUnit(const Vec3& n) {
  Vec3 ref = std::abs(n[0]) < 0.9 ? Vec3{1.0, 0.0, 0.0} : Vec3{0.0, 1.0, 0.0};
  Vec3 t = Cross(n, ref);
  if (Norm(t) <= kEps) {
    ref = {0.0, 0.0, 1.0};
    t = Cross(n, ref);
  }
  return Normalize(t, {1.0, 0.0, 0.0});
}

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string Trim(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

std::vector<std::string> Split(const std::string& s, const char delim) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    item = Trim(item);
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

std::vector<int> ParseIntList(const std::string& text) {
  std::vector<int> values;
  for (const auto& token : Split(text, ',')) {
    values.push_back(std::stoi(token));
  }
  return values;
}

std::array<double, 3> ParseTriple(const std::string& text) {
  const auto tokens = Split(text, ',');
  if (tokens.size() != 3) {
    throw std::runtime_error("Expected three comma-separated values, got: " + text);
  }
  return {std::stod(tokens[0]), std::stod(tokens[1]), std::stod(tokens[2])};
}

class KdTree3 {
 public:
  KdTree3(const std::vector<Vec3>& points, std::vector<int> point_ids)
      : points_(points), ids_(std::move(point_ids)) {
    if (ids_.empty()) {
      root_ = -1;
      return;
    }
    nodes_.reserve(ids_.size());
    root_ = Build(0, static_cast<int>(ids_.size()), 0);
  }

  int Nearest(const Vec3& q, double* dist2_out = nullptr) const {
    if (root_ < 0) {
      return -1;
    }
    int best_id = -1;
    double best_dist2 = std::numeric_limits<double>::infinity();
    NearestRec(root_, q, best_id, best_dist2);
    if (dist2_out) {
      *dist2_out = best_dist2;
    }
    return best_id;
  }

 private:
  struct Node {
    int point_id = -1;
    int axis = 0;
    int left = -1;
    int right = -1;
  };

  int Build(const int begin, const int end, const int depth) {
    if (begin >= end) {
      return -1;
    }
    const int axis = depth % 3;
    const int mid = begin + (end - begin) / 2;
    std::nth_element(ids_.begin() + begin,
                     ids_.begin() + mid,
                     ids_.begin() + end,
                     [&](const int a, const int b) {
                       return points_[static_cast<size_t>(a)][static_cast<size_t>(axis)] <
                              points_[static_cast<size_t>(b)][static_cast<size_t>(axis)];
                     });

    const int node_id = static_cast<int>(nodes_.size());
    nodes_.push_back(Node{});
    nodes_[static_cast<size_t>(node_id)].point_id = ids_[static_cast<size_t>(mid)];
    nodes_[static_cast<size_t>(node_id)].axis = axis;
    nodes_[static_cast<size_t>(node_id)].left = Build(begin, mid, depth + 1);
    nodes_[static_cast<size_t>(node_id)].right = Build(mid + 1, end, depth + 1);
    return node_id;
  }

  static double Dist2(const Vec3& a, const Vec3& b) {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
  }

  void NearestRec(const int node_id, const Vec3& q, int& best_id, double& best_dist2) const {
    if (node_id < 0) {
      return;
    }
    const Node& node = nodes_[static_cast<size_t>(node_id)];
    const Vec3& p = points_[static_cast<size_t>(node.point_id)];

    const double d2 = Dist2(q, p);
    if (d2 < best_dist2) {
      best_dist2 = d2;
      best_id = node.point_id;
    }

    const double diff = q[static_cast<size_t>(node.axis)] - p[static_cast<size_t>(node.axis)];
    const int near_child = diff < 0.0 ? node.left : node.right;
    const int far_child = diff < 0.0 ? node.right : node.left;

    NearestRec(near_child, q, best_id, best_dist2);
    if (diff * diff < best_dist2) {
      NearestRec(far_child, q, best_id, best_dist2);
    }
  }

  const std::vector<Vec3>& points_;
  std::vector<int> ids_;
  std::vector<Node> nodes_;
  int root_ = -1;
};

enum class AnchorKind {
  Ring,
  Path,
};

struct AnchorSpec {
  std::string name;
  AnchorKind kind = AnchorKind::Ring;
  std::string path;
};

struct AnchorData {
  AnchorSpec spec;
  std::vector<Vec3> raw_points;
  std::vector<int> mapped_vertices;
};

struct Options {
  std::string mesh_path;
  std::string out_dir = "benchmarks/regional_fibers";

  std::vector<int> atria_attrs;
  std::vector<int> ventricle_attrs;
  std::vector<AnchorSpec> atria_anchors;

  std::array<double, 3> csv_scale{1.0, 1.0, 1.0};
  std::array<double, 3> csv_shift{0.0, 0.0, 0.0};

  int atria_path_smooth_iters = 20;
  int atria_diffusion_iters = 300;
  int atria_defect_iters = 8;
  double atria_defect_threshold = 0.5;

  std::vector<int> vent_apex_bdr_attrs;
  std::vector<int> vent_base_bdr_attrs;
  std::vector<int> vent_epi_bdr_attrs;
  std::vector<int> vent_lv_bdr_attrs;
  std::vector<int> vent_rv_bdr_attrs;

  int vent_laplace_iters = 2000;
  double vent_laplace_tol = 1e-6;

  double a_endo = 40.0;
  double a_epi = -50.0;
  double b_endo = -65.0;
  double b_epi = 25.0;
};

void PrintUsage() {
  std::cout
      << "Usage: generate_regional_fibers --mesh <mesh> [options]\n"
      << "\n"
      << "Required (at least one region):\n"
      << "  --atria-attrs <a,b,...>            Atrial volume attributes\n"
      << "  --ventricle-attrs <a,b,...>        Ventricular volume attributes\n"
      << "\n"
      << "Atria anchors (repeat):\n"
      << "  --atria-anchor <name:kind:path>    kind in {ring,path}\n"
      << "\n"
      << "Atria options:\n"
      << "  --csv-scale <sx,sy,sz>             Applied to CSV landmarks only (default 1,1,1)\n"
      << "  --csv-shift <tx,ty,tz>             Applied to CSV landmarks only (default 0,0,0)\n"
      << "  --atria-path-smooth-iters <int>    Default 20\n"
      << "  --atria-diffusion-iters <int>      Default 300\n"
      << "  --atria-defect-iters <int>         Default 8\n"
      << "  --atria-defect-threshold <real>    Default 0.5\n"
      << "\n"
      << "Ventricles (Cardioid-like boundary setup):\n"
      << "  --vent-apex-bdr-attrs <a,b,...>\n"
      << "  --vent-base-bdr-attrs <a,b,...>\n"
      << "  --vent-epi-bdr-attrs <a,b,...>\n"
      << "  --vent-lv-bdr-attrs <a,b,...>\n"
      << "  --vent-rv-bdr-attrs <a,b,...>\n"
      << "  --vent-laplace-iters <int>         Default 2000\n"
      << "  --vent-laplace-tol <real>          Default 1e-6\n"
      << "  --a-endo <deg> --a-epi <deg>       Cardioid fiber angles (defaults 40,-50)\n"
      << "  --b-endo <deg> --b-epi <deg>       Cardioid sheet angles (defaults -65,25)\n"
      << "\n"
      << "Output:\n"
      << "  --out-dir <dir>                    Default benchmarks/regional_fibers\n"
      << "  Writes: fiber_f.gf, fiber_s.gf, fiber_n.gf\n";
}

std::string NeedValue(const int i, const int argc, char* argv[], const std::string& key) {
  if (i + 1 >= argc) {
    throw std::runtime_error("Missing value for " + key);
  }
  return std::string(argv[i + 1]);
}

AnchorSpec ParseAnchorSpec(const std::string& raw) {
  const auto first = raw.find(':');
  if (first == std::string::npos) {
    throw std::runtime_error("Invalid --atria-anchor (need name:kind:path): " + raw);
  }
  const auto second = raw.find(':', first + 1);
  if (second == std::string::npos) {
    throw std::runtime_error("Invalid --atria-anchor (need name:kind:path): " + raw);
  }

  AnchorSpec spec;
  spec.name = Trim(raw.substr(0, first));
  const std::string kind = ToLower(Trim(raw.substr(first + 1, second - first - 1)));
  spec.path = Trim(raw.substr(second + 1));

  if (spec.name.empty() || spec.path.empty()) {
    throw std::runtime_error("Invalid --atria-anchor (empty name/path): " + raw);
  }

  if (kind == "ring" || kind == "circ" || kind == "circumferential") {
    spec.kind = AnchorKind::Ring;
  } else if (kind == "path" || kind == "pathway") {
    spec.kind = AnchorKind::Path;
  } else {
    throw std::runtime_error("Unsupported anchor kind in --atria-anchor: " + kind);
  }
  return spec;
}

Options ParseOptions(int argc, char* argv[]) {
  Options opt;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      PrintUsage();
      std::exit(0);
    } else if (arg == "--mesh") {
      opt.mesh_path = NeedValue(i, argc, argv, arg);
      ++i;
    } else if (arg == "--out-dir") {
      opt.out_dir = NeedValue(i, argc, argv, arg);
      ++i;
    } else if (arg == "--atria-attrs") {
      opt.atria_attrs = ParseIntList(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--ventricle-attrs") {
      opt.ventricle_attrs = ParseIntList(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--atria-anchor") {
      opt.atria_anchors.push_back(ParseAnchorSpec(NeedValue(i, argc, argv, arg)));
      ++i;
    } else if (arg == "--csv-scale") {
      opt.csv_scale = ParseTriple(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--csv-shift") {
      opt.csv_shift = ParseTriple(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--atria-path-smooth-iters") {
      opt.atria_path_smooth_iters = std::stoi(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--atria-diffusion-iters") {
      opt.atria_diffusion_iters = std::stoi(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--atria-defect-iters") {
      opt.atria_defect_iters = std::stoi(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--atria-defect-threshold") {
      opt.atria_defect_threshold = std::stod(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--vent-apex-bdr-attrs") {
      opt.vent_apex_bdr_attrs = ParseIntList(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--vent-base-bdr-attrs") {
      opt.vent_base_bdr_attrs = ParseIntList(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--vent-epi-bdr-attrs") {
      opt.vent_epi_bdr_attrs = ParseIntList(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--vent-lv-bdr-attrs") {
      opt.vent_lv_bdr_attrs = ParseIntList(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--vent-rv-bdr-attrs") {
      opt.vent_rv_bdr_attrs = ParseIntList(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--vent-laplace-iters") {
      opt.vent_laplace_iters = std::stoi(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--vent-laplace-tol") {
      opt.vent_laplace_tol = std::stod(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--a-endo") {
      opt.a_endo = std::stod(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--a-epi") {
      opt.a_epi = std::stod(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--b-endo") {
      opt.b_endo = std::stod(NeedValue(i, argc, argv, arg));
      ++i;
    } else if (arg == "--b-epi") {
      opt.b_epi = std::stod(NeedValue(i, argc, argv, arg));
      ++i;
    } else {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }

  if (opt.mesh_path.empty()) {
    throw std::runtime_error("--mesh is required");
  }
  if (opt.atria_attrs.empty() && opt.ventricle_attrs.empty()) {
    throw std::runtime_error("At least one region is required: --atria-attrs and/or --ventricle-attrs");
  }
  if (!opt.atria_attrs.empty() && opt.atria_anchors.empty()) {
    throw std::runtime_error("Atria generation requires at least one --atria-anchor");
  }
  if (opt.atria_path_smooth_iters < 0 || opt.atria_diffusion_iters < 0 || opt.atria_defect_iters < 0) {
    throw std::runtime_error("Atria iteration counts must be >= 0");
  }
  if (opt.vent_laplace_iters <= 0 || opt.vent_laplace_tol <= 0.0) {
    throw std::runtime_error("Ventricular Laplace controls must be positive");
  }

  if (!opt.ventricle_attrs.empty()) {
    if (opt.vent_apex_bdr_attrs.empty() || opt.vent_base_bdr_attrs.empty() ||
        opt.vent_epi_bdr_attrs.empty() || opt.vent_lv_bdr_attrs.empty() ||
        opt.vent_rv_bdr_attrs.empty()) {
      throw std::runtime_error(
          "Ventricles generation requires all boundary attribute lists: "
          "--vent-apex-bdr-attrs, --vent-base-bdr-attrs, --vent-epi-bdr-attrs, "
          "--vent-lv-bdr-attrs, --vent-rv-bdr-attrs");
    }
  }

  return opt;
}

std::unordered_set<int> ToSet(const std::vector<int>& vals) {
  return std::unordered_set<int>(vals.begin(), vals.end());
}

std::vector<Vec3> CollectMeshVertices(const mfem::Mesh& mesh) {
  std::vector<Vec3> coords(static_cast<size_t>(mesh.GetNV()));
  for (int i = 0; i < mesh.GetNV(); ++i) {
    const double* v = mesh.GetVertex(i);
    coords[static_cast<size_t>(i)] = {v[0], v[1], v[2]};
  }
  return coords;
}

struct RegionData {
  std::vector<char> mask;
  std::vector<std::vector<int>> adjacency;
  std::vector<int> vertices;
  std::vector<int> elements;
};

RegionData BuildRegionData(const mfem::Mesh& mesh, const std::unordered_set<int>& attrs) {
  RegionData region;
  const int nv = mesh.GetNV();
  region.mask.assign(static_cast<size_t>(nv), 0);
  region.adjacency.resize(static_cast<size_t>(nv));

  for (int e = 0; e < mesh.GetNE(); ++e) {
    const int attr = mesh.GetAttribute(e);
    if (attrs.find(attr) == attrs.end()) {
      continue;
    }

    region.elements.push_back(e);
    mfem::Array<int> vtx;
    mesh.GetElementVertices(e, vtx);
    for (int i = 0; i < vtx.Size(); ++i) {
      region.mask[static_cast<size_t>(vtx[i])] = 1;
    }
    for (int i = 0; i < vtx.Size(); ++i) {
      for (int j = i + 1; j < vtx.Size(); ++j) {
        const int vi = vtx[i];
        const int vj = vtx[j];
        region.adjacency[static_cast<size_t>(vi)].push_back(vj);
        region.adjacency[static_cast<size_t>(vj)].push_back(vi);
      }
    }
  }

  for (int v = 0; v < nv; ++v) {
    if (region.mask[static_cast<size_t>(v)]) {
      auto& nbs = region.adjacency[static_cast<size_t>(v)];
      std::sort(nbs.begin(), nbs.end());
      nbs.erase(std::unique(nbs.begin(), nbs.end()), nbs.end());
      region.vertices.push_back(v);
    }
  }

  return region;
}

std::vector<Vec3> ReadCsvPoints(const std::string& path,
                                const std::array<double, 3>& scale,
                                const std::array<double, 3>& shift) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open CSV landmark file: " + path);
  }

  std::vector<Vec3> points;
  std::string line;
  while (std::getline(in, line)) {
    const auto comment = line.find('#');
    if (comment != std::string::npos) {
      line = line.substr(0, comment);
    }
    for (char& c : line) {
      if (c == ',' || c == ';' || c == '\t') {
        c = ' ';
      }
    }

    std::stringstream ss(line);
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!(ss >> x >> y >> z)) {
      continue;
    }

    Vec3 p = {x * scale[0] + shift[0], y * scale[1] + shift[1], z * scale[2] + shift[2]};
    points.push_back(p);
  }

  if (points.empty()) {
    throw std::runtime_error("No valid points read from CSV landmark file: " + path);
  }
  return points;
}

std::vector<Vec3> ReadLegacyVtkPoints(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open VTK landmark file: " + path);
  }

  std::string token;
  int npoints = -1;
  while (in >> token) {
    if (ToLower(token) == "points") {
      std::string dtype;
      in >> npoints >> dtype;
      break;
    }
  }

  if (npoints <= 0) {
    throw std::runtime_error("Could not find a valid POINTS section in VTK file: " + path);
  }

  std::vector<Vec3> points;
  points.reserve(static_cast<size_t>(npoints));
  for (int i = 0; i < npoints; ++i) {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!(in >> x >> y >> z)) {
      throw std::runtime_error("Failed reading POINTS payload in VTK file: " + path);
    }
    points.push_back({x, y, z});
  }

  return points;
}

std::vector<Vec3> LoadLandmarkPoints(const std::string& path,
                                     const std::array<double, 3>& csv_scale,
                                     const std::array<double, 3>& csv_shift) {
  const auto ext = ToLower(std::filesystem::path(path).extension().string());
  if (ext == ".csv") {
    return ReadCsvPoints(path, csv_scale, csv_shift);
  }
  if (ext == ".vtk") {
    return ReadLegacyVtkPoints(path);
  }
  throw std::runtime_error("Unsupported landmark file extension (expected .csv or .vtk): " + path);
}

AnchorData LoadAnchorData(const AnchorSpec& spec,
                          const KdTree3& atria_vertex_tree,
                          const std::array<double, 3>& csv_scale,
                          const std::array<double, 3>& csv_shift) {
  AnchorData anchor;
  anchor.spec = spec;
  anchor.raw_points = LoadLandmarkPoints(spec.path, csv_scale, csv_shift);

  anchor.mapped_vertices.reserve(anchor.raw_points.size());
  for (const auto& p : anchor.raw_points) {
    const int vid = atria_vertex_tree.Nearest(p);
    if (vid >= 0) {
      anchor.mapped_vertices.push_back(vid);
    }
  }

  std::sort(anchor.mapped_vertices.begin(), anchor.mapped_vertices.end());
  anchor.mapped_vertices.erase(std::unique(anchor.mapped_vertices.begin(), anchor.mapped_vertices.end()),
                               anchor.mapped_vertices.end());

  if (anchor.mapped_vertices.empty()) {
    throw std::runtime_error("Anchor produced no mapped vertices: " + spec.name);
  }
  return anchor;
}

Vec3 ComputeCentroid(const std::vector<Vec3>& pts) {
  Vec3 c = {0.0, 0.0, 0.0};
  for (const auto& p : pts) {
    AddInPlace(c, p);
  }
  if (!pts.empty()) {
    c = c / static_cast<double>(pts.size());
  }
  return c;
}

Vec3 ComputePcaPlaneNormal(const std::vector<Vec3>& pts) {
  if (pts.size() < 3) {
    return {0.0, 0.0, 1.0};
  }

  const Vec3 c = ComputeCentroid(pts);
  mfem::DenseMatrix cov(3, 3);
  cov = 0.0;

  for (const auto& p : pts) {
    const Vec3 d = p - c;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        cov(i, j) += d[static_cast<size_t>(i)] * d[static_cast<size_t>(j)];
      }
    }
  }

  const double inv_n = 1.0 / static_cast<double>(pts.size());
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      cov(i, j) *= inv_n;
    }
  }

  double lam[3] = {0.0, 0.0, 0.0};
  double vec[9] = {0.0};
  cov.CalcEigenvalues(lam, vec);

  int min_idx = 0;
  for (int i = 1; i < 3; ++i) {
    if (std::abs(lam[i]) < std::abs(lam[min_idx])) {
      min_idx = i;
    }
  }

  Vec3 axis = {vec[min_idx * 3 + 0], vec[min_idx * 3 + 1], vec[min_idx * 3 + 2]};
  if (Norm(axis) <= kEps) {
    const Vec3 a = pts[1] - pts[0];
    const Vec3 b = pts[2] - pts[0];
    axis = Cross(a, b);
  }
  return Normalize(axis, {0.0, 0.0, 1.0});
}

std::unordered_map<int, Vec3> BuildRingAnchorVectors(const AnchorData& anchor,
                                                      const std::vector<Vec3>& coords) {
  std::vector<Vec3> mesh_pts;
  mesh_pts.reserve(anchor.mapped_vertices.size());
  for (const int vid : anchor.mapped_vertices) {
    mesh_pts.push_back(coords[static_cast<size_t>(vid)]);
  }

  const Vec3 centroid = ComputeCentroid(mesh_pts);
  const Vec3 axis = ComputePcaPlaneNormal(mesh_pts);

  std::unordered_map<int, Vec3> out;
  out.reserve(anchor.mapped_vertices.size());
  for (const int vid : anchor.mapped_vertices) {
    const Vec3 radial = coords[static_cast<size_t>(vid)] - centroid;
    Vec3 f = Cross(axis, radial);
    if (Norm(f) <= kEps) {
      f = Cross(axis, OrthogonalUnit(axis));
    }
    out.emplace(vid, Normalize(f, {1.0, 0.0, 0.0}));
  }
  return out;
}

std::unordered_map<int, Vec3> BuildPathAnchorVectors(const AnchorData& anchor,
                                                      const RegionData& atria,
                                                      const std::vector<Vec3>& coords,
                                                      const int smooth_iters) {
  const int nv = static_cast<int>(coords.size());
  std::vector<char> is_anchor(static_cast<size_t>(nv), 0);
  for (const int vid : anchor.mapped_vertices) {
    is_anchor[static_cast<size_t>(vid)] = 1;
  }

  std::unordered_map<int, Vec3> tangents;
  tangents.reserve(anchor.mapped_vertices.size());

  for (const int v : anchor.mapped_vertices) {
    Vec3 sum = {0.0, 0.0, 0.0};
    int cnt = 0;
    for (const int nb : atria.adjacency[static_cast<size_t>(v)]) {
      if (!is_anchor[static_cast<size_t>(nb)]) {
        continue;
      }
      Vec3 dir = coords[static_cast<size_t>(nb)] - coords[static_cast<size_t>(v)];
      const double n = Norm(dir);
      if (n > kEps) {
        AddInPlace(sum, dir / n);
        ++cnt;
      }
    }
    if (cnt == 0) {
      sum = {1.0, 0.0, 0.0};
    }
    tangents[v] = Normalize(sum, {1.0, 0.0, 0.0});
  }

  for (int iter = 0; iter < smooth_iters; ++iter) {
    std::unordered_map<int, Vec3> next = tangents;
    for (const int v : anchor.mapped_vertices) {
      Vec3 sum = tangents[v];
      int cnt = 1;
      for (const int nb : atria.adjacency[static_cast<size_t>(v)]) {
        if (!is_anchor[static_cast<size_t>(nb)]) {
          continue;
        }
        AddInPlace(sum, tangents[nb]);
        ++cnt;
      }
      next[v] = Normalize(sum / static_cast<double>(cnt), tangents[v]);
    }
    tangents.swap(next);
  }

  std::vector<char> visited(static_cast<size_t>(nv), 0);
  std::queue<int> q;
  for (const int root : anchor.mapped_vertices) {
    if (visited[static_cast<size_t>(root)]) {
      continue;
    }
    visited[static_cast<size_t>(root)] = 1;
    q.push(root);

    while (!q.empty()) {
      const int v = q.front();
      q.pop();
      for (const int nb : atria.adjacency[static_cast<size_t>(v)]) {
        if (!is_anchor[static_cast<size_t>(nb)]) {
          continue;
        }
        if (Dot(tangents[v], tangents[nb]) < 0.0) {
          tangents[nb] = tangents[nb] * -1.0;
        }
        if (!visited[static_cast<size_t>(nb)]) {
          visited[static_cast<size_t>(nb)] = 1;
          q.push(nb);
        }
      }
    }
  }

  return tangents;
}

struct AtriaSeed {
  std::vector<Vec3> f;
  std::vector<char> fixed;
};

AtriaSeed BuildAtriaSeed(const std::vector<AnchorData>& anchors,
                         const RegionData& atria,
                         const std::vector<Vec3>& coords,
                         const Options& opt) {
  const int nv = static_cast<int>(coords.size());
  std::vector<Vec3> accum(static_cast<size_t>(nv), Vec3{0.0, 0.0, 0.0});
  std::vector<int> count(static_cast<size_t>(nv), 0);
  std::vector<char> fixed(static_cast<size_t>(nv), 0);

  for (const auto& anchor : anchors) {
    std::unordered_map<int, Vec3> vecs;
    if (anchor.spec.kind == AnchorKind::Ring) {
      vecs = BuildRingAnchorVectors(anchor, coords);
    } else {
      vecs = BuildPathAnchorVectors(anchor, atria, coords, opt.atria_path_smooth_iters);
    }

    for (const auto& kv : vecs) {
      const int vid = kv.first;
      if (!atria.mask[static_cast<size_t>(vid)]) {
        continue;
      }
      AddInPlace(accum[static_cast<size_t>(vid)], kv.second);
      ++count[static_cast<size_t>(vid)];
      fixed[static_cast<size_t>(vid)] = 1;
    }
  }

  int n_fixed = 0;
  for (const int v : atria.vertices) {
    if (!fixed[static_cast<size_t>(v)]) {
      continue;
    }
    accum[static_cast<size_t>(v)] = Normalize(accum[static_cast<size_t>(v)], {1.0, 0.0, 0.0});
    ++n_fixed;
  }

  if (n_fixed == 0) {
    throw std::runtime_error("Atria anchors produced no fixed seed vectors in atrial region");
  }

  return AtriaSeed{std::move(accum), std::move(fixed)};
}

void InitializeByNearestFixed(const RegionData& region,
                              const std::vector<Vec3>& coords,
                              std::vector<Vec3>& field,
                              const std::vector<char>& fixed) {
  std::vector<int> fixed_ids;
  fixed_ids.reserve(region.vertices.size());
  for (const int v : region.vertices) {
    if (fixed[static_cast<size_t>(v)]) {
      fixed_ids.push_back(v);
    }
  }
  if (fixed_ids.empty()) {
    throw std::runtime_error("No fixed vertices available for initialization");
  }

  KdTree3 tree(coords, fixed_ids);
  for (const int v : region.vertices) {
    if (fixed[static_cast<size_t>(v)]) {
      continue;
    }
    const int nearest = tree.Nearest(coords[static_cast<size_t>(v)]);
    if (nearest >= 0) {
      field[static_cast<size_t>(v)] = field[static_cast<size_t>(nearest)];
    } else {
      field[static_cast<size_t>(v)] = {1.0, 0.0, 0.0};
    }
  }
}

void DiffuseField(const RegionData& region,
                  const std::vector<char>& fixed,
                  std::vector<Vec3>& field,
                  const int iterations) {
  std::vector<Vec3> next = field;
  for (int iter = 0; iter < iterations; ++iter) {
    for (const int v : region.vertices) {
      if (fixed[static_cast<size_t>(v)]) {
        next[static_cast<size_t>(v)] = field[static_cast<size_t>(v)];
        continue;
      }

      Vec3 sum = {0.0, 0.0, 0.0};
      int cnt = 0;
      for (const int nb : region.adjacency[static_cast<size_t>(v)]) {
        if (!region.mask[static_cast<size_t>(nb)]) {
          continue;
        }
        AddInPlace(sum, field[static_cast<size_t>(nb)]);
        ++cnt;
      }
      if (cnt > 0) {
        next[static_cast<size_t>(v)] = Normalize(sum, field[static_cast<size_t>(v)]);
      } else {
        next[static_cast<size_t>(v)] = field[static_cast<size_t>(v)];
      }
    }
    field.swap(next);
  }
}

void RepairDefects(const RegionData& region,
                   const std::vector<char>& fixed,
                   std::vector<Vec3>& field,
                   const int iters,
                   const double defect_threshold) {
  std::vector<char> defect(field.size(), 0);
  std::vector<Vec3> next = field;

  for (int iter = 0; iter < iters; ++iter) {
    int defect_count = 0;
    for (const int v : region.vertices) {
      if (fixed[static_cast<size_t>(v)]) {
        defect[static_cast<size_t>(v)] = 0;
        continue;
      }
      Vec3 sum = {0.0, 0.0, 0.0};
      int cnt = 0;
      for (const int nb : region.adjacency[static_cast<size_t>(v)]) {
        if (!region.mask[static_cast<size_t>(nb)]) {
          continue;
        }
        AddInPlace(sum, field[static_cast<size_t>(nb)]);
        ++cnt;
      }
      const double coherence = cnt > 0 ? Norm(sum) / static_cast<double>(cnt) : 0.0;
      if (coherence < defect_threshold) {
        defect[static_cast<size_t>(v)] = 1;
        ++defect_count;
      } else {
        defect[static_cast<size_t>(v)] = 0;
      }
    }

    if (defect_count == 0) {
      break;
    }

    for (const int v : region.vertices) {
      if (!defect[static_cast<size_t>(v)] || fixed[static_cast<size_t>(v)]) {
        next[static_cast<size_t>(v)] = field[static_cast<size_t>(v)];
        continue;
      }
      Vec3 sum = {0.0, 0.0, 0.0};
      int cnt = 0;
      for (const int nb : region.adjacency[static_cast<size_t>(v)]) {
        if (!region.mask[static_cast<size_t>(nb)] || defect[static_cast<size_t>(nb)]) {
          continue;
        }
        AddInPlace(sum, field[static_cast<size_t>(nb)]);
        ++cnt;
      }
      if (cnt > 0) {
        next[static_cast<size_t>(v)] = Normalize(sum, field[static_cast<size_t>(v)]);
      } else {
        next[static_cast<size_t>(v)] = field[static_cast<size_t>(v)];
      }
    }

    field.swap(next);
  }
}

struct FaceKey {
  int a = -1;
  int b = -1;
  int c = -1;

  bool operator==(const FaceKey& other) const { return a == other.a && b == other.b && c == other.c; }
};

struct FaceKeyHash {
  std::size_t operator()(const FaceKey& k) const {
    std::size_t h = 1469598103934665603ull;
    h ^= static_cast<std::size_t>(k.a) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= static_cast<std::size_t>(k.b) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= static_cast<std::size_t>(k.c) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

FaceKey MakeFaceKey(int a, int b, int c) {
  std::array<int, 3> x = {a, b, c};
  std::sort(x.begin(), x.end());
  return FaceKey{x[0], x[1], x[2]};
}

struct FaceEntry {
  int count = 0;
  std::array<int, 3> oriented = {-1, -1, -1};
  int owner_elem = -1;
};

std::vector<Vec3> ComputeRegionBoundaryNormals(const mfem::Mesh& mesh,
                                               const RegionData& region,
                                               const std::vector<Vec3>& coords) {
  std::vector<Vec3> normals(coords.size(), Vec3{0.0, 0.0, 0.0});
  std::vector<char> has_normal(coords.size(), 0);

  std::unordered_map<FaceKey, FaceEntry, FaceKeyHash> faces;
  faces.reserve(region.elements.size() * 4);

  constexpr int tet_faces[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};

  for (const int e : region.elements) {
    mfem::Array<int> vtx;
    mesh.GetElementVertices(e, vtx);
    if (vtx.Size() != 4) {
      throw std::runtime_error("Atria boundary normal computation currently expects tetrahedral elements");
    }

    for (int fi = 0; fi < 4; ++fi) {
      const int va = vtx[tet_faces[fi][0]];
      const int vb = vtx[tet_faces[fi][1]];
      const int vc = vtx[tet_faces[fi][2]];
      const FaceKey key = MakeFaceKey(va, vb, vc);
      auto& entry = faces[key];
      if (entry.count == 0) {
        entry.oriented = {va, vb, vc};
        entry.owner_elem = e;
      }
      ++entry.count;
    }
  }

  for (const auto& kv : faces) {
    const FaceEntry& fe = kv.second;
    if (fe.count != 1) {
      continue;
    }

    const int a = fe.oriented[0];
    const int b = fe.oriented[1];
    const int c = fe.oriented[2];

    const Vec3 pa = coords[static_cast<size_t>(a)];
    const Vec3 pb = coords[static_cast<size_t>(b)];
    const Vec3 pc = coords[static_cast<size_t>(c)];

    Vec3 n = Cross(pb - pa, pc - pa);
    if (Norm(n) <= kEps) {
      continue;
    }

    mfem::Array<int> ev;
    mesh.GetElementVertices(fe.owner_elem, ev);
    Vec3 ec = {0.0, 0.0, 0.0};
    for (int i = 0; i < ev.Size(); ++i) {
      AddInPlace(ec, coords[static_cast<size_t>(ev[i])]);
    }
    ec = ec / static_cast<double>(ev.Size());

    const Vec3 fc = (pa + pb + pc) / 3.0;
    const Vec3 to_inside = ec - fc;
    if (Dot(n, to_inside) > 0.0) {
      n = n * -1.0;
    }

    n = Normalize(n, {0.0, 0.0, 1.0});
    AddInPlace(normals[static_cast<size_t>(a)], n);
    AddInPlace(normals[static_cast<size_t>(b)], n);
    AddInPlace(normals[static_cast<size_t>(c)], n);
    has_normal[static_cast<size_t>(a)] = 1;
    has_normal[static_cast<size_t>(b)] = 1;
    has_normal[static_cast<size_t>(c)] = 1;
  }

  std::vector<int> boundary_vertices;
  boundary_vertices.reserve(region.vertices.size());
  for (const int v : region.vertices) {
    if (has_normal[static_cast<size_t>(v)] && Norm(normals[static_cast<size_t>(v)]) > kEps) {
      normals[static_cast<size_t>(v)] = Normalize(normals[static_cast<size_t>(v)], {0.0, 0.0, 1.0});
      boundary_vertices.push_back(v);
    }
  }

  if (boundary_vertices.empty()) {
    for (const int v : region.vertices) {
      normals[static_cast<size_t>(v)] = {0.0, 0.0, 1.0};
    }
    return normals;
  }

  KdTree3 btree(coords, boundary_vertices);
  for (const int v : region.vertices) {
    if (has_normal[static_cast<size_t>(v)]) {
      continue;
    }
    const int nearest = btree.Nearest(coords[static_cast<size_t>(v)]);
    if (nearest >= 0) {
      normals[static_cast<size_t>(v)] = normals[static_cast<size_t>(nearest)];
    } else {
      normals[static_cast<size_t>(v)] = {0.0, 0.0, 1.0};
    }
  }

  return normals;
}

void BuildAtriaBasis(const RegionData& atria,
                     const std::vector<Vec3>& diffused_f,
                     const std::vector<Vec3>& normals,
                     std::vector<Vec3>& f,
                     std::vector<Vec3>& s,
                     std::vector<Vec3>& n) {
  for (const int v : atria.vertices) {
    Vec3 nv = Normalize(normals[static_cast<size_t>(v)], {0.0, 0.0, 1.0});
    Vec3 fv = Normalize(diffused_f[static_cast<size_t>(v)], OrthogonalUnit(nv));

    Vec3 fproj = fv - nv * Dot(fv, nv);
    if (Norm(fproj) <= kEps) {
      fproj = OrthogonalUnit(nv);
    }
    fv = Normalize(fproj, OrthogonalUnit(nv));

    Vec3 sv = Cross(nv, fv);
    if (Norm(sv) <= kEps) {
      sv = OrthogonalUnit(fv);
    }
    sv = Normalize(sv, OrthogonalUnit(fv));

    nv = Normalize(Cross(fv, sv), nv);

    f[static_cast<size_t>(v)] = fv;
    s[static_cast<size_t>(v)] = sv;
    n[static_cast<size_t>(v)] = nv;
  }
}

std::vector<int> CollectBoundaryVertices(const mfem::Mesh& mesh,
                                         const std::unordered_set<int>& bdr_attrs,
                                         const RegionData& region) {
  std::vector<char> mark(region.mask.size(), 0);

  for (int be = 0; be < mesh.GetNBE(); ++be) {
    const int attr = mesh.GetBdrAttribute(be);
    if (bdr_attrs.find(attr) == bdr_attrs.end()) {
      continue;
    }
    mfem::Array<int> vtx;
    mesh.GetBdrElementVertices(be, vtx);
    for (int i = 0; i < vtx.Size(); ++i) {
      const int v = vtx[i];
      if (region.mask[static_cast<size_t>(v)]) {
        mark[static_cast<size_t>(v)] = 1;
      }
    }
  }

  std::vector<int> out;
  for (const int v : region.vertices) {
    if (mark[static_cast<size_t>(v)]) {
      out.push_back(v);
    }
  }
  return out;
}

struct DirichletBC {
  std::vector<char> is_fixed;
  std::vector<double> value;
};

DirichletBC MakeDirichletBC(const int nv) {
  return DirichletBC{std::vector<char>(static_cast<size_t>(nv), 0), std::vector<double>(static_cast<size_t>(nv), 0.0)};
}

void ApplyDirichlet(const std::vector<int>& vertices, const double val, DirichletBC& bc) {
  for (const int v : vertices) {
    bc.is_fixed[static_cast<size_t>(v)] = 1;
    bc.value[static_cast<size_t>(v)] = val;
  }
}

std::vector<double> SolveGraphLaplace(const RegionData& region,
                                      const DirichletBC& bc,
                                      const int max_iters,
                                      const double tol) {
  const int nv = static_cast<int>(region.mask.size());
  std::vector<double> x(static_cast<size_t>(nv), 0.0);
  std::vector<double> next = x;

  for (const int v : region.vertices) {
    x[static_cast<size_t>(v)] = 0.5;
    if (bc.is_fixed[static_cast<size_t>(v)]) {
      x[static_cast<size_t>(v)] = bc.value[static_cast<size_t>(v)];
    }
  }

  for (int iter = 0; iter < max_iters; ++iter) {
    double max_delta = 0.0;

    for (const int v : region.vertices) {
      if (bc.is_fixed[static_cast<size_t>(v)]) {
        next[static_cast<size_t>(v)] = bc.value[static_cast<size_t>(v)];
        continue;
      }

      double sum = 0.0;
      int cnt = 0;
      for (const int nb : region.adjacency[static_cast<size_t>(v)]) {
        if (!region.mask[static_cast<size_t>(nb)]) {
          continue;
        }
        sum += x[static_cast<size_t>(nb)];
        ++cnt;
      }

      if (cnt > 0) {
        next[static_cast<size_t>(v)] = sum / static_cast<double>(cnt);
      } else {
        next[static_cast<size_t>(v)] = x[static_cast<size_t>(v)];
      }

      max_delta = std::max(max_delta, std::abs(next[static_cast<size_t>(v)] - x[static_cast<size_t>(v)]));
    }

    for (const int v : region.vertices) {
      x[static_cast<size_t>(v)] = next[static_cast<size_t>(v)];
    }

    if (max_delta < tol) {
      break;
    }
  }

  return x;
}

std::vector<Vec3> EstimateGradient(const RegionData& region,
                                   const std::vector<Vec3>& coords,
                                   const std::vector<double>& scalar) {
  std::vector<Vec3> grads(coords.size(), Vec3{0.0, 0.0, 0.0});

  for (const int v : region.vertices) {
    Vec3 g = {0.0, 0.0, 0.0};
    const Vec3 pv = coords[static_cast<size_t>(v)];
    const double uv = scalar[static_cast<size_t>(v)];

    for (const int nb : region.adjacency[static_cast<size_t>(v)]) {
      if (!region.mask[static_cast<size_t>(nb)]) {
        continue;
      }
      const Vec3 d = coords[static_cast<size_t>(nb)] - pv;
      const double d2 = Dot(d, d);
      if (d2 <= kEps) {
        continue;
      }
      const double w = (scalar[static_cast<size_t>(nb)] - uv) / d2;
      AddInPlace(g, d * w);
    }

    grads[static_cast<size_t>(v)] = g;
  }

  return grads;
}

Mat3 IdentityMat3() {
  Mat3 M;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      M.m[static_cast<size_t>(i)][static_cast<size_t>(j)] = (i == j ? 1.0 : 0.0);
    }
  }
  return M;
}

Vec3 GetCol(const Mat3& M, const int c) {
  return {M.m[0][static_cast<size_t>(c)], M.m[1][static_cast<size_t>(c)], M.m[2][static_cast<size_t>(c)]};
}

void SetCol(Mat3& M, const int c, const Vec3& v) {
  M.m[0][static_cast<size_t>(c)] = v[0];
  M.m[1][static_cast<size_t>(c)] = v[1];
  M.m[2][static_cast<size_t>(c)] = v[2];
}

Mat3 MatMul(const Mat3& A, const Mat3& B) {
  Mat3 C = IdentityMat3();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double v = 0.0;
      for (int k = 0; k < 3; ++k) {
        v += A.m[static_cast<size_t>(i)][static_cast<size_t>(k)] *
             B.m[static_cast<size_t>(k)][static_cast<size_t>(j)];
      }
      C.m[static_cast<size_t>(i)][static_cast<size_t>(j)] = v;
    }
  }
  return C;
}

bool VecParallel(const Vec3& a, const Vec3& b) {
  const Vec3 aa = Normalize(a, {1.0, 0.0, 0.0});
  const Vec3 bb = Normalize(b, {1.0, 0.0, 0.0});
  return std::abs(Dot(aa, bb)) > 0.9999;
}

Mat3 AxisFrame(const Vec3& psi, const Vec3& phi) {
  Vec3 e1 = Normalize(psi, {0.0, 1.0, 0.0});
  Vec3 p = Normalize(phi, OrthogonalUnit(e1));
  Vec3 e2 = p - e1 * Dot(e1, p);
  e2 = Normalize(e2, OrthogonalUnit(e1));
  Vec3 e0 = Normalize(Cross(e1, e2), OrthogonalUnit(e1));

  Mat3 Q = IdentityMat3();
  SetCol(Q, 0, e0);
  SetCol(Q, 1, e1);
  SetCol(Q, 2, e2);
  return Q;
}

Mat3 OrientFrame(const Mat3& Q, const double a_deg, const double b_deg) {
  const double a = a_deg * kPi / 180.0;
  const double b = b_deg * kPi / 180.0;

  Mat3 A = IdentityMat3();
  A.m[0][0] = std::cos(a);
  A.m[0][1] = -std::sin(a);
  A.m[1][0] = std::sin(a);
  A.m[1][1] = std::cos(a);

  Mat3 B = IdentityMat3();
  B.m[1][1] = std::cos(b);
  B.m[1][2] = std::sin(b);
  B.m[2][1] = -std::sin(b);
  B.m[2][2] = std::cos(b);

  return MatMul(MatMul(Q, A), B);
}

double QuatDot(const Quat& q1, const Quat& q2) {
  return q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3];
}

Quat NormalizeQuat(const Quat& q) {
  double n2 = 0.0;
  for (double v : q) {
    n2 += v * v;
  }
  const double n = std::sqrt(n2);
  if (n <= kEps) {
    return {1.0, 0.0, 0.0, 0.0};
  }
  return {q[0] / n, q[1] / n, q[2] / n, q[3] / n};
}

Quat RotToQuat(const Mat3& Q) {
  Quat q = {1.0, 0.0, 0.0, 0.0};
  const double M11 = Q.m[0][0];
  const double M21 = Q.m[1][0];
  const double M31 = Q.m[2][0];
  const double M12 = Q.m[0][1];
  const double M22 = Q.m[1][1];
  const double M32 = Q.m[2][1];
  const double M13 = Q.m[0][2];
  const double M23 = Q.m[1][2];
  const double M33 = Q.m[2][2];

  const double w2 = 0.25 * (1.0 + M11 + M22 + M33);
  constexpr double err = 1e-6;
  if (w2 > err) {
    const double w = std::sqrt(w2);
    q[0] = w;
    const double denom = 4.0 * w;
    q[1] = (M23 - M32) / denom;
    q[2] = (M31 - M13) / denom;
    q[3] = (M12 - M21) / denom;
  } else {
    q[0] = 0.0;
    const double x2 = -0.5 * (M22 + M33);
    if (x2 > err) {
      const double x = std::sqrt(x2);
      q[1] = x;
      q[2] = M12 / (2.0 * x);
      q[3] = M13 / (2.0 * x);
    } else {
      q[1] = 0.0;
      const double y2 = 0.5 * (1.0 - M33);
      if (y2 > err) {
        const double y = std::sqrt(y2);
        q[2] = y;
        q[3] = M23 / (2.0 * y);
      } else {
        q[2] = 0.0;
        q[3] = 1.0;
      }
    }
  }
  return NormalizeQuat(q);
}

Mat3 QuatToRot(const Quat& qin) {
  const Quat q = NormalizeQuat(qin);
  const double w = q[0];
  const double x = q[1];
  const double y = q[2];
  const double z = q[3];

  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  const double xy = x * y;
  const double xz = x * z;
  const double yz = y * z;
  const double wx = w * x;
  const double wy = w * y;
  const double wz = w * z;

  Mat3 Q = IdentityMat3();
  Q.m[0][0] = 1.0 - 2.0 * y2 - 2.0 * z2;
  Q.m[1][0] = 2.0 * xy - 2.0 * wz;
  Q.m[2][0] = 2.0 * xz + 2.0 * wy;

  Q.m[0][1] = 2.0 * xy + 2.0 * wz;
  Q.m[1][1] = 1.0 - 2.0 * x2 - 2.0 * z2;
  Q.m[2][1] = 2.0 * yz - 2.0 * wx;

  Q.m[0][2] = 2.0 * xz - 2.0 * wy;
  Q.m[1][2] = 2.0 * yz + 2.0 * wx;
  Q.m[2][2] = 1.0 - 2.0 * x2 - 2.0 * y2;
  return Q;
}

Quat Slerp(const Quat& q1_in, const Quat& q2_in, const double t) {
  Quat q1 = NormalizeQuat(q1_in);
  Quat q2 = NormalizeQuat(q2_in);

  double dot = QuatDot(q1, q2);
  if (dot < 0.0) {
    dot = -dot;
    q2 = {-q2[0], -q2[1], -q2[2], -q2[3]};
  }

  if (dot < 0.9999) {
    const double angle = std::acos(std::max(-1.0, std::min(1.0, dot)));
    const double s = std::sin(angle);
    const double a = std::sin((1.0 - t) * angle) / s;
    const double b = std::sin(t * angle) / s;
    Quat q = {a * q1[0] + b * q2[0], a * q1[1] + b * q2[1],
              a * q1[2] + b * q2[2], a * q1[3] + b * q2[3]};
    return NormalizeQuat(q);
  }

  Quat q = {(1.0 - t) * q1[0] + t * q2[0], (1.0 - t) * q1[1] + t * q2[1],
            (1.0 - t) * q1[2] + t * q2[2], (1.0 - t) * q1[3] + t * q2[3]};
  return NormalizeQuat(q);
}

Mat3 Bislerp(const Mat3& Qa, const Mat3& Qb, const double t) {
  const Quat qa = RotToQuat(Qa);
  const Quat qb = RotToQuat(Qb);

  const double a = qa[0];
  const double b = qa[1];
  const double c = qa[2];
  const double d = qa[3];

  std::array<Quat, 4> equiv = {{{a, b, c, d}, {-b, a, -d, c}, {-c, d, a, -b}, {-d, -c, b, a}}};

  double best = -1.0;
  Quat qm = equiv[0];
  for (const auto& q : equiv) {
    const double val = std::abs(QuatDot(q, qb));
    if (val > best) {
      best = val;
      qm = q;
    }
  }

  return QuatToRot(Slerp(qm, qb, t));
}

double a_s_f(const double a_endo, const double /*a_epi*/, const double d) {
  // Matches cardioid fiberp utils.cpp exactly.
  return a_endo * (1.0 - d) - a_endo * d;
}

double a_w_f(const double a_endo, const double a_epi, const double d) {
  return a_endo * (1.0 - d) + a_epi * d;
}

double b_s_f(const double b_endo, const double /*b_epi*/, const double d) {
  // Matches cardioid fiberp utils.cpp exactly.
  return b_endo * (1.0 - d) - b_endo * d;
}

double b_w_f(const double b_endo, const double b_epi, const double d) {
  return b_endo * (1.0 - d) + b_epi * d;
}

Mat3 VectorEigenFallback(const Vec3& psi_ab) {
  const Vec3 e1 = Normalize(psi_ab, {0.0, 1.0, 0.0});
  const Vec3 e2 = OrthogonalUnit(e1);
  const Vec3 e0 = Normalize(Cross(e1, e2), {1.0, 0.0, 0.0});
  Mat3 Q = IdentityMat3();
  SetCol(Q, 0, e0);
  SetCol(Q, 1, e1);
  SetCol(Q, 2, e2);
  return Q;
}

Mat3 BiSlerpCombo(const double psi_ab,
                  const Vec3& psi_ab_vec,
                  const double phi_epi,
                  const Vec3& phi_epi_vec,
                  const double phi_lv,
                  const Vec3& phi_lv_vec,
                  const double phi_rv,
                  const Vec3& phi_rv_vec,
                  const Options& options) {
  (void)psi_ab;

  if (!IsNonZero(psi_ab_vec)) {
    return VectorEigenFallback({1.0, 0.0, 0.0});
  }

  const double phi_v = phi_lv + phi_rv;
  const double frac = std::abs(phi_v) > kEps ? (phi_rv / phi_v) : 0.5;
  const double frac_epi = phi_epi;

  const double as = a_s_f(options.a_endo, options.a_epi, frac);
  const double bs = b_s_f(options.b_endo, options.b_epi, frac);
  const double aw = a_w_f(options.a_endo, options.a_epi, frac_epi);
  const double bw = b_w_f(options.b_endo, options.b_epi, frac_epi);

  bool lv_ok = IsNonZero(phi_lv_vec);
  bool rv_ok = IsNonZero(phi_rv_vec);
  bool epi_ok = IsNonZero(phi_epi_vec);

  Mat3 QPlv = IdentityMat3();
  if (lv_ok) {
    const Vec3 phi_lv_neg = phi_lv_vec * -1.0;
    if (VecParallel(psi_ab_vec, phi_lv_neg)) {
      lv_ok = false;
    } else {
      QPlv = OrientFrame(AxisFrame(psi_ab_vec, phi_lv_neg), as, bs);
    }
  }

  Mat3 QPrv = IdentityMat3();
  if (rv_ok) {
    if (VecParallel(psi_ab_vec, phi_rv_vec)) {
      rv_ok = false;
    } else {
      QPrv = OrientFrame(AxisFrame(psi_ab_vec, phi_rv_vec), as, bs);
    }
  }

  Mat3 QPepi = IdentityMat3();
  if (epi_ok) {
    if (VecParallel(psi_ab_vec, phi_epi_vec)) {
      epi_ok = false;
    } else {
      QPepi = OrientFrame(AxisFrame(psi_ab_vec, phi_epi_vec), aw, bw);
    }
  }

  if (lv_ok && rv_ok && epi_ok) {
    const Mat3 QPendo = Bislerp(QPlv, QPrv, frac);
    return Bislerp(QPendo, QPepi, frac_epi);
  }
  if (!lv_ok && rv_ok && epi_ok) {
    return Bislerp(QPrv, QPepi, frac_epi);
  }
  if (lv_ok && !rv_ok && epi_ok) {
    return Bislerp(QPlv, QPepi, frac_epi);
  }
  if (lv_ok && rv_ok && !epi_ok) {
    return Bislerp(QPlv, QPrv, frac);
  }

  return VectorEigenFallback(psi_ab_vec);
}

void BuildVentricleBasis(const RegionData& vent,
                         const std::vector<double>& psi_ab,
                         const std::vector<Vec3>& psi_ab_g,
                         const std::vector<double>& phi_epi,
                         const std::vector<Vec3>& phi_epi_g,
                         const std::vector<double>& phi_lv,
                         const std::vector<Vec3>& phi_lv_g,
                         const std::vector<double>& phi_rv,
                         const std::vector<Vec3>& phi_rv_g,
                         const Options& options,
                         std::vector<Vec3>& f,
                         std::vector<Vec3>& s,
                         std::vector<Vec3>& n) {
  for (const int v : vent.vertices) {
    const Mat3 Q = BiSlerpCombo(psi_ab[static_cast<size_t>(v)],
                                psi_ab_g[static_cast<size_t>(v)],
                                phi_epi[static_cast<size_t>(v)],
                                phi_epi_g[static_cast<size_t>(v)],
                                phi_lv[static_cast<size_t>(v)],
                                phi_lv_g[static_cast<size_t>(v)],
                                phi_rv[static_cast<size_t>(v)],
                                phi_rv_g[static_cast<size_t>(v)],
                                options);

    Vec3 fv = Normalize(GetCol(Q, 0), {1.0, 0.0, 0.0});
    Vec3 sv_raw = GetCol(Q, 1);
    Vec3 sv = sv_raw - fv * Dot(sv_raw, fv);
    if (Norm(sv) <= kEps) {
      sv = OrthogonalUnit(fv);
    }
    sv = Normalize(sv, OrthogonalUnit(fv));

    Vec3 nv = Cross(fv, sv);
    if (Norm(nv) <= kEps) {
      nv = Normalize(GetCol(Q, 2), OrthogonalUnit(fv));
      sv = Normalize(Cross(nv, fv), OrthogonalUnit(fv));
      nv = Normalize(Cross(fv, sv), nv);
    } else {
      nv = Normalize(nv, OrthogonalUnit(fv));
    }

    f[static_cast<size_t>(v)] = fv;
    s[static_cast<size_t>(v)] = sv;
    n[static_cast<size_t>(v)] = nv;
  }
}

void FillGridFunctionFromVertexVectors(mfem::FiniteElementSpace& vfes,
                                       const std::vector<Vec3>& vecs,
                                       mfem::GridFunction& gf) {
  gf = 0.0;
  mfem::Array<int> vdofs;
  const int nv = vfes.GetMesh()->GetNV();
  for (int v = 0; v < nv; ++v) {
    vfes.GetVertexVDofs(v, vdofs);
    if (vdofs.Size() < 3) {
      throw std::runtime_error("Unexpected VertexVDofs size while writing vector GF");
    }
    for (int c = 0; c < 3; ++c) {
      int dof = vdofs[c];
      double sign = 1.0;
      if (dof < 0) {
        dof = -1 - dof;
        sign = -1.0;
      }
      gf[dof] = sign * vecs[static_cast<size_t>(v)][static_cast<size_t>(c)];
    }
  }
}

void WriteFiberFiles(const mfem::Mesh& mesh,
                     const std::vector<Vec3>& f,
                     const std::vector<Vec3>& s,
                     const std::vector<Vec3>& n,
                     const std::string& out_dir) {
  std::filesystem::create_directories(out_dir);

  const int dim = mesh.Dimension();
  mfem::H1_FECollection fec(1, dim);
  mfem::FiniteElementSpace vfes(const_cast<mfem::Mesh*>(&mesh), &fec, dim, mfem::Ordering::byVDIM);

  mfem::GridFunction gf_f(&vfes);
  mfem::GridFunction gf_s(&vfes);
  mfem::GridFunction gf_n(&vfes);

  FillGridFunctionFromVertexVectors(vfes, f, gf_f);
  FillGridFunctionFromVertexVectors(vfes, s, gf_s);
  FillGridFunctionFromVertexVectors(vfes, n, gf_n);

  {
    std::ofstream out(std::filesystem::path(out_dir) / "fiber_f.gf");
    if (!out) {
      throw std::runtime_error("Cannot write fiber_f.gf");
    }
    gf_f.Save(out);
  }
  {
    std::ofstream out(std::filesystem::path(out_dir) / "fiber_s.gf");
    if (!out) {
      throw std::runtime_error("Cannot write fiber_s.gf");
    }
    gf_s.Save(out);
  }
  {
    std::ofstream out(std::filesystem::path(out_dir) / "fiber_n.gf");
    if (!out) {
      throw std::runtime_error("Cannot write fiber_n.gf");
    }
    gf_n.Save(out);
  }
}

void EnsureNonEmpty(const std::vector<int>& values, const std::string& label) {
  if (values.empty()) {
    throw std::runtime_error("Empty required set: " + label);
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);

  if (mfem::Mpi::WorldSize() != 1) {
    if (mfem::Mpi::WorldRank() == 0) {
      std::cerr << "generate_regional_fibers must run with -np 1" << std::endl;
    }
    return 1;
  }

  try {
    const Options opt = ParseOptions(argc, argv);

    mfem::Mesh mesh(opt.mesh_path.c_str(), 1, 1);
    if (mesh.Dimension() != 3) {
      throw std::runtime_error("Only 3D meshes are supported");
    }

    const std::vector<Vec3> coords = CollectMeshVertices(mesh);
    std::vector<Vec3> fiber_f(coords.size(), Vec3{1.0, 0.0, 0.0});
    std::vector<Vec3> fiber_s(coords.size(), Vec3{0.0, 1.0, 0.0});
    std::vector<Vec3> fiber_n(coords.size(), Vec3{0.0, 0.0, 1.0});

    RegionData atria;
    RegionData vent;

    if (!opt.atria_attrs.empty()) {
      std::cout << "[atria] building region graph..." << std::endl;
      atria = BuildRegionData(mesh, ToSet(opt.atria_attrs));
      if (atria.vertices.empty()) {
        throw std::runtime_error("Atria attributes selected no vertices");
      }

      KdTree3 atria_tree(coords, atria.vertices);

      std::vector<AnchorData> anchors;
      anchors.reserve(opt.atria_anchors.size());
      for (const auto& spec : opt.atria_anchors) {
        std::cout << "[atria] loading anchor: " << spec.name << " from " << spec.path << std::endl;
        anchors.push_back(LoadAnchorData(spec, atria_tree, opt.csv_scale, opt.csv_shift));
      }

      std::cout << "[atria] stage A/B: initialize anchor vectors..." << std::endl;
      AtriaSeed seed = BuildAtriaSeed(anchors, atria, coords, opt);

      std::vector<Vec3> atria_f = seed.f;
      InitializeByNearestFixed(atria, coords, atria_f, seed.fixed);

      std::cout << "[atria] stage C: volumetric diffusion..." << std::endl;
      DiffuseField(atria, seed.fixed, atria_f, opt.atria_diffusion_iters);

      std::cout << "[atria] stage D: defect repair..." << std::endl;
      RepairDefects(atria, seed.fixed, atria_f, opt.atria_defect_iters, opt.atria_defect_threshold);

      std::cout << "[atria] stage E: orthonormal basis construction..." << std::endl;
      const std::vector<Vec3> atria_normals = ComputeRegionBoundaryNormals(mesh, atria, coords);
      BuildAtriaBasis(atria, atria_f, atria_normals, fiber_f, fiber_s, fiber_n);
    }

    if (!opt.ventricle_attrs.empty()) {
      std::cout << "[ventricles] building region graph..." << std::endl;
      vent = BuildRegionData(mesh, ToSet(opt.ventricle_attrs));
      if (vent.vertices.empty()) {
        throw std::runtime_error("Ventricle attributes selected no vertices");
      }

      for (const int v : vent.vertices) {
        if (!atria.mask.empty() && atria.mask[static_cast<size_t>(v)]) {
          throw std::runtime_error("Atria and ventricle volume attributes overlap on vertices");
        }
      }

      const std::vector<int> apex = CollectBoundaryVertices(mesh, ToSet(opt.vent_apex_bdr_attrs), vent);
      const std::vector<int> base = CollectBoundaryVertices(mesh, ToSet(opt.vent_base_bdr_attrs), vent);
      const std::vector<int> epi = CollectBoundaryVertices(mesh, ToSet(opt.vent_epi_bdr_attrs), vent);
      const std::vector<int> lv = CollectBoundaryVertices(mesh, ToSet(opt.vent_lv_bdr_attrs), vent);
      const std::vector<int> rv = CollectBoundaryVertices(mesh, ToSet(opt.vent_rv_bdr_attrs), vent);

      EnsureNonEmpty(apex, "vent-apex-bdr-attrs (intersect vent region)");
      EnsureNonEmpty(base, "vent-base-bdr-attrs (intersect vent region)");
      EnsureNonEmpty(epi, "vent-epi-bdr-attrs (intersect vent region)");
      EnsureNonEmpty(lv, "vent-lv-bdr-attrs (intersect vent region)");
      EnsureNonEmpty(rv, "vent-rv-bdr-attrs (intersect vent region)");

      std::cout << "[ventricles] solving four Laplace-like fields..." << std::endl;
      DirichletBC bc = MakeDirichletBC(mesh.GetNV());

      bc = MakeDirichletBC(mesh.GetNV());
      ApplyDirichlet(base, 1.0, bc);
      ApplyDirichlet(apex, 0.0, bc);
      const std::vector<double> psi_ab = SolveGraphLaplace(vent, bc, opt.vent_laplace_iters, opt.vent_laplace_tol);

      bc = MakeDirichletBC(mesh.GetNV());
      ApplyDirichlet(apex, 1.0, bc);
      ApplyDirichlet(epi, 1.0, bc);
      ApplyDirichlet(lv, 0.0, bc);
      ApplyDirichlet(rv, 0.0, bc);
      const std::vector<double> phi_epi = SolveGraphLaplace(vent, bc, opt.vent_laplace_iters, opt.vent_laplace_tol);

      bc = MakeDirichletBC(mesh.GetNV());
      ApplyDirichlet(lv, 1.0, bc);
      ApplyDirichlet(apex, 0.0, bc);
      ApplyDirichlet(epi, 0.0, bc);
      ApplyDirichlet(rv, 0.0, bc);
      const std::vector<double> phi_lv = SolveGraphLaplace(vent, bc, opt.vent_laplace_iters, opt.vent_laplace_tol);

      bc = MakeDirichletBC(mesh.GetNV());
      ApplyDirichlet(rv, 1.0, bc);
      ApplyDirichlet(apex, 0.0, bc);
      ApplyDirichlet(epi, 0.0, bc);
      ApplyDirichlet(lv, 0.0, bc);
      const std::vector<double> phi_rv = SolveGraphLaplace(vent, bc, opt.vent_laplace_iters, opt.vent_laplace_tol);

      const std::vector<Vec3> psi_ab_g = EstimateGradient(vent, coords, psi_ab);
      const std::vector<Vec3> phi_epi_g = EstimateGradient(vent, coords, phi_epi);
      const std::vector<Vec3> phi_lv_g = EstimateGradient(vent, coords, phi_lv);
      const std::vector<Vec3> phi_rv_g = EstimateGradient(vent, coords, phi_rv);

      std::cout << "[ventricles] applying cardioid-style bislerp orientation..." << std::endl;
      BuildVentricleBasis(vent,
                          psi_ab,
                          psi_ab_g,
                          phi_epi,
                          phi_epi_g,
                          phi_lv,
                          phi_lv_g,
                          phi_rv,
                          phi_rv_g,
                          opt,
                          fiber_f,
                          fiber_s,
                          fiber_n);
    }

    std::cout << "[output] writing GF files to: " << opt.out_dir << std::endl;
    WriteFiberFiles(mesh, fiber_f, fiber_s, fiber_n, opt.out_dir);

    std::cout << "Done. Generated fiber_f.gf, fiber_s.gf, fiber_n.gf" << std::endl;
  } catch (const std::exception& ex) {
    std::cerr << "generate_regional_fibers failed: " << ex.what() << std::endl;
    return 2;
  }

  return 0;
}
