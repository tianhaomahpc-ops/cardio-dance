// Half-ellipsoid hollow shell (left ventricle approximation), OpenCASCADE CSG.
//
// Outer ellipsoid centered at origin, semi-axes (a_out, b_out, c_out).
// Inner cavity ellipsoid same center, semi-axes (a_in, b_in, c_in).
// Half: x >= 0 only (basal plane at x = 0).
//
// Output: a tet mesh of the shell, with boundary attributes:
//   1 = epicardium (outer surface)
//   2 = endocardium (inner cavity surface)
//   3 = base       (x = 0 cutting plane)
//
// Usage:
//   gmsh -3 tools/half_ellipsoid.geo -o benchmarks/half_ellipsoid/heart.msh \
//        -setnumber h_mm 1.5
// Defaults below match those of generate_half_ellipsoid_case.cpp.

SetFactory("OpenCASCADE");

If (!Exists(a_out)) a_out = 35.0; EndIf
If (!Exists(b_out)) b_out = 22.0; EndIf
If (!Exists(c_out)) c_out = 22.0; EndIf
If (!Exists(a_in))  a_in  = 28.0; EndIf
If (!Exists(b_in))  b_in  = 15.0; EndIf
If (!Exists(c_in))  c_in  = 15.0; EndIf
If (!Exists(h_mm))  h_mm  = 1.5;  EndIf

// Build outer and inner ellipsoids by scaling a unit sphere.
Sphere(1) = {0, 0, 0, 1};
Dilate {{0, 0, 0}, {a_out, b_out, c_out}} { Volume{1}; }

Sphere(2) = {0, 0, 0, 1};
Dilate {{0, 0, 0}, {a_in, b_in, c_in}} { Volume{2}; }

// Subtract cavity from outer ellipsoid -> closed shell (volume 3).
BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

// Cut to x >= 0 by intersecting with a half-space.
Box(4) = {0, -2*b_out, -2*c_out, 2*a_out, 4*b_out, 4*c_out};
BooleanIntersection(5) = { Volume{3}; Delete; }{ Volume{4}; Delete; };

// After intersection, gmsh assigns surface tags automatically. Classify by
// surface centroid to set physical groups.
Mesh.CharacteristicLengthMin = h_mm;
Mesh.CharacteristicLengthMax = h_mm;
Mesh.Algorithm   = 6;   // Frontal-Delaunay 2D
Mesh.Algorithm3D = 1;   // Delaunay 3D
Mesh.MshFileVersion = 2.2;  // MFEM reads v2.2 cleanly

// Use Plugin to walk all surfaces of volume 5 and group by axial position.
// We rely on a robust trick: at the base, every surface point has x ~ 0;
// elsewhere we test mean radius vs both ellipsoids.

// Get all surfaces of volume 5.
surfs() = Boundary{ Volume{5}; };

epi[]  = {};
endo[] = {};
base[] = {};

For i In {0:#surfs[]-1}
  s = surfs[i];
  // Probe surface center to classify: get its bounding box centroid.
  bb[] = BoundingBox Surface{ s };
  cx = 0.5 * (bb[0] + bb[3]);
  cy = 0.5 * (bb[1] + bb[4]);
  cz = 0.5 * (bb[2] + bb[5]);

  // Score: distance from each ellipsoid surface in normalized form.
  outer_norm = Sqrt((cx*cx)/(a_out*a_out) + (cy*cy)/(b_out*b_out) + (cz*cz)/(c_out*c_out));
  inner_norm = Sqrt((cx*cx)/(a_in*a_in)   + (cy*cy)/(b_in*b_in)   + (cz*cz)/(c_in*c_in));
  outer_dist = Fabs(outer_norm - 1.0);
  inner_dist = Fabs(inner_norm - 1.0);

  If (cx < 0.5*h_mm)
    base[] += s;
  ElseIf (outer_dist < inner_dist)
    epi[]  += s;
  Else
    endo[] += s;
  EndIf
EndFor

Physical Surface("epicardium",  1) = epi[];
Physical Surface("endocardium", 2) = endo[];
Physical Surface("base",        3) = base[];
Physical Volume ("myocardium",  1) = {5};
