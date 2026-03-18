#include <cmath>
#include <iostream>
#include <pybind11/pybind11.h>
using namespace std;
double getAngle(double x1, double y1, double x2, double y2) {
  double dot = x1 * x2 + y1 * y2;
  double sq1 = sqrt(x1 * x1 + y1 * y1);
  double sq2 = sqrt(x2 * x2 + y2 * y2);
  if (sq1 == 0 || sq2 == 0)
    throw std::runtime_error("Vector magnitude cannot be zero.");
  double cosTheta = dot / (sq1 * sq2);
  if (cosTheta > 1.0)
    cosTheta = 1.0;
  if (cosTheta < -1.0)
    cosTheta = -1.0;
  return acos(cosTheta);
}
// m -> py::module_
PYBIND11_MODULE(_vector, m) {
  // m.def(name, func, doc)
  m.def("getAngle", &getAngle, "Calculate the angle between two vectors.");
}