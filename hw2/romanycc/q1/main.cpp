#include <iostream>
#include <vector>
class Line {
public:
  /// Default constructor
  Line() = default;
  /// Copy constructor
  Line(Line const &) = default;
  /// Move constructor
  Line(Line &&) noexcept = default;
  /// Copy assignment operator
  Line &operator=(Line const &) = default;
  /// Move assignment operator
  Line &operator=(Line &&) noexcept = default;
  /// Constructor with size
  Line(size_t size) : _x(size), _y(size) {}
  /// Destructor
  ~Line() = default;
  /// Get the size of the line
  size_t size() const { return _x.size(); }
  /// Get the x-coordinate of the it-th point
  // .at() will throw std::out_of_range exception
  float const &x(size_t it) const { return _x.at(it); }
  float &x(size_t it) { return _x.at(it); }
  /// Get the y-coordinate of the it-th point
  float const &y(size_t it) const { return _y.at(it); }
  float &y(size_t it) { return _y.at(it); }

private:
  // Member data.
  std::vector<float> _x;
  std::vector<float> _y;
}; /* end class Line */

int main(int, char **) {
  Line line(3);
  line.x(0) = 0;
  line.y(0) = 1;
  line.x(1) = 1;
  line.y(1) = 3;
  line.x(2) = 2;
  line.y(2) = 5;

  Line line2(line);
  line2.x(0) = 9;

  std::cout << "line: number of points = " << line.size() << std::endl;
  for (size_t it = 0; it < line.size(); ++it) {
    std::cout << "point " << it << ":"
              << " x = " << line.x(it) << " y = " << line.y(it) << std::endl;
  }

  std::cout << "line2: number of points = " << line.size() << std::endl;
  for (size_t it = 0; it < line.size(); ++it) {
    std::cout << "point " << it << ":"
              << " x = " << line2.x(it) << " y = " << line2.y(it) << std::endl;
  }

  return 0;
}