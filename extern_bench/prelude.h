#pragma once
#include <functional>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <fstream>
#include <limits>
#include <stack>
#include <string>
// standalone mode forgets these forward decls (only declared #ifndef STANDALONE_CONVEX_CELL)
namespace GEO { class Mesh; class PeriodicDelaunay3d; }
#ifndef geo_assert
#define geo_assert(x) assert(x)
#define geo_debug_assert(x) assert(x)
#define geo_assert_not_reached assert(0)
#define geo_argused(x) (void)(x)
#define geo_debug(x)
#define geo_debug_abort()
#endif
