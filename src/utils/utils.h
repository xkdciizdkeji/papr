#pragma once

#include "geo.h"
#include "log.h"
#include "prettyprint.h"
#include "enum.h"
#include "parameters.h"
#include "robin_hood.h"

using utils::log;
using utils::logeol;
using utils::loghline;
using utils::logmem;

using DBU = std::int64_t;
using CostT = double;
using CapacityT = double;
