#ifdef ENABLE_CUDA
#include "gamer_utils.cuh"
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <algorithm>
#include <numeric>
#include <stack>

// -------------------------------------
// GR Tree
// -------------------------------------

std::shared_ptr<GRTreeNode> extractGRTree(const int *routes, int rootIdx, int DIRECTION, int N)
{
  auto [rootX, rootY, rootZ] = idxToXYZ(rootIdx, DIRECTION, N);
  auto root = std::make_shared<GRTreeNode>(rootZ, rootX, rootY);
  if (routes[0] < 2)
    return root;

  auto hash = [&](const int3 &p)
  {
    return std::hash<int>{}(xyzToIdx(p.x, p.y, p.z, DIRECTION, N));
  };
  auto equal = [&](const int3 &p, const int3 &q)
  {
    return p.x == q.x && p.y == q.y && p.z == q.z;
  };
  auto compw = [&](const int3 &p, const int3 &q)
  {
    if (p.z == q.z)
    {
      if ((p.z & 1) ^ DIRECTION) // Y-X
        return (p.y < q.y) || (p.y == q.y && p.x < q.x);
      else
        return (p.x < q.x) || (p.x == q.x && p.y < q.y);
    }
    else
      return p.z < q.z;
  };
  auto compv = [&](const int3 &p, const int3 &q)
  {
    return (p.x < q.x) || (p.x == q.x && p.y < q.y) || (p.x == q.x && p.y == q.y && p.z < q.z);
  };
  // 收集所有交点
  std::vector<int3> allpoints;
  for (int i = 0; i < routes[0]; i += 2)
  {
    allpoints.push_back(idxToXYZ(routes[1 + i], DIRECTION, N));
    allpoints.push_back(idxToXYZ(routes[2 + i], DIRECTION, N));
  }
  std::sort(allpoints.begin(), allpoints.end(), compw);
  auto last = std::unique(allpoints.begin(), allpoints.end(), equal);
  allpoints.erase(last, allpoints.end());
  // 根据交点拆分线段
  std::vector<std::pair<int3, int3>> segments;
  for (int i = 0; i < routes[0]; i += 2)
  {
    int3 start = idxToXYZ(routes[1 + i], DIRECTION, N);
    int3 end = idxToXYZ(routes[2 + i], DIRECTION, N);
    if (start.z == end.z)
    {
      auto startIt = std::lower_bound(allpoints.begin(), allpoints.end(), start, compw);
      auto endIt = std::upper_bound(allpoints.begin(), allpoints.end(), end, compw);
      for (auto it = startIt, nextIt = startIt + 1; nextIt != endIt; it++, nextIt++)
        segments.emplace_back(*it, *nextIt);
    }
  }
  std::sort(allpoints.begin(), allpoints.end(), compv);
  for (int i = 0; i < routes[0]; i += 2)
  {
    int3 start = idxToXYZ(routes[1 + i], DIRECTION, N);
    int3 end = idxToXYZ(routes[2 + i], DIRECTION, N);
    if (start.z != end.z)
    {
      auto startIt = std::lower_bound(allpoints.begin(), allpoints.end(), start, compv);
      auto endIt = std::upper_bound(allpoints.begin(), allpoints.end(), end, compv);
      for (auto it = startIt, nextIt = it + 1; nextIt != endIt; it++, nextIt++)
        segments.emplace_back(*it, *nextIt);
    }
  }
  // 建立连接
  std::unordered_map<int3, std::vector<int3>, decltype(hash), decltype(equal)> linkGraph(allpoints.size(), hash, equal);
  for (const auto &[s, e] : segments)
  {
    linkGraph[s].push_back(e);
    linkGraph[e].push_back(s);
  }
  // extract GRTree using `linkGraph`
  std::unordered_map<int3, bool, decltype(hash), decltype(equal)> mark(allpoints.size(), hash, equal);
  for (const auto &[p, link] : linkGraph)
    mark.emplace(p, false);
  std::stack<std::shared_ptr<GRTreeNode>> stack;
  stack.push(root);
  mark.at(idxToXYZ(rootIdx, DIRECTION, N)) = true;
  while (!stack.empty())
  {
    auto node = stack.top();
    stack.pop();
    const auto &link = linkGraph.at(make_int3(node->x, node->y, node->layerIdx));
    for (const int3 &q : link)
    {
      if (mark.at(q) == false)
      {
        auto child = std::make_shared<GRTreeNode>(q.z, q.x, q.y);
        node->children.push_back(child);
        stack.push(child);
        mark.at(q) = true;
      }
    }
  }
  return root;
}
#endif