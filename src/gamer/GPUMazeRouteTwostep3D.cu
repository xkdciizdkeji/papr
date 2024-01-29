#ifdef ENABLE_CUDA

#include "GPUMazeRouteTwostep3D.cuh"

GPUMazeRouteTwostep3D::GPUMazeRouteTwostep3D(const std::shared_ptr<GPURouteContext> &context)
    : context(context)
{
  int scaleX = std::min(context->getX() / 256, MAX_SCALE);
  int scaleY = std::min(context->getY() / 256, MAX_SCALE);
  scaler = std::make_unique<GridScaler>(context->getDIRECTION(), context->getN(), context->getX(), context->getY(), context->getLAYER(), scaleX, scaleY);
  scaler->setWireCost(context->getWireCost());
  scaler->setViaCost(context->getNonStackViaCost());
  coarseGamer = std::make_unique<BasicGamer>(context->getDIRECTION(), scaler->getCoarseN(), scaler->getCoarseX(), scaler->getCoarseY(), context->getLAYER(), context->getMaxNumPins());
  coarseGamer->setWireCost(scaler->getCoarseWireCost());
  coarseGamer->setNonStackViaCost(scaler->getCoarseViaCost());
  coarseGamer->setUnitViaCost(context->getUnitViaCost());
  fineGamer = std::make_unique<GuidedGamer>(context->getDIRECTION(), context->getN(), context->getX(), context->getY(), context->getLAYER(), context->getMaxNumPins());
  fineGamer->setWireCost(context->getWireCost());
  fineGamer->setNonStackViaCost(context->getNonStackViaCost());
  fineGamer->setUnitViaCost(context->getUnitViaCost());
}

void GPUMazeRouteTwostep3D::run(const std::vector<int> &netIndices, int numCoarseTurns, int numFineTurns, int margin)
{
  // reverse
  for (auto netId : netIndices)
  {
    context->reverse(netId);
  }

  // route
  std::vector<int> pin2DIndices(context->getMaxNumPins());
  std::vector<int> coarsePinIndices(context->getMaxNumPins());
  std::vector<int> coarseRoutes(context->getMaxNumPins() * MAX_ROUTE_LEN_PER_PIN);
  std::vector<std::array<int, 6>> guide;
  for (int netId : netIndices)
  {
    const std::vector<int> &pinIndices = context->getPinIndices(netId);
    int rootIdx = pinIndices.front();
    // bouding box
    utils::BoxT<int> box = context->getBoundingBox(netId);
    utils::BoxT<int> fineBox(
        std::max(0, box.lx() - margin), std::max(0, box.ly() - margin),
        std::min(context->getX(), box.hx() + margin), std::min(context->getY(), box.hy() + margin));
    utils::BoxT<int> coarseBox(
        fineBox.lx() / scaler->getScaleX(),
        fineBox.ly() / scaler->getScaleY(),
        std::min(scaler->getCoarseX(), (fineBox.hx() + scaler->getScaleX() - 1) / scaler->getScaleX()),
        std::min(scaler->getCoarseY(), (fineBox.hy() + scaler->getScaleY() - 1) / scaler->getScaleY()));
    fineBox.Set(
        coarseBox.lx() * scaler->getScaleX(),
        coarseBox.ly() * scaler->getScaleY(),
        std::min(context->getX(), coarseBox.hx() * scaler->getScaleX()),
        std::min(context->getY(), coarseBox.hy() * scaler->getScaleY()));
    // coarse 2d pins
    getCoarsePinIndices(coarsePinIndices, pinIndices);
    // compute guide
    guide.clear();
    context->updateCost(fineBox);
    if (box.width() < 5 * scaler->getScaleX() && box.height() < 5 * scaler->getScaleY())
      guide.push_back({fineBox.lx(), fineBox.ly(), 0, fineBox.hx() - 1, fineBox.hy() - 1, context->getLAYER() - 1});
    else
    {
      scaler->scale(coarseBox);
      coarseGamer->route(coarsePinIndices, numCoarseTurns, coarseBox);
      checkCudaErrors(cudaMemcpy(coarseRoutes.data(), coarseGamer->getRoutes().get(),
                                 coarseRoutes.size() * sizeof(int), cudaMemcpyDeviceToHost));
      getGuideFromCoarseRoutes(guide, coarseRoutes);
    }

    // fine routing using guides
    fineGamer->setGuide(guide);
    fineGamer->route(pinIndices, numFineTurns);
    // if (!fineGamer->getIsRouted())
    // {
    //   utils::log() << "gamer error: route net(id=" << netId << ") failed\n";
    //   exit(-1);
    // }
    context->commit(netId, rootIdx, fineGamer->getRoutes());
  }
}

void GPUMazeRouteTwostep3D::getCoarsePinIndices(std::vector<int> &coarsePinIndices, const std::vector<int> &pinIndices)
{
  // transform 3d pin to coarse 2d pin
  coarsePinIndices.resize(pinIndices.size());
  std::unordered_map<int, int> order(pinIndices.size());
  for (int i = 0; i < pinIndices.size(); i++)
  {
    auto [x, y, z] = idxToXYZ(pinIndices[i], context->getDIRECTION(), context->getN());
    int cx = x / scaler->getScaleX();
    int cy = y / scaler->getScaleY();
    coarsePinIndices[i] = xyzToIdx(cx, cy, z, context->getDIRECTION(), scaler->getCoarseN());
    order.try_emplace(coarsePinIndices[i], i);
  }
  // remove duplicate coarse 2d pins
  std::sort(coarsePinIndices.begin(), coarsePinIndices.end());
  auto last = std::unique(coarsePinIndices.begin(), coarsePinIndices.end());
  coarsePinIndices.erase(last, coarsePinIndices.end());
  // sort coarse 2d pins according to order
  std::sort(coarsePinIndices.begin(), coarsePinIndices.end(), [&](int left, int right)
            { return order[left] < order[right]; });
}

void GPUMazeRouteTwostep3D::getGuideFromCoarseRoutes(std::vector<std::array<int, 6>> &guide, const std::vector<int> &coarseRoutes)
{
  guide.clear();
  for (int i = 0; i < coarseRoutes[0]; i += 2)
  {
    auto [startX, startY, startZ] = idxToXYZ(coarseRoutes[1 + i], context->getDIRECTION(), scaler->getCoarseN());
    auto [endX, endY, endZ] = idxToXYZ(coarseRoutes[2 + i], context->getDIRECTION(), scaler->getCoarseN());
    guide.push_back({startX * scaler->getScaleX(),
                     startY * scaler->getScaleY(),
                     0,
                     std::min(context->getX(), (endX + 1) * scaler->getScaleX()) - 1,
                     std::min(context->getY(), (endY + 1) * scaler->getScaleY()) - 1,
                     context->getLAYER() - 1});
  }
}

#endif