#ifdef ENABLE_CUDA

#include "GPUMazeRouteTwostep.cuh" extractor

GPUMazeRouteTwostep::GPUMazeRouteTwostep(const std::shared_ptr<GPURouteContext> &context)
    : context(context)
{
  int scaleX = std::min(context->getX() / 256, MAX_SCALE);
  int scaleY = std::min(context->getY() / 256, MAX_SCALE);
  extractor = std::make_unique<Grid2DExtractor>(context->getDIRECTION(), context->getN(), context->getX(), context->getY(), context->getLAYER());
  extractor->setWireCost(context->getWireCost());
  scaler = std::make_unique<GridScaler2D>(context->getX(), context->getY(), scaleX, scaleY);
  scaler->setCost2D(extractor->getCost2D());
  coarseGamer = std::make_unique<BasicGamer2D>(scaler->getCoarseX(), scaler->getCoarseY(), context->getMaxNumPins());
  coarseGamer->setCost2D(scaler->getCoarseCost2D());
  fineGamer = std::make_unique<GuidedGamer>(context->getDIRECTION(), context->getN(), context->getX(), context->getY(), context->getLAYER(), context->getMaxNumPins());
  fineGamer->setWireCost(context->getWireCost());
  fineGamer->setNonStackViaCost(context->getNonStackViaCost());
  fineGamer->setUnitViaCost(context->getUnitViaCost());
}

void GPUMazeRouteTwostep::run(const std::vector<int> &netIndices, int numCoarseTurns, int numFineTurns, int margin)
{
  // reverse
  for (auto netId : netIndices)
  {
    context->reverse(netId);
  }

  // route
  std::vector<int> pin2DIndices(context->getMaxNumPins());
  std::vector<int> coarsePin2DIndices(context->getMaxNumPins());
  std::vector<int> coarseRoutes2D(context->getMaxNumPins() * MAX_ROUTE_LEN_PER_PIN);
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
    getCoarsePin2DIndices(coarsePin2DIndices, pinIndices);
    // compute guide2D
    guide.clear();
    context->updateCost(fineBox);
    if (box.width() < 5 * scaler->getScaleX() && box.height() < 5 * scaler->getScaleY())
      guide.push_back({fineBox.lx(), fineBox.ly(), 0, fineBox.hx() - 1, fineBox.hy() - 1, context->getLAYER() - 1});
    else
    {
      extractor->extract(fineBox);
      scaler->scale(coarseBox);
      coarseGamer->route(coarsePin2DIndices, numCoarseTurns, coarseBox);
      checkCudaErrors(cudaMemcpy(coarseRoutes2D.data(), coarseGamer->getRoutes2D().get(),
                                 coarseRoutes2D.size() * sizeof(int), cudaMemcpyDeviceToHost));
      getGuideFromCoarseRoutes2D(guide, coarseRoutes2D);
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

void GPUMazeRouteTwostep::getCoarsePin2DIndices(std::vector<int> &coarsePin2DIndices, const std::vector<int> &pinIndices)
{
  // transform 3d pin to coarse 2d pin
  coarsePin2DIndices.resize(pinIndices.size());
  std::unordered_map<int, int> order(pinIndices.size());
  for (int i = 0; i < pinIndices.size(); i++)
  {
    auto [x, y, z] = idxToXYZ(pinIndices[i], context->getDIRECTION(), context->getN());
    int cx = x / scaler->getScaleX();
    int cy = y / scaler->getScaleY();
    coarsePin2DIndices[i] = cx + cy * scaler->getCoarseX();
    order.try_emplace(coarsePin2DIndices[i], i);
  }
  // remove duplicate coarse 2d pins
  std::sort(coarsePin2DIndices.begin(), coarsePin2DIndices.end());
  auto last = std::unique(coarsePin2DIndices.begin(), coarsePin2DIndices.end());
  coarsePin2DIndices.erase(last, coarsePin2DIndices.end());
  // sort coarse 2d pins according to order
  std::sort(coarsePin2DIndices.begin(), coarsePin2DIndices.end(), [&](int left, int right)
            { return order[left] < order[right]; });
}

void GPUMazeRouteTwostep::getGuideFromCoarseRoutes2D(std::vector<std::array<int, 6>> &guide, const std::vector<int> &coarseRoutes2D)
{
  guide.clear();
  for (int i = 0; i < coarseRoutes2D[0]; i += 2)
  {
    int startX = coarseRoutes2D[1 + i] % scaler->getCoarseX();
    int startY = coarseRoutes2D[1 + i] / scaler->getCoarseX();
    int endX = coarseRoutes2D[2 + i] % scaler->getCoarseX();
    int endY = coarseRoutes2D[2 + i] / scaler->getCoarseX();
    guide.push_back({startX * scaler->getScaleX(),
                     startY * scaler->getScaleY(),
                     0,
                     std::min(context->getX(), (endX + 1) * scaler->getScaleX()) - 1,
                     std::min(context->getY(), (endY + 1) * scaler->getScaleY()) - 1,
                     context->getLAYER() - 1});
  }
}

#endif