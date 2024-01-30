#ifdef ENABLE_CUDA

#include "GPUMazeRouteBasic.cuh"

GPUMazeRouteBasic::GPUMazeRouteBasic(const std::shared_ptr<GPURouteContext> &context)
    : context(context)
{
  gamer = std::make_unique<BasicGamer>(context->getDIRECTION(), context->getN(), context->getX(), context->getY(), context->getLAYER(), context->getMaxNumPins());
  gamer->setWireCost(context->getWireCost());
  gamer->setNonStackViaCost(context->getNonStackViaCost());
  gamer->setUnitViaCost(context->getUnitViaCost());
}

void GPUMazeRouteBasic::run(const std::vector<int> &netIndices, int numTurns, int margin)
{
  // reverse
  for (auto netId : netIndices)
  {
    context->reverse(netId);
  }
  // route
  for (int netId : netIndices)
  {
    const auto &pinIndices = context->getPinIndices(netId);
    int rootIdx = pinIndices.front();
    utils::BoxT<int> box = context->getBoundingBox(netId);
    box.Set(std::max(0, box.lx() - margin), std::max(0, box.ly() - margin),
            std::min(context->getX(), box.hx() + margin), std::min(context->getY(), box.hy() + margin));
    context->updateCost(box);
    gamer->route(pinIndices, numTurns, box);
    // if (!gamer->getIsRouted())
    // {
    //   utils::log() << "gamer error: route net(id=" << netId << ") failed\n";
    //   exit(-1);
    // }
    context->commit(netId, rootIdx, gamer->getRoutes());
    std::vector<int> routes(context->getMaxNumPins() * MAX_ROUTE_LEN_PER_PIN);
    checkCudaErrors(cudaMemcpy(routes.data(), gamer->getRoutes().get(), routes.size() * sizeof(int), cudaMemcpyDeviceToHost));
  }
}

#endif