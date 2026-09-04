/**
 * @file fv/distributed.hpp
 * \brief Track C, rung C5 (Voronoi methods plan): the distributed static solvers — the ghost
 * field exchange over VoronoiHalo (core's ORB decomposition + particle halo) and the global
 * reductions, installed on CollocatedNS / PressureSolver through their hooks.
 *
 * Layout: this rank's tessellation is built for its owned seeds [0, nOwned) with the gathered
 * ghost seeds [nOwned, nCombined) as candidates (DistributedMovingTessellation's layout); the face
 * mesh (`buildFaceMesh(view, aux, nOwned)`) keeps the facets toward ghosts as INTERFACE faces
 * owned here. Cell fields are sized nCombined; the owned part is stepped, the ghost part is
 * refreshed from the owning ranks by `exchange` (VoronoiHalo::forward over the established
 * topology) — velocity, its gradient, the pressure, the transpose moments — so an interface face
 * computes the same flux on both ranks from identical inputs. Pressure PCG: the matvec exchanges
 * the search direction, the dot products are all-reduced, the AMG preconditioner is the per-rank
 * owned block (block Jacobi). np = 1 is the single-rank code path (no ghosts, identity hooks).
 *
 * The exchange packs on the device (gather kernel over the flat topology's send list) and
 * scatters the receives into the ghost slots on the device; only the packed buffers cross to the
 * host for MPI (host-staged, like core's GPU-resident GridHalo variant). CUDA-aware MPI on the
 * device buffers is the remaining step.
 */
#ifndef PECLET_VORO_FV_DISTRIBUTED_HPP
#define PECLET_VORO_FV_DISTRIBUTED_HPP

#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <type_traits>
#include <vector>

#include "peclet/voro/fv/operators.hpp"
#include "peclet/voro/mpi/voronoi_halo.hpp"

namespace peclet::voro::fv {

template <class Real>
struct GhostExchange {
  using Mem = peclet::core::MemSpace;
  using Exec = peclet::core::ExecSpace;
  peclet::voro::mpi::VoronoiHalo<Real>* halo = nullptr;
  int nOwned = 0, nGhost = 0;
  /// Device-packed path (default): the sends are gathered on the device from the flat topology's
  /// index list into one buffer, only the packed send/recv buffers cross to the host for MPI, and
  /// the receives land in the ghost slots by a device scatter — no whole-field round trip. The
  /// host path (`devicePack = false`, VoronoiHalo::forward per component) is the reference; the
  /// two are gated bitwise-equal in tests/kokkos_mpi/test_flow_mpi.
  bool devicePack = true;
  // flat topology (host) + device send index list
  std::vector<int> sendRanks, sendCounts, sendOffsets, recvRanks, recvCounts, recvOffsets;
  int nSend = 0, nReceived = 0;
  Kokkos::View<int*, Mem> dSendIdx, dSelfIdx;
  Kokkos::View<Real*, Mem> dSendBuf, dRecvBuf;
  std::vector<Real> hSendBuf, hRecvBuf;

  void init(peclet::voro::mpi::VoronoiHalo<Real>& h, int owned) {
    halo = &h;
    nOwned = owned;
    nGhost = h.numGhost();
    auto t = h.flatTopology();
    sendRanks = t.sendRanks;
    sendCounts = t.sendCounts;
    sendOffsets = t.sendOffsets;
    recvRanks = t.recvRanks;
    recvCounts = t.recvCounts;
    recvOffsets.assign(t.recvOffsets.begin(), t.recvOffsets.end());
    nSend = (int)t.sendIdx.size();
    nReceived = (int)t.numReceived;
    dSendIdx = Kokkos::View<int*, Mem>("gx.sendIdx", nSend);
    {
      auto hh = Kokkos::create_mirror_view(dSendIdx);
      for (int i = 0; i < nSend; ++i)
        hh(i) = (int)t.sendIdx[i];
      Kokkos::deep_copy(dSendIdx, hh);
    }
    dSelfIdx = Kokkos::View<int*, Mem>("gx.selfIdx", t.selfIdx.size());
    {
      auto hh = Kokkos::create_mirror_view(dSelfIdx);
      for (std::size_t i = 0; i < t.selfIdx.size(); ++i)
        hh(i) = (int)t.selfIdx[i];
      Kokkos::deep_copy(dSelfIdx, hh);
    }
  }
  /// Refresh the ghost entries [nOwned, nOwned + nGhost) of an `nc`-component cell field.
  void exchange(const DV<Real>& f, int nc) const {
    if (!halo || nGhost == 0)
      return;
    if (!devicePack) {
      exchangeHost(f, nc);
      return;
    }
    auto* self = const_cast<GhostExchange*>(this);
    const std::size_t sb = (std::size_t)nc * nSend, rb = (std::size_t)nc * nReceived;
    if (self->dSendBuf.extent(0) < sb)
      self->dSendBuf = Kokkos::View<Real*, Mem>("gx.sendBuf", sb);
    if (self->dRecvBuf.extent(0) < rb)
      self->dRecvBuf = Kokkos::View<Real*, Mem>("gx.recvBuf", rb);
    self->hSendBuf.resize(sb);
    self->hRecvBuf.resize(rb);
    // pack on the device: buffer[nc*k + c] = f[nc*sendIdx[k] + c]
    {
      const auto idx = dSendIdx;
      const auto buf = dSendBuf;
      const int ncl = nc;
      Kokkos::parallel_for(
          "gx.pack", Kokkos::RangePolicy<Exec>(0, nSend), KOKKOS_LAMBDA(const int k) {
            for (int c = 0; c < ncl; ++c)
              buf(ncl * k + c) = f(ncl * idx(k) + c);
          });
    }
    Kokkos::deep_copy(Kokkos::View<Real*, Kokkos::HostSpace>(self->hSendBuf.data(), sb),
                      Kokkos::subview(dSendBuf, std::make_pair((std::size_t)0, sb)));
    const int ns = (int)sendRanks.size(), nr = (int)recvRanks.size();
    std::vector<MPI_Request> sreq(ns, MPI_REQUEST_NULL), rreq(nr, MPI_REQUEST_NULL);
    const MPI_Datatype T = std::is_same_v<Real, double> ? MPI_DOUBLE : MPI_FLOAT;
    for (int p = 0; p < nr; ++p)
      MPI_Irecv(self->hRecvBuf.data() + (std::size_t)nc * recvOffsets[p], nc * recvCounts[p], T,
                recvRanks[p], 7601, halo->comm(), &rreq[p]);
    for (int k = 0; k < ns; ++k)
      MPI_Isend(self->hSendBuf.data() + (std::size_t)nc * sendOffsets[k], nc * sendCounts[k], T,
                sendRanks[k], 7601, halo->comm(), &sreq[k]);
    MPI_Waitall(nr, rreq.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(ns, sreq.data(), MPI_STATUSES_IGNORE);
    // the received ghosts are contiguous in ghost order: scatter straight into the ghost slots
    if (rb > 0) {
      Kokkos::deep_copy(Kokkos::subview(dRecvBuf, std::make_pair((std::size_t)0, rb)),
                        Kokkos::View<const Real*, Kokkos::HostSpace>(self->hRecvBuf.data(), rb));
      const auto buf = dRecvBuf;
      const int ncl = nc, base = nc * nOwned;
      Kokkos::parallel_for(
          "gx.unpack", Kokkos::RangePolicy<Exec>(0, (int)rb),
          KOKKOS_LAMBDA(const int i) { f(base + i) = buf(i); });
    }
    // local periodic self-ghosts (none with includePeriodicSelf = false)
    const int nSelf = (int)dSelfIdx.extent(0);
    if (nSelf > 0) {
      const auto sidx = dSelfIdx;
      const int ncl = nc, base = nc * (nOwned + nReceived);
      Kokkos::parallel_for(
          "gx.self", Kokkos::RangePolicy<Exec>(0, nSelf), KOKKOS_LAMBDA(const int j) {
            for (int c = 0; c < ncl; ++c)
              f(base + ncl * j + c) = f(ncl * sidx(j) + c);
          });
    }
    Kokkos::fence();
  }
  /// The host reference path: VoronoiHalo::forward per component through a host mirror.
  void exchangeHost(const DV<Real>& f, int nc) const {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f);
    std::vector<Real> own(nOwned), gh(nGhost);
    for (int c = 0; c < nc; ++c) {
      for (int i = 0; i < nOwned; ++i)
        own[i] = h(nc * i + c);
      halo->forward(own.data(), gh.data());
      for (int j = 0; j < nGhost; ++j)
        h(nc * (nOwned + j) + c) = gh[j];
    }
    Kokkos::deep_copy(f, h);
  }
  Real sum(Real v) const {
    if (!halo)
      return v;
    Real g = 0;
    MPI_Allreduce(&v, &g, 1, std::is_same_v<Real, double> ? MPI_DOUBLE : MPI_FLOAT, MPI_SUM,
                  halo->comm());
    return g;
  }
  Real max(Real v) const {
    if (!halo)
      return v;
    Real g = 0;
    MPI_Allreduce(&v, &g, 1, std::is_same_v<Real, double> ? MPI_DOUBLE : MPI_FLOAT, MPI_MAX,
                  halo->comm());
    return g;
  }
};

}  // namespace peclet::voro::fv

#endif  // PECLET_VORO_FV_DISTRIBUTED_HPP
