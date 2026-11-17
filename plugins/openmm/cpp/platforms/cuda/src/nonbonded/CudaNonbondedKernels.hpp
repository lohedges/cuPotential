// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NV_CUDA_KERNELS_H_
#define NV_CUDA_KERNELS_H_

#include "nvopenmm/Kernels.h"
#include "InteractionModel.hpp"
#include "nvopenmm/internal/NonbondedForceImpl.h"   // calcDispersionCorrection
#include "openmm/common/ComputeContext.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "../bonded/GroupedBonded.hpp"
#include "cupo/box.h"                // CUPO_BOX_FLOATS, the device box layout
#include "cupo/cell_list.h"          // cupo_cell_list_handle
#include "cupo/dspace.h"  // cupo_dspace_handle
#include "cupo/kspace.h"               // cupo_kspace_handle
#include "cupo/splitting_plan.h"       // cupo_splitting_plan_handle
#include "nvopenmm/NonbondedForce.h"   // NonbondedForce::NonbondedMethod

namespace NvOpenMM {

class CustomPairParameters;
class ExpressionTailCorrection;

/**
 * CUDA implementation of CalcNonbondedForceKernel -- our analogue of OpenMM's
 * CommonCalcNonbondedForceKernel.
 *
 * Standard and cutoff Custom modes share cell-list construction, direct-space
 * execution, CUDA streams/graphs and deferred energy collection. Each force owns
 * one DSpace and optional independent Coulomb/dispersion reciprocal workspaces
 * with shared mesh dimensions, in either parameter mode.
 * CustomPairParameters adapts OpenMM inputs to libcupo's expression instance.
 * Device workspaces are allocated lazily on the first execute().
 */
class CudaCalcNonbondedForce : public CalcNonbondedForceKernel {
public:
    CudaCalcNonbondedForce(const std::string& name, const OpenMM::Platform& platform,
                          OpenMM::ComputeContext& cc);
    ~CudaCalcNonbondedForce();

    void initialize(const OpenMM::System& system, const NonbondedForce& force,
                    float boxSafetyFactor);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const NonbondedForce& force,
                                 int firstParticle, int lastParticle, int firstException, int lastException);

    /**
     * Called by ReorderListener after OpenMM permutes atom order: re-permutes and
     * re-uploads the per-atom data. The cell list needs no help (see the definition).
     */
    void applyAtomReorder();

    /**
     * Deferred graph join for bonded-overlap mode. Invoked by the registered
     * ForcePostComputation after OpenMM enqueues its own bonded kernel, so the
     * bonded kernel overlaps our side-stream nonbonded graph.
     */
    void graphJoin(cudaStream_t ws);

    /**
     * Read back and return this evaluation's potential energy, or 0 if none was
     * requested. Invoked from the ForcePostComputation rather than from execute()
     * so the host does not block before OpenMM has submitted its bonded kernel --
     * see the comment at the end of execute().
     */
    double finishPendingEnergy();


private:
    struct GraphSlot;   // defined with the graph state below

    // Canonical Standard parameters are prepared on the host, then committed
    // together. A bad effective parameter or divergent tail cannot leave a
    // partially updated live Context behind.
    struct ParameterOffset {
        int parameter, index;
        double scale[3];
    };
    // Base tables and offset laws change only through a public parameter update.
    // Live proposals share this immutable definition and own their numeric state.
    struct BuiltinParameterDefinition {
        std::vector<double> baseParticles, baseExceptions;
        std::vector<ParameterOffset> particleOffsets, exceptionOffsets;
        std::vector<std::string> globalNames, watchedNames;
        CutoffInteraction::TailPolicy tail = CutoffInteraction::NoTail;
        CutoffInteraction tailReference;
        bool tailFollowsParameters = true;
        double defaultTailCoefficient = 0.0;
    };
    struct BuiltinParameters {
        std::shared_ptr<const BuiltinParameterDefinition> definition;
        std::vector<double> particles, exceptions, values;
        double tailCoefficient = 0.0;
        // Set by a ranged update: only these rows differ from the active packed records.
        bool ranged = false;
        int firstParticle = 0, lastParticle = -1, firstException = 0, lastException = -1;
    };
    enum ParameterChange : unsigned {
        ParticlesChanged = 1u << 0,
        ExceptionsChanged = 1u << 1,
    };

    // ---- host-side setup and parameter updates ----
    BuiltinParameters readBuiltinParameters(const NonbondedForce& force) const;
    void readOffsetValues(BuiltinParameters& parameters, OpenMM::ContextImpl& context) const;
    unsigned applyOffsets(BuiltinParameters& parameters,
                          const std::vector<double>& before) const;
    std::vector<float> buildCupoParams(const BuiltinParameters& parameters) const;
    std::vector<cupo_tuples_pair_parameters> buildBondedPairParameters(
        const BuiltinParameters& parameters) const;
    double builtinTailCoefficient(const BuiltinParameters& parameters) const;
    void allocateHostData(const std::vector<int>& exclusion_pairs);
    void pushBuiltinParameters(BuiltinParameters&& parameters, unsigned changed,
                               bool invalidateMolecules);
    void updateOffsetParameters(OpenMM::ContextImpl& context);
    bool refreshLatticeInputs();
    void waitForSteps() const;

    // ---- device lifecycle ----
    void createCudaResources();
    void destroyCudaResources();
    void freeDevice();
    void setupDevice(int natoms, const float box[CUPO_BOX_FLOATS]);

    // ---- per-step compute (decomposed) ----
    void runStep(const float4* posq, unsigned long long* force, int natoms, int paddedN,
                 const float box[CUPO_BOX_FLOATS], bool computeEnergy, bool computeForces,
                 cudaStream_t stream);
    void updateBoxState(const float box[CUPO_BOX_FLOATS]);
    void replayCapturedGraph(cudaGraphExec_t exec, cudaStream_t stream);
    void uploadBoxIfChanged(cudaStream_t s);
    void dispatchStep(const float4* posq, unsigned long long* force, int natoms, int paddedN,
                      bool computeEnergy, bool computeForces, cudaStream_t ws, bool wholeCapture = false);
    void endCaptureAndInstantiate(GraphSlot& slot, bool computeEnergy, bool computeForces,
                                  cudaStream_t ws);
    double readEnergy();

    // ---- atom reorder / parameter update ----
    void reorderHostData(const int* atomIndex, int natoms);
    void applyAtomReorderImpl();
    void validateExclusionTopology();
    void copyBuiltinParameters(OpenMM::ContextImpl& context, const NonbondedForce& force);
    bool copyChangedBuiltinParameters(OpenMM::ContextImpl& context, const NonbondedForce& force,
                                      int firstParticle, int lastParticle, int firstException, int lastException);

    // ======================= state =======================
    OpenMM::ComputeContext& cc_;
    std::optional<InteractionModel> model_;
    bool box_triclinic_ = false;  // fixed cell-list geometry
    std::unique_ptr<CustomPairParameters> custom_parameters_;
    // One implicit expression tail plus explicitly retained source definitions.
    // These are CPU scalar corrections and introduce no GPU traversal.
    std::vector<std::unique_ptr<ExpressionTailCorrection>> expression_tails_;
    int additional_tail_count_ = 0;
    bool implicit_expression_tail_ = false;
    bool device_initialized_ = false;   // first-execute lazy setup done
    bool setup_started_ = false;        // a partial setup requires a fresh Context
    const void* particle_input_ = nullptr;  // borrowed from standard/custom parameters
    size_t particle_stride_bytes_ = 0;

    // Standard parameter ownership is independent of lattice terms. CustomPairParameters
    // owns their expression records and device globals.
    BuiltinParameters builtin_parameters_;

    // System-order copies: what a reorder and a parameter update re-derive the
    // GPU-order arrays from (applyAtomReorderImpl).
    std::vector<float> sysorder_params_;
    std::vector<float> sysorder_dispersion_weights_, reordered_dispersion_weights_;
    std::vector<float> reordered_charge_weights_;
    uint64_t charge_weights_version_ = 0, dispersion_weights_version_ = 0;
    std::vector<int> sysorder_exclusion_pairs_;
    // Native cancellation topology is fixed in GPU input order. Custom owns a
    // live remapped pair list. Bare physical pairs always use logical atom IDs.
    std::unordered_set<std::uint64_t> native_exclusion_pairs_;
    bool exclusion_topology_dirty_ = true;
    // Structural convention for physical replacements only. Cancellation retains
    // ordinary periodic pair geometry; reciprocal corrections always use MIC.
    bool exceptions_use_pbc_ = false;
    std::vector<cupo_tuples_pair_parameters> exception_parameters_;
    std::unique_ptr<GroupedBondedPairs> bonded_pairs_;

    // One direct workspace and optional Coulomb/dispersion workspaces share cells.
    // Energy slots are direct, Coulomb reciprocal and dispersion reciprocal.
    cupo_cell_list_handle cell_list_ = nullptr;
    cupo_dspace_handle dspace_ = nullptr;
    cupo_kspace_handle kspace_ = nullptr;
    cupo_kspace_handle dispersion_kspace_ = nullptr;
    bool piece_tuning_attempted_ = false;
    double* d_energy_ = nullptr;

    // Private CUDA streams + events for the per-step multi-stream overlap. The
    // misc stream clears direct accumulators while dspace builds the cell list.
    // Cancellation/correction and towerplate then run on dspace; reciprocal work
    // runs on kspace. The misc clear is joined through dspace before accumulation.
    cudaStream_t dspace_stream_ = nullptr;
    cudaStream_t kspace_stream_ = nullptr;
    cudaStream_t misc_stream_ = nullptr;
    cudaEvent_t step_fork_event_ = nullptr;
    cudaEvent_t force_clear_done_event_ = nullptr;
    cudaEvent_t cells_ready_event_ = nullptr;
    cudaEvent_t bonded_cells_ready_event_ = nullptr;
    cudaEvent_t dspace_done_event_ = nullptr;
    cudaEvent_t kspace_done_event_ = nullptr;

    // CUDA-graph capture of the per-step compute (default path;
    // CUPO_PLUGIN_CUDA_GRAPH=0 disables). NVT / fixed-box + NPT band-cross resume.
    //
    // One graph per (computeEnergy, computeForces) combination, indexed
    // ce*2 + cf. A single slot is not enough, because a run asks for more than
    // one combination: the integrator wants force-only every step, and under NPT
    // a barostat wants energy-only twice per move. With one slot the first
    // combination to warm up claims it and every other combination then runs
    // EAGERLY FOREVER -- the capture gate requires an uncaptured slot, so it
    // never fires again. Measured on apoA1 NPT, that left the barostat's two
    // evaluations at ~356 us each against a 246 us replay of strictly more work,
    // most of the gap being per-launch overhead the graph amortizes (a one-thread
    // kernel, cuda_kspace_begin_spread, averaged 32.5 us eager).
    //
    // It also removes a latent cliff: the warmup counter is shared, so with one
    // slot whichever combination happened to land on step 51 won it. Force-only
    // wins in practice only because it is ~25x more frequent.
    static const int GRAPH_SLOTS = 4;
    struct GraphSlot {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        bool captured = false;
        int seen = 0;          // eager runs of this combination so far
    };
    GraphSlot graph_slots_[GRAPH_SLOTS];
    bool graph_enabled_ = false;
    int  graph_warmup_ = 0;    // eager steps of ANY combination; gates all capture
    cudaStream_t graph_origin = nullptr;   // capturable origin (default stream can't capture)
    cudaEvent_t bridge_in_event_ = nullptr;
    cudaEvent_t bridge_out_event_ = nullptr;
    bool graph_side_launched_ = false;
    // Set by execute() when this evaluation asked for energy; consumed by
    // finishPendingEnergy() at the end of the force evaluation.
    bool energy_pending_ = false;
    // This step's box and the box last uploaded to d_box_, both in the cupo/box.h
    // layout (diagonal first, then b_x, c_x, c_y).
    float h_box_[CUPO_BOX_FLOATS] = {};
    float box_on_device_[CUPO_BOX_FLOATS] = {};
    cudaEvent_t box_upload_done_ = nullptr;
    bool box_upload_pending_ = false;

    long debug_graph_capture_count = 0;      // (re)captures so far; asserted by the NPT test
    bool debug_graph_trace_ = false;          // CUPO_PLUGIN_GRAPH_TRACE=1
    // Atom reorders applied so far. Traced under CUPO_PLUGIN_GRAPH_TRACE so a test
    // can prove a reorder actually fired -- OpenMM exposes no such signal, and a
    // reorder test that cannot see one passes vacuously.
    long reorder_count_ = 0;

    // Mesh dimensions also depend on the initial box and allocation headroom.
    int kspace_nx_ = 0, kspace_ny_ = 0, kspace_nz_ = 0;
    float box_safety_factor_ = 1.0f;   // stashed at init; consumed at setupDevice
    int natoms_ = 0;
    int n_exclusion_pairs_ = 0;

    // Particle values and pair endpoints are permuted into current GPU order.
    // Exception parameters stay in the original system exception order.
    float* h_params_ = nullptr;
    int* h_exclusion_pairs_ = nullptr;

    // device buffers (owned)
    float* d_box_ = nullptr;             // CUPO_BOX_FLOATS, cupo/box.h layout
    float* d_params_ = nullptr;
    float* d_charge_weights_ = nullptr;
    float* d_dispersion_weights_ = nullptr;
    int* d_exclusion_pairs_ = nullptr;
    // Caller-owned arenas for the cuPo workspaces.
    void* d_cl_arena_ = nullptr;
    void* d_dspace_arena_ = nullptr;
    void* d_kspace_arena_ = nullptr;
    void* d_dispersion_kspace_arena_ = nullptr;

};

} // namespace NvOpenMM

#endif // NV_CUDA_KERNELS_H_
