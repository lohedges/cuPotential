// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * CudaNonbondedKernels.cu
 *
 * CUDA implementation of the NonbondedForce nonbonded kernel, backed by the cupo runtime
 * C ABI (cell list + direct-space tower-plate + reciprocal-space SPME). Our
 * analogue of OpenMM's CommonCalcNonbondedForceKernel: both modes share one
 * execution pipeline, with CustomPairParameters preparing custom expressions and
 * particle data. Device buffers are allocated lazily on the first execute(), and
 * both modes capture per-step work in CUDA graphs.
 *
 * Compiled by nvcc, so the device kernels and their <<<>>> launches sit directly in
 * the method bodies, and the translation unit exports no C symbols.
 */

#include "CudaNonbondedKernels.hpp"
#include "CustomPairParameters.hpp"
#include "ExpressionTailCorrection.hpp"
#include "common/CudaFaultHelp.hpp"
#include "common/CudaGraphEnv.hpp"
#include "common/CupoStatus.hpp"
#include "nvopenmm/NonbondedForce.h"

#include "cupo/cell_list.h"
#include "cupo/dspace.h"
#include "cupo/splitting_plan.h"
#include "cupo/kspace.h"
#include "cupo/version.h"

#include "openmm/OpenMMException.h"
#include "openmm/MonteCarloFlexibleBarostat.h"
#include "openmm/NonbondedForce.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/common/ComputeForceInfo.h"
#include "openmm/common/ContextSelector.h"
#include "CudaContext.h"
#include "CudaArray.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>


// Definitions live in the plugin's namespace, so our NonbondedForce et al.
// win unqualified lookup over OpenMM's same-named classes rather than
// being ambiguous with them.
namespace NvOpenMM {
using namespace std;

// A failed CUDA call: one stderr line, and no abort -- matches prior behavior.
//
// If the error means a kernel died on the device, also the advice on finding it
// (nvDeviceFaultHelp), once per process: after such a fault every later call
// fails too, and one copy of the advice is enough. This call may be the first to
// notice a fault in cupo's kernels, since a trap surfaces at whatever CUDA call
// syncs next.
static void nvReportCudaError(cudaError_t err, const char* file, int line)
{
    fprintf(stderr, "[nv-plugin cupo] CUDA error %s:%d: %s\n",
            file, line, cudaGetErrorString(err));
    const std::string help = nvDeviceFaultHelp(err);
    static std::once_flag advised;
    if (!help.empty())
        std::call_once(advised, [&help] { fprintf(stderr, "%s\n", help.c_str()); });
}

#define NV_CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) \
        nvReportCudaError(err__, __FILE__, __LINE__); \
} while (0)

// ===========================================================================
// Device kernels + small helpers
// ===========================================================================
namespace {

std::uint64_t exclusionPairKey(int first, int second) {
    return (std::uint64_t(std::min(first, second)) << 32) |
           std::uint32_t(std::max(first, second));
}


// cupo evaluates the cell OpenMM describes: lower triangular, in reduced form.
//
// OpenMM keeps a periodic box as a = (a_x, 0, 0), b = (b_x, b_y, 0), c = (c_x, c_y,
// c_z) with a positive diagonal and |b_x| <= a_x/2, |c_x| <= a_x/2, |c_y| <= b_y/2
// (System::setDefaultPeriodicBoxVectors reduces what it is given), and that is the
// form the six-float cupo/box.h layout holds and every cupo stage decomposes: a
// cell list binned in fractional coordinates, direct space shifting by lattice
// vectors, a mesh along the lattice vectors. A rectangular cell is the special
// case with zero off-diagonals and takes the orthorhombic kernels unchanged.
//
// Checked at Context creation and again per step, because setPeriodicBoxVectors
// can hand over anything. The upper-triangle check is exact: a.y, a.z and b.z are
// zero BY CONSTRUCTION in OpenMM, so a non-zero there is not a cell at all.
static void requireReducedBox(const OpenMM::Vec3& a, const OpenMM::Vec3& b,
                              const OpenMM::Vec3& c, const char* where)
{
    char msg[640];
    if (a[1] != 0.0 || a[2] != 0.0 || b[2] != 0.0) {
        std::snprintf(msg, sizeof(msg),
            "NonbondedForce: the periodic box is not lower triangular (%s): a=(%g, %g, %g) "
            "b=(%g, %g, %g). OpenMM's reduced form has a along x and b in the xy plane; "
            "pass the vectors through System.setDefaultPeriodicBoxVectors, which reduces "
            "them.", where, a[0], a[1], a[2], b[0], b[1], b[2]);
        throw OpenMM::OpenMMException(msg);
    }
    float box[CUPO_BOX_FLOATS];
    cupo_box_from_vectors(a[0], b[0], b[1], c[0], c[1], c[2], box);
    if (cupo_box_is_reduced(box))
        return;
    std::snprintf(msg, sizeof(msg),
        "NonbondedForce: the periodic box is not in reduced form (%s): a=(%g, 0, 0) "
        "b=(%g, %g, 0) c=(%g, %g, %g). cupo needs a positive diagonal with |b.x| <= a.x/2, "
        "|c.x| <= a.x/2 and |c.y| <= b.y/2, which is what OpenMM itself requires of a "
        "box; pass the vectors through System.setDefaultPeriodicBoxVectors.",
        where, a[0], b[0], b[1], c[0], c[1], c[2]);
    throw OpenMM::OpenMMException(msg);
}

// OpenMM's box vectors in the cupo/box.h device layout. OpenMM keeps the cell in
// reduced lower-triangular form, so a.y, a.z and b.z are zero by construction and
// the six components below are the whole cell.
static void boxFromVectors(const OpenMM::Vec3& a, const OpenMM::Vec3& b,
                           const OpenMM::Vec3& c, float out[CUPO_BOX_FLOATS])
{
    cupo_box_from_vectors(a[0], b[0], b[1], c[0], c[1], c[2], out);
}

}  // namespace

// ===========================================================================
// OpenMM helper objects
// ===========================================================================
namespace {

// Molecules are interchangeable only when their base parameters AND complete
// offset laws agree. In particular, matching just the last offset could exchange
// particles that follow different earlier globals after a later lambda change.
class NonbondedForceInfo : public OpenMM::ComputeForceInfo {
public:
    NonbondedForceInfo(const NonbondedForce& force) : force(force) {
        if (force.getNumParticleParameterOffsets())
            particleOffsets.resize(force.getNumParticles());
        if (force.getNumExceptionParameterOffsets())
            exceptionOffsets.resize(force.getNumExceptions());
        for (int k = 0; k < force.getNumParticleParameterOffsets(); ++k) {
            std::string parameter;
            int index;
            double q, sigma, epsilon;
            force.getParticleParameterOffset(k, parameter, index, q, sigma, epsilon);
            particleOffsets[index].push_back(k);
        }
        for (int k = 0; k < force.getNumExceptionParameterOffsets(); ++k) {
            std::string parameter;
            int index;
            double q, sigma, epsilon;
            force.getExceptionParameterOffset(k, parameter, index, q, sigma, epsilon);
            exceptionOffsets[index].push_back(k);
        }
    }
    int getNumParticleGroups() override { return force.getNumExceptions(); }
    void getParticlesInGroup(int index, vector<int>& particles) override {
        int first, second;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(index, first, second, chargeProd, sigma, epsilon);
        particles = {first, second};
    }
    bool areGroupsIdentical(int group1, int group2) override {
        int first, second;
        double q1, s1, e1, q2, s2, e2;
        force.getExceptionParameters(group1, first, second, q1, s1, e1);
        force.getExceptionParameters(group2, first, second, q2, s2, e2);
        return q1 == q2 && s1 == s2 && e1 == e2 &&
               sameOffsets(exceptionOffsets, group1, group2, true);
    }
    bool areParticlesIdentical(int particle1, int particle2) override {
        double q1, s1, e1, q2, s2, e2;
        force.getParticleParameters(particle1, q1, s1, e1);
        force.getParticleParameters(particle2, q2, s2, e2);
        return q1 == q2 && s1 == s2 && e1 == e2 &&
               sameOffsets(particleOffsets, particle1, particle2, false);
    }
private:
    bool sameOffsets(const vector<vector<int>>& offsets, int first, int second,
                     bool exceptions) const {
        if (offsets.empty()) return true;
        const auto& a = offsets[first];
        const auto& b = offsets[second];
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            std::string name1, name2;
            int index;
            double q1, s1, e1, q2, s2, e2;
            if (exceptions) {
                force.getExceptionParameterOffset(a[i], name1, index, q1, s1, e1);
                force.getExceptionParameterOffset(b[i], name2, index, q2, s2, e2);
            } else {
                force.getParticleParameterOffset(a[i], name1, index, q1, s1, e1);
                force.getParticleParameterOffset(b[i], name2, index, q2, s2, e2);
            }
            if (name1 != name2 || q1 != q2 || s1 != s2 || e1 != e2) return false;
        }
        return true;
    }
    const NonbondedForce& force;
    vector<vector<int>> particleOffsets, exceptionOffsets;
};

// ReorderListener: called by OpenMM after it permutes atom order on device.
class ReorderListener : public OpenMM::ComputeContext::ReorderListener {
public:
    ReorderListener(CudaCalcNonbondedForce& kernel) : kernel(kernel) {}
    void execute() { kernel.applyAtomReorder(); }
private:
    CudaCalcNonbondedForce& kernel;
};

// PostComputation: deferred graph join for bonded-overlap mode. Runs at the very
// end of force evaluation (after OpenMM enqueues its own bonded kernel) and makes
// the current stream wait for our side-stream nonbonded graph -- letting bonded
// overlap the graph. No-op unless a replay step launched the graph off the current
// stream this step.
class GraphJoinPostComputation : public OpenMM::ComputeContext::ForcePostComputation {
public:
    GraphJoinPostComputation(OpenMM::CudaContext& cu, CudaCalcNonbondedForce* kernel)
        : cu(cu), kernel(kernel) {}
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        kernel->graphJoin(reinterpret_cast<cudaStream_t>(cu.getCurrentStream()));
        return kernel->finishPendingEnergy();
    }
private:
    OpenMM::CudaContext& cu;
    CudaCalcNonbondedForce* kernel;
};

}  // namespace

// ===========================================================================
// Construction and destruction
// ===========================================================================
CudaCalcNonbondedForce::CudaCalcNonbondedForce(const std::string& name,
        const OpenMM::Platform& platform, OpenMM::ComputeContext& cc)
    : CalcNonbondedForceKernel(name, platform), cc_(cc) {}

CudaCalcNonbondedForce::~CudaCalcNonbondedForce()
{
    // Graph nodes borrow workspaces and parameter/module storage. Drain every
    // producer, destroy graphs, then consumers, then their shared inputs.
    OpenMM::ContextSelector selector(cc_);
    auto& cu = static_cast<OpenMM::CudaContext&>(cc_);
    NV_CUDA_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cu.getCurrentStream())));
    for (cudaStream_t stream : {graph_origin, dspace_stream_, kspace_stream_, misc_stream_})
        if (stream) NV_CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < GRAPH_SLOTS; i++) {
        if (graph_slots_[i].exec) NV_CUDA_CHECK(cudaGraphExecDestroy(graph_slots_[i].exec));
        if (graph_slots_[i].graph) NV_CUDA_CHECK(cudaGraphDestroy(graph_slots_[i].graph));
    }
    if (dspace_) cupoCleanup(cupo_dspace_destroy(dspace_));
    if (kspace_) cupoCleanup(cupo_kspace_destroy(kspace_));
    if (dispersion_kspace_) cupoCleanup(cupo_kspace_destroy(dispersion_kspace_));
    if (cell_list_) cupoCleanup(cupo_cell_list_destroy(cell_list_));
    destroyCudaResources();
    freeDevice();
    custom_parameters_.reset();

    delete[] h_params_;
    delete[] h_exclusion_pairs_;
}

// ===========================================================================
// Host-side setup
// ===========================================================================

// Physical records at the C boundary: q, sigma (nm), epsilon (kJ/mol).
// All offsets are applied in double first, in declaration order. Native packing
// still belongs to the Standard pair descriptor, not to this update path.
vector<float> CudaCalcNonbondedForce::buildCupoParams(const BuiltinParameters& parameters) const
{
    const bool ranged = parameters.ranged && sysorder_params_.size() == parameters.particles.size();
    vector<float> result = ranged ? sysorder_params_ : vector<float>(parameters.particles.size());
    const size_t first = ranged ? 3*static_cast<size_t>(parameters.firstParticle) : 0;
    const size_t end = ranged ? 3*static_cast<size_t>(parameters.lastParticle+1) : parameters.particles.size();
    for (size_t i = first; i < end; ++i) {
        const double value = parameters.particles[i];
        result[i] = static_cast<float>(value);
        if (!std::isfinite(value) || !std::isfinite(result[i]) ||
                (i % 3 != 0 && value < 0))
            throw OpenMM::OpenMMException("NonbondedForce: effective particle parameters must "
                "be finite in FP32, with nonnegative sigma and epsilon");
    }
    return result;
}

// Physical replacement parameters have no relationship to ordinary pair
// mixing or reciprocal weights. Keep them in logical order for bonded updates.
vector<cupo_tuples_pair_parameters> CudaCalcNonbondedForce::buildBondedPairParameters(
        const BuiltinParameters& parameters) const
{
    const bool sparse = parameters.definition == builtin_parameters_.definition &&
        exception_parameters_.size() == static_cast<size_t>(n_exclusion_pairs_) &&
        parameters.definition->exceptionOffsets.size() < static_cast<size_t>(n_exclusion_pairs_);
    auto result = sparse ? exception_parameters_ :
        vector<cupo_tuples_pair_parameters>(n_exclusion_pairs_);
    const double coupling = model_->latticeSum ? model_->latticeSum->definition().kernel.coupling : 0.;
    const auto set = [&](int k) {
        const double charge = parameters.exceptions[3*k];
        const double sigma = parameters.exceptions[3*k+1];
        const double epsilon = parameters.exceptions[3*k+2];
        if (!std::isfinite(charge) || !std::isfinite(sigma) || !std::isfinite(epsilon) ||
                sigma < 0 || epsilon < 0)
            throw OpenMM::OpenMMException("NonbondedForce: effective exception parameters must "
                "be finite, with nonnegative sigma and epsilon");
        // Preserve no-P1 semantics: unused exception charge columns stay inert.
        const double coulomb = coupling*charge;
        for (double value : {charge, coulomb, sigma, 4.0*epsilon})
            if (!std::isfinite(static_cast<float>(value)))
                throw OpenMM::OpenMMException("NonbondedForce: effective exception parameters "
                                              "must remain finite in FP32");
        result[k] = {coulomb, sigma, epsilon};
    };
    if (parameters.ranged && exception_parameters_.size() == static_cast<size_t>(n_exclusion_pairs_)) {
        result = exception_parameters_;
        for (int k = parameters.firstException; k <= parameters.lastException; ++k) set(k);
    } else if (sparse) {
        for (const auto& offset : parameters.definition->exceptionOffsets) set(offset.index);
    } else {
        for (int k=0; k<n_exclusion_pairs_; ++k) set(k);
    }
    return result;
}

void CudaCalcNonbondedForce::allocateHostData(const vector<int>& exclusion_pairs)
{
    n_exclusion_pairs_ = static_cast<int>(exclusion_pairs.size()/2);
    sysorder_exclusion_pairs_ = exclusion_pairs;
    h_params_ = new float[static_cast<size_t>(natoms_) * 3]{};
    if (n_exclusion_pairs_ > 0) {
        h_exclusion_pairs_ = new int[exclusion_pairs.size()];
        std::copy(exclusion_pairs.begin(), exclusion_pairs.end(), h_exclusion_pairs_);
    }
}

CudaCalcNonbondedForce::BuiltinParameters
CudaCalcNonbondedForce::readBuiltinParameters(const NonbondedForce& force) const
{
    BuiltinParameters parameters;
    auto definition = std::make_shared<BuiltinParameterDefinition>();
    definition->baseParticles.resize(static_cast<size_t>(natoms_) * 3);
    for (int i = 0; i < natoms_; ++i)
        force.getParticleParameters(i, definition->baseParticles[3*i],
            definition->baseParticles[3*i+1], definition->baseParticles[3*i+2]);
    definition->baseExceptions.resize(static_cast<size_t>(n_exclusion_pairs_) * 3);
    for (int k = 0; k < n_exclusion_pairs_; ++k) {
        int a, b;
        force.getExceptionParameters(k, a, b, definition->baseExceptions[3*k],
            definition->baseExceptions[3*k+1], definition->baseExceptions[3*k+2]);
    }
    for (int i = 0; i < force.getNumGlobalParameters(); ++i)
        definition->globalNames.push_back(force.getGlobalParameterName(i));
    const auto watchedIndex = [&](const string& name) {
        auto it = std::find(definition->watchedNames.begin(), definition->watchedNames.end(), name);
        if (it != definition->watchedNames.end())
            return static_cast<int>(it - definition->watchedNames.begin());
        auto global = std::find(definition->globalNames.begin(), definition->globalNames.end(), name);
        if (global == definition->globalNames.end())
            throw OpenMM::OpenMMException("NonbondedForce: unknown offset parameter " + name);
        definition->watchedNames.push_back(name);
        parameters.values.push_back(force.getGlobalParameterDefaultValue(
            static_cast<int>(global - definition->globalNames.begin())));
        return static_cast<int>(definition->watchedNames.size()) - 1;
    };
    for (int k = 0; k < force.getNumParticleParameterOffsets(); ++k) {
        string name;
        ParameterOffset offset;
        force.getParticleParameterOffset(k, name, offset.index,
                                         offset.scale[0], offset.scale[1], offset.scale[2]);
        if (offset.index < 0 || offset.index >= natoms_)
            throw OpenMM::OpenMMException("NonbondedForce: illegal particle index for a parameter offset");
        offset.parameter = watchedIndex(name);
        definition->particleOffsets.push_back(offset);
    }
    for (int k = 0; k < force.getNumExceptionParameterOffsets(); ++k) {
        string name;
        ParameterOffset offset;
        force.getExceptionParameterOffset(k, name, offset.index,
                                          offset.scale[0], offset.scale[1], offset.scale[2]);
        if (offset.index < 0 || offset.index >= n_exclusion_pairs_)
            throw OpenMM::OpenMMException("NonbondedForce: illegal exception index for a parameter offset");
        offset.parameter = watchedIndex(name);
        definition->exceptionOffsets.push_back(offset);
    }
    for (const auto* term : {model_->latticeSum ? &*model_->latticeSum : nullptr,
                             model_->dispersion ? &*model_->dispersion : nullptr})
        if (term)
            for (const auto& name : term->globalDependencies()) watchedIndex(name);
    definition->tailReference = force.getDispersionCorrectionInteraction();
    definition->tail = definition->tailReference.tail;
    definition->tailFollowsParameters = force.getDispersionCorrectionFollowsParameters();
    // The compatibility policy uses global defaults. Live tails are computed
    // after applying the current Context globals, and can reuse the active tail.
    if (!definition->tailFollowsParameters)
        definition->defaultTailCoefficient = model_->cutoffInteraction.tailCoefficientFor(force);
    parameters.definition = std::move(definition);
    return parameters;
}

void CudaCalcNonbondedForce::readOffsetValues(BuiltinParameters& parameters,
                                            OpenMM::ContextImpl& context) const
{
    const auto& names = parameters.definition->watchedNames;
    for (size_t i = 0; i < names.size(); ++i)
        parameters.values[i] = context.getParameter(names[i]);
}

unsigned CudaCalcNonbondedForce::applyOffsets(BuiltinParameters& parameters,
                                             const vector<double>& before) const
{
    const auto& definition = *parameters.definition;
    // Watched names feed an offset law or a lattice-weight expression.
    // Validate once before deriving either local effective table.
    for (double value : parameters.values)
        if (!std::isfinite(value))
            throw OpenMM::OpenMMException("NonbondedForce: offset parameter value must be finite");
    const auto moved = [&](const vector<ParameterOffset>& offsets) {
        if (before.empty()) return true;
        for (const auto& offset : offsets)
            if (parameters.values[offset.parameter] != before[offset.parameter]) return true;
        return false;
    };
    const auto apply = [&](vector<double>& values, const vector<double>& base,
                           const vector<ParameterOffset>& offsets) {
        values = base;
        for (const auto& offset : offsets) {
            const double value = parameters.values[offset.parameter];
            for (int c = 0; c < 3; ++c)
                values[3*offset.index+c] += value * offset.scale[c];
        }
    };
    unsigned changed = 0;
    if (moved(definition.particleOffsets)) {
        apply(parameters.particles, definition.baseParticles, definition.particleOffsets);
        changed |= ParticlesChanged;
    }
    if (moved(definition.exceptionOffsets)) {
        apply(parameters.exceptions, definition.baseExceptions, definition.exceptionOffsets);
        changed |= ExceptionsChanged;
    }
    return changed;
}

double CudaCalcNonbondedForce::builtinTailCoefficient(const BuiltinParameters& parameters) const
{
    const auto& definition = *parameters.definition;
    if (definition.tail == CutoffInteraction::NoTail) return 0.0;
    double coefficient = definition.defaultTailCoefficient;
    if (definition.tailFollowsParameters) {
        // Only effective particle LJ parameters enter the native tail. Ignore
        // charge and exception changes, including explicit parameter updates.
        // Cutoff, mixing and switching settings are immutable in this Context.
        const auto& previous = builtin_parameters_;
        // Initialization has no active definition to reuse yet.
        bool sameLJ = previous.definition && previous.definition->tail == definition.tail &&
            previous.definition->tailFollowsParameters &&
            previous.particles.size() == parameters.particles.size();
        for (int i = 0; sameLJ && i < natoms_; ++i)
            sameLJ = previous.particles[3*i+1] == parameters.particles[3*i+1] &&
                     previous.particles[3*i+2] == parameters.particles[3*i+2];
        if (sameLJ)
            return previous.tailCoefficient;
        vector<double> sigma(natoms_), epsilon(natoms_);
        for (int i = 0; i < natoms_; ++i) {
            sigma[i] = parameters.particles[3*i+1];
            epsilon[i] = parameters.particles[3*i+2];
        }
        const auto& cutoff = definition.tailReference;
        coefficient = NonbondedForceImpl::calcDispersionCorrection(
            natoms_, sigma.data(), epsilon.data(), model_->cutoff,
            cutoff.combiningRule, cutoff.useSwitching, cutoff.switchingDistance);
    }
    if (!std::isfinite(coefficient))
        throw OpenMM::OpenMMException("NonbondedForce: dispersion correction must be finite");
    return coefficient;
}

void CudaCalcNonbondedForce::waitForSteps() const
{
    auto& cu = static_cast<OpenMM::CudaContext&>(cc_);
    NV_CUDA_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cu.getCurrentStream())));
    for (cudaStream_t stream : {graph_origin, dspace_stream_, kspace_stream_, misc_stream_})
        if (stream) NV_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void CudaCalcNonbondedForce::pushBuiltinParameters(BuiltinParameters&& parameters,
                                                  unsigned changed, bool invalidateMolecules)
{
    const bool particles = (changed & ParticlesChanged) != 0;
    const bool exceptions = n_exclusion_pairs_ > 0 &&
        (changed & ExceptionsChanged) != 0;
    // All physical rows, weight expressions and corrections are validated before
    // committing any live input. The same FP32 weights feed both spaces.
    vector<float> packed = particles ? buildCupoParams(parameters) : vector<float>{};
    const auto& particleRecords = particles ? packed : sysorder_params_;
    auto special = exceptions ? buildBondedPairParameters(parameters) :
        vector<cupo_tuples_pair_parameters>();
    std::map<std::string,double> globals;
    for (size_t i=0;i<parameters.values.size();++i)
        globals[parameters.definition->watchedNames[i]]=parameters.values[i];
    std::optional<LatticeSumEvaluation::PreparedWeights> charges, dispersion;
    const auto prepare = [&](const auto& term, auto& proposal) {
        if (!term) return;
        bool globalsChanged = !builtin_parameters_.definition ||
            parameters.definition != builtin_parameters_.definition;
        for (const auto& name : term->globalDependencies()) {
            const auto& names=parameters.definition->watchedNames;
            const size_t i=std::find(names.begin(),names.end(),name)-names.begin();
            globalsChanged=globalsChanged || parameters.values[i]!=builtin_parameters_.values[i];
        }
        if (particles || globalsChanged) {
            proposal=term->prepareWeights(particleRecords,3,globals);
            for (int k=0;k<n_exclusion_pairs_;++k)
                if (!std::isfinite(proposal->values[sysorder_exclusion_pairs_[2*k]]*
                                   proposal->values[sysorder_exclusion_pairs_[2*k+1]]))
                    throw OpenMM::OpenMMException("LatticeSumInteraction: excluded weight products must be finite FP32");
        }
    };
    prepare(model_->latticeSum,charges);
    prepare(model_->dispersion,dispersion);
    parameters.tailCoefficient = builtinTailCoefficient(parameters);
    // Another force group can already have prepared the shared bonded plan
    // even before this nonbonded workspace's first evaluation.
    if (device_initialized_ || (exceptions && bonded_pairs_)) waitForSteps();

    if (exceptions && bonded_pairs_) bonded_pairs_->update(special);
    builtin_parameters_ = std::move(parameters);
    if (particles) {
        sysorder_params_ = std::move(packed);
        if (!device_initialized_)
            std::copy(sysorder_params_.begin(), sysorder_params_.end(), h_params_);
    }
    if (exceptions)
        exception_parameters_ = std::move(special);
    if (charges) model_->latticeSum->commitWeights(std::move(*charges));
    if (dispersion) model_->dispersion->commitWeights(std::move(*dispersion));
    const bool weightsChanged=refreshLatticeInputs();
    if (custom_parameters_ && (particles || weightsChanged))
        custom_parameters_->setManagedParameters(sysorder_params_);
    if (model_->usesExpression())
        model_->cutoffInteraction.updateTailCoefficient(builtin_parameters_.tailCoefficient);
    else
        model_->cutoffInteraction.updateTail(builtin_parameters_.definition->tail, builtin_parameters_.tailCoefficient);

    // Explicit base/scale updates may split molecule equivalence classes. Put all
    // canonical records in place before invalidation invokes reorder listeners.
    // A live global does not change those classes: ForceInfo compares offset laws.
    const long previousReorder = reorder_count_;
    if (invalidateMolecules) cc_.invalidateMolecules();
    if (device_initialized_ && reorder_count_ == previousReorder) {
        if (particles || weightsChanged)
            applyAtomReorderImpl();
    }
    // The next evaluation runs on nonblocking streams. Complete pageable uploads
    // even if CUDA's synchronous-copy call returned before the DMA finished.
    if (device_initialized_) NV_CUDA_CHECK(cudaStreamSynchronize(0));
}

bool CudaCalcNonbondedForce::refreshLatticeInputs()
{
    const bool charges=model_->latticeSum &&
        model_->latticeSum->weightsVersion()!=charge_weights_version_;
    const bool dispersion=model_->dispersion &&
        model_->dispersion->weightsVersion()!=dispersion_weights_version_;
    if (!charges && !dispersion) return false;
    for (const auto* term : {charges ? &*model_->latticeSum : nullptr,
                             dispersion ? &*model_->dispersion : nullptr}) {
        if (!term) continue;
        if (term->weights().size()!=static_cast<size_t>(natoms_))
            throw OpenMM::OpenMMException("LatticeSumInteraction: wrong weight count");
        for (int k=0;k<n_exclusion_pairs_;++k)
            if (!std::isfinite(term->weights()[sysorder_exclusion_pairs_[2*k]]*
                               term->weights()[sysorder_exclusion_pairs_[2*k+1]]))
                throw OpenMM::OpenMMException("LatticeSumInteraction: excluded weight products must be finite FP32");
    }
    if (charges) {
        charge_weights_version_=model_->latticeSum->weightsVersion();
    }
    if (dispersion) {
        sysorder_dispersion_weights_=model_->dispersion->weights();
        dispersion_weights_version_=model_->dispersion->weightsVersion();
    }
    return true;
}

void CudaCalcNonbondedForce::updateOffsetParameters(OpenMM::ContextImpl& context)
{
    // The normal Standard path has no offsets and keeps its no-update fast path.
    const auto& names = builtin_parameters_.definition->watchedNames;
    for (size_t i = 0; i < names.size(); ++i) {
        if (context.getParameter(names[i]) != builtin_parameters_.values[i]) {
            BuiltinParameters next;
            next.definition = builtin_parameters_.definition;
            next.values = builtin_parameters_.values;
            readOffsetValues(next, context);
            const unsigned changed = applyOffsets(next, builtin_parameters_.values);
            // Changed categories already contain base + all current offsets.
            // Preserve only the effective arrays that applyOffsets did not fill.
            if (!(changed & ParticlesChanged))
                next.particles = builtin_parameters_.particles;
            if (!(changed & ExceptionsChanged))
                next.exceptions = builtin_parameters_.exceptions;
            pushBuiltinParameters(std::move(next), changed, false);
            return;
        }
    }
}

// ===========================================================================
// Device lifecycle
// ===========================================================================

void CudaCalcNonbondedForce::createCudaResources()
{
    const char* setting = std::getenv("CUPO_PLUGIN_BONDED_SCHEDULE");
    const std::string requested = setting ? setting : "auto";
    if (requested != "auto" && requested != "overlap" && requested != "after-sort")
        throw OpenMM::OpenMMException("CUPO_PLUGIN_BONDED_SCHEDULE must be auto, overlap or after-sort");
    // usesExpression() is the execution model's counterpart of hasCutoffExpression():
    // both Custom mode and a Standard force with setCutoffInteraction(expression).
    // These forces retain component execution and benefit from the original overlap.
    const std::string schedule = requested == "auto"
        ? (model_->usesExpression() ? "overlap" : "after-sort") : requested;
    if (debug_graph_trace_)
        std::fprintf(stderr, "[nv-plugin cupo] bonded schedule requested=%s resolved=%s cutoff_expression=%d\n",
                     requested.c_str(), schedule.c_str(), model_->usesExpression());
    // Keep bonded work off the shared cell-list build, then overlap it with
    // direct and reciprocal force calculation. The integrator still joins all.
    if (schedule == "after-sort" && graph_enabled_)
        NV_CUDA_CHECK(cudaEventCreateWithFlags(&bonded_cells_ready_event_, cudaEventDisableTiming));

    NV_CUDA_CHECK(cudaStreamCreateWithFlags(&dspace_stream_, cudaStreamNonBlocking));
    if (model_->hasLatticeSum())
        NV_CUDA_CHECK(cudaStreamCreateWithFlags(&kspace_stream_, cudaStreamNonBlocking));
    NV_CUDA_CHECK(cudaStreamCreateWithFlags(&misc_stream_, cudaStreamNonBlocking));
    NV_CUDA_CHECK(cudaEventCreateWithFlags(&step_fork_event_, cudaEventDisableTiming));
    NV_CUDA_CHECK(cudaEventCreateWithFlags(&box_upload_done_, cudaEventDisableTiming));
    NV_CUDA_CHECK(cudaEventCreateWithFlags(&force_clear_done_event_, cudaEventDisableTiming));
    NV_CUDA_CHECK(cudaEventCreateWithFlags(&cells_ready_event_, cudaEventDisableTiming));
    NV_CUDA_CHECK(cudaEventCreateWithFlags(&dspace_done_event_, cudaEventDisableTiming));
    if (model_->hasLatticeSum())
        NV_CUDA_CHECK(cudaEventCreateWithFlags(&kspace_done_event_, cudaEventDisableTiming));
    if (graph_enabled_) {
        NV_CUDA_CHECK(cudaStreamCreateWithFlags(&graph_origin, cudaStreamNonBlocking));
        NV_CUDA_CHECK(cudaEventCreateWithFlags(&bridge_in_event_, cudaEventDisableTiming));
        NV_CUDA_CHECK(cudaEventCreateWithFlags(&bridge_out_event_, cudaEventDisableTiming));
    }
}

void CudaCalcNonbondedForce::destroyCudaResources()
{
    if (bonded_cells_ready_event_) NV_CUDA_CHECK(cudaEventDestroy(bonded_cells_ready_event_));
    bonded_cells_ready_event_ = nullptr;
    if (bridge_in_event_) NV_CUDA_CHECK(cudaEventDestroy(bridge_in_event_));
    if (bridge_out_event_) NV_CUDA_CHECK(cudaEventDestroy(bridge_out_event_));
    if (graph_origin) NV_CUDA_CHECK(cudaStreamDestroy(graph_origin));
    if (step_fork_event_) NV_CUDA_CHECK(cudaEventDestroy(step_fork_event_));
    if (box_upload_done_) NV_CUDA_CHECK(cudaEventDestroy(box_upload_done_));
    if (force_clear_done_event_) NV_CUDA_CHECK(cudaEventDestroy(force_clear_done_event_));
    if (cells_ready_event_) NV_CUDA_CHECK(cudaEventDestroy(cells_ready_event_));
    if (dspace_done_event_) NV_CUDA_CHECK(cudaEventDestroy(dspace_done_event_));
    if (kspace_done_event_) NV_CUDA_CHECK(cudaEventDestroy(kspace_done_event_));
    if (dspace_stream_) NV_CUDA_CHECK(cudaStreamDestroy(dspace_stream_));
    if (kspace_stream_) NV_CUDA_CHECK(cudaStreamDestroy(kspace_stream_));
    if (misc_stream_) NV_CUDA_CHECK(cudaStreamDestroy(misc_stream_));
    bridge_in_event_ = bridge_out_event_ = nullptr;
    step_fork_event_ = force_clear_done_event_ = nullptr;
    box_upload_done_ = nullptr;
    box_upload_pending_ = false;
    cells_ready_event_ = dspace_done_event_ = kspace_done_event_ = nullptr;
    graph_origin = nullptr;
    dspace_stream_ = kspace_stream_ = misc_stream_ = nullptr;
}

void CudaCalcNonbondedForce::freeDevice()
{
#define NV_FREE(ptr) do { if (ptr) NV_CUDA_CHECK(cudaFree(ptr)); ptr = nullptr; } while (0)
    NV_FREE(d_box_);
    NV_FREE(d_params_);
    NV_FREE(d_charge_weights_);
    NV_FREE(d_dispersion_weights_);
    NV_FREE(d_exclusion_pairs_);
    NV_FREE(d_energy_);
    NV_FREE(d_dspace_arena_);
    NV_FREE(d_kspace_arena_);
    NV_FREE(d_dispersion_kspace_arena_);
    NV_FREE(d_cl_arena_);
    particle_input_ = nullptr;
#undef NV_FREE
}

// Allocate all device buffers and create the cupo primitives. Called once on the
// first execute(), when the atom count and box are known.
void CudaCalcNonbondedForce::setupDevice(int n, const float box[CUPO_BOX_FLOATS])
{
    if (setup_started_)
        throw OpenMM::OpenMMException("NonbondedForce: previous workspace setup already started; "
                                      "reinitialize Context rather than retrying partial setup");
    setup_started_ = true;
    natoms_ = n;

    NV_CUDA_CHECK(cudaMalloc(&d_box_, CUPO_BOX_FLOATS * sizeof(float)));
    NV_CUDA_CHECK(cudaMemcpy(d_box_, box, CUPO_BOX_FLOATS * sizeof(float), cudaMemcpyHostToDevice));
    std::copy(box, box + CUPO_BOX_FLOATS, box_on_device_);
    box_triclinic_ = !cupo_box_is_orthorhombic(box);

    const auto plan = model_->latticeSum ? model_->latticeSum->plan() : nullptr;
    cupo_pair_desc pair{};
    cupo_dspace_desc desc{};
    desc.struct_bytes = sizeof(desc);
    desc.abi_version = CUPO_DSPACE_DESC_VERSION;
    if (h_params_) {
        NV_CUDA_CHECK(cudaMalloc(&d_params_, static_cast<size_t>(n) * 3 * sizeof(float)));
        NV_CUDA_CHECK(cudaMemcpy(d_params_, h_params_, static_cast<size_t>(n) * 3 * sizeof(float),
                                cudaMemcpyHostToDevice));
        if (model_->usesExpression() && model_->latticeSum) {
            reordered_charge_weights_.resize(n);
            const auto& order=cc_.getAtomIndex();
            for (int i=0;i<n;++i)
                reordered_charge_weights_[i]=model_->latticeSum->weights()[order[i]];
            NV_CUDA_CHECK(cudaMalloc(&d_charge_weights_,static_cast<size_t>(n)*sizeof(float)));
            NV_CUDA_CHECK(cudaMemcpy(d_charge_weights_,reordered_charge_weights_.data(),
                                    static_cast<size_t>(n)*sizeof(float),cudaMemcpyHostToDevice));
        }
        if (model_->dispersion) {
            reordered_dispersion_weights_.resize(n);
            const auto& order=cc_.getAtomIndex();
            for (int i=0;i<n;++i)
                reordered_dispersion_weights_[i]=sysorder_dispersion_weights_[order[i]];
            NV_CUDA_CHECK(cudaMalloc(&d_dispersion_weights_,static_cast<size_t>(n)*sizeof(float)));
            NV_CUDA_CHECK(cudaMemcpy(d_dispersion_weights_,reordered_dispersion_weights_.data(),
                                    static_cast<size_t>(n)*sizeof(float),cudaMemcpyHostToDevice));
        }
        if (n_exclusion_pairs_ > 0 && !model_->usesExpression()) {
            native_exclusion_pairs_.reserve(n_exclusion_pairs_);
            for (int k=0; k<n_exclusion_pairs_; ++k)
                if (!native_exclusion_pairs_.insert(exclusionPairKey(h_exclusion_pairs_[2*k],
                        h_exclusion_pairs_[2*k+1])).second)
                    throw OpenMM::OpenMMException("NonbondedForce: duplicate exception pair");
            exclusion_topology_dirty_=false;
            NV_CUDA_CHECK(cudaMalloc(&d_exclusion_pairs_,static_cast<size_t>(n_exclusion_pairs_)*2*sizeof(int)));
            NV_CUDA_CHECK(cudaMemcpy(d_exclusion_pairs_,h_exclusion_pairs_,
                static_cast<size_t>(n_exclusion_pairs_)*2*sizeof(int),cudaMemcpyHostToDevice));
        }
        desc.n_exclusion_pairs=n_exclusion_pairs_;
        desc.d_exclusion_pairs=d_exclusion_pairs_;
    }
    desc.reciprocal_exclusions=1;
    if (model_->usesExpression()) {
        custom_parameters_->setup();
        pair = custom_parameters_->pairDescriptor();
        particle_input_ = custom_parameters_->particles();
        particle_stride_bytes_ = custom_parameters_->particleStride();
        desc.n_exclusion_pairs=custom_parameters_->exclusionCount();
        desc.d_exclusion_pairs=custom_parameters_->exclusions();
        if (model_->latticeSum)
            desc.coulomb_exclusions={plan,d_charge_weights_,sizeof(float),static_cast<uint64_t>(n)};
        if (model_->dispersion)
            desc.dispersion_exclusions={model_->dispersion->plan(),d_dispersion_weights_,
                                        sizeof(float),static_cast<uint64_t>(n)};
    } else {
        // The fully built-in model keeps its compact records and native kernel.
        particle_input_ = d_params_;
        particle_stride_bytes_ = 3 * sizeof(float);
        pair.struct_bytes = sizeof(pair);
        pair.abi_version = CUPO_PAIR_ABI_VERSION;
        pair.kind = CUPO_PAIR_STANDARD;
        pair.model.standard.coulomb_plan = plan;
        pair.model.standard.dispersion_plan=model_->dispersion ? model_->dispersion->plan() : nullptr;
        pair.model.standard.combining_rule = model_->cutoffInteraction.definition().combiningRule == NonbondedForce::Geometric
            ? CUPO_LJ_GEOMETRIC : CUPO_LJ_LORENTZ_BERTHELOT;
        pair.model.standard.use_switching = model_->cutoffInteraction.definition().useSwitching;
        pair.model.standard.switch_distance = model_->cutoffInteraction.definition().switchingDistance;
    }

    unsigned long long cl_bytes = 0, cl_host = 0;
    cupoCheck(cupo_cell_list_workspace_size(model_->cutoff, box_safety_factor_, n, box,
                                          &pair, &cl_bytes, &cl_host));
    NV_CUDA_CHECK(cudaMalloc(&d_cl_arena_, static_cast<size_t>(cl_bytes)));
    cupoCheck(cupo_cell_list_create(model_->cutoff, box_safety_factor_, n, box,
                                   &pair, d_cl_arena_, cl_bytes, &cell_list_));

    // Fixed energy slots preserve one deferred readback for all GPU outputs.
    const size_t energyBytes=(model_->dispersion ? 3 : 2)*sizeof(double);
    NV_CUDA_CHECK(cudaMalloc(&d_energy_, energyBytes));
    NV_CUDA_CHECK(cudaMemset(d_energy_, 0, energyBytes));
    desc.pair = pair;
    desc.cell_list = cell_list_;
    unsigned long long dspace_bytes = 0;
    cupoCheck(cupo_dspace_workspace_size(&desc, &dspace_bytes));
    NV_CUDA_CHECK(cudaMalloc(&d_dspace_arena_, static_cast<size_t>(dspace_bytes)));
    cupoCheck(cupo_dspace_create(&desc, d_dspace_arena_, dspace_bytes, &dspace_));

    if (model_->hasLatticeSum()) {
        unsigned long long bytes = 0;
        cupo_kspace_desc reciprocal{};
        reciprocal.struct_bytes = sizeof(reciprocal);
        reciprocal.abi_version = CUPO_KSPACE_DESC_VERSION;
        reciprocal.grid_nx = kspace_nx_;
        reciprocal.grid_ny = kspace_ny_;
        reciprocal.grid_nz = kspace_nz_;
        reciprocal.cell_list = cell_list_;
        if (model_->latticeSum) {
            reciprocal.plan = plan;
            if (model_->usesExpression()) {
                reciprocal.d_charges = d_charge_weights_;
                reciprocal.charge_stride_bytes = sizeof(float);
                reciprocal.charge_count = n;
            }
            cupoCheck(cupo_kspace_workspace_size(&reciprocal, &bytes, nullptr));
            NV_CUDA_CHECK(cudaMalloc(&d_kspace_arena_, static_cast<size_t>(bytes)));
            cupoCheck(cupo_kspace_create(&reciprocal, d_kspace_arena_, bytes, &kspace_));
        }
        if (model_->dispersion) {
            reciprocal.plan=model_->dispersion->plan();
            reciprocal.d_charges=d_dispersion_weights_;
            reciprocal.charge_stride_bytes=sizeof(float);
            reciprocal.charge_count=n;
            cupoCheck(cupo_kspace_workspace_size(&reciprocal,&bytes,nullptr));
            NV_CUDA_CHECK(cudaMalloc(&d_dispersion_kspace_arena_,static_cast<size_t>(bytes)));
            cupoCheck(cupo_kspace_create(&reciprocal,d_dispersion_kspace_arena_,bytes,&dispersion_kspace_));
        }
    }

    // Select before graph capture. Automatic support covers the cell list's
    // entire accepted box envelope, so NPT refreshes never change the layout.
    // One setting with two names: the stable option overrides the legacy alias.
    // Unset/empty means auto; only the P1 workspace uses compact convolution.
    // P6 keeps dense cuFFT on the shared mesh.
    const char* compact = std::getenv("CUPO_COMPACT_CONVOLUTION");
    if (!compact) compact = std::getenv("CUPO_EXPERIMENT_COMPACT_CONVOLUTION");
    const bool automatic = !compact || !*compact || std::strcmp(compact, "auto") == 0;
    if (kspace_) {
        cupo_kspace_convolution_desc configuration{};
        configuration.struct_bytes = sizeof(configuration);
        configuration.abi_version = CUPO_KSPACE_CONVOLUTION_DESC_VERSION;
        configuration.backend = CUPO_KSPACE_BACKEND_CUFFT;
        configuration.spectrum = CUPO_KSPACE_SPECTRUM_FULL;
        if (automatic) {
            configuration.backend = CUPO_KSPACE_BACKEND_AUTO;
            if (model_->latticeSum->definition().splitting == LatticeSumInteraction::Prolate)
                configuration.spectrum = CUPO_KSPACE_SPECTRUM_TRUNCATED_RADIAL;
        } else if (std::strcmp(compact, "off") != 0) {
            configuration.backend = CUPO_KSPACE_BACKEND_COMPACT;
            if (std::strcmp(compact, "full") != 0) {
                char extra;
                if (std::sscanf(compact, "%d,%d,%d%c", &configuration.support[0],
                                &configuration.support[1], &configuration.support[2], &extra) != 3)
                    throw OpenMM::OpenMMException(
                        "CUPO_COMPACT_CONVOLUTION must be auto, off, full, or sx,sy,sz");
                configuration.spectrum = CUPO_KSPACE_SPECTRUM_TRUNCATED_BOX;
            }
        }
        cupoCheck(cupo_kspace_configure_convolution(kspace_, &configuration));
        cupo_kspace_convolution_info info{};
        info.struct_bytes = sizeof(info);
        info.abi_version = CUPO_KSPACE_CONVOLUTION_INFO_VERSION;
        cupoCheck(cupo_kspace_get_convolution_info(kspace_, &info));
        std::fprintf(stderr, "[nv-plugin cupo] convolution mesh=%d,%d,%d support=%d,%d,%d compact=%d mode=%s\n",
                     kspace_nx_, kspace_ny_, kspace_nz_, info.support[0], info.support[1], info.support[2],
                     info.compact_available, automatic ? "auto" : compact);
    }

    if (model_->hasLatticeSum()) {
        // Select numerical support before the first influence refresh, avoiding
        // a transient full table and a second setup-time build for auto Prolate.
        // This and uploadBoxIfChanged pair every box upload with a refresh.
        if (kspace_) cupoCheck(cupo_kspace_refresh_influence(kspace_, d_box_, 0));
        if (dispersion_kspace_)
            cupoCheck(cupo_kspace_refresh_influence(dispersion_kspace_,d_box_,0));
        NV_CUDA_CHECK(cudaStreamSynchronize(0));
        if (automatic && kspace_) {
            const char* tune = std::getenv("CUPO_COMPACT_AUTOTUNE");
            if (tune && *tune && std::strcmp(tune, "0") != 0 && std::strcmp(tune, "1") != 0)
                throw OpenMM::OpenMMException("CUPO_COMPACT_AUTOTUNE must be 0 or 1");
            if (!tune || !*tune || std::strcmp(tune, "0") != 0) {
                cupoCheck(cupo_kspace_tune_convolution(kspace_, 0));
                cupo_kspace_convolution_info info{};
                info.struct_bytes = sizeof(info);
                info.abi_version = CUPO_KSPACE_CONVOLUTION_INFO_VERSION;
                cupoCheck(cupo_kspace_get_convolution_info(kspace_, &info));
                if (info.tuned) {
                    std::fprintf(stderr,
                        "[nv-plugin cupo] convolution selection potential=%s energy=%s both=%s "
                        "compact_us=%.3f,%.3f,%.3f cufft_us=%.3f,%.3f,%.3f setup_ms=%.3f\n",
                        info.backend[0] == CUPO_KSPACE_BACKEND_COMPACT ? "compact" : "cuFFT",
                        info.backend[1] == CUPO_KSPACE_BACKEND_COMPACT ? "compact" : "cuFFT",
                        info.backend[2] == CUPO_KSPACE_BACKEND_COMPACT ? "compact" : "cuFFT",
                        info.compact_us[0], info.compact_us[1], info.compact_us[2],
                        info.cufft_us[0], info.cufft_us[1], info.cufft_us[2], info.setup_ms);
                }
            }
        }
    }
    NV_CUDA_CHECK(cudaStreamSynchronize(0));
}

// ===========================================================================
// OpenMM kernel interface
// ===========================================================================

void CudaCalcNonbondedForce::initialize(const OpenMM::System& system, const NonbondedForce& force,
                                        float boxSafetyFactor)
{
    OpenMM::ContextSelector selector(cc_);

    // The kernels are single precision throughout, and this class hands OpenMM's posq
    // to cupo as a float4 pointer. At double precision that array is double4, so the
    // pointer walks it at half stride and reads coordinates that are halves of
    // neighbouring doubles -- which does not degrade the answer, it kills the context:
    // the SPME spread kernel fails to launch and takes the process down with a
    // CUDA_ERROR_LAUNCH_FAILED from whatever OpenMM frees next. Refuse it here, where
    // the message can say why, as the integrator already does.
    if (cc_.getUseDoublePrecision())
        throw OpenMM::OpenMMException("NonbondedForce supports only single or mixed "
                              "precision, not double.");

    // CUDA-graph capture is the default path (supports NPT/barostat via the
    // box-as-graph-parameter + band-cross resume machinery); CUPO_PLUGIN_CUDA_GRAPH=0
    // disables it here and in the integrator alike.
    graph_enabled_ = nvCudaGraphEnabled();
    const char* gt = std::getenv("CUPO_PLUGIN_GRAPH_TRACE");
    debug_graph_trace_ = (gt != nullptr && gt[0] == '1');

    // Fatal, not a warning. Every one of these breaks is a signature change, so a
    // stale runtime does not fail to link -- it mis-stacks arguments and computes
    // silent garbage. A message on stderr is not a defence against that; the
    // process ending is.
    if (cupo_abi_version() != CUPO_ABI_VERSION) {
        throw OpenMM::OpenMMException(
            "NonbondedForce: cupo ABI mismatch (plugin built against " +
            std::to_string(CUPO_ABI_VERSION) + ", runtime is " +
            std::to_string(cupo_abi_version()) +
            "). The libcupo.so on the library path is not the one this "
            "plugin was built against; rebuild the plugin or fix LD_LIBRARY_PATH.");
    }

    model_.emplace(InteractionModel::fromForce(force));
    if (model_->usesExpression()) {
        OpenMM::Vec3 a, b, c;
        system.getDefaultPeriodicBoxVectors(a, b, c);
        requireReducedBox(a, b, c, "System default box vectors");
    }
    box_safety_factor_ = boxSafetyFactor;
    natoms_ = system.getNumParticles();
    if (model_->hasLatticeSum()) {
        OpenMM::Vec3 box[3];
        system.getDefaultPeriodicBoxVectors(box[0],box[1],box[2]);
        requireReducedBox(box[0],box[1],box[2],"System default box vectors");
        // Same precedence/default as convolution setup: stable name, legacy
        // alias, then auto. Retain the alias for existing launch scripts.
        const char* compact=std::getenv("CUPO_COMPACT_CONVOLUTION");
        if (!compact) compact=std::getenv("CUPO_EXPERIMENT_COMPACT_CONVOLUTION");
        // Retain the native P1 mesh policy. Dispersion shares it for identical
        // splitting; otherwise take the conservative maximum of both plans.
        for (const auto* term : {model_->latticeSum ? &*model_->latticeSum : nullptr,
                                 model_->dispersion ? &*model_->dispersion : nullptr}) {
            if (!term) continue;
            if (term== (model_->dispersion ? &*model_->dispersion : nullptr) &&
                    model_->latticeSum && term->definition().splitting==model_->latticeSum->definition().splitting)
                continue;
            const bool prolate=term->definition().splitting==LatticeSumInteraction::Prolate;
            const auto extent=[&](int i) { return prolate ? std::sqrt(box[i].dot(box[i])) : box[i][i]; };
            // NPT headroom is an application choice. Keep Ewald's stock OpenMM
            // grid convention; Prolate resolves the declared enlarged box.
            const double mesh_scale = prolate ? box_safety_factor_ : 1.0;
            const double mesh_box[3] = {extent(0)*mesh_scale, extent(1)*mesh_scale, extent(2)*mesh_scale};
            int mesh[3];
            cupoCheck(cupo_plan_grid(term->plan(), mesh_box, CUPO_GRID_DENSE, mesh));
            if (prolate && term->definition().kernel.power==1 &&
                    (!compact || std::strcmp(compact,"off")!=0)) {
                int candidate[3];
                cupo_status status = cupo_plan_grid(
                    term->plan(), mesh_box, CUPO_GRID_COMPACT, candidate);
                if (status == CUPO_SUCCESS) {
                    status = cupo_kspace_check_compact_mesh(candidate);
                    if (status == CUPO_SUCCESS) std::copy(candidate, candidate+3, mesh);
                    else if (status != CUPO_NOT_SUPPORTED) cupoCheck(status);
                } else if (status != CUPO_INVALID_SIZE) cupoCheck(status);
            }
            kspace_nx_=std::max(kspace_nx_,mesh[0]);
            kspace_ny_=std::max(kspace_ny_,mesh[1]);
            kspace_nz_=std::max(kspace_nz_,mesh[2]);
        }
    }
    if (model_->hasBuiltinParameters()) {
        exceptions_use_pbc_ = force.getExceptionsUsePeriodicBoundaryConditions();
        vector<string> globalNames;
        for (int i=0;i<force.getNumGlobalParameters();++i)
            globalNames.push_back(force.getGlobalParameterName(i));
        for (auto* term : {model_->latticeSum ? &*model_->latticeSum : nullptr,
                           model_->dispersion ? &*model_->dispersion : nullptr})
            if (term) term->initializeWeightSchema({"q","sigma","epsilon"},globalNames);
        vector<int> exclusion_pairs(static_cast<size_t>(force.getNumExceptions())*2);
        for (int k=0;k<force.getNumExceptions();++k) {
            int ai,aj; double q,sigma,epsilon;
            force.getExceptionParameters(k,ai,aj,q,sigma,epsilon);
            if (ai<0 || ai>=natoms_ || aj<0 || aj>=natoms_ || ai==aj)
                throw OpenMM::OpenMMException("NonbondedForce: illegal exception particle index");
            exclusion_pairs[2*k]=ai;
            exclusion_pairs[2*k+1]=aj;
        }
        allocateHostData(exclusion_pairs);
        auto parameters=readBuiltinParameters(force);
        const unsigned changed=applyOffsets(parameters,{});
        pushBuiltinParameters(std::move(parameters),changed,false);
        cc_.addForce(new NonbondedForceInfo(force));
        if (n_exclusion_pairs_ > 0) {
            vector<cupo_tuples_pair> terms(n_exclusion_pairs_);
            for (int k=0;k<n_exclusion_pairs_;++k)
                terms[k]={{sysorder_exclusion_pairs_[2*k],sysorder_exclusion_pairs_[2*k+1]},
                          exception_parameters_[k],int(exceptions_use_pbc_),force.getForceGroup()};
            bonded_pairs_=std::make_unique<GroupedBondedPairs>(
                static_cast<OpenMM::CudaContext&>(cc_),std::move(terms));
        }
    }

    if (model_->usesExpression()) {
        custom_parameters_ = std::make_unique<CustomPairParameters>(static_cast<OpenMM::CudaContext&>(cc_));
        custom_parameters_->initialize(system,force,*model_);
        if (model_->hasBuiltinParameters())
            custom_parameters_->setManagedParameters(sysorder_params_);
        else if (model_->hasLatticeSum()) {
            const auto& expression=force.getCustomDefinition();
            vector<int> pairs(static_cast<size_t>(expression.getNumExclusions())*2);
            for (int k=0;k<expression.getNumExclusions();++k)
                expression.getExclusionParticles(k,pairs[2*k],pairs[2*k+1]);
            allocateHostData(pairs);
            sysorder_params_.assign(static_cast<size_t>(natoms_)*3,0.0f);
            refreshLatticeInputs();
        }
        if (force.getCutoffInteraction().evaluator==CutoffInteraction::Expression) {
            const auto& expression=force.getCutoffExpressionDefinition();
            implicit_expression_tail_=expression.getUseLongRangeCorrection();
            if (implicit_expression_tail_)
                expression_tails_.push_back(std::make_unique<ExpressionTailCorrection>(expression));
        }
    }

    // Standalone Custom records also follow OpenMM's permutation. Notifications
    // cover reorders, identity resets, checkpoint loads and barostat restores.
    cc_.addReorderListener(new ReorderListener(*this));
    additional_tail_count_ = force.getNumExpressionTailCorrections();
    for (int i=0;i<additional_tail_count_;++i)
        expression_tails_.push_back(std::make_unique<ExpressionTailCorrection>(
            force.getExpressionTailCorrectionDefinition(i)));
    // The cell list fixes its KIND when it is created, from whether the first box
    // has off-diagonals. The rectangular list reads only the diagonal and cannot
    // follow a cell that acquires off-diagonals later; the triclinic list re-derives
    // everything from the device box each step and follows any reduced cell. The
    // one OpenMM force that tilts a cell -- the flexible barostat varies the
    // off-diagonals by construction -- is therefore refused only when the Context
    // starts from a rectangular box, here rather than on the step that first tilts
    // it. From a tilted box it runs on the triclinic list like any other box change
    // (the cell-count envelope the box safety factor sizes still applies). The
    // isotropic and anisotropic barostats scale the vectors and keep a rectangular
    // cell rectangular (and a tilted one tilted).
    {
        OpenMM::Vec3 a, b, c;
        system.getDefaultPeriodicBoxVectors(a, b, c);
        float box0[CUPO_BOX_FLOATS];
        boxFromVectors(a, b, c, box0);
        if (cupo_box_is_orthorhombic(box0)) {
            for (int i = 0; i < system.getNumForces(); i++) {
                if (dynamic_cast<const OpenMM::MonteCarloFlexibleBarostat*>(&system.getForce(i)) != nullptr)
                    throw OpenMM::OpenMMException(
                        "NonbondedForce: MonteCarloFlexibleBarostat on a rectangular box is not "
                        "supported. It changes the box's off-diagonals every move, and cupo fixed a "
                        "rectangular cell decomposition when the Context was created. Start from a "
                        "tilted box (any off-diagonal non-zero), or use MonteCarloBarostat or "
                        "MonteCarloAnisotropicBarostat.");
            }
        }
    }

    createCudaResources();
    // Both pair modes finish on the common execution lanes. Defer the graph join
    // and energy readback until OpenMM has submitted its bonded work.
    cc_.addPostComputation(new GraphJoinPostComputation(static_cast<OpenMM::CudaContext&>(cc_), this));
}

// applyAtomReorder(): called by ReorderListener after OpenMM permutes atom order.
//
// OpenMM reorders for cache locality every 250 steps, but only when some force
// registered a cutoff interaction (ComputeContext::reorderAtoms gates on
// getNonbondedUtilities().getUseCutoff()). A System whose only nonbonded force is
// NonbondedForce registers none, so this fires only when a cutoff-using force coexists.
//
// Everything the plugin indexes BY ATOM has to follow the permutation; everything
// else is either rebuilt per step or indexed by something a reorder does not touch:
//
//   positions             not ours -- the cell list reads OpenMM's posq directly
//   d_params_             re-permuted and re-uploaded here
//   d_exclusion_pairs_    fixed construction-time GPU pair set
//   bonded pair parameters remain in logical order; their adapter maps positions
//   the cell list         self-heals; see below
//   cancellation geometry derived from current cell slots during direct execution
//   the captured graph    holds these buffers by POINTER, so no recapture
//
// The cell list needs nothing. A reorder is the relabel case: it reads posq, which
// OpenMM has already permuted, so atom i simply has a different position, and the
// next migrate re-derives every atom's cell from the live grid. It is also the WORST case for the transient
// row occupancy -- a relabel leaves the density field identical, so every cell sheds
// and regains a full row at once -- which is exactly the bound the per-cell arrival
// lists provide (see cuda_matrix_migrate, and the protein-density relabel test).
//
// Safe to do synchronous copies here: ComputeContext::reorderAtomsImpl opens with a
// BLOCKING download of posq/velm, so the stream is drained before listeners run, and
// atomIndex is uploaded before them too -- getAtomIndex() below is the NEW mapping.
void CudaCalcNonbondedForce::applyAtomReorder()
{
    exclusion_topology_dirty_ = true;
    if (custom_parameters_) custom_parameters_->invalidateAtomOrder();
    if (!device_initialized_ || !h_params_) return;

    // ForceInfo includes exception topology and parameters, so a permutation of
    // equivalent molecules preserves the GPU pair set. Parameter updates can
    // change equivalence classes and reset atom order, however: both particle
    // values and exception corrections must follow their logical system atoms.
    // Homogeneous-water reorder tests alone cannot exercise that association.
    applyAtomReorderImpl();

    reorder_count_++;
    if (debug_graph_trace_) {
        fprintf(stderr, "[nv-plugin cupo] atom reorder #%ld applied\n", reorder_count_);
    }
}

// The GPU buffers are set up in the first execute(), not at Context creation
// (OpenMM's deferred initialization). This function puts the parameters into those
// buffers, in the CudaContext's atom order. It runs after every atom reorder, and
// after a parameter update if the buffers already exist. An update before the
// first execute() only changes the system-order copies, and setupDevice uploads
// them later.
//
//   1. gather particle parameters into the CudaContext's current atom order;
//   2. rebuild exclusion endpoints in that order only if the permutation is dirty;
//   3. copy particle values and live reciprocal weights into GPU input order.
//      Native cancellation topology remains fixed; Custom remaps its own pairs.
//
// Both gathers read separate canonical system-order sources, so an earlier
// permutation cannot change their source order. Numeric updates reuse endpoints.
void CudaCalcNonbondedForce::applyAtomReorderImpl()
{
    reorderHostData(cc_.getAtomIndex().data(), natoms_);

    NV_CUDA_CHECK(cudaMemcpy(d_params_, h_params_,
                             static_cast<size_t>(natoms_) * 3 * sizeof(float),
                             cudaMemcpyHostToDevice));
    if (d_charge_weights_) {
        const auto& atomIndex=cc_.getAtomIndex();
        for (int gpu=0;gpu<natoms_;++gpu)
            reordered_charge_weights_[gpu]=model_->latticeSum->weights()[atomIndex[gpu]];
        NV_CUDA_CHECK(cudaMemcpy(d_charge_weights_,reordered_charge_weights_.data(),
                                static_cast<size_t>(natoms_)*sizeof(float),cudaMemcpyHostToDevice));
    }
    if (d_dispersion_weights_) {
        const auto& atomIndex=cc_.getAtomIndex();
        for (int gpu=0;gpu<natoms_;++gpu)
            reordered_dispersion_weights_[gpu]=sysorder_dispersion_weights_[atomIndex[gpu]];
        NV_CUDA_CHECK(cudaMemcpy(d_dispersion_weights_,reordered_dispersion_weights_.data(),
                                static_cast<size_t>(natoms_)*sizeof(float),cudaMemcpyHostToDevice));
    }
    validateExclusionTopology();
}

void CudaCalcNonbondedForce::validateExclusionTopology()
{
    if (!exclusion_topology_dirty_ || model_->usesExpression()) return;
    for (int k=0;k<n_exclusion_pairs_;++k)
        if (!native_exclusion_pairs_.count(exclusionPairKey(h_exclusion_pairs_[2*k],h_exclusion_pairs_[2*k+1])))
            throw OpenMM::OpenMMException("NonbondedForce: atom reorder changed the exclusion pair set");
    exclusion_topology_dirty_=false;
}

double CudaCalcNonbondedForce::execute(OpenMM::ContextImpl& context,
                                      bool includeForces, bool includeEnergy)
{
    if (!includeForces && !includeEnergy) return 0.0;
    OpenMM::CudaContext& cu = static_cast<OpenMM::CudaContext&>(cc_);
    OpenMM::ContextSelector selector(cu);
    const auto stream = reinterpret_cast<cudaStream_t>(cu.getCurrentStream());
    graphJoin(stream);

    OpenMM::Vec3 a, b, c;
    context.getPeriodicBoxVectors(a, b, c);
    // Again, per step: the check in initialize() sees the System's DEFAULT box, and
    // the context's can be anything after setPeriodicBoxVectors. Host-side compares
    // against a box we have already fetched.
    requireReducedBox(a, b, c, "context box vectors");
    float box[CUPO_BOX_FLOATS];
    boxFromVectors(a, b, c, box);
    // The decomposition's kind was fixed by the first box (setupDevice). A
    // rectangular cell list reads only the diagonal of the device box, so a cell
    // that acquired off-diagonals would be evaluated as if they were zero: refuse
    // it, loudly, before cupo aborts on the same condition. The other direction --
    // a tilted cell whose off-diagonals happen to reach exactly zero -- is fine; the
    // triclinic kernels evaluate a rectangular cell correctly.
    if (device_initialized_ && !box_triclinic_ && !cupo_box_is_orthorhombic(box)) {
        char msg[400];
        std::snprintf(msg, sizeof(msg),
            "NonbondedForce: the periodic box acquired off-diagonals (b.x=%g, c.x=%g, c.y=%g) "
            "after the Context was created with a rectangular one. cupo fixed a rectangular "
            "cell decomposition at creation; create the Context with the tilted box instead.",
            b[0], c[0], c[1]);
        throw OpenMM::OpenMMException(msg);
    }

    // Validate live tail globals on every request, but integrate only for energy:
    // the homogeneous correction contributes no particle forces. The next energy
    // request must observe any globals changed during force-only propagation.
    for (auto& tail : expression_tails_) tail->prepareStep(context.getOwner(), includeEnergy);
    // Apply live globals before lazy setup as well as graph replay. The graph
    // borrows persistent buffers, so a parameter change needs no recapture.
    if (model_->hasBuiltinParameters())
        updateOffsetParameters(context);

    if (!device_initialized_) {
        if (h_params_) {
            const vector<int>& atomIndex = cc_.getAtomIndex();
            reorderHostData(atomIndex.data(), cu.getNumAtoms());
        }
        setupDevice(cu.getNumAtoms(), box);
        device_initialized_ = true;
    }
    // Global changes, reordered input records and dirty preprocessing run outside
    // capture. The common fork orders both execution streams after these writes.
    if (custom_parameters_) {
        custom_parameters_->prepareStep(context, stream, d_params_);
        if (!model_->hasBuiltinParameters() && refreshLatticeInputs()) {
            // Weight updates are rare; drain prior graph readers before writing
            // the persistent reciprocal and sparse-correction inputs.
            waitForSteps();
            applyAtomReorderImpl();
            NV_CUDA_CHECK(cudaStreamSynchronize(0));
        }
    }

    if (bonded_pairs_) bonded_pairs_->prepare();

    const auto* posq = reinterpret_cast<const float4*>(
        static_cast<uintptr_t>(cu.unwrap(cu.getPosq()).getDevicePointer()));
    auto* force = reinterpret_cast<unsigned long long*>(
        static_cast<uintptr_t>(cu.unwrap(cu.getForce()).getDevicePointer()));

    runStep(posq, force, cu.getNumAtoms(), cu.getPaddedNumAtoms(),
            box, includeEnergy, includeForces, stream);

    // Deliberately does NOT read the energy back here, even when it was asked for.
    //
    // ContextImpl::calcForcesAndEnergy runs every ForceImpl first and only then
    // calls finishComputation(), which is where OpenMM enqueues its own bonded
    // kernel. Blocking on our energy inside execute() therefore stalls the host
    // BEFORE bonded is even submitted, so bonded cannot overlap our graph -- on a
    // barostat move it lands 14 us after our graph ends instead of alongside it,
    // and costs ~17.7 us of critical path twice per move. Force-only steps never
    // showed this because they never block here.
    //
    // So the readback moves to the post-computation (finishPendingEnergy), which
    // OpenMM invokes after the bonded submission and whose return value it adds to
    // the total. Nothing about the number changes; only when we wait for it.
    energy_pending_ = includeEnergy;
    return 0.0;
}

// Called from the post-computation, after OpenMM has enqueued its bonded kernel.
// Returns our contribution to the potential energy, or 0 if this evaluation did
// not ask for energy (or did not include our force group, in which case execute()
// never ran and the flag is still false).
double CudaCalcNonbondedForce::finishPendingEnergy()
{
    if (!energy_pending_)
        return 0.0;
    energy_pending_ = false;
    double energy = readEnergy();
    const double volume = cupo_box_volume(h_box_);
    if (model_->latticeSum)
        energy += model_->latticeSum->correctionEnergy(volume);
    if (model_->dispersion)
        energy += model_->dispersion->correctionEnergy(volume);
    energy += model_->cutoffInteraction.correctionEnergy(volume);
    for (const auto& tail : expression_tails_) energy += tail->coefficient()/volume;
    return energy;
}



void CudaCalcNonbondedForce::copyParametersToContext(OpenMM::ContextImpl& context,
                                                    const NonbondedForce& force,
                                                    int firstParticle, int lastParticle,
                                                    int firstException, int lastException)
{
    model_->validateUpdate(force);
    if (model_->hasBuiltinParameters() &&
            force.getExceptionsUsePeriodicBoundaryConditions() != exceptions_use_pbc_)
        throw OpenMM::OpenMMException(
            "updateParametersInContext: exception periodic boundary convention changed; reinitialize the Context");
    OpenMM::ContextSelector selector(cc_);
    const auto stream = reinterpret_cast<cudaStream_t>(
        static_cast<OpenMM::CudaContext&>(cc_).getCurrentStream());
    graphJoin(stream);
    if (custom_parameters_)
        custom_parameters_->validateDefinition(force);
    if (force.getNumExpressionTailCorrections()!=additional_tail_count_)
        throw OpenMM::OpenMMException("updateParametersInContext: tail definitions changed; reinitialize Context");
    std::vector<std::unique_ptr<ExpressionTailCorrection>> next_tails;
    for (size_t i=0;i<expression_tails_.size();++i) {
        const auto& definition = implicit_expression_tail_ && i==0
            ? force.getCutoffExpressionDefinition()
            : force.getExpressionTailCorrectionDefinition(int(i)-int(implicit_expression_tail_));
        next_tails.push_back(expression_tails_[i]->prepareUpdate(definition, context.getOwner()));
    }
    if (model_->hasBuiltinParameters() &&
            !copyChangedBuiltinParameters(context, force, firstParticle, lastParticle,
                                          firstException, lastException))
        copyBuiltinParameters(context, force);
    if (custom_parameters_)
        custom_parameters_->copyParametersToContext(context, force);
    expression_tails_.swap(next_tails);
}

// Update only the particles and exceptions in the changed ranges. Returns false,
// without changing anything, when the full update is needed instead.
bool CudaCalcNonbondedForce::copyChangedBuiltinParameters(OpenMM::ContextImpl& context,
                                                        const NonbondedForce& force,
                                                        int firstParticle, int lastParticle,
                                                        int firstException, int lastException)
{
    const auto previous = builtin_parameters_.definition;
    if (!previous || force.getNumParticles() != natoms_ || force.getNumExceptions() != n_exclusion_pairs_)
        return false;
    // Offsets may refer to any particle or exception, so they use the full update.
    if (force.getNumParticleParameterOffsets() > 0 || force.getNumExceptionParameterOffsets() > 0 ||
            !previous->particleOffsets.empty() || !previous->exceptionOffsets.empty())
        return false;
    if (force.getNumGlobalParameters() != static_cast<int>(previous->globalNames.size()))
        return false;
    for (int i = 0; i < force.getNumGlobalParameters(); ++i)
        if (force.getGlobalParameterName(i) != previous->globalNames[i])
            return false;
    const auto tail = force.getDispersionCorrectionInteraction();
    const auto& previousTail = previous->tailReference;
    if (tail.tail != previousTail.tail || tail.combiningRule != previousTail.combiningRule ||
            tail.useSwitching != previousTail.useSwitching ||
            (tail.useSwitching && tail.switchingDistance != previousTail.switchingDistance) ||
            force.getDispersionCorrectionFollowsParameters() != previous->tailFollowsParameters ||
            (previous->tail != CutoffInteraction::NoTail && !previous->tailFollowsParameters))
        return false;

    firstParticle = std::max(firstParticle, 0);
    lastParticle = std::min(lastParticle, natoms_ - 1);
    firstException = std::max(firstException, 0);
    lastException = std::min(lastException, n_exclusion_pairs_ - 1);
    const bool particles = firstParticle <= lastParticle;
    const bool exceptions = firstException <= lastException;

    BuiltinParameters next;
    next.values = builtin_parameters_.values;
    next.particles = builtin_parameters_.particles;
    next.exceptions = builtin_parameters_.exceptions;
    if (particles || exceptions) {
        auto definition = std::make_shared<BuiltinParameterDefinition>(*previous);
        for (int i = firstParticle; i <= lastParticle; ++i)
            force.getParticleParameters(i, definition->baseParticles[3*i],
                definition->baseParticles[3*i+1], definition->baseParticles[3*i+2]);
        for (int k = firstException; k <= lastException; ++k) {
            int first, second;
            force.getExceptionParameters(k, first, second, definition->baseExceptions[3*k],
                definition->baseExceptions[3*k+1], definition->baseExceptions[3*k+2]);
            if (first != sysorder_exclusion_pairs_[2*k] || second != sysorder_exclusion_pairs_[2*k+1])
                throw OpenMM::OpenMMException(
                    "updateParametersInContext: exception particles changed; reinitialize the Context");
        }
        // Without offsets, the effective parameters are the base parameters.
        if (particles)
            std::copy(definition->baseParticles.begin() + 3*firstParticle,
                      definition->baseParticles.begin() + 3*(lastParticle+1),
                      next.particles.begin() + 3*firstParticle);
        if (exceptions)
            std::copy(definition->baseExceptions.begin() + 3*firstException,
                      definition->baseExceptions.begin() + 3*(lastException+1),
                      next.exceptions.begin() + 3*firstException);
        next.definition = std::move(definition);
    }
    else {
        next.definition = previous;
    }
    readOffsetValues(next, context);
    next.ranged = true;
    next.firstParticle = firstParticle;
    next.lastParticle = lastParticle;
    next.firstException = firstException;
    next.lastException = lastException;
    const unsigned changed = (particles ? ParticlesChanged : 0u) | (exceptions ? ExceptionsChanged : 0u);
    pushBuiltinParameters(std::move(next), changed, particles || exceptions);
    return true;
}

void CudaCalcNonbondedForce::copyBuiltinParameters(OpenMM::ContextImpl& context,
                                                 const NonbondedForce& force)
{
    if (force.getNumParticles() != natoms_)
        throw OpenMM::OpenMMException("updateParametersInContext: The number of particles has changed");
    if (force.getNumExceptions() != n_exclusion_pairs_)
        throw OpenMM::OpenMMException("updateParametersInContext: The set of exclusions has changed");
    for (int k = 0; k < n_exclusion_pairs_; ++k) {
        int first, second;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(k, first, second, chargeProd, sigma, epsilon);
        if (first != sysorder_exclusion_pairs_[2*k] ||
                second != sysorder_exclusion_pairs_[2*k+1])
            throw OpenMM::OpenMMException(
                "updateParametersInContext: exception particles changed; reinitialize the Context");
    }

    auto next = readBuiltinParameters(force);
    const auto& definition = *next.definition;
    const auto& previous = *builtin_parameters_.definition;
    const auto& beforeTail = previous.tailReference;
    if (definition.tailReference.combiningRule!=beforeTail.combiningRule ||
            definition.tailReference.useSwitching!=beforeTail.useSwitching ||
            (definition.tailReference.useSwitching &&
             definition.tailReference.switchingDistance!=beforeTail.switchingDistance))
        throw OpenMM::OpenMMException("updateParametersInContext: reference LJ tail policy changed; reinitialize Context");
    if (definition.globalNames != previous.globalNames)
        throw OpenMM::OpenMMException(
            "updateParametersInContext: global parameters changed; reinitialize the Context");
    const auto checkOffsets = [&](const vector<ParameterOffset>& before,
                                  const vector<ParameterOffset>& after, const char* kind) {
        if (before.size() != after.size())
            throw OpenMM::OpenMMException(string("updateParametersInContext: the number of ") +
                kind + " parameter offsets changed; reinitialize the Context");
        for (size_t i = 0; i < before.size(); ++i) {
            if (before[i].index != after[i].index ||
                    previous.watchedNames[before[i].parameter] !=
                        definition.watchedNames[after[i].parameter])
                throw OpenMM::OpenMMException(string("updateParametersInContext: ") + kind +
                    " offset changed its " + kind + " or its parameter; reinitialize the Context");
        }
    };
    checkOffsets(previous.particleOffsets, definition.particleOffsets, "particle");
    checkOffsets(previous.exceptionOffsets, definition.exceptionOffsets, "exception");

    // Prepare a complete candidate using the current Context globals. Validation
    // and tail integration happen before the active buffers or canonical records
    // change, including updates made before the first force evaluation.
    readOffsetValues(next, context);
    const unsigned changed = applyOffsets(next, {});
    pushBuiltinParameters(std::move(next), changed, true);
}

// ===========================================================================
// Per-step compute
// ===========================================================================

// Host-only per-step box update, kept out of any capture. Recomputes the cell grid
// if the box crossed a cell-count boundary (NPT); NVT always stays in-band.
// Unsupported boxes raise a host error before a kernel or graph is launched.
void CudaCalcNonbondedForce::updateBoxState(const float box[CUPO_BOX_FLOATS])
{
    cupo_cell_list_box_status box_status = CUPO_CELL_LIST_BOX_IN_BAND;
    const cupo_status status =
        cupo_cell_list_update_box(cell_list_, box, &box_status);
    if (box_status == CUPO_CELL_LIST_BOX_BELOW_FLOOR)
        throw OpenMM::OpenMMException(
            "NonbondedForce: box shrank below the direct-space cell-grid minimum "
            "for its neighbor stencil; increase the box size or reduce the cutoff. " +
            std::string(cupo_last_error()));
    if (box_status == CUPO_CELL_LIST_BOX_ABOVE_ENVELOPE)
        throw OpenMM::OpenMMException(
            "NonbondedForce: box expanded past the allocated cell-list envelope; "
            "raise the box safety factor and reinitialize Context. " +
            std::string(cupo_last_error()));
    cupoCheck(status);
    // A BAND_CROSS needs nothing here. update_box recomputed ncx/ncy/ncz in place
    // (every buffer is envelope-pre-sized, so no realloc), the captured kernels
    // derive the grid from d_box on-device with persistent launch dims, and the
    // cell list absorbs the regrid on its next ordinary build. So the graph keeps
    // replaying straight through a cross -- no eager step, no recapture.
    if (!std::equal(h_box_, h_box_ + CUPO_BOX_FLOATS, box)) {
        if (box_upload_pending_) {
            NV_CUDA_CHECK(cudaEventSynchronize(box_upload_done_));
            box_upload_pending_ = false;
        }
        std::copy(box, box + CUPO_BOX_FLOATS, h_box_);
    }
}

// Refresh d_box on the given stream only if the box moved (skipped for NVT).
void CudaCalcNonbondedForce::uploadBoxIfChanged(cudaStream_t s)
{
    if (!std::equal(h_box_, h_box_ + CUPO_BOX_FLOATS, box_on_device_)) {
        NV_CUDA_CHECK(cudaMemcpyAsync(d_box_, h_box_, CUPO_BOX_FLOATS * sizeof(float),
                                      cudaMemcpyHostToDevice, s));
        NV_CUDA_CHECK(cudaEventRecord(box_upload_done_, s));
        box_upload_pending_ = true;
        std::copy(h_box_, h_box_ + CUPO_BOX_FLOATS, box_on_device_);
        // Rebuild the reciprocal influence-function table for the new box, on this
        // stream so it is ordered ahead of the graph launch (or the eager path)
        // that follows. Paired with the upload deliberately: the table is the one
        // piece of reciprocal state that does not re-derive itself from d_box each
        // step, and a missed rebuild evaluates the OLD box silently -- no error, no
        // NaN. Cost is one pass over the spectrum, on barostat moves only.
        if (kspace_)
            cupoCheck(cupo_kspace_refresh_influence(kspace_, d_box_, s));
        if (dispersion_kspace_)
            cupoCheck(cupo_kspace_refresh_influence(dispersion_kspace_,d_box_,s));
    }
}

// Replay on a side stream so OpenMM can submit bonded work before joining the
// nonbonded graph. The same ordering applies to Standard and Custom modes.
void CudaCalcNonbondedForce::replayCapturedGraph(cudaGraphExec_t exec, cudaStream_t stream)
{
    // Box as a graph parameter: the captured kernels read d_box by pointer, so an
    // in-band barostat move only needs d_box refreshed before replay. Upload on the
    // launch stream (ordered before both the graph launch and the bonded-overlap
    // bridge_in below). A band-crossing box change was caught in updateBoxState.
    uploadBoxIfChanged(stream);

    // Fork the graph onto graph_origin so OpenMM's current stream stays free for its
    // bonded kernel. bridge_in_event (recorded on `stream`) captures this step's
    // force-clear + posq write so the graph's roots wait for them. The ws-join is
    // deferred to the ForcePostComputation (graphJoin), which runs after OpenMM
    // enqueues bonded -- so bonded overlaps the graph.
    NV_CUDA_CHECK(cudaEventRecord(bridge_in_event_, stream));
    NV_CUDA_CHECK(cudaStreamWaitEvent(graph_origin, bridge_in_event_, 0));
    NV_CUDA_CHECK(cudaGraphLaunch(exec, graph_origin));
    if (bonded_cells_ready_event_)
        NV_CUDA_CHECK(cudaStreamWaitEvent(stream, bonded_cells_ready_event_, 0));
    NV_CUDA_CHECK(cudaEventRecord(bridge_out_event_, graph_origin));
    graph_side_launched_ = true;
}

void CudaCalcNonbondedForce::runStep(const float4* posq, unsigned long long* force,
                                    int n, int paddedN, const float box[CUPO_BOX_FLOATS],
                                    bool computeEnergy, bool computeForces,
                                    cudaStream_t stream)
{
    // Graph mode runs on a dedicated capturable origin stream (OpenMM hands us the
    // un-capturable default/nil stream); normal mode ws == OpenMM's stream.
    cudaStream_t ws = graph_enabled_ ? graph_origin : stream;
    const int ce = computeEnergy ? 1 : 0;
    const int cf = computeForces ? 1 : 0;

    updateBoxState(box);

    // An enclosing whole-step capture owns the graph. OpenMM's normal default
    // stream cannot be capturing; query only when it supplies an explicit queue.
    cudaStreamCaptureStatus outerStatus = cudaStreamCaptureStatusNone;
    if (stream) NV_CUDA_CHECK(cudaStreamIsCapturing(stream, &outerStatus));
    if (outerStatus == cudaStreamCaptureStatusActive) {
        NV_CUDA_CHECK(cudaEventRecord(bridge_in_event_, stream));
        NV_CUDA_CHECK(cudaStreamWaitEvent(graph_origin, bridge_in_event_, 0));
        dispatchStep(posq, force, n, paddedN, computeEnergy, computeForces, graph_origin, true);
        // Honor the same bonded schedule as component replay.
        if (bonded_cells_ready_event_)
            NV_CUDA_CHECK(cudaStreamWaitEvent(stream, bonded_cells_ready_event_, 0));
        NV_CUDA_CHECK(cudaEventRecord(bridge_out_event_, graph_origin));
        graph_side_launched_ = true;
        return;
    }

    // Overflow needs no host poll: the captured migrate/rowsort kernels __trap() on
    // any cell that exceeds the fixed occupancy envelope, aborting directly.

    // One graph per (energy, forces) combination -- see GraphSlot in the header for
    // why a single slot silently condemns every other combination to run eagerly.
    GraphSlot& slot = graph_slots_[ce * 2 + cf];

    if (slot.captured) {
        replayCapturedGraph(slot.exec, stream);
        return;
    }
    // Bridge IN (capture/warmup only): origin waits OpenMM's stream (fresh positions).
    if (graph_enabled_) {
        NV_CUDA_CHECK(cudaEventRecord(bridge_in_event_, stream));
        NV_CUDA_CHECK(cudaStreamWaitEvent(ws, bridge_in_event_, 0));
    }
    // Box upload on ws BEFORE BeginCapture so it is NOT recorded into the graph
    // (replay refreshes d_box itself on the launch stream). Skipped when unchanged.
    uploadBoxIfChanged(ws);

    // Two gates before capturing a combination. graph_warmup_ counts eager steps of
    // ANY combination and says the SYSTEM is warm -- allocations settled, occupancy
    // envelope locked -- which is what the original 50-step threshold was protecting.
    // slot.seen says THIS combination has already run, because its first run is where
    // any lazily-sized buffer it alone touches gets allocated, and an allocation
    // inside a capture is illegal. Two prior runs is cheap insurance: energy-only
    // occurs twice per barostat move, so it still captures within a few moves.
    bool capturing = false;
    if (graph_enabled_ && graph_warmup_ >= 50 && slot.seen >= 2) {
        if (cudaStreamBeginCapture(ws, cudaStreamCaptureModeThreadLocal) == cudaSuccess)
            capturing = true;
        else
            cudaGetLastError();
    }
    if (!capturing) {
        graph_warmup_++;
        slot.seen++;
    }

    dispatchStep(posq, force, n, paddedN, computeEnergy, computeForces, ws);

    if (capturing)
        endCaptureAndInstantiate(slot, computeEnergy, computeForces, ws);

    // Bridge OUT (warmup + capture step; replay path bridged above).
    if (graph_enabled_) {
        NV_CUDA_CHECK(cudaEventRecord(bridge_out_event_, ws));
        NV_CUDA_CHECK(cudaStreamWaitEvent(stream, bridge_out_event_, 0));
    }
}


void CudaCalcNonbondedForce::dispatchStep(const float4* posq, unsigned long long* force,
                                         int n, int paddedN,
                                         bool computeEnergy, bool computeForces,
                                         cudaStream_t ws, bool wholeCapture)
{
    cudaStream_t dS = dspace_stream_;
    cudaStream_t kS = kspace_stream_;
    cudaStream_t mS = misc_stream_;

    // All input writes and OpenMM's force clear precede this fork. Retain the
    // Standard overlap: direct clears run beside the one shared cell-list build.
    NV_CUDA_CHECK(cudaEventRecord(step_fork_event_, ws));
    NV_CUDA_CHECK(cudaStreamWaitEvent(mS, step_fork_event_, 0));
    cupoCheck(cupo_dspace_prepare(dspace_, computeEnergy, computeForces, mS));
    NV_CUDA_CHECK(cudaEventRecord(force_clear_done_event_, mS));

    NV_CUDA_CHECK(cudaStreamWaitEvent(dS, step_fork_event_, 0));
    cupoCheck(cupo_cell_list_execute(cell_list_, d_box_,
        reinterpret_cast<const float*>(posq), 4,
        particle_input_, particle_stride_bytes_, dS));

    // First-use tuning measures both production branches through their join.
    // Wait for the asynchronous direct clear before borrowing its accumulators.
    // Keep steady-state and captured stream dependencies unchanged.
    if ((kspace_ || dispersion_kspace_) && !piece_tuning_attempted_) {
        NV_CUDA_CHECK(cudaStreamWaitEvent(dS, force_clear_done_event_, 0));
        cupo_kspace_handle reciprocals[2];
        int reciprocal_count = 0;
        if (kspace_) reciprocals[reciprocal_count++] = kspace_;
        if (dispersion_kspace_) reciprocals[reciprocal_count++] = dispersion_kspace_;
        cupoCheck(cupo_dspace_tune_with_kspaces(
            dspace_, reciprocals, reciprocal_count, d_box_, paddedN, dS));
        piece_tuning_attempted_ = true;
    }

    if (bonded_cells_ready_event_) {
        // Keep a real event record node: a wait outside the graph must observe
        // completion of sorting in this replay, not only a capture dependency.
        cudaStreamCaptureStatus status;
        NV_CUDA_CHECK(cudaStreamIsCapturing(dS,&status));
        if (status==cudaStreamCaptureStatusActive && !wholeCapture)
            NV_CUDA_CHECK(cudaEventRecordWithFlags(bonded_cells_ready_event_,dS,cudaEventRecordExternal));
        else
            NV_CUDA_CHECK(cudaEventRecord(bonded_cells_ready_event_,dS));
    }


    // Direct and reciprocal work overlap. Their force outputs use the same
    // fixed64 atomics and therefore commute.
    if (kspace_ || dispersion_kspace_) {
        NV_CUDA_CHECK(cudaEventRecord(cells_ready_event_, dS));
        NV_CUDA_CHECK(cudaStreamWaitEvent(kS, cells_ready_event_, 0));
        if (kspace_) cupoCheck(cupo_kspace_execute(kspace_, d_box_,
            computeEnergy ? d_energy_ + 1 : nullptr,
            computeForces ? force : nullptr, paddedN, kS));
        if (dispersion_kspace_)
            cupoCheck(cupo_kspace_execute(dispersion_kspace_,d_box_,
                computeEnergy ? d_energy_+2 : nullptr,
                computeForces ? force : nullptr,paddedN,kS));
        NV_CUDA_CHECK(cudaEventRecord(kspace_done_event_, kS));
    }

    NV_CUDA_CHECK(cudaStreamWaitEvent(dS, force_clear_done_event_, 0));
    cupoCheck(cupo_dspace_execute(dspace_, d_box_, computeEnergy, computeForces, dS));
    cupoCheck(cupo_dspace_finish(dspace_,
        computeEnergy ? d_energy_ : nullptr,
        computeForces ? force : nullptr, paddedN, 1.0f, dS));
    if (kspace_ || dispersion_kspace_)
        NV_CUDA_CHECK(cudaStreamWaitEvent(ws, kspace_done_event_, 0));
    NV_CUDA_CHECK(cudaEventRecord(dspace_done_event_, dS));
    NV_CUDA_CHECK(cudaStreamWaitEvent(ws, dspace_done_event_, 0));

    // Each producer writes its own contiguous energy slot directly. The joins
    // above make one later host readback sufficient; no packing kernel is needed.
}

void CudaCalcNonbondedForce::endCaptureAndInstantiate(GraphSlot& slot, bool computeEnergy,
                                                     bool computeForces, cudaStream_t ws)
{
    cudaGraph_t captured = nullptr;
    cudaError_t cap = cudaStreamEndCapture(ws, &captured);
    if (cap != cudaSuccess || captured == nullptr) {
        if (captured) cudaGraphDestroy(captured);
        graph_enabled_ = false;
        cudaGetLastError();
        throw OpenMM::OpenMMException(std::string("NonbondedForce: CUDA graph capture failed: ") +
                                      (cap == cudaSuccess ? "no graph was returned" : cudaGetErrorString(cap)));
    }
    if (slot.exec) NV_CUDA_CHECK(cudaGraphExecDestroy(slot.exec));
    if (slot.graph) NV_CUDA_CHECK(cudaGraphDestroy(slot.graph));
    slot.graph = captured;
    const cudaError_t instantiated = cudaGraphInstantiate(&slot.exec, slot.graph, 0);
    if (instantiated != cudaSuccess) {
        graph_enabled_ = false;
        throw OpenMM::OpenMMException(std::string("NonbondedForce: CUDA graph instantiation failed: ") +
                                      cudaGetErrorString(instantiated));
    }
    slot.captured = true;
    debug_graph_capture_count++;
    if (debug_graph_trace_) {
        fprintf(stderr, "[nv-plugin cupo] graph capture #%ld (energy=%d forces=%d)\n",
                debug_graph_capture_count, computeEnergy ? 1 : 0, computeForces ? 1 : 0);
    }
    NV_CUDA_CHECK(cudaGraphLaunch(slot.exec, ws));
}

double CudaCalcNonbondedForce::readEnergy()
{
    // Both eager dispatch and graph post-computation join the producers onto
    // OpenMM's stream before this one readback.
    auto& cu = static_cast<OpenMM::CudaContext&>(cc_);
    const auto stream = reinterpret_cast<cudaStream_t>(cu.getCurrentStream());
    double energies[3] = {};
    NV_CUDA_CHECK(cudaMemcpyAsync(energies, d_energy_, (dispersion_kspace_ ? 3 : 2)*sizeof(double),
                                 cudaMemcpyDeviceToHost, stream));
    NV_CUDA_CHECK(cudaStreamSynchronize(stream));
    double energy = energies[0];
    if (kspace_) energy += energies[1];
    if (dispersion_kspace_) energy += energies[2];
    return energy;
}

// ===========================================================================
// Atom reorder / parameter update (host bookkeeping + device re-upload)
// ===========================================================================

void CudaCalcNonbondedForce::reorderHostData(const int* atomIndex, int n)
{
    // Gather from canonical system order into the distinct GPU-order buffer.
    for (int gpu = 0; gpu < n; gpu++) {
        const int sys = atomIndex[gpu];
        h_params_[gpu * 3 + 0] = sysorder_params_[sys * 3 + 0];
        h_params_[gpu * 3 + 1] = sysorder_params_[sys * 3 + 1];
        h_params_[gpu * 3 + 2] = sysorder_params_[sys * 3 + 2];
    }

    // Numeric edits preserve endpoints. Only a new atom permutation requires
    // the inverse map; keep the dirty flag until row validation completes.
    if (exclusion_topology_dirty_ && n_exclusion_pairs_ > 0 && h_exclusion_pairs_) {
        std::vector<int> sys2gpu(n);
        for (int gpu = 0; gpu < n; gpu++)
            sys2gpu[atomIndex[gpu]] = gpu;
        for (int k = 0; k < n_exclusion_pairs_; k++) {
            h_exclusion_pairs_[k * 2 + 0] = sys2gpu[sysorder_exclusion_pairs_[k * 2 + 0]];
            h_exclusion_pairs_[k * 2 + 1] = sys2gpu[sysorder_exclusion_pairs_[k * 2 + 1]];
        }
    }
}

// Deferred graph join (bonded-overlap mode): make ws wait for the side-stream graph's
// completion so OpenMM's bonded kernel overlaps it. No-op unless a replay step
// launched the graph off ws this step.
void CudaCalcNonbondedForce::graphJoin(cudaStream_t ws)
{
    if (!graph_side_launched_) return;
    NV_CUDA_CHECK(cudaStreamWaitEvent(ws, bridge_out_event_, 0));
    graph_side_launched_ = false;
}

} // namespace NvOpenMM
