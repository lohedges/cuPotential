// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENMM_NVFORCEIMPL_H_
#define OPENMM_NVFORCEIMPL_H_

#include "nvopenmm/NonbondedForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <string>
#include <vector>

namespace OpenMM {
class System;
class CustomNonbondedForce;
}

namespace NvOpenMM {

/**
 * This is the internal implementation of NonbondedForce.
 */
class NonbondedForceImpl : public OpenMM::ForceImpl {
public:
    NonbondedForceImpl(const NonbondedForce& owner);
    ~NonbondedForceImpl();

    void initialize(OpenMM::ContextImpl& context);

    const NonbondedForce& getOwner() const { return owner; }

    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {}

    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces,
                               bool includeEnergy, int groups);

    std::map<std::string, double> getDefaultParameters();

    std::vector<std::string> getKernelNames();

    void updateParametersInContext(OpenMM::ContextImpl& context, int firstParticle, int lastParticle,
                                   int firstException, int lastException);

    /** Validate the supported cutoff expression contract before copying or use. */
    static void validateCustomDefinition(const OpenMM::CustomNonbondedForce& force);
    static void validateCutoffReplacement(const NonbondedForce& owner,
                                             const OpenMM::CustomNonbondedForce& definition);
    static void validateTailDefinition(const NonbondedForce& owner,
                                      const OpenMM::CustomNonbondedForce& definition);

    /**
     * Analytic long-range LJ dispersion correction for @p force, the counterpart of
     * NonbondedForceImpl::calcDispersionCorrection.
     *
     * Lives here rather than on NonbondedForce for the same reason it lives on the Impl
     * upstream: it is a derived quantity, not state, and the platform recomputes it
     * whenever the parameters it depends on change.
     *
     * Needs no System. Upstream divides by system.getNumParticles(); OpenMM refuses
     * to build a Context where that differs from the force's own particle count, so
     * the force alone is enough.
     */
    static double calcDispersionCorrection(const NonbondedForce& force);

    /** The same correction from effective parameters, retaining the force's
     * combining rule and switching policy. The caller supplies either global
     * default values or the current Context values according to its tail policy.
     */
    static double calcDispersionCorrection(int numParticles, const double* sigma,
        const double* epsilon, double cutoff, NonbondedForce::LJCombiningRule combiningRule,
        bool useSwitching, double switchingDistance);

private:
    /**
     * Determine the cell-grid headroom the platform will actually use.
     *
     * Standard AutoBoxSafety uses 1.0 without an active barostat. Custom mode
     * and active barostats use DefaultNPTBoxSafety. Explicit 1.0 under a barostat
     * throws because expansion requires preallocated headroom.
     */
    float resolveBoxSafetyFactor(const OpenMM::System& system);

    const NonbondedForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace NvOpenMM

#endif // OPENMM_NVFORCEIMPL_H_
