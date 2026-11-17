// The kernel interfaces and their documentation comments follow OpenMM
// olla/include/openmm/kernels.h:
// Copyright (c) 2008-2025 Stanford University and the Authors.
// Author: Peter Eastman.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT

#ifndef NV_KERNELS_H_
#define NV_KERNELS_H_

#include "nvopenmm/NonbondedForce.h"
#include "nvopenmm/LangevinMiddleIntegrator.h"
#include "nvopenmm/MonteCarloBarostat.h"
#include "openmm/Kernel.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include <iosfwd>
#include <string>

namespace NvOpenMM {

// ---------------------------------------------------------------------------
// The `Nv` in the Name() strings below is LOAD-BEARING. Do not drop it to match
// the un-prefixed class names.
//
// The class names need no prefix because `NvOpenMM` disambiguates them from
// OpenMM's own CalcNonbondedForceKernel, ApplyMonteCarloBarostatKernel and
// IntegrateLangevinMiddleStepKernel at the language level. These strings get no
// such protection: they are keys in OpenMM's kernel-factory registry, which is a
// single flat map<string, KernelFactory*> shared with OpenMM and every other
// loaded plugin. A namespace cannot disambiguate a std::string.
//
// Registering under OpenMM's own names does not fail loudly -- our factory simply
// wins the key, and the next time OpenMM asks its registry for "CalcNonbondedForce"
// it gets ours and dynamic_casts it to its own kernel type. That surfaces as
// `std::bad_cast` out of Context construction, nowhere near the cause. (Observed:
// dropping the prefix here failed 126 of 141 plugin tests exactly this way.)
// ---------------------------------------------------------------------------

/**
 * Shared execution interface for Standard and Custom NonbondedForce modes.
 */
class CalcNonbondedForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "NvCalcNonbondedForce";
    }
    CalcNonbondedForceKernel(std::string name, const OpenMM::Platform& platform)
        : OpenMM::KernelImpl(name, platform) {}

    /**
     * Initialize the kernel.
     *
     * @param system   the System this kernel will be applied to
     * @param force    the NonbondedForce this kernel will be used for
     * @param boxSafetyFactor  the RESOLVED cell-grid headroom. Passed rather than
     *        read off @p force because the force may carry
     *        NonbondedForce::AutoBoxSafety, which is a request and not a factor;
     *        resolving it needs the System, so it is done once by
     *        NonbondedForceImpl and the answer travels with the call.
     */
    virtual void initialize(const OpenMM::System& system, const NonbondedForce& force,
                            float boxSafetyFactor) = 0;

    /**
     * Execute the kernel to calculate forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;

    /**
     * Copy changed parameters to a context.
     *
     * @param context   the context to copy parameters to
     * @param force     the NonbondedForce to copy the parameters from
     * @param firstParticle   the index of the first particle whose parameters might have changed
     * @param lastParticle    the index of the last particle whose parameters might have changed
     * @param firstException  the index of the first exception whose parameters might have changed
     * @param lastException   the index of the last exception whose parameters might have changed
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const NonbondedForce& force,
                                         int firstParticle, int lastParticle,
                                         int firstException, int lastException) = 0;

};

/**
 * This kernel is invoked by LangevinMiddleIntegrator to take one time step.
 */
class IntegrateLangevinMiddleStepKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "NvIntegrateLangevinMiddleStep";
    }
    IntegrateLangevinMiddleStepKernel(std::string name, const OpenMM::Platform& platform)
        : OpenMM::KernelImpl(name, platform) {}

    /**
     * Initialize the kernel.
     *
     * @param system      the System this kernel will be applied to
     * @param integrator  the LangevinMiddleIntegrator this kernel is used for
     */
    virtual void initialize(const OpenMM::System& system, const LangevinMiddleIntegrator& integrator) = 0;

    /**
     * Execute the kernel to advance the simulation by one time step.
     *
     * @param context     the context in which to execute this kernel
     * @param integrator  the LangevinMiddleIntegrator this kernel is used for
     */
    virtual void execute(OpenMM::ContextImpl& context, const LangevinMiddleIntegrator& integrator) = 0;

    /** Optional whole-step graph. Returns false without advancing when unsupported. */
    virtual bool executeSteps(OpenMM::ContextImpl& context, const LangevinMiddleIntegrator& integrator, int steps) {
        return false;
    }
    virtual long long getWholeStepGraphCaptureCount() const { return 0; }
    virtual long long getWholeStepGraphReplayCount() const { return 0; }

    /**
     * Compute the kinetic energy.
     */
    virtual double computeKineticEnergy(OpenMM::ContextImpl& context, const LangevinMiddleIntegrator& integrator) = 0;
    /** Save/restore private integrator state not owned by OpenMM's platform. */
    virtual void createCheckpoint(std::ostream& stream) const = 0;
    virtual void loadCheckpoint(std::istream& stream) = 0;
};

/**
 * This kernel is invoked by MonteCarloBarostat to carry out one trial volume move.
 *
 * The host side owns the Monte Carlo decision; this kernel owns only the device
 * state it acts on -- snapshot it, scale the coordinates, and put it back if the
 * move is rejected.
 */
class ApplyMonteCarloBarostatKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "NvApplyMonteCarloBarostat";
    }
    ApplyMonteCarloBarostatKernel(std::string name, const OpenMM::Platform& platform)
        : OpenMM::KernelImpl(name, platform) {}

    /**
     * Initialize the kernel.
     *
     * @param system    the System this kernel will be applied to
     * @param barostat  the MonteCarloBarostat this kernel is used for
     */
    virtual void initialize(const OpenMM::System& system, const MonteCarloBarostat& barostat) = 0;

    /**
     * Snapshot everything a rejected move has to undo: positions, velocities,
     * forces, cell offsets and atom order. More than scaleCoordinates touches,
     * because the trial energy evaluation between save and restore is allowed
     * to reorder atoms.
     */
    virtual void saveState(OpenMM::ContextImpl& context) = 0;

    /**
     * Scale the system isotropically by displacing each molecule's centroid,
     * leaving intramolecular geometry rigid.
     *
     * @param context  the context to operate on
     * @param scale    the linear scale factor applied to all three axes
     */
    virtual void scaleCoordinates(OpenMM::ContextImpl& context, double scale) = 0;

    /** Undo a rejected move, restoring the state captured by saveState(). */
    virtual void restoreState(OpenMM::ContextImpl& context) = 0;
};

} // namespace NvOpenMM

#endif // NV_KERNELS_H_
