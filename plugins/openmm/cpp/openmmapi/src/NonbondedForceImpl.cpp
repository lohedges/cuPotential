// The dispersion correction (calcDispersionCorrection) follows OpenMM
// openmmapi/src/NonbondedForceImpl.cpp:
// Copyright (c) 2008-2024 Stanford University and the Authors.
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

#include "nvopenmm/internal/NonbondedForceImpl.h"
#include "nvopenmm/Kernels.h"
#include "nvopenmm/MonteCarloBarostat.h"
#include "lj_tail.hpp"
#include "openmm/internal/ContextImpl.h"
#include "openmm/CustomNonbondedForce.h"
#include "openmm/MonteCarloAnisotropicBarostat.h"
#include "openmm/MonteCarloBarostat.h"
#include "openmm/MonteCarloFlexibleBarostat.h"
#include "openmm/MonteCarloMembraneBarostat.h"
#include "openmm/OpenMMException.h"
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>


// Definitions live in the plugin's namespace, so our NonbondedForce et al.
// win unqualified lookup over OpenMM's same-named classes rather than
// being ambiguous with them.
namespace NvOpenMM {
using namespace std;

NonbondedForceImpl::NonbondedForceImpl(const NonbondedForce& owner) : owner(owner) {
}

NonbondedForceImpl::~NonbondedForceImpl() {
}

static void validateParameterOffsets(const NonbondedForce& force) {
    for (int k = 0; k < force.getNumParticleParameterOffsets(); k++) {
        string parameter;
        int particle;
        double chargeScale, sigmaScale, epsilonScale;
        force.getParticleParameterOffset(k, parameter, particle, chargeScale,
                                         sigmaScale, epsilonScale);
        if (particle < 0 || particle >= force.getNumParticles()) {
            stringstream msg;
            msg << "NonbondedForce: illegal particle index for a particle parameter offset: "
                << particle;
            throw OpenMM::OpenMMException(msg.str());
        }
    }
    for (int k = 0; k < force.getNumExceptionParameterOffsets(); k++) {
        string parameter;
        int exception;
        double chargeProdScale, sigmaScale, epsilonScale;
        force.getExceptionParameterOffset(k, parameter, exception, chargeProdScale,
                                          sigmaScale, epsilonScale);
        if (exception < 0 || exception >= force.getNumExceptions()) {
            stringstream msg;
            msg << "NonbondedForce: illegal exception index for an exception parameter "
                   "offset: " << exception;
            throw OpenMM::OpenMMException(msg.str());
        }
    }
}

static void validateSwitchingInterval(double start, double cutoff) {
    const float start32 = static_cast<float>(start);
    const float cutoff32 = static_cast<float>(cutoff);
    if (!std::isfinite(start) || !std::isfinite(cutoff) ||
            !(start >= 0 && start < cutoff))
        throw OpenMM::OpenMMException("NonbondedForce: switching distance must satisfy "
                                      "0 <= r_switch < r_cutoff and be finite");
    if (!std::isfinite(cutoff32) || !(start32 < cutoff32) ||
            !std::isfinite(1.0f/(cutoff32-start32)))
        throw OpenMM::OpenMMException("NonbondedForce: switching interval must remain "
                                      "finite and nonzero in FP32");
}

void NonbondedForceImpl::initialize(OpenMM::ContextImpl& context) {
    const OpenMM::System& system = context.getSystem();
    if (owner.getNumParticles() != system.getNumParticles()) {
        stringstream msg;
        msg << "NonbondedForce: number of particles (" << owner.getNumParticles()
            << ") does not match the System (" << system.getNumParticles() << ")";
        throw OpenMM::OpenMMException(msg.str());
    }
    if (owner.getPairMode() == NonbondedForce::Custom)
        validateCustomDefinition(owner.getCustomDefinition());
    else if (owner.getUseSwitchingFunction())
        validateSwitchingInterval(owner.getSwitchingDistance(), owner.getCutoffDistance());
    if (owner.getPairMode() == NonbondedForce::Standard && owner.hasCutoffExpression())
        validateCutoffReplacement(owner, owner.getCutoffExpressionDefinition());
    for (int i = 0; i < owner.getNumExpressionTailCorrections(); ++i)
        validateTailDefinition(owner, owner.getExpressionTailCorrectionDefinition(i));
    validateParameterOffsets(owner);
    kernel = context.getPlatform().createKernel(CalcNonbondedForceKernel::Name(), context);
    kernel.getAs<CalcNonbondedForceKernel>().initialize(context.getSystem(), owner,
                                                        resolveBoxSafetyFactor(system));
}

void NonbondedForceImpl::validateCustomDefinition(const OpenMM::CustomNonbondedForce& force) {
    const auto reject = [](const char* message) {
        throw OpenMM::OpenMMException(std::string("CustomNonbondedForce: ") + message);
    };
    if (force.getNonbondedMethod() != OpenMM::CustomNonbondedForce::CutoffPeriodic)
        reject("requires CutoffPeriodic; non-periodic geometry is not supported");
    const float cutoff = static_cast<float>(force.getCutoffDistance());
    if (!(cutoff > 0) || !std::isfinite(cutoff))
        reject("cutoff must be positive and finite in FP32");
    if (force.getNumInteractionGroups())
        reject("interaction groups are not supported; encode pair selection in the custom expression");
    if (force.getNumTabulatedFunctions())
        reject("tabulated functions are not supported");
    if (force.getNumComputedValues())
        reject("computed values are not supported; expand them in the pair expression");
    if (force.getNumEnergyParameterDerivatives())
        reject("energy parameter derivatives are not supported");
    if (force.getUseSwitchingFunction())
        validateSwitchingInterval(force.getSwitchingDistance(), force.getCutoffDistance());
    std::set<std::pair<int, int>> exclusions;
    for (int i = 0; i < force.getNumExclusions(); ++i) {
        int a, b;
        force.getExclusionParticles(i, a, b);
        if (a < 0 || b < 0 || a >= force.getNumParticles() ||
                b >= force.getNumParticles() || a == b)
            reject("exclusion has invalid particle indices");
        if (a > b) std::swap(a, b);
        if (!exclusions.emplace(a, b).second)
            reject("duplicate exclusion");
    }
}

void NonbondedForceImpl::validateCutoffReplacement(
        const NonbondedForce& owner, const OpenMM::CustomNonbondedForce& definition) {
    if (owner.getPairMode() != NonbondedForce::Standard)
        throw OpenMM::OpenMMException("NonbondedForce: cutoff replacement requires Standard mode");
    validateCustomDefinition(definition);
    if (definition.getNumParticles() != owner.getNumParticles())
        throw OpenMM::OpenMMException("NonbondedForce: cutoff expression particle count must match the base force");
    if (definition.getCutoffDistance() != owner.getCutoffDistance())
        throw OpenMM::OpenMMException("NonbondedForce: cutoff expression must share the base cutoff");
    std::set<std::pair<int, int>> exceptions, exclusions;
    for (int i = 0; i < owner.getNumExceptions(); ++i) {
        int first, second;
        double charge, sigma, epsilon;
        owner.getExceptionParameters(i, first, second, charge, sigma, epsilon);
        if (first > second) std::swap(first, second);
        if (!exceptions.emplace(first, second).second)
            throw OpenMM::OpenMMException("NonbondedForce: duplicate exception pair");
    }
    for (int i = 0; i < definition.getNumExclusions(); ++i) {
        int first, second;
        definition.getExclusionParticles(i, first, second);
        if (first > second) std::swap(first, second);
        exclusions.emplace(first, second);
    }
    if (exceptions != exclusions)
        throw OpenMM::OpenMMException("NonbondedForce: cutoff expression exclusions must "
                                      "exactly match Standard exception pairs");
}

void NonbondedForceImpl::validateTailDefinition(
        const NonbondedForce& owner, const OpenMM::CustomNonbondedForce& definition) {
    if (!definition.getUseLongRangeCorrection() ||
            definition.getNonbondedMethod() != OpenMM::CustomNonbondedForce::CutoffPeriodic)
        throw OpenMM::OpenMMException("Expression tail requires periodic cutoff and long-range correction");
    if (definition.getNumParticles() != owner.getNumParticles() ||
            definition.getCutoffDistance() != owner.getCutoffDistance())
        throw OpenMM::OpenMMException("Expression tail particle count and cutoff must match the force");
    if (definition.getNumTabulatedFunctions() || definition.getNumComputedValues() ||
            definition.getNumEnergyParameterDerivatives())
        throw OpenMM::OpenMMException("Expression tail does not support tables, computed values or parameter derivatives");
    if (definition.getUseSwitchingFunction() &&
            !(definition.getSwitchingDistance() >= 0 &&
              definition.getSwitchingDistance() < definition.getCutoffDistance()))
        throw OpenMM::OpenMMException("Expression tail has an invalid switching interval");
    for (int i = 0; i < definition.getNumParticles(); ++i) {
        std::vector<double> values;
        definition.getParticleParameters(i, values);
        if (values.size() != (size_t)definition.getNumPerParticleParameters())
            throw OpenMM::OpenMMException("Expression tail has the wrong particle record size");
        for (double value : values)
            if (!std::isfinite(static_cast<float>(value)))
                throw OpenMM::OpenMMException("Expression tail parameters must be finite in FP32");
    }
    for (int i = 0; i < definition.getNumInteractionGroups(); ++i) {
        std::set<int> a, b;
        definition.getInteractionGroupParameters(i, a, b);
        for (const auto& group : {a, b})
            for (int atom : group)
                if (atom < 0 || atom >= definition.getNumParticles())
                    throw OpenMM::OpenMMException("Expression tail has an invalid group particle index");
    }
}

// Is there a barostat that will actually move the box?
//
// Frequency 0 means present but disabled, which is how callers switch NPT off without
// rebuilding the System. A barostat that moves nothing needs no headroom.
static bool hasBarostat(const OpenMM::System& system) {
    for (int i = 0; i < system.getNumForces(); i++) {
        const OpenMM::Force& f = system.getForce(i);
        if (const NvOpenMM::MonteCarloBarostat* b =
                dynamic_cast<const NvOpenMM::MonteCarloBarostat*>(&f))
            { if (b->getFrequency() > 0) return true; }
        else if (const OpenMM::MonteCarloBarostat* b =
                dynamic_cast<const OpenMM::MonteCarloBarostat*>(&f))
            { if (b->getFrequency() > 0) return true; }
        else if (const OpenMM::MonteCarloAnisotropicBarostat* b =
                dynamic_cast<const OpenMM::MonteCarloAnisotropicBarostat*>(&f))
            { if (b->getFrequency() > 0) return true; }
        else if (const OpenMM::MonteCarloMembraneBarostat* b =
                dynamic_cast<const OpenMM::MonteCarloMembraneBarostat*>(&f))
            { if (b->getFrequency() > 0) return true; }
        else if (const OpenMM::MonteCarloFlexibleBarostat* b =
                dynamic_cast<const OpenMM::MonteCarloFlexibleBarostat*>(&f))
            { if (b->getFrequency() > 0) return true; }
    }
    return false;
}


float NonbondedForceImpl::resolveBoxSafetyFactor(const OpenMM::System& system) {
    const float requested = owner.getBoxSafetyFactor();
    const bool npt = hasBarostat(system);

    if (requested == NonbondedForce::AutoBoxSafety) {
        // 1. No safety factor given: pick one. NVT needs none, NPT gets the validated default.
        // Preserve the custom path's historical headroom for explicit box moves
        // as well as barostats. Standard NVT keeps its smaller allocation.
        return npt || owner.getPairMode() == NonbondedForce::Custom
            ? NonbondedForce::DefaultNPTBoxSafety : 1.0f;
    } else {
        // 2. A factor was given, some possible reasons include:
        // a) User explicitly calls setPeriodicBoxVectors
        // b) The default headroom of 1.15 is not enough for NPT simulations, etc.
        //
        // An active barostat requires headroom: the envelope is allocated once
        // and never resized, so a later expansion beyond it would be rejected.
        // Diagnose the missing headroom at Context creation.
        if (npt && requested <= 1.0f)
            throw OpenMM::OpenMMException("NonbondedForce: an active barostat needs box safety "
                                          "factor > 1.0, or AutoBoxSafety to have one chosen");
        return requested;
    }

}

double NonbondedForceImpl::calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces,
                                             bool includeEnergy, int groups) {
    if ((groups & (1 << owner.getForceGroup())) != 0)
        return kernel.getAs<CalcNonbondedForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

vector<string> NonbondedForceImpl::getKernelNames() {
    return {CalcNonbondedForceKernel::Name()};
}

std::map<std::string, double> NonbondedForceImpl::getDefaultParameters() {
    std::map<std::string, double> result;
    for (int i = 0; i < owner.getNumGlobalParameters(); ++i)
        result.emplace(owner.getGlobalParameterName(i), owner.getGlobalParameterDefaultValue(i));
    return result;
}

void NonbondedForceImpl::updateParametersInContext(OpenMM::ContextImpl& context, int firstParticle,
                                                   int lastParticle, int firstException, int lastException) {
    if (owner.getPairMode() == NonbondedForce::Custom)
        validateCustomDefinition(owner.getCustomDefinition());
    if (owner.getPairMode() == NonbondedForce::Standard && owner.hasCutoffExpression())
        validateCutoffReplacement(owner, owner.getCutoffExpressionDefinition());
    for (int i = 0; i < owner.getNumExpressionTailCorrections(); ++i)
        validateTailDefinition(owner, owner.getExpressionTailCorrectionDefinition(i));
    validateParameterOffsets(owner);
    kernel.getAs<CalcNonbondedForceKernel>().copyParametersToContext(context, owner, firstParticle, lastParticle,
                                                                     firstException, lastException);
    context.systemChanged();
}



// Match OpenMM's class-pair averaging, extended to geometric sigma. Preserve map
// iteration order and the unswitched arithmetic. Switching adds the missing LJ
// energy inside the switching interval to the ordinary beyond-cutoff correction.
double NonbondedForceImpl::calcDispersionCorrection(const NonbondedForce& force) {
    std::vector<double> sigma(force.getNumParticles()), epsilon(force.getNumParticles());
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge;
        force.getParticleParameters(i, charge, sigma[i], epsilon[i]);
    }
    std::map<std::string, double> parameters;
    for (int i = 0; i < force.getNumGlobalParameters(); ++i)
        parameters[force.getGlobalParameterName(i)] = force.getGlobalParameterDefaultValue(i);
    for (int i = 0; i < force.getNumParticleParameterOffsets(); ++i) {
        std::string parameter;
        int particle;
        double chargeScale, sigmaScale, epsilonScale;
        force.getParticleParameterOffset(i, parameter, particle, chargeScale, sigmaScale, epsilonScale);
        if (particle < 0 || particle >= force.getNumParticles())
            throw OpenMM::OpenMMException("NonbondedForce: illegal particle index for a particle parameter offset");
        sigma[particle] += parameters.at(parameter)*sigmaScale;
        epsilon[particle] += parameters.at(parameter)*epsilonScale;
    }
    const auto reference = force.getDispersionCorrectionInteraction();
    return calcDispersionCorrection(force.getNumParticles(), sigma.data(), epsilon.data(),
                                    force.getCutoffDistance(), reference.combiningRule,
                                    reference.useSwitching, reference.switchingDistance);
}

double NonbondedForceImpl::calcDispersionCorrection(int numParticles, const double* sigma,
        const double* epsilon, double cutoff, NonbondedForce::LJCombiningRule combiningRule,
        bool useSwitching, double switchingDistance) {
    if (numParticles == 0) return 0.0;
    std::map<std::pair<double, double>, int> classCounts;
    for (int i = 0; i < numParticles; i++) {
        if (!std::isfinite(sigma[i]) || !std::isfinite(epsilon[i]) || sigma[i] < 0 || epsilon[i] < 0)
            throw OpenMM::OpenMMException("NonbondedForce: effective sigma and epsilon must be finite and nonnegative");
        std::pair<double, double> key = std::make_pair(sigma[i], epsilon[i]);
        std::map<std::pair<double, double>, int>::iterator entry = classCounts.find(key);
        if (entry == classCounts.end())
            classCounts[key] = 1;
        else
            entry->second++;
    }

    double sum1 = 0, sum2 = 0;
    const bool geometric = combiningRule == NonbondedForce::Geometric;
    for (std::map<std::pair<double, double>, int>::const_iterator entry = classCounts.begin();
            entry != classCounts.end(); ++entry) {
        double sig = entry->first.first;
        double eps = entry->first.second;
        double count = (double) entry->second;
        count *= (count + 1) / 2;
        double sig2 = sig*sig;
        double sig6 = sig2*sig2*sig2;
        sum1 += count*eps*sig6*sig6;
        sum2 += count*eps*sig6;
    }
    for (std::map<std::pair<double, double>, int>::const_iterator class1 = classCounts.begin();
            class1 != classCounts.end(); ++class1)
        for (std::map<std::pair<double, double>, int>::const_iterator class2 = classCounts.begin();
                class2 != class1; ++class2) {
            double sig = 0.5*(class1->first.first+class2->first.first);
            double eps = std::sqrt(class1->first.second*class2->first.second);
            double count = (double) class1->second;
            count *= (double) class2->second;
            // Both rules retain geometric epsilon. Preserve the arithmetic
            // branch's operation order; geometric sigma needs no square root.
            double sig2 = geometric ? class1->first.first*class2->first.first : sig*sig;
            double sig6 = sig2*sig2*sig2;
            sum1 += count*eps*sig6*sig6;
            sum2 += count*eps*sig6;
        }
    const double particleCount = static_cast<double>(numParticles);
    double numInteractions = (particleCount*(particleCount+1))/2;
    sum1 /= numInteractions;
    sum2 /= numInteractions;
    if (useSwitching && (sum1 != 0 || sum2 != 0)) {
        const double start = switchingDistance;
        if (!(start > 0))
            throw OpenMM::OpenMMException("NonbondedForce: switched LJ dispersion correction "
                                          "requires a positive switching distance when LJ is nonzero");
        const auto integrals = cupo::lj_switched_tail_integrals(start, cutoff);
        return 8*particleCount*particleCount*M_PI*(sum1*integrals.twelfth
                                                -sum2*integrals.sixth);
    }
    return 8*particleCount*particleCount*M_PI*(sum1/(9*std::pow(cutoff, 9))
                                            -sum2/(3*std::pow(cutoff, 3)));
}

} // namespace NvOpenMM
