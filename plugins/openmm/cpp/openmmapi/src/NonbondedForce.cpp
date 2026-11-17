// The particle, exception, global-parameter and offset storage and their accessors
// follow OpenMM openmmapi/src/NonbondedForce.cpp:
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

#include "nvopenmm/NonbondedForce.h"
#include "nvopenmm/internal/NonbondedForceImpl.h"
#include "openmm/NonbondedForce.h"
#include "openmm/CustomNonbondedForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/System.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include <algorithm>
#include <set>
#include <cmath>
#include <map>
#include <string>
#include <utility>


// Definitions live in the plugin's namespace, so our NonbondedForce et al.
// win unqualified lookup over OpenMM's same-named classes rather than
// being ambiguous with them.
// OpenMM's ASSERT_VALID_INDEX macro expands to an unqualified throwException(),
// so that one name has to be visible here. A using-DECLARATION, not a directive:
// it names exactly what the macro needs and nothing else.
using OpenMM::throwException;

namespace NvOpenMM {
using namespace std;

namespace {
void validateRadialKernel(const RadialKernel& kernel) {
    if (kernel.power != 1 && kernel.power != 6)
        throw OpenMM::OpenMMException("RadialKernel: only inverse powers 1 and 6 are supported");
    const float coupling = static_cast<float>(kernel.coupling);
    if (!std::isfinite(coupling) || coupling == 0 ||
            (kernel.power == 1 ? coupling < 0 : coupling > 0))
        throw OpenMM::OpenMMException("RadialKernel: P1 coupling must be positive and P6 coupling negative, finite and nonzero in FP32");
}

void validateLatticeInteraction(const LatticeSumInteraction& interaction,
        const std::vector<LatticeSumInteraction>& existing, int replaced = -1) {
    validateRadialKernel(interaction.kernel);
    if (interaction.splitting != LatticeSumInteraction::Ewald &&
            interaction.splitting != LatticeSumInteraction::Prolate)
        throw OpenMM::OpenMMException("LatticeSumInteraction: unsupported splitting");
    if (interaction.weightExpression.find_first_not_of(" \t\r\n") == std::string::npos)
        throw OpenMM::OpenMMException("LatticeSumInteraction: weightExpression must not be empty");
    for (int index = 0; index < (int)existing.size(); ++index)
        if (index != replaced && existing[index].kernel.power == interaction.kernel.power)
            throw OpenMM::OpenMMException("LatticeSumInteraction: at most one term per inverse power is supported");
}

void validateExpressionParameterName(const std::string& name) {
    const auto alpha = [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    };
    if (name.empty() || !alpha(name[0]) ||
            !std::all_of(name.begin()+1, name.end(), [&](char c) {
                return alpha(c) || (c >= '0' && c <= '9');
            }))
        throw OpenMM::OpenMMException("CutoffInteraction: invalid expression parameter name " + name);
}
}

RadialKernel::RadialKernel(int power, double coupling) : power(power), coupling(coupling) {
    validateRadialKernel(*this);
}

RadialKernel RadialKernel::inversePower(int power, double coupling) {
    return RadialKernel(power, coupling);
}

RadialKernel CoulombRadialKernel(double coupling) {
    return RadialKernel(1, coupling);
}

RadialKernel DispersionRadialKernel(double coupling) {
    return RadialKernel(6, coupling);
}

NonbondedForce::NonbondedForce(NonbondedMethod method, double cutoff,
                               float boxSafetyFactor)
        : cutoffDistance(cutoff),
          boxSafetyFactor(boxSafetyFactor) {
    if (boxSafetyFactor != AutoBoxSafety && boxSafetyFactor < 1.0f)
        throw OpenMM::OpenMMException("NonbondedForce: box safety factor must be >= 1.0, "
                                      "or AutoBoxSafety to have it chosen from the System");
    if (method != PME && method != Prolate)
        throw OpenMM::OpenMMException("NonbondedForce: unsupported electrostatics method");
    latticeSums.emplace_back(CoulombRadialKernel(), "q");
    latticeSums.back().splitting = method == PME ? LatticeSumInteraction::Ewald : LatticeSumInteraction::Prolate;
    cutoffInteraction.tail = CutoffInteraction::HomogeneousTail;
}

int NonbondedForce::addParticle(double charge, double sigma, double epsilon) {
    requireMode(Standard, "addParticle(charge, sigma, epsilon)");
    particles.push_back(ParticleInfo(charge, sigma, epsilon));
    return (int)particles.size() - 1;
}

void NonbondedForce::getParticleParameters(int index, double& charge,
                                         double& sigma, double& epsilon) const {
    requireMode(Standard, "getParticleParameters(charge, sigma, epsilon)");
    ASSERT_VALID_INDEX(index, particles);
    charge  = particles[index].charge;
    sigma   = particles[index].sigma;
    epsilon = particles[index].epsilon;
}

void NonbondedForce::setParticleParameters(int index, double charge,
                                         double sigma, double epsilon) {
    requireMode(Standard, "setParticleParameters(charge, sigma, epsilon)");
    ASSERT_VALID_INDEX(index, particles);
    particles[index].charge  = charge;
    particles[index].sigma   = sigma;
    particles[index].epsilon = epsilon;
    if (numContexts > 0) {
        firstChangedParticle = std::min(index, firstChangedParticle);
        lastChangedParticle = std::max(index, lastChangedParticle);
    }
}

void NonbondedForce::updateParametersInContext(OpenMM::Context& context) {
    dynamic_cast<NonbondedForceImpl&>(getImplInContext(context))
        .updateParametersInContext(getContextImpl(context), firstChangedParticle, lastChangedParticle,
                                   firstChangedException, lastChangedException);
    if (numContexts == 1) {
        // The only Context using this force is now up to date.
        firstChangedParticle = (int)particles.size();
        lastChangedParticle = -1;
        firstChangedException = (int)exceptions.size();
        lastChangedException = -1;
    }
}

void NonbondedForce::setLJCombiningRule(LJCombiningRule rule) {
    requireMode(Standard, "setLJCombiningRule");
    if (hasCutoffExpression())
        throw OpenMM::OpenMMException("NonbondedForce: combining rules belong in the cutoff expression");
    if (rule != LorentzBerthelot && rule != Geometric)
        throw OpenMM::OpenMMException("NonbondedForce: unsupported LJ combining rule");
    cutoffInteraction.combiningRule = rule;
}

int NonbondedForce::addException(int particle1, int particle2,
                          double chargeProd, double sigma, double epsilon) {
    requireMode(Standard, "addException");
    ExceptionInfo e;
    e.particle1 = particle1;
    e.particle2 = particle2;
    e.chargeProd = chargeProd;
    e.sigma = sigma;
    e.epsilon = epsilon;
    exceptions.push_back(e);
    return (int)exceptions.size() - 1;
}

void NonbondedForce::getExceptionParameters(int index, int& particle1, int& particle2,
                                     double& chargeProd, double& sigma,
                                     double& epsilon) const {
    requireMode(Standard, "getExceptionParameters");
    ASSERT_VALID_INDEX(index, exceptions);
    const ExceptionInfo& e = exceptions[index];
    particle1 = e.particle1;
    particle2 = e.particle2;
    chargeProd = e.chargeProd;
    sigma = e.sigma;
    epsilon = e.epsilon;
}

void NonbondedForce::setExceptionParameters(int index, int particle1, int particle2,
                                     double chargeProd, double sigma, double epsilon) {
    requireMode(Standard, "setExceptionParameters");
    ASSERT_VALID_INDEX(index, exceptions);
    ExceptionInfo& e = exceptions[index];
    e.particle1 = particle1;
    e.particle2 = particle2;
    e.chargeProd = chargeProd;
    e.sigma = sigma;
    e.epsilon = epsilon;
    if (numContexts > 0) {
        firstChangedException = std::min(index, firstChangedException);
        lastChangedException = std::max(index, lastChangedException);
    }
}

OpenMM::ForceImpl* NonbondedForce::createImpl() const {
    if (numContexts == 0) {
        firstChangedParticle = (int)particles.size();
        lastChangedParticle = -1;
        firstChangedException = (int)exceptions.size();
        lastChangedException = -1;
    }
    numContexts++;
    return new NonbondedForceImpl(*this);
}

/* ---- Conversion from a stock NonbondedForce -------------------------------- */

NonbondedForce::NonbondedForce(const OpenMM::NonbondedForce& nonbonded,
                               float boxSafetyFactor)
        : NonbondedForce(nonbonded, boxSafetyFactor, Prolate) {}

NonbondedForce::NonbondedForce(const OpenMM::NonbondedForce& nonbonded,
                               float boxSafetyFactor, NonbondedMethod method)
        : cutoffDistance(nonbonded.getCutoffDistance()),
          boxSafetyFactor(boxSafetyFactor) {
    if (boxSafetyFactor != AutoBoxSafety && boxSafetyFactor < 1.0f)
        throw OpenMM::OpenMMException("NonbondedForce: box safety factor must be >= 1.0, "
                                      "or AutoBoxSafety to have it chosen from the System");

    if (nonbonded.getNonbondedMethod() == OpenMM::NonbondedForce::CutoffPeriodic)
        throw OpenMM::OpenMMException("NonbondedForce: Standard conversion requires a PME source; "
                                      "use CustomNonbondedForce with an explicit cutoff expression");
    if (nonbonded.getNonbondedMethod() != OpenMM::NonbondedForce::PME &&
            nonbonded.getNonbondedMethod() != OpenMM::NonbondedForce::LJPME)
        throw OpenMM::OpenMMException("NonbondedForce: only PME or LJPME source forces are supported");
    if (method != PME && method != Prolate)
        throw OpenMM::OpenMMException("NonbondedForce: unsupported electrostatics method");
    latticeSums.emplace_back(CoulombRadialKernel(), "q");
    latticeSums.back().splitting = method == PME ? LatticeSumInteraction::Ewald : LatticeSumInteraction::Prolate;
    cutoffInteraction.tail = CutoffInteraction::HomogeneousTail;
    setForceGroup(nonbonded.getForceGroup());
    setName(nonbonded.getName());

    // 5e-4 is the only accuracy target implemented, for both methods (see
    // NonbondedForce::ForceErrorTolerance). Silently substituting it would change the
    // splitting a caller asked for without changing any number it can read back, so a
    // source force carrying a different one is refused rather than reinterpreted.
    // Explicitly comparing against a stored default, which is exact.
    //
    // The source force's alpha and mesh settings are checked below; the selected
    // method derives its own values internally. The current API does not expose
    // a general accuracy-driven tuning procedure.
    if (nonbonded.getEwaldErrorTolerance() != ForceErrorTolerance)
        throw OpenMM::OpenMMException("NonbondedForce: only an Ewald error tolerance of "
                              "5e-4 is supported by the current mesh-sizing policies "
                              "for both PME and C2 Prolate. Call "
                              "setEwaldErrorTolerance(5e-4) on the source force.");

    // Validate source PME settings against the plugin's automatic parameter policy.
    // An explicit alpha is accepted only if it matches the default derived below.
    double srcAlpha;
    int srcNx, srcNy, srcNz;
    nonbonded.getPMEParameters(srcAlpha, srcNx, srcNy, srcNz);
    if (srcAlpha != 0.0) {
        // Matches OpenMM's calcPMEParameters and cuPo's EwaldPlan::default_alpha.
        const double derived = (1.0 / cutoffDistance)
                             * std::sqrt(-std::log(2.0 * ForceErrorTolerance));
        // Allow rounding differences from serialization.
        if (std::fabs(srcAlpha - derived) > 1e-9 * derived)
            throw OpenMM::OpenMMException("NonbondedForce: the source force pins an Ewald "
                                  "alpha of " + std::to_string(srcAlpha) + " nm^-1, but "
                                  "this plugin derives alpha from the cutoff at an error "
                                  "tolerance of 5e-4 and would use " +
                                  std::to_string(derived) + " nm^-1. The pinned value "
                                  "cannot be honoured -- under Prolate there is no alpha "
                                  "at all -- so it is refused rather than ignored. Call "
                                  "setPMEParameters(0, 0, 0, 0) on the source force to let "
                                  "both implementations derive it.");
    }
    // Mesh dimensions are derived from the periodic box at Context creation.
    // This constructor has no box with which to validate explicit dimensions.
    if (srcNx != 0 || srcNy != 0 || srcNz != 0)
        throw OpenMM::OpenMMException("NonbondedForce: the source force pins a PME mesh of ("
                              + std::to_string(srcNx) + ", " + std::to_string(srcNy) + ", "
                              + std::to_string(srcNz) + "). Explicit mesh dimensions are "
                              "not supported; the plugin derives them at Context creation. Call "
                              "setPMEParameters(0, 0, 0, 0) on the source force.");

    setUseSwitchingFunction(nonbonded.getUseSwitchingFunction());
    setSwitchingDistance(nonbonded.getSwitchingDistance());

    const int numParticles = nonbonded.getNumParticles();
    for (int i = 0; i < numParticles; i++) {
        double charge, sigma, epsilon;
        nonbonded.getParticleParameters(i, charge, sigma, epsilon);
        addParticle(charge, sigma, epsilon);
    }

    for (int g = 0; g < nonbonded.getNumGlobalParameters(); ++g)
        addGlobalParameter(nonbonded.getGlobalParameterName(g),
                           nonbonded.getGlobalParameterDefaultValue(g));
    for (int k = 0; k < nonbonded.getNumParticleParameterOffsets(); ++k) {
        std::string parameter;
        int particle;
        double chargeScale, sigmaScale, epsilonScale;
        nonbonded.getParticleParameterOffset(k, parameter, particle, chargeScale,
                                             sigmaScale, epsilonScale);
        addParticleParameterOffset(parameter, particle, chargeScale, sigmaScale, epsilonScale);
    }

    // Carried, not evaluated: the platform derives the coefficient from this, at
    // Context creation and again whenever the parameters change.
    setUseDispersionCorrection(nonbonded.getUseDispersionCorrection());
    if (nonbonded.getNonbondedMethod() == OpenMM::NonbondedForce::LJPME) {
        double alpha;
        int nx, ny, nz;
        nonbonded.getLJPMEParameters(alpha, nx, ny, nz);
        const double derived = std::sqrt(-std::log(2.0*ForceErrorTolerance))/cutoffDistance;
        if (!std::isfinite(alpha) || (alpha != 0.0 && std::fabs(alpha-derived) > 1e-9*derived) || nx || ny || nz)
            throw OpenMM::OpenMMException("NonbondedForce: pinned LJPME parameters are unsupported; "
                "use setLJPMEParameters(0, 0, 0, 0). Dispersion shares the Coulomb mesh.");
        setUseLJPME(true);
    }

    // Preserve the source's physical exception convention, including its default
    // false setting. Reciprocal exclusion corrections remain periodic either way.
    setExceptionsUsePeriodicBoundaryConditions(
        nonbonded.getExceptionsUsePeriodicBoundaryConditions());
    // Preserve source exceptions, including currently inactive pairs, so later
    // explicit particle updates cannot accidentally turn on an excluded pair.
    for (int ex = 0; ex < nonbonded.getNumExceptions(); ex++) {
        int i, j;
        double chargeProd, sigma, epsilon;
        nonbonded.getExceptionParameters(ex, i, j, chargeProd, sigma, epsilon);
        addException(i, j, chargeProd, sigma, epsilon);
    }
    for (int k = 0; k < nonbonded.getNumExceptionParameterOffsets(); ++k) {
        std::string parameter;
        int exception;
        double chargeProdScale, sigmaScale, epsilonScale;
        nonbonded.getExceptionParameterOffset(k, parameter, exception, chargeProdScale,
                                              sigmaScale, epsilonScale);
        addExceptionParameterOffset(parameter, exception, chargeProdScale, sigmaScale, epsilonScale);
    }

}

NonbondedForce::~NonbondedForce() = default;

void NonbondedForce::requireMode(PairMode mode, const char* operation) const {
    if (pairMode != mode)
        throw OpenMM::OpenMMException(std::string("NonbondedForce: ") + operation +
            " requires " + (mode == Standard ? "Standard" : "Custom") + " mode");
}

NonbondedForce::NonbondedForce(const OpenMM::CustomNonbondedForce& nonbonded,
                               float boxSafetyFactor)
        : pairMode(Custom), cutoffDistance(nonbonded.getCutoffDistance()), boxSafetyFactor(boxSafetyFactor) {
    if (boxSafetyFactor != AutoBoxSafety && boxSafetyFactor < 1.0f)
        throw OpenMM::OpenMMException("NonbondedForce: box safety factor must be >= 1.0, "
                                      "or AutoBoxSafety to have it chosen from the System");
    // Validate before copying: unsupported tabulated functions own pointers, so
    // a stock force's implicit copy is only safe after ruling those out.
    NonbondedForceImpl::validateCustomDefinition(nonbonded);
    customDefinition.reset(new OpenMM::CustomNonbondedForce(nonbonded));
    setForceGroup(nonbonded.getForceGroup());
    setName(nonbonded.getName());
}

NonbondedForce::NonbondedForce(const std::string& energy, float boxSafetyFactor)
        : pairMode(Custom), cutoffDistance(0.9),
          boxSafetyFactor(boxSafetyFactor) {
    if (boxSafetyFactor != AutoBoxSafety && boxSafetyFactor < 1.0f)
        throw OpenMM::OpenMMException("NonbondedForce: box safety factor must be >= 1.0, "
                                      "or AutoBoxSafety to have it chosen from the System");
    customDefinition.reset(new OpenMM::CustomNonbondedForce(energy));
    customDefinition->setNonbondedMethod(OpenMM::CustomNonbondedForce::CutoffPeriodic);
    customDefinition->setCutoffDistance(cutoffDistance);
}

const OpenMM::CustomNonbondedForce& NonbondedForce::getCustomDefinition() const {
    requireMode(Custom, "getCustomDefinition");
    return *customDefinition;
}

NonbondedForce* NonbondedForce::cast(OpenMM::Force& force) {
    return dynamic_cast<NonbondedForce*>(&force);
}

void NonbondedForce::setCutoffInteraction(const OpenMM::CustomNonbondedForce& source,
                                         bool preserveDispersionCorrection) {
    if (pairMode == Custom) {
        if (preserveDispersionCorrection)
            throw OpenMM::OpenMMException("CutoffInteraction: Custom mode has no native LJ tail to preserve");
        NonbondedForceImpl::validateCustomDefinition(source);
        if (source.getNumParticles() != getNumParticles() ||
                source.getCutoffDistance() != getCutoffDistance())
            throw OpenMM::OpenMMException("CutoffInteraction: definition must match the existing particle count and cutoff");
    } else {
        NonbondedForceImpl::validateCutoffReplacement(*this, source);
    }
    auto globals = globalParameters;
    if (pairMode == Custom)
        for (int i = 0; i < getNumGlobalParameters(); ++i)
            globals.push_back(GlobalParameterInfo{getGlobalParameterName(i), getGlobalParameterDefaultValue(i)});
    std::set<std::string> seen;
    for (int i = 0; i < source.getNumGlobalParameters(); ++i) {
        const auto& name = source.getGlobalParameterName(i);
        const double value = source.getGlobalParameterDefaultValue(i);
        if (!seen.insert(name).second || !std::isfinite(static_cast<float>(value)))
            throw OpenMM::OpenMMException("NonbondedForce: duplicate or nonfinite expression global " + name);
        auto found = std::find_if(globals.begin(), globals.end(), [&](const auto& g) { return g.name == name; });
        if (found != globals.end() && found->defaultValue != value)
            throw OpenMM::OpenMMException("NonbondedForce: conflicting default for global parameter " + name);
        if (found == globals.end()) globals.push_back({name, value});
    }
    auto definition = std::make_unique<OpenMM::CustomNonbondedForce>(source);
    if (pairMode == Custom) {
        // Preserve globals used only by additional tails. Validate and construct
        // the whole candidate before publishing any part of the replacement.
        for (const auto& global : globals)
            if (!seen.count(global.name))
                definition->addGlobalParameter(global.name, global.defaultValue);
        customDefinition = std::move(definition);
        return;
    }
    CutoffInteraction interaction;
    interaction.evaluator = CutoffInteraction::Expression;
    interaction.expression = source.getEnergyFunction();
    interaction.useSwitching = source.getUseSwitchingFunction();
    interaction.switchingDistance = source.getSwitchingDistance();
    interaction.tail = source.getUseLongRangeCorrection()
        ? CutoffInteraction::HomogeneousTail : CutoffInteraction::NoTail;
    auto reference = hasCutoffExpression() ? referenceLJInteraction : cutoffInteraction;
    if (!preserveDispersionCorrection) reference.tail = CutoffInteraction::NoTail;
    globalParameters.swap(globals);
    referenceLJInteraction = std::move(reference);
    cutoffExpressionDefinition = std::move(definition);
    cutoffInteraction = std::move(interaction);
}

void NonbondedForce::setCutoffInteraction(const std::string& expression,
        const std::vector<std::string>& parameterNames,
        const std::vector<std::vector<double>>& particleParameters,
        const std::map<std::string, double>& expressionGlobals,
        bool preserveDispersionCorrection, bool tailCorrection) {
    if (preserveDispersionCorrection && tailCorrection)
        throw OpenMM::OpenMMException("CutoffInteraction: tailCorrection and preserveDispersionCorrection cannot both be enabled");
    if (tailCorrection && getUseLJPME())
        throw OpenMM::OpenMMException("LJPME: switching and homogeneous dispersion tails are incompatible with the lattice sum");
    if (particleParameters.size() != static_cast<size_t>(getNumParticles()))
        throw OpenMM::OpenMMException("CutoffInteraction: particle parameter row count must match the existing particle count");
    OpenMM::CustomNonbondedForce definition(expression);
    definition.setNonbondedMethod(OpenMM::CustomNonbondedForce::CutoffPeriodic);
    definition.setCutoffDistance(getCutoffDistance());
    definition.setUseSwitchingFunction(getUseSwitchingFunction());
    definition.setSwitchingDistance(getSwitchingDistance());
    definition.setUseLongRangeCorrection(tailCorrection);
    std::set<std::string> names;
    for (const auto& name : parameterNames) {
        validateExpressionParameterName(name);
        if (!names.insert(name).second)
            throw OpenMM::OpenMMException("CutoffInteraction: duplicate per-particle parameter name " + name);
        definition.addPerParticleParameter(name);
    }
    for (const auto& row : particleParameters) {
        if (row.size() != parameterNames.size())
            throw OpenMM::OpenMMException("CutoffInteraction: particle parameter row width must match parameterNames");
        for (double value : row)
            if (!std::isfinite(static_cast<float>(value)))
                throw OpenMM::OpenMMException("CutoffInteraction: particle parameters must be finite in FP32");
        definition.addParticle(row);
    }
    for (const auto& global : expressionGlobals) {
        validateExpressionParameterName(global.first);
        if (!std::isfinite(static_cast<float>(global.second)))
            throw OpenMM::OpenMMException("CutoffInteraction: global defaults must be finite in FP32");
        definition.addGlobalParameter(global.first, global.second);
    }
    if (tailCorrection) {
        // Direct evaluation already binds retained owner globals. The tail
        // evaluator reads its own definition, so expose the same names there.
        for (int i = 0; i < getNumGlobalParameters(); ++i) {
            const auto& name = getGlobalParameterName(i);
            if (!expressionGlobals.count(name))
                definition.addGlobalParameter(name, getGlobalParameterDefaultValue(i));
        }
    }
    if (pairMode == Standard) {
        for (const auto& exception : exceptions)
            definition.addExclusion(exception.particle1, exception.particle2);
    } else {
        for (int i = 0; i < getNumExclusions(); ++i) {
            int first, second;
            getExclusionParticles(i, first, second);
            definition.addExclusion(first, second);
        }
    }
    // Delegate physical validation, global merging and the atomic replacement.
    setCutoffInteraction(definition, preserveDispersionCorrection);
}

const OpenMM::CustomNonbondedForce& NonbondedForce::getCutoffExpressionDefinition() const {
    if (pairMode == Custom) return *customDefinition;
    if (!cutoffExpressionDefinition)
        throw OpenMM::OpenMMException("NonbondedForce: no cutoff expression is configured");
    return *cutoffExpressionDefinition;
}

OpenMM::CustomNonbondedForce NonbondedForce::getCutoffExpression() const {
    return getCutoffExpressionDefinition();
}

void NonbondedForce::mergeGlobalParameters(const OpenMM::CustomNonbondedForce& definition) {
    std::set<std::string> seen;
    for (int i = 0; i < definition.getNumGlobalParameters(); ++i) {
        const auto& name = definition.getGlobalParameterName(i);
        if (!std::isfinite(static_cast<float>(definition.getGlobalParameterDefaultValue(i))))
            throw OpenMM::OpenMMException("NonbondedForce: expression global must be finite in FP32");
        if (!seen.insert(name).second)
            throw OpenMM::OpenMMException("NonbondedForce: duplicate global parameter " + name);
        for (int j = 0; j < getNumGlobalParameters(); ++j)
            if (getGlobalParameterName(j) == name &&
                    getGlobalParameterDefaultValue(j) != definition.getGlobalParameterDefaultValue(i))
                throw OpenMM::OpenMMException("NonbondedForce: conflicting default for global parameter " + name);
    }
    for (int i = 0; i < definition.getNumGlobalParameters(); ++i) {
        const auto& name = definition.getGlobalParameterName(i);
        bool found = false;
        for (int j = 0; j < getNumGlobalParameters(); ++j)
            found = found || getGlobalParameterName(j) == name;
        if (!found) addGlobalParameter(name, definition.getGlobalParameterDefaultValue(i));
    }
}

int NonbondedForce::addExpressionTailCorrection(const OpenMM::CustomNonbondedForce& definition) {
    NonbondedForceImpl::validateTailDefinition(*this, definition);
    auto copy = std::make_unique<OpenMM::CustomNonbondedForce>(definition);
    mergeGlobalParameters(definition);
    tailDefinitions.push_back(std::move(copy));
    return (int)tailDefinitions.size()-1;
}

const OpenMM::CustomNonbondedForce& NonbondedForce::getExpressionTailCorrectionDefinition(int index) const {
    ASSERT_VALID_INDEX(index, tailDefinitions);
    return *tailDefinitions[index];
}

OpenMM::CustomNonbondedForce NonbondedForce::getExpressionTailCorrection(int index) const {
    return getExpressionTailCorrectionDefinition(index);
}

void NonbondedForce::setExpressionTailCorrection(int index, const OpenMM::CustomNonbondedForce& definition) {
    ASSERT_VALID_INDEX(index, tailDefinitions);
    NonbondedForceImpl::validateTailDefinition(*this, definition);
    auto copy = std::make_unique<OpenMM::CustomNonbondedForce>(definition);
    mergeGlobalParameters(definition);
    tailDefinitions[index] = std::move(copy);
}

CutoffInteraction NonbondedForce::getDispersionCorrectionInteraction() const {
    requireMode(Standard, "getDispersionCorrectionInteraction");
    return hasCutoffExpression() ? referenceLJInteraction : cutoffInteraction;
}

int NonbondedForce::getNumParticles() const {
    return pairMode == Custom ? customDefinition->getNumParticles() : (int)particles.size();
}

std::vector<double> NonbondedForce::getParticleParameters(int index) const {
    std::vector<double> parameters;
    if (pairMode == Custom)
        customDefinition->getParticleParameters(index, parameters);
    else {
        double charge, sigma, epsilon;
        getParticleParameters(index, charge, sigma, epsilon);
        parameters = {charge, sigma, epsilon};
    }
    return parameters;
}

int NonbondedForce::addParticle(const std::vector<double>& parameters) {
    requireMode(Custom, "addParticle(parameters)");
    return customDefinition->addParticle(parameters);
}

void NonbondedForce::getParticleParameters(int index, std::vector<double>& parameters) const {
    requireMode(Custom, "getParticleParameters(parameters)");
    customDefinition->getParticleParameters(index, parameters);
}

void NonbondedForce::setParticleParameters(int index, const std::vector<double>& parameters) {
    requireMode(Custom, "setParticleParameters(parameters)");
    customDefinition->setParticleParameters(index, parameters);
}

const std::string& NonbondedForce::getEnergyFunction() const {
    requireMode(Custom, "getEnergyFunction");
    return customDefinition->getEnergyFunction();
}

void NonbondedForce::setEnergyFunction(const std::string& energy) {
    requireMode(Custom, "setEnergyFunction");
    customDefinition->setEnergyFunction(energy);
}

int NonbondedForce::getNumPerParticleParameters() const {
    return pairMode == Custom ? customDefinition->getNumPerParticleParameters() : 0;
}

int NonbondedForce::addPerParticleParameter(const std::string& name) {
    requireMode(Custom, "addPerParticleParameter");
    return customDefinition->addPerParticleParameter(name);
}

const std::string& NonbondedForce::getPerParticleParameterName(int index) const {
    requireMode(Custom, "getPerParticleParameterName");
    return customDefinition->getPerParticleParameterName(index);
}

void NonbondedForce::setPerParticleParameterName(int index, const std::string& name) {
    requireMode(Custom, "setPerParticleParameterName");
    customDefinition->setPerParticleParameterName(index, name);
}

int NonbondedForce::getNumGlobalParameters() const {
    return pairMode == Custom ? customDefinition->getNumGlobalParameters() : (int)globalParameters.size();
}

int NonbondedForce::addGlobalParameter(const std::string& name, double defaultValue) {
    for (int i = 0; i < getNumGlobalParameters(); ++i)
        if (getGlobalParameterName(i) == name)
            throw OpenMM::OpenMMException("NonbondedForce: duplicate global parameter " + name);
    if (pairMode == Custom)
        return customDefinition->addGlobalParameter(name, defaultValue);
    globalParameters.push_back(GlobalParameterInfo{name, defaultValue});
    return (int)globalParameters.size() - 1;
}

const std::string& NonbondedForce::getGlobalParameterName(int index) const {
    if (pairMode == Custom)
        return customDefinition->getGlobalParameterName(index);
    ASSERT_VALID_INDEX(index, globalParameters);
    return globalParameters[index].name;
}

void NonbondedForce::setGlobalParameterName(int index, const std::string& name) {
    const std::string previous = getGlobalParameterName(index);
    for (int i = 0; i < getNumGlobalParameters(); ++i)
        if (i != index && getGlobalParameterName(i) == name)
            throw OpenMM::OpenMMException("NonbondedForce: duplicate global parameter " + name);
    if (pairMode == Custom)
        customDefinition->setGlobalParameterName(index, name);
    else {
        ASSERT_VALID_INDEX(index, globalParameters);
        globalParameters[index].name = name;
    }
    const auto rename = [&](OpenMM::CustomNonbondedForce& definition) {
        for (int i = 0; i < definition.getNumGlobalParameters(); ++i)
            if (definition.getGlobalParameterName(i) == previous)
                definition.setGlobalParameterName(i, name);
    };
    if (cutoffExpressionDefinition) rename(*cutoffExpressionDefinition);
    for (auto& tail : tailDefinitions) rename(*tail);
}

double NonbondedForce::getGlobalParameterDefaultValue(int index) const {
    if (pairMode == Custom)
        return customDefinition->getGlobalParameterDefaultValue(index);
    ASSERT_VALID_INDEX(index, globalParameters);
    return globalParameters[index].defaultValue;
}

void NonbondedForce::setGlobalParameterDefaultValue(int index, double defaultValue) {
    const std::string name = getGlobalParameterName(index);
    if (pairMode == Custom)
        customDefinition->setGlobalParameterDefaultValue(index, defaultValue);
    else {
        ASSERT_VALID_INDEX(index, globalParameters);
        globalParameters[index].defaultValue = defaultValue;
    }
    const auto update = [&](OpenMM::CustomNonbondedForce& definition) {
        for (int i = 0; i < definition.getNumGlobalParameters(); ++i)
            if (definition.getGlobalParameterName(i) == name)
                definition.setGlobalParameterDefaultValue(i, defaultValue);
    };
    if (cutoffExpressionDefinition) update(*cutoffExpressionDefinition);
    for (auto& tail : tailDefinitions) update(*tail);
}

int NonbondedForce::getGlobalParameterIndex(const std::string& parameter) const {
    for (int i = 0; i < (int)globalParameters.size(); i++)
        if (globalParameters[i].name == parameter)
            return i;
    throw OpenMM::OpenMMException("NonbondedForce: there is no global parameter called '" +
                                  parameter + "'");
}

int NonbondedForce::addParticleParameterOffset(const std::string& parameter, int particleIndex,
                                               double chargeScale, double sigmaScale,
                                               double epsilonScale) {
    requireMode(Standard, "addParticleParameterOffset");
    particleOffsets.push_back(ParticleOffsetInfo{getGlobalParameterIndex(parameter), particleIndex,
                                                 chargeScale, sigmaScale, epsilonScale});
    return (int)particleOffsets.size() - 1;
}

void NonbondedForce::getParticleParameterOffset(int index, std::string& parameter,
                                                int& particleIndex, double& chargeScale,
                                                double& sigmaScale, double& epsilonScale) const {
    requireMode(Standard, "getParticleParameterOffset");
    ASSERT_VALID_INDEX(index, particleOffsets);
    const ParticleOffsetInfo& o = particleOffsets[index];
    parameter = globalParameters[o.parameter].name;
    particleIndex = o.particle;
    chargeScale = o.chargeScale;
    sigmaScale = o.sigmaScale;
    epsilonScale = o.epsilonScale;
}

void NonbondedForce::setParticleParameterOffset(int index, const std::string& parameter,
                                                int particleIndex, double chargeScale,
                                                double sigmaScale, double epsilonScale) {
    requireMode(Standard, "setParticleParameterOffset");
    ASSERT_VALID_INDEX(index, particleOffsets);
    particleOffsets[index] = ParticleOffsetInfo{getGlobalParameterIndex(parameter), particleIndex,
                                                chargeScale, sigmaScale, epsilonScale};
}

int NonbondedForce::addExceptionParameterOffset(const std::string& parameter, int exceptionIndex,
                                                double chargeProdScale, double sigmaScale,
                                                double epsilonScale) {
    requireMode(Standard, "addExceptionParameterOffset");
    exceptionOffsets.push_back(ExceptionOffsetInfo{getGlobalParameterIndex(parameter), exceptionIndex,
                                                   chargeProdScale, sigmaScale, epsilonScale});
    return (int)exceptionOffsets.size() - 1;
}

void NonbondedForce::getExceptionParameterOffset(int index, std::string& parameter,
                                                 int& exceptionIndex, double& chargeProdScale,
                                                 double& sigmaScale, double& epsilonScale) const {
    requireMode(Standard, "getExceptionParameterOffset");
    ASSERT_VALID_INDEX(index, exceptionOffsets);
    const ExceptionOffsetInfo& o = exceptionOffsets[index];
    parameter = globalParameters[o.parameter].name;
    exceptionIndex = o.exception;
    chargeProdScale = o.chargeProdScale;
    sigmaScale = o.sigmaScale;
    epsilonScale = o.epsilonScale;
}

void NonbondedForce::setExceptionParameterOffset(int index, const std::string& parameter,
                                                 int exceptionIndex, double chargeProdScale,
                                                 double sigmaScale, double epsilonScale) {
    requireMode(Standard, "setExceptionParameterOffset");
    ASSERT_VALID_INDEX(index, exceptionOffsets);
    exceptionOffsets[index] = ExceptionOffsetInfo{getGlobalParameterIndex(parameter), exceptionIndex,
                                                  chargeProdScale, sigmaScale, epsilonScale};
}

int NonbondedForce::getNumExclusions() const {
    return pairMode == Custom ? customDefinition->getNumExclusions() : 0;
}

int NonbondedForce::addExclusion(int particle1, int particle2) {
    requireMode(Custom, "addExclusion");
    return customDefinition->addExclusion(particle1, particle2);
}

void NonbondedForce::getExclusionParticles(int index, int& particle1, int& particle2) const {
    requireMode(Custom, "getExclusionParticles");
    customDefinition->getExclusionParticles(index, particle1, particle2);
}

void NonbondedForce::setExclusionParticles(int index, int particle1, int particle2) {
    requireMode(Custom, "setExclusionParticles");
    customDefinition->setExclusionParticles(index, particle1, particle2);
}

void NonbondedForce::createExclusionsFromBonds(
        const std::vector<std::pair<int, int>>& bonds, int bondCutoff) {
    requireMode(Custom, "createExclusionsFromBonds");
    customDefinition->createExclusionsFromBonds(bonds, bondCutoff);
}

bool NonbondedForce::getUseLongRangeCorrection() const {
    return pairMode == Custom && customDefinition->getUseLongRangeCorrection();
}

void NonbondedForce::setUseLongRangeCorrection(bool use) {
    requireMode(Custom, "setUseLongRangeCorrection");
    customDefinition->setUseLongRangeCorrection(use);
}

CutoffInteraction NonbondedForce::getCutoffInteraction() const {
    if (pairMode == Standard) return cutoffInteraction;
    CutoffInteraction interaction;
    interaction.evaluator = CutoffInteraction::Expression;
    interaction.expression = customDefinition->getEnergyFunction();
    interaction.useSwitching = customDefinition->getUseSwitchingFunction();
    interaction.switchingDistance = customDefinition->getSwitchingDistance();
    interaction.tail = customDefinition->getUseLongRangeCorrection()
        ? CutoffInteraction::HomogeneousTail : CutoffInteraction::NoTail;
    return interaction;
}

void NonbondedForce::setCutoffInteraction(const CutoffInteraction& interaction) {
    const auto reject = [](const char* reason) {
        throw OpenMM::OpenMMException(std::string("CutoffInteraction: ")+reason);
    };
    if (interaction.tail != CutoffInteraction::NoTail &&
            interaction.tail != CutoffInteraction::HomogeneousTail)
        reject("unsupported tail policy (expected NoTail or HomogeneousTail)");
    if (interaction.evaluator == CutoffInteraction::Expression) {
        if (interaction.combiningRule != CutoffInteraction::LorentzBerthelot)
            reject("include combining and shifting in the custom expression");
        if (!hasCutoffExpression())
            reject("pass a complete CustomNonbondedForce definition to setCutoffInteraction to configure particle records");
        auto next = getCutoffExpressionDefinition();
        next.setEnergyFunction(interaction.expression);
        next.setUseSwitchingFunction(interaction.useSwitching);
        next.setSwitchingDistance(interaction.switchingDistance);
        next.setUseLongRangeCorrection(interaction.tail == CutoffInteraction::HomogeneousTail);
        setCutoffInteraction(next, pairMode == Standard && getUseDispersionCorrection());
        return;
    }
    if (pairMode == Custom) reject("Custom mode requires an Expression");
    if (interaction.evaluator != CutoffInteraction::BuiltinLJ || !interaction.expression.empty())
        reject("unsupported evaluator");
    if (interaction.combiningRule != CutoffInteraction::LorentzBerthelot &&
            interaction.combiningRule != CutoffInteraction::Geometric)
        reject("unsupported LJ combining rule");
    cutoffInteraction = interaction;
    cutoffExpressionDefinition.reset();
}

int NonbondedForce::addLatticeSumInteraction(const LatticeSumInteraction& interaction) {
    validateLatticeInteraction(interaction, latticeSums);
    latticeSums.push_back(interaction);
    applyLatticePolicy(interaction);
    return (int)latticeSums.size()-1;
}

LatticeSumInteraction NonbondedForce::getLatticeSumInteraction(int index) const {
    ASSERT_VALID_INDEX(index, latticeSums);
    return latticeSums[index];
}

void NonbondedForce::setLatticeSumInteraction(int index, const LatticeSumInteraction& interaction) {
    ASSERT_VALID_INDEX(index, latticeSums);
    validateLatticeInteraction(interaction, latticeSums, index);
    latticeSums[index] = interaction;
    applyLatticePolicy(interaction);
}

void NonbondedForce::removeLatticeSumInteraction(int index) {
    ASSERT_VALID_INDEX(index, latticeSums);
    latticeSums.erase(latticeSums.begin()+index);
}

LatticeSumInteraction NonbondedForce::getLatticeSumInteraction() const {
    for (const auto& interaction : latticeSums)
        if (interaction.kernel.power == 1) return interaction;
    throw OpenMM::OpenMMException("NonbondedForce: no LatticeSumInteraction for Coulomb");
}

void NonbondedForce::setLatticeSumInteraction(const LatticeSumInteraction& interaction) {
    if (interaction.kernel.power != 1)
        throw OpenMM::OpenMMException("setLatticeSumInteraction: unindexed convenience requires a Coulomb/P1 kernel");
    for (int index = 0; index < (int)latticeSums.size(); ++index) {
        if (latticeSums[index].kernel.power == 1) {
            setLatticeSumInteraction(index, interaction);
            return;
        }
    }
    addLatticeSumInteraction(interaction);
}

void NonbondedForce::applyLatticePolicy(const LatticeSumInteraction& interaction) {
    if (interaction.kernel.power == 6) {
        setUseSwitchingFunction(false);
        if (pairMode == Standard) setUseDispersionCorrection(false);
        else setUseLongRangeCorrection(false);
        if (cutoffExpressionDefinition) {
            cutoffExpressionDefinition->setUseLongRangeCorrection(false);
            cutoffInteraction.tail=CutoffInteraction::NoTail;
        }
    }
}

void NonbondedForce::setUseLJPME(bool use) {
    for (int index = 0; index < (int)latticeSums.size(); ++index) {
        if (latticeSums[index].kernel.power == 6) {
            if (!use) removeLatticeSumInteraction(index);
            else applyLatticePolicy(latticeSums[index]);
            return;
        }
    }
    if (!use) return;
    LatticeSumInteraction interaction(DispersionRadialKernel(), "2*sqrt(epsilon)*sigma^3");
    for (const auto& existing : latticeSums)
        if (existing.kernel.power == 1) interaction.splitting = existing.splitting;
    addLatticeSumInteraction(interaction);
}

bool NonbondedForce::getUseLJPME() const {
    return std::any_of(latticeSums.begin(), latticeSums.end(),
        [](const LatticeSumInteraction& interaction) { return interaction.kernel.power == 6; });
}

NonbondedForce::NonbondedMethod NonbondedForce::getNonbondedMethod() const {
    if (latticeSums.empty()) return CutoffPeriodic;
    const auto splitting = latticeSums.front().splitting;
    for (const auto& interaction : latticeSums)
        if (interaction.splitting != splitting)
            throw OpenMM::OpenMMException("NonbondedForce: lattice terms use different splittings; query each indexed interaction");
    return splitting == LatticeSumInteraction::Ewald ? PME : Prolate;
}

void NonbondedForce::setNonbondedMethod(NonbondedMethod method) {
    requireMode(Custom, "setNonbondedMethod (use setLatticeSumInteraction for Standard)");
    if (method != CutoffPeriodic)
        throw OpenMM::OpenMMException("NonbondedForce: setNonbondedMethod accepts only CutoffPeriodic; configure lattice terms explicitly");
    customDefinition->setNonbondedMethod(OpenMM::CustomNonbondedForce::CutoffPeriodic);
    latticeSums.clear();
}

double NonbondedForce::getCutoffDistance() const {
    return pairMode == Custom ? customDefinition->getCutoffDistance() : cutoffDistance;
}

void NonbondedForce::setCutoffDistance(double distance) {
    requireMode(Custom, "setCutoffDistance (Standard cutoff is fixed at construction)");
    customDefinition->setCutoffDistance(distance);
}

NonbondedForce::LJCombiningRule NonbondedForce::getLJCombiningRule() const {
    requireMode(Standard, "getLJCombiningRule");
    return cutoffInteraction.combiningRule;
}

bool NonbondedForce::getUseSwitchingFunction() const {
    return pairMode == Custom ? customDefinition->getUseSwitchingFunction() : cutoffInteraction.useSwitching;
}

void NonbondedForce::setUseSwitchingFunction(bool use) {
    if (pairMode == Custom)
        customDefinition->setUseSwitchingFunction(use);
    else {
        cutoffInteraction.useSwitching = use;
        if (cutoffExpressionDefinition)
            cutoffExpressionDefinition->setUseSwitchingFunction(use);
    }
}

double NonbondedForce::getSwitchingDistance() const {
    return pairMode == Custom ? customDefinition->getSwitchingDistance() : cutoffInteraction.switchingDistance;
}

void NonbondedForce::setSwitchingDistance(double distance) {
    if (pairMode == Custom)
        customDefinition->setSwitchingDistance(distance);
    else {
        cutoffInteraction.switchingDistance = distance;
        if (cutoffExpressionDefinition)
            cutoffExpressionDefinition->setSwitchingDistance(distance);
    }
}

bool NonbondedForce::getUseDispersionCorrection() const {
    requireMode(Standard, "getUseDispersionCorrection");
    return getDispersionCorrectionInteraction().tail == CutoffInteraction::HomogeneousTail;
}

void NonbondedForce::setUseDispersionCorrection(bool use) {
    requireMode(Standard, "setUseDispersionCorrection");
    auto& reference = hasCutoffExpression() ? referenceLJInteraction : cutoffInteraction;
    reference.tail = use ? CutoffInteraction::HomogeneousTail : CutoffInteraction::NoTail;
}

bool NonbondedForce::getDispersionCorrectionFollowsParameters() const {
    requireMode(Standard, "getDispersionCorrectionFollowsParameters");
    return dispersionCorrectionFollowsParameters;
}

void NonbondedForce::setDispersionCorrectionFollowsParameters(bool follow) {
    requireMode(Standard, "setDispersionCorrectionFollowsParameters");
    dispersionCorrectionFollowsParameters = follow;
}

bool NonbondedForce::getExceptionsUsePeriodicBoundaryConditions() const {
    return exceptionsUsePeriodicBoundaryConditions;
}

void NonbondedForce::setExceptionsUsePeriodicBoundaryConditions(bool use) {
    requireMode(Standard, "setExceptionsUsePeriodicBoundaryConditions");
    exceptionsUsePeriodicBoundaryConditions = use;
}

} // namespace NvOpenMM
