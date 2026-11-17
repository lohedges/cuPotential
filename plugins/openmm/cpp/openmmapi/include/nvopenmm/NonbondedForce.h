// The particle, exception, global-parameter and offset declarations and data members
// follow OpenMM openmmapi/include/openmm/NonbondedForce.h:
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

#ifndef NVOPENMM_NONBONDEDFORCE_H_
#define NVOPENMM_NONBONDEDFORCE_H_

#include "nvopenmm/Interactions.h"
#include "openmm/Context.h"
#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
namespace OpenMM {
class System;
class CustomNonbondedForce;
class NonbondedForce;
}

namespace NvOpenMM {

/**
 * Periodic standard or custom pair interactions using one cupo direct traversal.
 *
 * Standard defaults to PME or Prolate electrostatics with a reciprocal mesh.
 * Custom mode evaluates a complete user expression with a periodic cutoff,
 * per-particle parameters, global parameters and exclusions. Pair selection can
 * be expressed with per-particle flags; interaction groups are not supported.
 * An optional expression tail correction follows global parameters and volume.
 * Either mode may configure independent lattice terms. Custom supplies the full
 * unsplit potential and the lattice weights; each declared reciprocal term is
 * subtracted from direct evaluation and evaluated on its matching mesh.
 * Standard may replace its ordinary LJ by a custom cutoff expression while
 * retaining its Coulomb and optional dispersion lattice sums, live parameters,
 * and independent exceptions.
 *
 * Usually built from a stock NonbondedForce by the converting constructor below.
 * To assemble one by hand: addParticle() once per system atom (system-index
 * order), then addException() for every 1-2/1-3/1-4 exception, all before
 * creating a Context.
 *
 * WHAT IS NOT HERE: the splitting's own parameters -- the Ewald alpha and the
 * reciprocal mesh. They are cupo's, derived inside it from the cutoff and the box
 * (cupo/splitting_plan.h), and this class neither stores nor forwards them. They
 * used to sit here as a pair of settable fields, which put this class in the
 * position of describing a splitting it does not implement: the mesh alone is
 * sized by two different formulas whose answers differ by ~5x in grid points
 * depending on which method is running, so every value carried through here was
 * one that could be carried through WRONG.
 */
class NonbondedForce : public OpenMM::Force {
public:
    enum PairMode { Standard = 0, Custom = 1 };

    /**
     * Compatibility selectors for OpenMM-shaped constructors and accessors.
     * Physical configuration uses CutoffInteraction and LatticeSumInteraction;
     * CutoffPeriodic is not a splitting method.
     */
    enum NonbondedMethod {
        /**
         * Smooth particle-mesh Ewald: erfc(alpha*r)/r direct, Gaussian reciprocal,
         * 5th-order B-spline spreading. Numerically what OpenMM's own PME does --
         * same alpha, same mesh, same rounding -- which makes it the arm to ask for
         * when the point is to be COMPARED with the stock platform rather than to
         * be fast. Not the default; see Prolate.
         */
        PME = 0,
        /**
         * C2 Prolate P=1 splitting, the default, with the C2-order-3
         * spreading window. The short-range potential and its first two radial
         * derivatives vanish at the cutoff. Its matching Fourier complement has
         * nonzero tails and is evaluated on every mesh mode. The design bandwidth
         * c=10.75 and modal coefficients are compiled in; no alpha is used.
         *
         * PME and Prolate represent the same Coulomb interaction with different
         * finite-mesh errors, so they need not agree to floating-point roundoff.
         * The plugin's mesh policy targets ForceErrorTolerance. This choice
         * does not itself enable LJPME; setUseLJPME selects dispersion separately.
         */
        Prolate = 1,
        /** Periodic cutoff for Custom expressions, without a reciprocal sum. */
        CutoffPeriodic = 2,
    };

    /** Compatibility aliases for the cutoff LJ combining policy. */
    using LJCombiningRule = CutoffInteraction::CombiningRule;
    static constexpr LJCombiningRule LorentzBerthelot = CutoffInteraction::LorentzBerthelot;
    static constexpr LJCombiningRule Geometric = CutoffInteraction::Geometric;

    /**
     * The force accuracy target for the PME and Prolate methods.
     *
     * Not "Ewald error tolerance", though 5e-4 is exactly what OpenMM means by that
     * and the converting constructor checks a source force's setting against it.
     * There are two splittings now, and only one of them has an Ewald anything --
     * under Prolate it is the target used by the mesh-sizing policy for the
     * compiled C2 Prolate model.
     *
     * Fixed rather than settable: the plugin currently exposes one validated
     * mesh policy for each method. This is an API policy, not a nonzero accuracy
     * floor of the full C2 Prolate Fourier complement. 5e-4 is also OpenMM's default.
     *
     * Stated here, but not USED here: cupo derives alpha and the mesh at this same
     * target (EwaldPlan::kErrorTolerance). This copy is what the converting
     * constructor below checks a source force against, so that a force asking for
     * a tolerance nobody implements is refused rather than silently reinterpreted.
     */
    static constexpr double ForceErrorTolerance = 5e-4;

    /**
     * Pass this as the box safety factor to have it chosen at Context creation:
     * In Standard mode this is 1.0 without an active barostat and
     * DefaultNPTBoxSafety with one. Custom mode uses DefaultNPTBoxSafety.
     * This default depends on information available when the System is complete.
     *
     * Any explicit factor wins. Choosing one is worth doing for a long or unusual
     * run: the envelope is allocated once and never resized, so the factor has to
     * cover the run's PEAK volume, not its average.
     */
    static constexpr float AutoBoxSafety = 0.0f;

    /**
     * What AutoBoxSafety resolves to under a barostat. 1.15 linear, i.e. 1.52x by
     * volume, which is the value every bundled NPT benchmark and tolerance in this
     * tree is validated at.
     *
     * Headroom is not free and not smooth: the envelope grows with the CELL COUNT,
     * floor(box * factor / cutoff), so raising the factor buys nothing until it
     * crosses an integer and then costs a whole shell at once. On dhfr (box 6.223,
     * cutoff 0.9) this factor is the difference between 6 and 7 cells per axis --
     * 1.6x the cell-matrix allocation.
     */
    static constexpr float DefaultNPTBoxSafety = 1.15f;

    /**
     * Assemble one by hand: addParticle() once per system atom in system-index
     * order, then addException() for every 1-2/1-3/1-4 exception, all before
     * creating a Context.
     *
     * The cutoff and NPT headroom are fixed here. The splitting can be replaced
     * with setLatticeSumInteraction(), requiring Context::reinitialize(). The
     * platform builds its cell list, splitting plan, mesh and captured graphs
     * from these settings at Context creation.
     *
     * @param method           electrostatics model; defaults to Prolate
     * @param cutoff           real-space cutoff (nm)
     * @param boxSafetyFactor  NPT cell-grid headroom (>= 1.0; 1.0 == NVT), or
     *                          AutoBoxSafety to decide it from the System
     * @throws OpenMMException if the safety factor is below 1.0 and is not
     *                         AutoBoxSafety
     */
    explicit NonbondedForce(NonbondedMethod method = Prolate, double cutoff = 0.9,
                            float boxSafetyFactor = AutoBoxSafety);

    /**
     * Copy a stock force's particles, exceptions, cutoff, force group, LJ switching
     * and tail settings. PME and LJPME sources select Prolate by default. The
     * source is unchanged. LJPME enables the dispersion lattice sum and ignores
     * the source's switching and homogeneous tail flags, as stock OpenMM does.
     * The LJ combining rule defaults to LorentzBerthelot, matching the source.
     *
     * Sources must use ForceErrorTolerance and unpinned meshes; a pinned
     * alpha must match the cutoff-derived value.
     *
     * Global parameters, particle/exception offsets, and the source's exception
     * periodic-boundary flag are copied. The flag controls physical exception
     * replacements only; reciprocal exclusion corrections always use periodic
     * minimum-image distances (see setExceptionsUsePeriodicBoundaryConditions()).
     * Other source methods are rejected; use CustomNonbondedForce with an explicit
     * reaction-field expression for CutoffPeriodic interactions.
     */
    NonbondedForce(const OpenMM::NonbondedForce& nonbonded,
                   float boxSafetyFactor = AutoBoxSafety);

    /** Explicit splitting: PME and LJPME sources accept PME or Prolate.
     * Other source methods are unsupported. */
    NonbondedForce(const OpenMM::NonbondedForce& nonbonded,
                   float boxSafetyFactor, NonbondedMethod method);

    /** Copy a periodic-cutoff CustomNonbondedForce definition, including switching and tail settings. */
    explicit NonbondedForce(const OpenMM::CustomNonbondedForce& nonbonded,
                            float boxSafetyFactor = AutoBoxSafety);

    /** Create a Custom expression with periodic cutoff 0.9 nm. */
    explicit NonbondedForce(const std::string& energy,
                            float boxSafetyFactor = AutoBoxSafety);
    ~NonbondedForce() override;

    /** Configure the complete cutoff expression, particle records and globals.
     * In Standard mode this replaces every ordinary LJ pair.
     * Coulomb direct/reciprocal terms and exceptions retain their Standard laws.
     * The definition must share the particle count, cutoff and exception pairs
     * (as exclusions). An explicit switch multiplies only this cutoff expression;
     * its unswitched definition is retained for tail integration. By default the native
     * LJ tail is removed too. preserveDispersionCorrection retains the original
     * native LJ tail, including its original combining and switching policies;
     * this is useful when the expression contains that native LJ contribution.
     * The expression's particle values are independent of the Standard sigma/
     * epsilon records; its cutoff parameter laws belong in the expression.
     * In Custom mode this replaces the full expression while retaining declared
     * lattice terms; preserving a native LJ tail is unsupported. Both modes require the existing particle count and
     * cutoff. Globals merge by name with matching defaults, retaining names used
     * by additional tail definitions.
     * This is one direct traversal. Changing code/layout requires reinitialization;
     * edit particle values in a copy, pass it back to this setter, then call
     * updateParametersInContext(). Preserve the native-tail flag when doing so.
     */
    void setCutoffInteraction(const OpenMM::CustomNonbondedForce& definition,
                              bool preserveDispersionCorrection = false);
    /** Configure a complete cutoff expression directly from its parameter table.
     * This has the same replacement semantics as the complete-definition overload:
     * Standard replaces all ordinary LJ pairs, retaining Coulomb and exceptions;
     * Custom replaces its full pair expression, retaining declared lattice terms.
     * The current cutoff, switching settings and exclusions are inherited. In
     * Standard mode every exception pair becomes an expression exclusion.
     *
     * particleParameters has one row per existing particle in system-index order,
     * with columns in parameterNames order. Names must be unique identifiers, and
     * parameter values and global defaults must be finite in FP32. Values use the
     * units expected by the expression: r is in nm and energy is in kJ/mol; no unit
     * conversion is performed. Globals merge with existing globals and must have
     * matching defaults for names already present.
     *
     * tailCorrection defaults to false. When true, one homogeneous correction is
     * computed for the complete expression using its particle records, inherited
     * cutoff and switch, and live global parameters. It uses the ungrouped
     * CustomNonbondedForce averaging convention, including the switching deficit.
     * The original Standard LJ tail is disabled by default. To retain that tail
     * instead, set preserveDispersionCorrection, as in the other overload; it is
     * unsupported in Custom mode and cannot be combined with tailCorrection.
     * tailCorrection is incompatible with a dispersion lattice sum (LJPME).
     * Existing independent expression tail corrections are retained and add to
     * the total; callers must avoid including the same contribution twice.
     * All validation performed by this call is atomic; expression compilation
     * still occurs when a Context is initialized. Code/layout or tail-policy
     * changes require reinitialization; particle-value changes can be applied
     * with updateParametersInContext().
     */
    void setCutoffInteraction(const std::string& expression,
                              const std::vector<std::string>& parameterNames,
                              const std::vector<std::vector<double>>& particleParameters,
                              const std::map<std::string, double>& globalParameters = {},
                              bool preserveDispersionCorrection = false,
                              bool tailCorrection = false);
    /** Whether either mode evaluates a caller-defined cutoff expression. */
    bool hasCutoffExpression() const { return pairMode == Custom || cutoffExpressionDefinition != nullptr; }
    /** Return an owned copy of either mode's complete expression definition.
     * Pass an edited copy to setCutoffInteraction(), then updateParametersInContext().
     * In Standard mode, preserve the native-tail flag when replacing it. */
    OpenMM::CustomNonbondedForce getCutoffExpression() const;
    const OpenMM::CustomNonbondedForce& getCutoffExpressionDefinition() const;

    /** Add an independent CPU expression tail correction, C(globals)/volume.
     * The source must enable long-range correction and match the particle count
     * and cutoff. Its interaction groups and automatic switch are used only to
     * reproduce its tail averaging/integral; they add no pair traversal. This
     * does not enable interaction groups or change the evaluator's switching settings.
     * Coefficients are cached by referenced live globals and divided by the current
     * volume. Force-only evaluations validate globals but defer integration until
     * energy is requested; an uncached global vector can still require CPU work.
     * Code/layout/group changes require reinitialization. Particle parameters may
     * be replaced with setExpressionTailCorrection and updateParametersInContext,
     * which prepares the new correction and discards its old coefficient cache.
     */
    int addExpressionTailCorrection(const OpenMM::CustomNonbondedForce& definition);
    int getNumExpressionTailCorrections() const { return (int)tailDefinitions.size(); }
    OpenMM::CustomNonbondedForce getExpressionTailCorrection(int index) const;
    void setExpressionTailCorrection(int index, const OpenMM::CustomNonbondedForce& definition);
    const OpenMM::CustomNonbondedForce& getExpressionTailCorrectionDefinition(int index) const;
    // Internal physical definition used by the native analytical-tail evaluator.
    CutoffInteraction getDispersionCorrectionInteraction() const;

    PairMode getPairMode() const { return pairMode; }

    /** Return an independent physical definition of the cutoff term. */
    CutoffInteraction getCutoffInteraction() const;
    /** Replace the cutoff definition within the existing force mode.
     * Standard accepts BuiltinLJ or an already configured cutoff Expression;
     * Custom accepts Expression with optional quintic switching. Expressions may
     * request numerical tail integration, including the switching deficit.
     * Code, combining and active switching changes require
     * Context::reinitialize(). Native LJ tail-policy changes can be applied with
     * updateParametersInContext(), like setUseDispersionCorrection(). Changing
     * an expression's tail policy requires Context::reinitialize().
     */
    void setCutoffInteraction(const CutoffInteraction& interaction);
    /** Both force modes may contain zero, one or two independently configured
     * lattice terms: at most one P1 and one P6. Getters return independent copies.
     * Construct a term with an explicit kernel and weight expression. Standard
     * starts with LatticeSumInteraction(CoulombRadialKernel(), "q"); Custom starts
     * without lattice terms.
     * Structural changes require Context::reinitialize(). Weight expressions
     * refer to original per-particle parameters and live globals, without r.
     * Adding or replacing P6 disables automatic switching and homogeneous tails.
     */
    int getNumLatticeSumInteractions() const { return (int)latticeSums.size(); }
    int addLatticeSumInteraction(const LatticeSumInteraction& interaction);
    LatticeSumInteraction getLatticeSumInteraction(int index) const;
    void setLatticeSumInteraction(int index, const LatticeSumInteraction& interaction);
    void removeLatticeSumInteraction(int index);
    bool hasLatticeSumInteraction() const { return !latticeSums.empty(); }
    /** Compatibility getter for the Coulomb/P1 entry; throws if absent. */
    LatticeSumInteraction getLatticeSumInteraction() const;
    /** Compatibility setter: add or replace the Coulomb/P1 entry. */
    void setLatticeSumInteraction(const LatticeSumInteraction& interaction);
    /** Enable the geometric reciprocal dispersion sum (LJPME). Native direct
     * LJ defaults to Lorentz-Berthelot; setLJCombiningRule(Geometric) selects
     * geometric direct mixing. In Standard with a cutoff expression, effective native
     * sigma/epsilon define the dispersion weights, while the expression supplies
     * the full unsplit non-Coulomb pair potential. The plugin shifts that expression
     * at the cutoff and adds the matching dispersion splitting correction.
     * The cutoff shift follows the selected splitting: Prolate omits Ewald's
     * small per-ordinary-pair constant b6_i*b6_j*SR6_Ewald(cutoff).
     * Enabling LJPME disables switching and homogeneous LJ tails; changing it
     * requires Context reinitialization. This convenience configures P6 with
     * coupling -1 and weight 2*sqrt(epsilon)*sigma^3; Custom schemas must declare
     * sigma/epsilon or configure their own weight through the collection API. */
    void setUseLJPME(bool use);
    bool getUseLJPME() const;
    /** Read-only definition for Custom mode; Standard mode throws. */
    const OpenMM::CustomNonbondedForce& getCustomDefinition() const;
    /** Borrow the plugin type from an OpenMM Force; nullptr for other force types. */
    static NonbondedForce* cast(OpenMM::Force& force);

    int getNumParticles() const;

    /** Read either mode's particle record (Standard order: charge, sigma, epsilon). */
    std::vector<double> getParticleParameters(int index) const;
    /** Custom particle records follow the declared per-particle parameter order. */
    int addParticle(const std::vector<double>& parameters);
    void getParticleParameters(int index, std::vector<double>& parameters) const;
    void setParticleParameters(int index, const std::vector<double>& parameters);

    const std::string& getEnergyFunction() const;
    void setEnergyFunction(const std::string& energy);
    /** The base expression's per-particle count is zero in Standard mode. */
    int getNumPerParticleParameters() const;
    int addPerParticleParameter(const std::string& name);
    const std::string& getPerParticleParameterName(int index) const;
    void setPerParticleParameterName(int index, const std::string& name);
    /** Live Context globals used by Standard offsets or the Custom expression. */
    int getNumGlobalParameters() const;
    int addGlobalParameter(const std::string& name, double defaultValue);
    const std::string& getGlobalParameterName(int index) const;
    void setGlobalParameterName(int index, const std::string& name);
    double getGlobalParameterDefaultValue(int index) const;
    void setGlobalParameterDefaultValue(int index, double defaultValue);

    int getNumExclusions() const;
    int addExclusion(int particle1, int particle2);
    void getExclusionParticles(int index, int& particle1, int& particle2) const;
    void setExclusionParticles(int index, int particle1, int particle2);
    /** Exclude Custom pairs separated by at most bondCutoff bonds. */
    void createExclusionsFromBonds(const std::vector<std::pair<int, int>>& bonds,
                                   int bondCutoff);
    bool getUseLongRangeCorrection() const;
    /** Numerically integrate the Custom expression's tail. The coefficient follows
     * live globals and explicit particle updates, and energy scales as 1/volume.
     * A bounded cache reuses coefficients for repeated referenced-global values;
     * particle updates discard that cache. Force-only evaluations validate globals
     * but defer integration and its convergence check until energy is requested.
     * Explicit particle updates still validate the replacement correction eagerly.
     * The supplied expression is the reference model. Automatic switching adds
     * its removed contribution; a switch written into the expression does not
     * imply an unswitched potential. The integral must converge. Enabling or
     * disabling this option after Context creation requires reinitialization.
     * A full Custom source containing bare Coulomb has a divergent tail: leave
     * this disabled and add an independent integrable cutoff definition through
     * addExpressionTailCorrection(). Lattice declarations do not separate tails.
     */
    void setUseLongRangeCorrection(bool use);

    /**
     * Add a Standard particle in system-atom order.
     * @param charge   partial charge (e)
     * @param sigma    LJ sigma (nm)
     * @param epsilon  LJ well depth (kJ/mol)
     */
    int addParticle(double charge, double sigma, double epsilon);

    void getParticleParameters(int index, double& charge,
                               double& sigma, double& epsilon) const;
    void setParticleParameters(int index, double charge,
                               double sigma, double epsilon);
    void updateParametersInContext(OpenMM::Context& context);

    // ---- Simulation parameters ----
    //
    // Named after their NonbondedForce counterparts, and stored in double as that
    // class stores them; the platform casts when it loads them. The precision is not
    // cosmetic: alpha is inversely proportional to the cutoff, so deriving it from a
    // float-rounded cutoff shifts alpha in its last bits and moves the energy.
    //
    // There is no mesh and no alpha among them, deliberately, and the reason is not
    // that they are uninteresting -- it is that this class cannot answer for them.
    // Both are functions of the SPLITTING and the box: alpha exists under one method
    // and not the other, and the mesh is sized by a different question under each.
    // cupo owns both, and derives them at Context creation, where the box is known --
    // which is also the earliest they COULD be derived, since the box belongs to the
    // Context and under a barostat is not even constant.

    /** Compatibility selector: CutoffPeriodic with no lattice terms, or their
     * common splitting. Throws if independently selected splittings differ. */
    NonbondedMethod getNonbondedMethod() const;
    /** Custom compatibility adapter accepting CutoffPeriodic and clearing lattice
     * terms. Configure lattice splittings through their indexed definitions. */
    void setNonbondedMethod(NonbondedMethod method);

    /** Real-space cutoff (nm). Standard cutoff is fixed at construction. */
    double getCutoffDistance() const;
    /** Custom cutoff changes require Context::reinitialize(). */
    void setCutoffDistance(double distance);

    /** Ordinary-pair LJ combining rule; defaults to LorentzBerthelot. */
    LJCombiningRule getLJCombiningRule() const;

    /**
     * Set the ordinary-pair LJ combining rule before creating a Context.
     * Changing it later requires Context::reinitialize(); it cannot be changed
     * by updateParametersInContext(). Explicit exception sigma and epsilon are
     * used as supplied and are never recombined. The analytic LJ tail correction
     * uses this rule as well. After installing a cutoff expression, combining belongs
     * in that expression and this setter is rejected.
     */
    void setLJCombiningRule(LJCombiningRule rule);

    /** Quintic switching for the cutoff interaction; default false.
     * Multiplies native LJ or the entire custom cutoff expression. The plugin
     * compiles switched expressions while retaining their unswitched definition
     * for tail integration. Separately managed lattice-sum electrostatics and
     * explicit exceptions are unchanged. Disable this option when the expression
     * already contains the intended switch.
     */
    bool getUseSwitchingFunction() const;
    void setUseSwitchingFunction(bool use);

    /** Switching start (nm), default -1. Active switching requires 0 <= start < cutoff.
     * Switching changes require Context::reinitialize(). Explicit exceptions are
     * never switched. Separately managed electrostatics is unchanged. When
     * switching is disabled this distance is inactive metadata.
     * The Standard analytic LJ tail includes
     * the switching interval and requires a positive start when LJ is nonzero.
     */
    double getSwitchingDistance() const;
    void setSwitchingDistance(double distance);

    /**
     * NPT cell-grid headroom (>= 1.0; 1.0 == NVT). Fixed at construction.
     *
     * It sizes the cell-list buffers for a box up to (initial * this), and those
     * are carved once from an arena with no grow path -- so a value changed after
     * the fact would describe an allocation that no longer exists.
     */
    float getBoxSafetyFactor() const { return boxSafetyFactor; }

    /**
     * Whether to add the analytic long-range LJ tail correction to the energy.
     *
     * The coefficient itself is NOT stored here. Like OpenMM, the platform derives
     * it -- at Context creation and again on updateParametersInContext -- so that
     * changing a particle's sigma or epsilon actually moves it, instead of leaving a
     * value frozen at construction.
     * This option applies only to Standard mode. By default, the coefficient also
     * follows live Context globals when they change effective particle sigma or
     * epsilon. Unlike stock OpenMM, offsets use current values rather than global
     * defaults; setDispersionCorrectionFollowsParameters(false) restores that policy.
     * setCutoffInteraction(custom_definition) removes this correction by default. Passing
     * preserveDispersionCorrection=true retains it as an independent reference
     * tail, with the original native mixing and switching policies.
     */
    bool getUseDispersionCorrection() const;
    void setUseDispersionCorrection(bool use);

    /**
     * Whether the Standard dispersion correction follows live Context globals.
     * True by default, including after conversion from a stock NonbondedForce.
     * The platform recomputes the coefficient from current effective particle
     * sigma and epsilon when they change, before the next force or energy
     * evaluation. Charge-only and exception-only global changes reuse the cached
     * coefficient. Fixed-lambda steps do not recompute it; energy always uses the
     * current box volume. No CUDA graph recapture is required.
     * False uses global default values, matching stock OpenMM's NonbondedForce.
     * Changing this policy on an existing Context requires
     * updateParametersInContext(). It does not enable the correction itself.
     */
    bool getDispersionCorrectionFollowsParameters() const;
    void setDispersionCorrectionFollowsParameters(bool follow);

    /**
     * Get whether physical Standard exceptions use periodic minimum-image
     * distances. The default is false, matching stock OpenMM's default. False
     * uses the direct displacement between the supplied atom positions; true
     * uses its periodic minimum image. Both Coulomb and Lennard-Jones exception
     * replacements are evaluated without a cutoff in either case.
     *
     * Direct-space cancellation continues to reproduce the ordinary periodic
     * pair interaction. Reciprocal exclusion corrections always use minimum-image
     * distances, independently of this flag. This differs from stock OpenMM,
     * whose flag also controls the reciprocal exclusion correction's distance
     * convention.
     */
    bool getExceptionsUsePeriodicBoundaryConditions() const;

    /**
     * Set whether physical Standard Coulomb and Lennard-Jones exceptions use
     * periodic minimum-image distances (true) or direct coordinate displacements
     * (false, the default). Exception replacements have no cutoff. Direct-space
     * cancellation continues to reproduce the ordinary periodic pair interaction.
     * Reciprocal exclusion corrections always use minimum-image distances for
     * both settings, unlike stock OpenMM's reciprocal exclusion correction
     * convention.
     *
     * Ordinary molecules represented with contiguous coordinates can use the
     * default false setting; examples need not force this flag to true. Use true
     * when physical exceptions should connect periodic images, such as in a
     * periodically bonded network or an exception-based NBFix construction.
     * A separate CustomBondForce supplying the matching LJ override must also
     * enable periodic boundary conditions for nearest-image behavior.
     * False does not unwrap individually wrapped
     * atom coordinates. Conversion from a stock NonbondedForce preserves the
     * source flag without modifying the source.
     *
     * This option applies only to Standard mode. Changing it after Context
     * creation requires Context::reinitialize(); updateParametersInContext()
     * rejects a changed flag before modifying the Context's parameters.
     */
    void setExceptionsUsePeriodicBoundaryConditions(bool use);

    // ---- Exceptions (exclusions and scaled 1-4 pairs) ----
    //
    // Signatures and parameter order follow NonbondedForce's exception API, and
    // carry the same raw quantities: chargeProd, sigma, epsilon, exactly as the
    // source force gave them. Nothing derived and nothing pre-scaled is stored --
    // the kernel wants qi*qj and 4*epsilon, and both are formed by the platform
    // when it uploads, because a stored copy is stale the moment a particle
    // parameter moves.

    int getNumExceptions() const { return (int)exceptions.size(); }

    /**
     * Add an exception: 1-2/1-3 pairs are fully excluded (chargeProd and epsilon
     * both 0), 1-4 pairs supply the desired scaled charge product, sigma and epsilon.
     * Exception sigma and epsilon are used directly, independently of the
     * ordinary-pair LJ combining rule.
     *
     * The pair kernel evaluates every in-cutoff pair including excluded ones, so
     * every exception must be registered for the cancellation pass. Keep even
     * currently inactive pairs: particle updates can activate an interaction
     * later without changing its exclusion topology.
     *
     * @return the index of the exception that was added
     */
    int addException(int particle1, int particle2,
                     double chargeProd, double sigma, double epsilon);

    void getExceptionParameters(int index, int& particle1, int& particle2,
                                double& chargeProd, double& sigma, double& epsilon) const;
    void setExceptionParameters(int index, int particle1, int particle2,
                                double chargeProd, double sigma, double epsilon);

    bool usesPeriodicBoundaryConditions() const { return true; }

    // ---- Standard parameter offsets ----
    //
    // The API and the meaning are OpenMM's NonbondedForce's: a global parameter is
    // a Context parameter, and an offset adds param*scale to a particle's charge,
    // sigma and epsilon. The effective particle parameters are
    //
    //     charge  = baseCharge  + sum over offsets of param*chargeScale
    //     sigma   = baseSigma   + sum over offsets of param*sigmaScale
    //     epsilon = baseEpsilon + sum over offsets of param*epsilonScale
    //
    // where "base" is what addParticle / setParticleParameters stored. Several
    // offsets may name the same particle, and one parameter may drive many.
    // Exception offsets do the same to an exception's chargeProd, sigma and
    // epsilon.
    //
    // A Context recomputes the effective parameters on the host and uploads them
    // when an evaluation sees a parameter that moved. For particle offsets that
    // costs a pass over every particle and an upload, about what
    // updateParametersInContext costs; for exception offsets alone, a re-ingest of
    // the exception list. Either is built for parameters that change between
    // windows or exchanges, not every step. Energy derivatives with respect to the
    // parameters are not computed.

    int getNumParticleParameterOffsets() const { return (int)particleOffsets.size(); }
    int getNumExceptionParameterOffsets() const { return (int)exceptionOffsets.size(); }

    /**
     * Make a particle's parameters depend on a global parameter. The parameter
     * must already have been added with addGlobalParameter.
     *
     * @return the index of the offset that was added
     */
    int addParticleParameterOffset(const std::string& parameter, int particleIndex,
                                   double chargeScale, double sigmaScale, double epsilonScale);
    void getParticleParameterOffset(int index, std::string& parameter, int& particleIndex,
                                    double& chargeScale, double& sigmaScale,
                                    double& epsilonScale) const;
    void setParticleParameterOffset(int index, const std::string& parameter, int particleIndex,
                                    double chargeScale, double sigmaScale, double epsilonScale);

    /**
     * Make an exception's parameters depend on a global parameter. The parameter
     * must already have been added with addGlobalParameter. `exceptionIndex` is an
     * index into this force's exceptions. Conversion preserves every source
     * exception and its index, including currently inactive exceptions.
     *
     * @return the index of the offset that was added
     */
    int addExceptionParameterOffset(const std::string& parameter, int exceptionIndex,
                                    double chargeProdScale, double sigmaScale,
                                    double epsilonScale);
    void getExceptionParameterOffset(int index, std::string& parameter, int& exceptionIndex,
                                     double& chargeProdScale, double& sigmaScale,
                                     double& epsilonScale) const;
    void setExceptionParameterOffset(int index, const std::string& parameter, int exceptionIndex,
                                     double chargeProdScale, double sigmaScale,
                                     double epsilonScale);

protected:
    OpenMM::ForceImpl* createImpl() const override;

private:
    friend class NonbondedForceProxy;
    void requireMode(PairMode mode, const char* operation) const;
    void applyLatticePolicy(const LatticeSumInteraction& interaction);
    const PairMode pairMode = Standard;
    std::unique_ptr<OpenMM::CustomNonbondedForce> customDefinition;

    struct ParticleInfo {
        double charge, sigma, epsilon;
        ParticleInfo() : charge(0), sigma(0), epsilon(0) {}
        ParticleInfo(double q, double s, double e)
            : charge(q), sigma(s), epsilon(e) {}
    };
    std::vector<ParticleInfo> particles;

    // Cutoff and headroom are structural. The component definitions are copied
    // into immutable execution policies at Context creation: the platform builds
    // a cell list, splitting plan, mesh and captured graphs from them. (The CUTOFF default is
    // 0.9 nm rather than NonbondedForce's 1.0: it is what every bundled system,
    // benchmark and tolerance in this tree is calibrated at, so it is the value a
    // hand-assembled force is least likely to want changed.)
    const double cutoffDistance;
    CutoffInteraction cutoffInteraction;
    bool dispersionCorrectionFollowsParameters = true;
    std::vector<LatticeSumInteraction> latticeSums;
    // (boxSafetyFactor is const too; it is declared with the NPT note below.)
    // NPT: pre-size the cell-grid buffers for a box up to (initial * this), so a
    // barostat expansion recomputes the grid without reallocating. 1.0 = NVT.
    //
    // May be AutoBoxSafety, which is NOT a factor -- it is a request to pick one at
    // Context creation, where the System is complete and the barostat is visible.
    // getBoxSafetyFactor() therefore returns what the caller passed, sentinel
    // included, and NOT the value the platform ended up using: a getter whose
    // meaning changes once a Context exists is worse than one that only ever reports
    // the request. NonbondedForceImpl::resolveBoxSafetyFactor() does the resolving
    // and hands the result straight to the kernel.
    const float boxSafetyFactor;

    // Special pairs
    bool exceptionsUsePeriodicBoundaryConditions = false;
    // Particles and exceptions changed since the last updateParametersInContext(), as in OpenMM.
    mutable int numContexts = 0, firstChangedParticle = 0, lastChangedParticle = -1,
                firstChangedException = 0, lastChangedException = -1;
    struct ExceptionInfo {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
    };
    std::vector<ExceptionInfo> exceptions;
    std::unique_ptr<OpenMM::CustomNonbondedForce> cutoffExpressionDefinition;
    CutoffInteraction referenceLJInteraction;
    std::vector<std::unique_ptr<OpenMM::CustomNonbondedForce>> tailDefinitions;
    void mergeGlobalParameters(const OpenMM::CustomNonbondedForce& definition);

    struct GlobalParameterInfo {
        std::string name;
        double defaultValue;
    };
    std::vector<GlobalParameterInfo> globalParameters;
    // `parameter` is an index into globalParameters, as in OpenMM's NonbondedForce,
    // so renaming a parameter renames it for every offset at once.
    struct ParticleOffsetInfo {
        int parameter, particle;
        double chargeScale, sigmaScale, epsilonScale;
    };
    std::vector<ParticleOffsetInfo> particleOffsets;
    struct ExceptionOffsetInfo {
        int parameter, exception;
        double chargeProdScale, sigmaScale, epsilonScale;
    };
    std::vector<ExceptionOffsetInfo> exceptionOffsets;
    int getGlobalParameterIndex(const std::string& parameter) const;
};

} // namespace NvOpenMM

#endif // NVOPENMM_NONBONDEDFORCE_H_
