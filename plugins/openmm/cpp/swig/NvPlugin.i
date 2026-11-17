// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

%module(package="nv_openmm_plugin") nvplugin

%include "std_string.i"

/*
 * OpenMM's typemaps, not SWIG's stock ones -- note the "swig/" on the path.
 *
 * OpenMM installs this file to <prefix>/include/swig for plugins to use, and its
 * `double` typemap runs Py_StripOpenMMUnits() before SWIG_AsVal_double: units come
 * off in C, so every double argument in this interface takes a unit.Quantity, and
 * takes it in whatever unit the caller spelled. `9.0*angstrom` reaches a cutoff
 * declared in nm as 0.9; `1.0*bar` reaches a pressure declared in bar as 1.0.
 *
 * Plain `typemaps.i` resolves to SWIG's own copy instead. The include path here
 * comes from USE_TARGET_INCLUDE_DIRECTORIES, which contributes <prefix>/include --
 * one level above where OpenMM puts this -- so the bare name silently found the
 * wrong file and every Quantity was refused by the SWIG layer. What made that
 * survivable, and hard to see, is that the two hand-written wrappers in
 * nv_openmm_plugin/__init__.py convert Quantities themselves before calling in;
 * the setters, which have no wrapper, simply raised TypeError on a Quantity.
 */
%include "swig/typemaps.i"

%{
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include "openmm/RPMDMonteCarloAnisotropicBarostat.h"
#include "openmm/RPMDMonteCarloFlexibleBarostat.h"
#include "openmm/RPMDMonteCarloMembraneBarostat.h"
#include "openmm/DPDIntegrator.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/serialization/SerializationProxy.h"
#include "openmm/serialization/XmlSerializer.h"
#include "nvopenmm/NonbondedForce.h"
#include "nvopenmm/MonteCarloBarostat.h"
#include "nvopenmm/LangevinMiddleIntegrator.h"
using namespace OpenMM;
%}

/*
 * Translate C++ exceptions into Python ones. Without this SWIG emits a bare call,
 * so anything thrown out of the plugin -- an ASSERT_VALID_INDEX on a bad particle
 * index, the box-safety-factor check in NonbondedForce's converting constructor -- unwinds
 * past the interpreter and calls std::terminate, aborting the process with SIGABRT
 * instead of raising.
 */
%exception {
    try {
        $action
    }
    catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        SWIG_fail;
    }
}

/*
 * Release the GIL around integration. step() runs the whole MD loop in C++ and
 * touches no Python object, so holding the GIL through it blocks every other thread
 * for the duration.
 *
 * The class lives in NvOpenMM, so that is the name SWIG matches here; spelled
 * OpenMM::LangevinMiddleIntegrator::step this applied to nothing (that class is only
 * %imported, never wrapped) and the generated step() held the GIL.
 *
 * The thread state is restored before the handler touches the error indicator:
 * PyErr_SetString needs the GIL, and an exception out of $action would otherwise
 * skip the restore.
 */
%exception NvOpenMM::LangevinMiddleIntegrator::step {
    PyThreadState* _save = PyEval_SaveThread();
    try {
        $action
    }
    catch (std::exception& e) {
        PyEval_RestoreThread(_save);
        PyErr_SetString(PyExc_Exception, e.what());
        SWIG_fail;
    }
    PyEval_RestoreThread(_save);
}

/*
 * Import the core OpenMM SWIG interface so our integrator becomes a subtype of
 * the *same* openmm.Integrator the pip-installed openmm module uses.  This is
 * what lets mm.Context(system, LangevinMiddleIntegrator(...), platform)
 * accept it (cross-module SWIG type sharing).  No wrappers are generated for the
 * imported types; they are only made known.
 */
%import(module="openmm") "swig/OpenMMSwigHeaders.i"

/*
 * The two getters return their results through non-const references. Without these
 * SWIG treats them as inputs and the methods become uncallable from Python; with
 * them the results come back as a list, which is what every stock OpenMM getter of
 * this shape does.
 */
%apply double& OUTPUT { double& charge, double& sigma, double& epsilon,
                        double& chargeProd };
%apply int&    OUTPUT { int& particle1, int& particle2 };

namespace NvOpenMM {

// The plugin's own namespace. OpenMM's names are visible
// unqualified; the ones we shadow (NonbondedForce, MonteCarloBarostat,
// LangevinMiddleIntegrator, ...) are spelled OpenMM:: at their use sites.
using namespace OpenMM;


/* Custom records accept numeric sequences, including numpy arrays and OpenMM
 * quantities, without relying on numpy C helpers private to OpenMM's module. */
%typemap(in, fragment="Py_StripOpenMMUnits") const std::vector<double>& parameters (std::vector<double> temp) {
    PyObject* stripped = Py_StripOpenMMUnits($input);
    if (!stripped) SWIG_fail;
    if (PyBytes_Check(stripped) || PyByteArray_Check(stripped) || PyUnicode_Check(stripped)) {
        Py_DECREF(stripped);
        PyErr_SetString(PyExc_TypeError, "particle parameters must be a sequence of numbers");
        SWIG_fail;
    }
    PyObject* seq = PySequence_Fast(stripped, "particle parameters must be a sequence of numbers");
    Py_DECREF(stripped);
    if (!seq) SWIG_fail;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    temp.reserve((size_t)n);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* value = Py_StripOpenMMUnits(PySequence_Fast_GET_ITEM(seq, i));
        if (!value) { Py_DECREF(seq); SWIG_fail; }
        const double v = PyFloat_AsDouble(value);
        Py_DECREF(value);
        if (v == -1.0 && PyErr_Occurred()) { Py_DECREF(seq); SWIG_fail; }
        temp.push_back(v);
    }
    Py_DECREF(seq);
    $1 = &temp;
}
%typemap(freearg) const std::vector<double>& parameters { }
%typemap(typecheck, precedence=SWIG_TYPECHECK_DOUBLE_ARRAY, fragment="Py_StripOpenMMUnits") const std::vector<double>& parameters {
    PyObject* stripped = Py_StripOpenMMUnits($input);
    $1 = stripped && PySequence_Check(stripped) && !PyBytes_Check(stripped) &&
         !PyByteArray_Check(stripped) && !PyUnicode_Check(stripped);
    Py_XDECREF(stripped);
    if (!$1) PyErr_Clear();
}
%typemap(out) std::vector<double> getParticleParameters {
    $result = PyList_New((Py_ssize_t)$1.size());
    if (!$result) SWIG_fail;
    for (size_t i = 0; i < $1.size(); i++)
        PyList_SET_ITEM($result, (Py_ssize_t)i, PyFloat_FromDouble($1[i]));
}

/* Bulk expression inputs own their converted data until the C++ setter returns.
 * Named typemaps keep these conversions local to this overload. */
%typemap(in) const std::vector<std::string>& parameterNames (std::vector<std::string> temp) {
    if (PyUnicode_Check($input) || PyBytes_Check($input)) {
        PyErr_SetString(PyExc_TypeError, "parameterNames must be a sequence of strings");
        SWIG_fail;
    }
    PyObject* seq = PySequence_Fast($input, "parameterNames must be a sequence of strings");
    if (!seq) SWIG_fail;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq); ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        Py_ssize_t size;
        const char* name = PyUnicode_AsUTF8AndSize(item, &size);
        if (!name) { Py_DECREF(seq); SWIG_fail; }
        temp.emplace_back(name, (size_t)size);
    }
    Py_DECREF(seq);
    $1 = &temp;
}
%typemap(freearg) const std::vector<std::string>& parameterNames { }
%typemap(typecheck, precedence=SWIG_TYPECHECK_STRING_ARRAY) const std::vector<std::string>& parameterNames {
    $1 = PySequence_Check($input) && !PyUnicode_Check($input) && !PyBytes_Check($input);
}

%typemap(in, fragment="Py_StripOpenMMUnits") const std::vector<std::vector<double>>& particleParameters (std::vector<std::vector<double>> temp) {
    PyObject* stripped = Py_StripOpenMMUnits($input);
    if (!stripped) SWIG_fail;
    PyObject* rows = PySequence_Fast(stripped, "particleParameters must be a sequence of numeric rows");
    Py_DECREF(stripped);
    if (!rows) SWIG_fail;
    temp.reserve((size_t)PySequence_Fast_GET_SIZE(rows));
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(rows); ++i) {
        PyObject* strippedRow = Py_StripOpenMMUnits(PySequence_Fast_GET_ITEM(rows, i));
        if (!strippedRow) { Py_DECREF(rows); SWIG_fail; }
        if (PyUnicode_Check(strippedRow) || PyBytes_Check(strippedRow) || PyByteArray_Check(strippedRow)) {
            Py_DECREF(strippedRow);
            Py_DECREF(rows);
            PyErr_SetString(PyExc_TypeError, "each particleParameters row must be a sequence of numbers");
            SWIG_fail;
        }
        PyObject* row = PySequence_Fast(strippedRow, "each particleParameters row must be a sequence of numbers");
        Py_DECREF(strippedRow);
        if (!row) { Py_DECREF(rows); SWIG_fail; }
        temp.emplace_back();
        temp.back().reserve((size_t)PySequence_Fast_GET_SIZE(row));
        for (Py_ssize_t j = 0; j < PySequence_Fast_GET_SIZE(row); ++j) {
            PyObject* value = Py_StripOpenMMUnits(PySequence_Fast_GET_ITEM(row, j));
            if (!value) { Py_DECREF(row); Py_DECREF(rows); SWIG_fail; }
            const double v = PyFloat_AsDouble(value);
            Py_DECREF(value);
            if (PyErr_Occurred()) { Py_DECREF(row); Py_DECREF(rows); SWIG_fail; }
            temp.back().push_back(v);
        }
        Py_DECREF(row);
    }
    Py_DECREF(rows);
    $1 = &temp;
}
%typemap(freearg) const std::vector<std::vector<double>>& particleParameters { }
%typemap(typecheck, precedence=SWIG_TYPECHECK_DOUBLE_ARRAY, fragment="Py_StripOpenMMUnits") const std::vector<std::vector<double>>& particleParameters {
    PyObject* stripped = Py_StripOpenMMUnits($input);
    $1 = stripped && PySequence_Check(stripped) && !PyUnicode_Check(stripped) &&
         !PyBytes_Check(stripped) && !PyByteArray_Check(stripped);
    Py_XDECREF(stripped);
    if (!$1) PyErr_Clear();
}

%typemap(in, fragment="Py_StripOpenMMUnits") const std::map<std::string, double>& globalParameters (std::map<std::string, double> temp) {
    if (!PyDict_Check($input)) {
        PyErr_SetString(PyExc_TypeError, "globalParameters must be a dict of names and defaults");
        SWIG_fail;
    }
    PyObject* key;
    PyObject* item;
    Py_ssize_t pos = 0;
    while (PyDict_Next($input, &pos, &key, &item)) {
        Py_ssize_t size;
        const char* name = PyUnicode_AsUTF8AndSize(key, &size);
        if (!name) SWIG_fail;
        PyObject* value = Py_StripOpenMMUnits(item);
        if (!value) SWIG_fail;
        const double v = PyFloat_AsDouble(value);
        Py_DECREF(value);
        if (PyErr_Occurred()) SWIG_fail;
        temp.emplace(std::string(name, (size_t)size), v);
    }
    $1 = &temp;
}
%typemap(freearg) const std::map<std::string, double>& globalParameters { }
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) const std::map<std::string, double>& globalParameters {
    $1 = PyDict_Check($input);
}

/* Exclusion construction takes an ordinary Python sequence of integer pairs. */
%typemap(in) const std::vector<std::pair<int, int>>& bonds (std::vector<std::pair<int, int>> temp) {
    PyObject* seq = PySequence_Fast($input, "bonds must be a sequence of integer pairs");
    if (!seq) SWIG_fail;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    temp.reserve((size_t)n);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* pair = PySequence_Fast(PySequence_Fast_GET_ITEM(seq, i),
                                       "each bond must be an integer pair");
        if (!pair) { Py_DECREF(seq); SWIG_fail; }
        if (PySequence_Fast_GET_SIZE(pair) != 2) {
            Py_DECREF(pair);
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "each bond must contain exactly two particle indices");
            SWIG_fail;
        }
        int indices[2];
        for (int j = 0; j < 2; j++) {
            PyObject* index = PyNumber_Index(PySequence_Fast_GET_ITEM(pair, j));
            if (!index) { Py_DECREF(pair); Py_DECREF(seq); SWIG_fail; }
            const int status = SWIG_AsVal_int(index, &indices[j]);
            Py_DECREF(index);
            if (!SWIG_IsOK(status)) {
                Py_DECREF(pair);
                Py_DECREF(seq);
                PyErr_SetString(PyExc_OverflowError, "bond particle index exceeds the C int range");
                SWIG_fail;
            }
        }
        temp.emplace_back(indices[0], indices[1]);
        Py_DECREF(pair);
    }
    Py_DECREF(seq);
    $1 = &temp;
}
%typemap(freearg) const std::vector<std::pair<int, int>>& bonds { }

struct CutoffInteraction {
    enum Evaluator { BuiltinLJ = 0, Expression = 1 };
    enum CombiningRule { LorentzBerthelot = 0, Geometric = 1 };
    enum TailPolicy { NoTail = 0, HomogeneousTail = 1 };
    CutoffInteraction();
    Evaluator evaluator;
    CombiningRule combiningRule;
    bool useSwitching;
    double switchingDistance;
    TailPolicy tail;
    std::string expression;
};

struct RadialKernel {
    RadialKernel(int power, double coupling);
    int power;
    double coupling;
    static RadialKernel inversePower(int power, double coupling);
};

RadialKernel CoulombRadialKernel(double coupling = 138.93545764498);
RadialKernel DispersionRadialKernel(double coupling = -1.0);

struct LatticeSumInteraction {
    enum SplittingMethod { Ewald = 0, Prolate = 1 };
    LatticeSumInteraction(const RadialKernel& kernel, const std::string& weightExpression,
                          SplittingMethod splitting = Prolate);
    RadialKernel kernel;
    SplittingMethod splitting;
    std::string weightExpression;
    %pythoncode %{
    def _get_kernel(self):
        kernel = _nvplugin.LatticeSumInteraction_kernel_get(self)
        kernel._lattice_owner = self
        return kernel
    kernel = property(_get_kernel, _nvplugin.LatticeSumInteraction_kernel_set)
    %}
};

%feature("docstring") NonbondedForce::getExceptionsUsePeriodicBoundaryConditions "
Get whether physical Coulomb/LJ exception replacements use periodic distances.

The default is False (raw displacements), and conversion preserves the source
setting. True selects periodic minimum-image displacements for the physical
replacements only. Reciprocal exclusion corrections always use minimum-image
distances, unlike stock OpenMM, where this flag also affects those corrections.
See setExceptionsUsePeriodicBoundaryConditions for details.
";
%feature("docstring") NonbondedForce::setExceptionsUsePeriodicBoundaryConditions "
Set the distance convention for Standard physical Coulomb/LJ exceptions.

False, the default, uses raw displacements. True uses periodic minimum-image
displacements. Conversion preserves the source setting; users no longer need
to explicitly call setExceptionsUsePeriodicBoundaryConditions(True) to convert
a stock NonbondedForce. No cutoff is applied to physical exception replacements.

This flag affects only the splitting-independent replacements evaluated by the
tuples API. Direct exclusion cancellation remains periodic, and Coulomb and
dispersion reciprocal exclusion corrections always use minimum-image distances.
Stock OpenMM also applies this flag to reciprocal exclusion corrections, so
results can differ when raw and minimum-image distances differ.

Use True for physical pair overrides that should follow the nearest periodic
image (for example, exception-based NBFix) or bonded networks connected across
periodic copies. A separate CustomBondForce supplying the matching LJ override
must also enable setUsesPeriodicBoundaryConditions(True) for nearest-image
behavior. A membrane of finite lipid molecules does not inherently need
True. Other bonded terms must use compatible periodic geometry. This setting
applies to every physical exception in this force. Changing it on an existing
Context requires reinitialize(); updateParametersInContext() rejects changes.
";

class NonbondedForce : public OpenMM::Force {
public:
    enum PairMode { Standard = 0, Custom = 1 };

    // Exposed as NonbondedForce.PME / NonbondedForce.Prolate, matching how stock
    // OpenMM surfaces its own NonbondedMethod values.
    enum NonbondedMethod {
        PME = 0,
        Prolate = 1,
        CutoffPeriodic = 2,
    };

    enum LJCombiningRule {
        LorentzBerthelot = 0,
        Geometric = 1,
    };

    // Kept in step with NonbondedForce.h BY HAND -- SWIG parses this file, not the
    // header, so a default that drifts here is a Python API that quietly disagrees
    // with the C++ one.
    static const float AutoBoxSafety;
    static const float DefaultNPTBoxSafety;

    NonbondedForce(NonbondedMethod method = Prolate, double cutoff = 0.9,
                   float boxSafetyFactor = NvOpenMM::NonbondedForce::AutoBoxSafety);
    NonbondedForce(const OpenMM::NonbondedForce& nonbonded,
                   float boxSafetyFactor = NvOpenMM::NonbondedForce::AutoBoxSafety);
    NonbondedForce(const OpenMM::NonbondedForce& nonbonded,
                   float boxSafetyFactor, NonbondedMethod method);

    NonbondedForce(const OpenMM::CustomNonbondedForce& nonbonded,
                   float boxSafetyFactor = NvOpenMM::NonbondedForce::AutoBoxSafety);
    NonbondedForce(const std::string& energy,
                   float boxSafetyFactor = NvOpenMM::NonbondedForce::AutoBoxSafety);

    PairMode getPairMode() const;
    void setCutoffInteraction(const OpenMM::CustomNonbondedForce& definition,
                              bool preserveDispersionCorrection = false);
    void setCutoffInteraction(const std::string& expression,
                              const std::vector<std::string>& parameterNames,
                              const std::vector<std::vector<double>>& particleParameters,
                              const std::map<std::string, double>& globalParameters = std::map<std::string, double>(),
                              bool preserveDispersionCorrection = false,
                              bool tailCorrection = false);
    bool hasCutoffExpression() const;
    OpenMM::CustomNonbondedForce getCutoffExpression() const;
    int addExpressionTailCorrection(const OpenMM::CustomNonbondedForce& definition);
    int getNumExpressionTailCorrections() const;
    OpenMM::CustomNonbondedForce getExpressionTailCorrection(int index) const;
    void setExpressionTailCorrection(int index, const OpenMM::CustomNonbondedForce& definition);
    CutoffInteraction getCutoffInteraction() const;
    void setCutoffInteraction(const CutoffInteraction& interaction);
    // SWIG's overload dispatcher takes positional arguments. Keep the public
    // keywords while dispatching to the C++ overloads.
    %pythoncode %{
    def setCutoffInteraction(self, interaction=None, preserveDispersionCorrection=None,
                             *, expression=None, parameterNames=None,
                             particleParameters=None, globalParameters=None,
                             tailCorrection=None):
        """Set a cutoff policy, a complete force definition, or bulk expression data.

        The bulk form requires expression, parameterNames, and particleParameters
        as keywords. Rows follow System particle order and parameterNames column
        order; globalParameters is an optional dict of names and default values.
        Numbers use OpenMM units; Quantity inputs are converted automatically.

        Bulk expressions inherit the current cutoff, switching, and exclusions.
        In Standard mode they replace all ordinary LJ, retaining Coulomb/PME and
        exceptions. Include ordinary LJ in the expression wherever it is needed.
        tailCorrection defaults to False. When True, one homogeneous correction
        covers the complete expression, using its inherited cutoff and switch,
        particle records, and live global parameters. It uses the ungrouped
        CustomNonbondedForce averaging convention, including the switch deficit.
        The original LJ tail is disabled by default. preserveDispersionCorrection
        retains that original tail instead, not its pair energy; it cannot be
        combined with tailCorrection. tailCorrection is incompatible with LJPME.
        Existing independent expression corrections remain additive, so avoid
        duplicating a contribution already covered by the complete expression.
        Invalid bulk data leave the force unchanged. Expression compilation and
        Context compatibility are checked at Context creation or update.
        """
        bulk = any(value is not None for value in
                   (expression, parameterNames, particleParameters, globalParameters,
                    tailCorrection))
        if bulk:
            if interaction is not None:
                raise TypeError("interaction cannot be combined with bulk expression arguments")
            if expression is None or parameterNames is None or particleParameters is None:
                raise TypeError("bulk cutoff requires expression, parameterNames, and particleParameters")
            return _nvplugin.NonbondedForce_setCutoffInteraction(
                self, expression, parameterNames, particleParameters,
                {} if globalParameters is None else globalParameters,
                False if preserveDispersionCorrection is None else preserveDispersionCorrection,
                False if tailCorrection is None else tailCorrection)
        if interaction is None:
            raise TypeError("provide interaction or bulk expression arguments")
        if preserveDispersionCorrection is None:
            return _nvplugin.NonbondedForce_setCutoffInteraction(self, interaction)
        return _nvplugin.NonbondedForce_setCutoffInteraction(
            self, interaction, preserveDispersionCorrection)
    %}

    bool hasLatticeSumInteraction() const;
    int getNumLatticeSumInteractions() const;
    int addLatticeSumInteraction(const LatticeSumInteraction& interaction);
    LatticeSumInteraction getLatticeSumInteraction(int index) const;
    void setLatticeSumInteraction(int index, const LatticeSumInteraction& interaction);
    void removeLatticeSumInteraction(int index);
    LatticeSumInteraction getLatticeSumInteraction() const;
    void setLatticeSumInteraction(const LatticeSumInteraction& interaction);
    void setUseLJPME(bool use);
    bool getUseLJPME() const;
    static NonbondedForce* cast(OpenMM::Force& force);
    int getNumParticles() const;
    int addParticle(const std::vector<double>& parameters);
    std::vector<double> getParticleParameters(int index) const;
    void setParticleParameters(int index, const std::vector<double>& parameters);

    const std::string& getEnergyFunction() const;
    void setEnergyFunction(const std::string& energy);
    int getNumPerParticleParameters() const;
    int addPerParticleParameter(const std::string& name);
    const std::string& getPerParticleParameterName(int index) const;
    void setPerParticleParameterName(int index, const std::string& name);
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
    void createExclusionsFromBonds(const std::vector<std::pair<int, int>>& bonds, int bondCutoff);
    bool getUseLongRangeCorrection() const;
    void setUseLongRangeCorrection(bool use);
    int addParticle(double charge, double sigma, double epsilon);
    void setParticleParameters(int index, double charge, double sigma, double epsilon);
    void updateParametersInContext(OpenMM::Context& context);

    NonbondedMethod getNonbondedMethod() const;
    void setNonbondedMethod(NonbondedMethod method);
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);
    LJCombiningRule getLJCombiningRule() const;
    void setLJCombiningRule(LJCombiningRule rule);
    bool getUseSwitchingFunction() const;
    void setUseSwitchingFunction(bool use);
    double getSwitchingDistance() const;
    void setSwitchingDistance(double distance);
    float getBoxSafetyFactor() const;
    bool getUseDispersionCorrection() const;
    void setUseDispersionCorrection(bool use);
    bool getDispersionCorrectionFollowsParameters() const;
    void setDispersionCorrectionFollowsParameters(bool follow);

    bool getExceptionsUsePeriodicBoundaryConditions() const;
    void setExceptionsUsePeriodicBoundaryConditions(bool use);

    int getNumExceptions() const;
    int addException(int particle1, int particle2,
                     double chargeProd, double sigma, double epsilon);
    void getExceptionParameters(int index, int& particle1, int& particle2,
                                double& chargeProd, double& sigma, double& epsilon) const;
    void setExceptionParameters(int index, int particle1, int particle2,
                                double chargeProd, double sigma, double epsilon);

    bool usesPeriodicBoundaryConditions() const;

    int getNumParticleParameterOffsets() const;
    int addParticleParameterOffset(const std::string& parameter, int particleIndex,
                                   double chargeScale, double sigmaScale, double epsilonScale);
    // The five results come back through references. Applied around this one
    // declaration and cleared after it, as OpenMM's own interface does: the setter
    // and adder below take the same names as inputs, and an OUTPUT typemap left in
    // force turns their `parameter` into a dangling string pointer.
    %apply std::string& OUTPUT { std::string& parameter };
    %apply int&    OUTPUT { int& particleIndex };
    %apply double& OUTPUT { double& chargeScale, double& sigmaScale, double& epsilonScale };
    void getParticleParameterOffset(int index, std::string& parameter, int& particleIndex,
                                    double& chargeScale, double& sigmaScale,
                                    double& epsilonScale) const;
    %clear std::string& parameter;
    %clear int& particleIndex;
    %clear double& chargeScale, double& sigmaScale, double& epsilonScale;
    void setParticleParameterOffset(int index, const std::string& parameter, int particleIndex,
                                    double chargeScale, double sigmaScale, double epsilonScale);

    int getNumExceptionParameterOffsets() const;
    int addExceptionParameterOffset(const std::string& parameter, int exceptionIndex,
                                    double chargeProdScale, double sigmaScale,
                                    double epsilonScale);
    // Applied and cleared around the one declaration, for the reason given above.
    %apply std::string& OUTPUT { std::string& parameter };
    %apply int&    OUTPUT { int& exceptionIndex };
    %apply double& OUTPUT { double& chargeProdScale, double& sigmaScale, double& epsilonScale };
    void getExceptionParameterOffset(int index, std::string& parameter, int& exceptionIndex,
                                     double& chargeProdScale, double& sigmaScale,
                                     double& epsilonScale) const;
    %clear std::string& parameter;
    %clear int& exceptionIndex;
    %clear double& chargeProdScale, double& sigmaScale, double& epsilonScale;
    void setExceptionParameterOffset(int index, const std::string& parameter, int exceptionIndex,
                                     double chargeProdScale, double sigmaScale,
                                     double epsilonScale);
};

// Casts borrow existing objects; their owners retain responsibility for deletion.
%feature("newobject", "0") MonteCarloBarostat::cast;
%feature("newobject", "0") LangevinMiddleIntegrator::cast;

class MonteCarloBarostat : public OpenMM::Force {
public:
    static const std::string& Pressure();
    static const std::string& Temperature();
    MonteCarloBarostat(double defaultPressure, double defaultTemperature, int frequency=25);
    static MonteCarloBarostat* cast(OpenMM::Force& force);

    double getDefaultPressure() const;
    void setDefaultPressure(double pressure);
    double getDefaultTemperature() const;
    void setDefaultTemperature(double temp);
    int getFrequency() const;
    void setFrequency(int freq);
    int getRandomNumberSeed() const;
    void setRandomNumberSeed(int seed);
    bool usesPeriodicBoundaryConditions() const;
};

class LangevinMiddleIntegrator : public OpenMM::Integrator {
public:
    LangevinMiddleIntegrator(double temperature, double frictionCoeff, double stepSize, int comFrequency=100);
    static LangevinMiddleIntegrator* cast(OpenMM::Integrator& integrator);
    double getTemperature() const;
    void setTemperature(double temp);
    double getFriction() const;
    void setFriction(double coeff);
    int getRandomNumberSeed() const;
    void setRandomNumberSeed(int seed);
    int getComFrequency() const;
    void setComFrequency(int frequency);
    virtual void step(int steps);
    long long getWholeStepGraphCaptureCount() const;
    long long getWholeStepGraphReplayCount() const;
};

}

%clear double& charge, double& sigma, double& epsilon, double& chargeProd;
%clear int& particle1, int& particle2;
