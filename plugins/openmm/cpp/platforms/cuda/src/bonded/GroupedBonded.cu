// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Adapter to the CUPO C ABI. Numerical evaluators and scheduling live in
// libcupo; no OpenMM GPU numerical implementation is embedded in this file.
#include "GroupedBonded.hpp"
#include "nvopenmm/NonbondedForce.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/PeriodicTorsionForce.h"
#include "openmm/System.h"
#include "openmm/kernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/ComputeForceInfo.h"
#include "CudaContext.h"
#include "CudaPlatform.h"
#include <openmm/cuda/CudaKernelFactory.h>
#include <cupo/tuples.h>
#include <cupo/detail/tuples_adapter.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <vector>
#include <variant>
#include <type_traits>
#include <iterator>

namespace NvOpenMM {
namespace {
void checked(cupo_status status) {
    if (status!=CUPO_SUCCESS) throw OpenMM::OpenMMException(cupo_last_error());
}
void checked(cudaError_t status) {
    if (status!=cudaSuccess) throw OpenMM::OpenMMException(cudaGetErrorString(status));
}
using Terms = std::variant<std::vector<cupo_tuples_bond>,
    std::vector<cupo_tuples_angle>, std::vector<cupo_tuples_torsion>,
    std::vector<cupo_tuples_pair>>;
using Parameters = std::variant<std::vector<cupo_tuples_bond_parameters>,
    std::vector<cupo_tuples_angle_parameters>, std::vector<cupo_tuples_torsion_parameters>,
    std::vector<cupo_tuples_pair_parameters>>;

std::vector<cupo_tuples_bond> extract(const OpenMM::HarmonicBondForce& force) {
    std::vector<cupo_tuples_bond> terms(force.getNumBonds());
    for (int i=0;i<int(terms.size());++i) {
        auto& t=terms[i]; t.group=force.getForceGroup();
        force.getBondParameters(i,t.atoms[0],t.atoms[1],t.parameters.length,t.parameters.k);
    }
    return terms;
}
std::vector<cupo_tuples_angle> extract(const OpenMM::HarmonicAngleForce& force) {
    std::vector<cupo_tuples_angle> terms(force.getNumAngles());
    for (int i=0;i<int(terms.size());++i) {
        auto& t=terms[i]; t.group=force.getForceGroup();
        force.getAngleParameters(i,t.atoms[0],t.atoms[1],t.atoms[2],t.parameters.angle,t.parameters.k);
    }
    return terms;
}
std::vector<cupo_tuples_torsion> extract(const OpenMM::PeriodicTorsionForce& force) {
    std::vector<cupo_tuples_torsion> terms(force.getNumTorsions());
    for (int i=0;i<int(terms.size());++i) {
        auto& t=terms[i]; t.group=force.getForceGroup();
        force.getTorsionParameters(i,t.atoms[0],t.atoms[1],t.atoms[2],t.atoms[3],
            t.parameters.periodicity,t.parameters.phase,t.parameters.k);
    }
    return terms;
}
constexpr cupo_tuples_kind kind(const cupo_tuples_bond_parameters&) { return CUPO_TUPLES_HARMONIC_BOND; }
constexpr cupo_tuples_kind kind(const cupo_tuples_angle_parameters&) { return CUPO_TUPLES_HARMONIC_ANGLE; }
constexpr cupo_tuples_kind kind(const cupo_tuples_torsion_parameters&) { return CUPO_TUPLES_PERIODIC_TORSION; }
constexpr cupo_tuples_kind kind(const cupo_tuples_pair_parameters&) { return CUPO_TUPLES_COULOMB_LJ_PAIR; }
bool equal(const cupo_tuples_bond_parameters& a,const cupo_tuples_bond_parameters& b) {
    return a.length==b.length && a.k==b.k;
}
bool equal(const cupo_tuples_angle_parameters& a,const cupo_tuples_angle_parameters& b) {
    return a.angle==b.angle && a.k==b.k;
}
bool equal(const cupo_tuples_torsion_parameters& a,const cupo_tuples_torsion_parameters& b) {
    return a.k==b.k && a.phase==b.phase && a.periodicity==b.periodicity;
}
bool equal(const cupo_tuples_pair_parameters& a,const cupo_tuples_pair_parameters& b) {
    return a.coulomb==b.coulomb && a.sigma==b.sigma && a.epsilon==b.epsilon;
}
cupo_tuples_term_batch termBatch(const Terms& terms) {
    return std::visit([](const auto& values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        return cupo_tuples_term_batch{kind(T{}.parameters),int(values.size()),values.data()};
    },terms);
}
Parameters parameters(const Terms& terms) {
    return std::visit([](const auto& values)->Parameters {
        using P = std::decay_t<decltype(values[0].parameters)>;
        std::vector<P> result;
        result.reserve(values.size());
        for (const auto& term:values) result.push_back(term.parameters);
        return result;
    },terms);
}
cupo_tuples_parameter_batch parameterBatch(const Parameters& params) {
    return std::visit([](const auto& values) {
        using P = typename std::decay_t<decltype(values)>::value_type;
        return cupo_tuples_parameter_batch{kind(P{}),int(values.size()),values.data()};
    },params);
}
struct Entry {
    Terms terms;
    Parameters cachedParameters;
    OpenMM::ComputeForceInfo* info=nullptr; // owned by CudaContext
};
class TopologyInfo final : public OpenMM::ComputeForceInfo {
    Entry& entry;
public:
    explicit TopologyInfo(Entry& e):entry(e) {}
    int getNumParticleGroups() override { return termBatch(entry.terms).count; }
    bool areParticlesIdentical(int,int) override { return true; }
    void getParticlesInGroup(int index,std::vector<int>& atoms) override {
        std::visit([&](const auto& terms) {
            const auto& t=terms[index]; atoms.assign(std::begin(t.atoms),std::end(t.atoms));
        },entry.terms);
    }
    bool areGroupsIdentical(int a,int b) override {
        return std::visit([&](const auto& terms) {
            return equal(terms[a].parameters,terms[b].parameters);
        },entry.terms);
    }
};
__global__ void addSingleEnergy(const double* value,float* buffer) {
    buffer[0]+=float(*value);
}

class GroupedContext final : public OpenMM::ComputeContext::ForcePostComputation {
    OpenMM::CudaContext& cu;
    std::vector<std::unique_ptr<Entry>> entries;
    cupo_tuples_handle plan=nullptr;
    double* singleEnergy=nullptr;
    double* deviceBox=nullptr;
    double hostBox[6]{};
    cudaEvent_t boxUploaded=nullptr;
    bool boxUploadPending=false;
    bool usePbc=false;
public:
    explicit GroupedContext(OpenMM::CudaContext& context):cu(context) {}
    ~GroupedContext() override {
        OpenMM::ContextSelector current(cu);
        if (plan) cupo_tuples_destroy(plan);
        if (singleEnergy) cudaFree(singleEnergy);
        if (boxUploaded) cudaEventDestroy(boxUploaded);
        if (deviceBox) cudaFree(deviceBox);
    }
    Entry& add(Terms terms,bool registerTopology=true) {
        if (plan) throw OpenMM::OpenMMException("Cannot add bonded topology after initialization");
        auto entry=std::make_unique<Entry>(); entry->terms=std::move(terms);
        entry->cachedParameters=parameters(entry->terms);
        std::visit([&](const auto& values) {
            for (const auto& term:values) usePbc|=term.use_pbc!=0;
        },entry->terms);
        if (registerTopology) {
            entry->info=new TopologyInfo(*entry);
            cu.addForce(entry->info);
        }
        entries.push_back(std::move(entry));
        return *entries.back();
    }
    void reorder() {
        if (!plan) return;
        OpenMM::ContextSelector current(cu);
        std::vector<int> inverse(cu.getNumAtoms());
        const auto& order=cu.getAtomIndex();
        for (int slot=0;slot<int(inverse.size());++slot) inverse[order[slot]]=slot;
        checked(cupo_tuples_set_atom_mapping(plan,int(inverse.size()),inverse.data()));
    }
    void prepare() {
        if (plan) return;
        OpenMM::ContextSelector current(cu);
        std::vector<cupo_tuples_term_batch> batches;
        for (const auto& e:entries) batches.push_back(termBatch(e->terms));
        checked(cupo_tuples_create(cu.getNumAtoms(),int(batches.size()),batches.data(),CUPO_TUPLES_FLOAT32,&plan));
        if (!cu.getUseMixedPrecision()) checked(cudaMalloc(&singleEnergy,sizeof(double)));
        if (usePbc) {
            checked(cudaMalloc(&deviceBox,6*sizeof(double)));
            checked(cudaEventCreateWithFlags(&boxUploaded,cudaEventDisableTiming));
        }
        reorder();
    }
    void uploadParameters(const Entry& entry,const Parameters& updated) {
        if (!plan) return;
        std::vector<cupo_tuples_parameter_batch> batches;
        batches.reserve(entries.size());
        for (const auto& e:entries)
            batches.push_back(parameterBatch(e.get()==&entry ? updated : e->cachedParameters));
        checked(cupo_tuples_update_parameters(plan,int(batches.size()),batches.data()));
    }
    void update(Entry& entry,Terms terms,int first,int last) {
        auto updated=entry.terms;
        std::visit([&](auto& values) {
            using T = std::decay_t<decltype(values)>;
            const auto& incoming=std::get<T>(terms);
            if (incoming.size()!=values.size())
                throw OpenMM::OpenMMException("Bonded interaction count changed");
            for (int i=0;i<int(values.size());++i) {
                if (!std::equal(std::begin(incoming[i].atoms),std::end(incoming[i].atoms),std::begin(values[i].atoms)))
                    throw OpenMM::OpenMMException("Bonded topology changes require Context reinitialization");
                if (i>=first && i<=last) values[i].parameters=incoming[i].parameters;
            }
        },updated);
        OpenMM::ContextSelector current(cu);
        auto params=parameters(updated);
        uploadParameters(entry,params);
        entry.terms.swap(updated);
        entry.cachedParameters.swap(params);
        cu.invalidateMolecules(entry.info,false,true);
    }
    void updatePairs(Entry& entry,const std::vector<cupo_tuples_pair_parameters>& values) {
        auto& terms=std::get<std::vector<cupo_tuples_pair>>(entry.terms);
        if (terms.size()!=values.size())
            throw OpenMM::OpenMMException("Exception pair count changed");
        // Pair topology is immutable. A parameter-only update needs neither
        // copies of all term descriptors nor extraction of unchanged entries.
        Parameters proposed=values;
        OpenMM::ContextSelector current(cu);
        uploadParameters(entry,proposed);
        for (size_t i=0;i<terms.size();++i)
            if (!equal(terms[i].parameters,values[i])) terms[i].parameters=values[i];
        entry.cachedParameters.swap(proposed);
    }
    double computeForceAndEnergy(bool forces,bool energy,int groups) override {
        if (!forces && !energy) return 0;
        OpenMM::ContextSelector current(cu);
        prepare();
        const auto stream=reinterpret_cast<cudaStream_t>(cu.getCurrentStream());
        if (usePbc) {
            OpenMM::Vec3 a,b,c;
            cu.getPeriodicBoxVectors(a,b,c);
            const double next[6]={a[0],b[1],c[2],b[0],c[0],c[1]};
            if (!std::equal(std::begin(next),std::end(next),hostBox)) {
                // Whole-step replay reads deviceBox in place. Ordinary force
                // evaluations refresh it after accepted and rejected NPT moves.
                if (boxUploadPending) checked(cudaEventSynchronize(boxUploaded));
                std::copy(std::begin(next),std::end(next),hostBox);
                checked(cudaMemcpyAsync(deviceBox,hostBox,sizeof(hostBox),cudaMemcpyHostToDevice,stream));
                checked(cudaEventRecord(boxUploaded,stream));
                boxUploadPending=true;
            }
        }
        double* output=nullptr;
        if (energy) {
            if (cu.getUseMixedPrecision()) output=reinterpret_cast<double*>(cu.unwrap(cu.getEnergyBuffer()).getDevicePointer());
            else { output=singleEnergy; checked(cudaMemsetAsync(output,0,sizeof(double),stream)); }
        }
        checked(cupo_tuples_execute(plan,reinterpret_cast<void*>(cu.unwrap(cu.getPosq()).getDevicePointer()),deviceBox,
            forces ? reinterpret_cast<int64_t*>(cu.unwrap(cu.getLongForceBuffer()).getDevicePointer()) : nullptr,
            cu.getPaddedNumAtoms(),output,uint32_t(groups),stream));
        if (energy && !cu.getUseMixedPrecision()) {
            addSingleEnergy<<<1,1,0,stream>>>(singleEnergy,reinterpret_cast<float*>(cu.unwrap(cu.getEnergyBuffer()).getDevicePointer()));
            checked(cudaGetLastError());
        }
        return 0;
    }
};
class OrderListener final : public OpenMM::ComputeContext::ReorderListener {
    GroupedContext& owner;
public:
    explicit OrderListener(GroupedContext& value):owner(value) {}
    void execute() override { owner.reorder(); }
};
GroupedContext& sharedContext(OpenMM::CudaContext& cu) {
    auto& hooks=cu.getPostComputations();
    for (auto* hook:hooks) if (auto* value=dynamic_cast<GroupedContext*>(hook)) return *value;
    auto* value=new GroupedContext(cu);
    // Evaluate while the nonbonded side stream is still running, before the
    // plugin's deferred nonbonded join/readback hook.
    hooks.insert(hooks.begin(),value);
    cu.addReorderListener(new OrderListener(*value));
    return *value;
}

template<class Interface,class Force> class Adapter final : public Interface {
    OpenMM::CudaContext& cu;
    OpenMM::ContextImpl& context;
    const OpenMM::Platform& platform;
    std::string name;
    std::unique_ptr<Interface> fallback;
    GroupedContext* grouped=nullptr;
    Entry* entry=nullptr;
public:
    Adapter(const std::string& name,const OpenMM::Platform& platform,OpenMM::ContextImpl& context,OpenMM::CudaContext& cu)
        :Interface(name,platform),cu(cu),context(context),platform(platform),name(name) {}
    void initialize(const OpenMM::System& system,const Force& force) override {
        if (force.usesPeriodicBoundaryConditions()) {
            OpenMM::CudaKernelFactory stock;
            fallback.reset(dynamic_cast<Interface*>(stock.createKernelImpl(name,platform,context)));
            if (!fallback) throw OpenMM::OpenMMException("Missing stock bonded fallback");
            fallback->initialize(system,force);
        } else {
            grouped=&sharedContext(cu);
            entry=&grouped->add(extract(force));
        }
    }
    double execute(OpenMM::ContextImpl& context,bool forces,bool energy) override {
        if (fallback) return fallback->execute(context,forces,energy);
        grouped->prepare();
        return 0; // one aggregate evaluation in the post-computation hook
    }
    void copyParametersToContext(OpenMM::ContextImpl& context,const Force& force,int first,int last) override {
        if (fallback) fallback->copyParametersToContext(context,force,first,last);
        else grouped->update(*entry,extract(force),first,last);
    }
};
} // namespace

struct GroupedBondedPairs::Impl {
    GroupedContext& owner;
    Entry& entry;
    Impl(OpenMM::CudaContext& cu,std::vector<cupo_tuples_pair> terms)
        :owner(sharedContext(cu)),entry(owner.add(std::move(terms),false)) {}
};
GroupedBondedPairs::GroupedBondedPairs(OpenMM::CudaContext& cu,std::vector<cupo_tuples_pair> terms)
    :impl(std::make_unique<Impl>(cu,std::move(terms))) {}
GroupedBondedPairs::~GroupedBondedPairs()=default;
void GroupedBondedPairs::prepare() { impl->owner.prepare(); }
void GroupedBondedPairs::update(const std::vector<cupo_tuples_pair_parameters>& parameters) {
    // Nonbonded ForceInfo owns molecule equivalence, including offset laws.
    impl->owner.updatePairs(impl->entry,parameters);
}

OpenMM::KernelImpl* createGroupedBondedKernel(const std::string& name,
    const OpenMM::Platform& platform,OpenMM::ContextImpl& context) {
    const bool bond=name==OpenMM::CalcHarmonicBondForceKernel::Name();
    const bool angle=name==OpenMM::CalcHarmonicAngleForceKernel::Name();
    const bool torsion=name==OpenMM::CalcPeriodicTorsionForceKernel::Name();
    if (!bond && !angle && !torsion) return nullptr;
    const char* setting=std::getenv("CUPO_PLUGIN_BONDED");
    if (setting && std::string(setting)!="0" && std::string(setting)!="1")
        throw OpenMM::OpenMMException("CUPO_PLUGIN_BONDED must be 0 or 1");
    auto& data=*static_cast<OpenMM::CudaPlatform::PlatformData*>(context.getPlatformData());
    auto& cu=*data.contexts[0];
    bool cupoSystem=false;
    const auto& system=context.getSystem();
    for (int i=0;i<system.getNumForces();++i)
        cupoSystem|=dynamic_cast<const NvOpenMM::NonbondedForce*>(&system.getForce(i))!=nullptr;
    if (!cupoSystem || (setting && setting[0]=='0') || cu.getUseDoublePrecision() || data.contexts.size()!=1) {
        OpenMM::CudaKernelFactory stock;
        return stock.createKernelImpl(name,platform,context);
    }
    if (bond) return new Adapter<OpenMM::CalcHarmonicBondForceKernel,OpenMM::HarmonicBondForce>(name,platform,context,cu);
    if (angle) return new Adapter<OpenMM::CalcHarmonicAngleForceKernel,OpenMM::HarmonicAngleForce>(name,platform,context,cu);
    return new Adapter<OpenMM::CalcPeriodicTorsionForceKernel,OpenMM::PeriodicTorsionForce>(name,platform,context,cu);
}
} // namespace NvOpenMM
