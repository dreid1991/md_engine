#include "FixSpringStatic.h"
#include "boost_for_export.h"
#include "FixHelpers.h"
#include "GPUData.h"
#include "State.h"

namespace py = boost::python;

const std::string springStaticType = "SpringStatic";

FixSpringStatic::FixSpringStatic(boost::shared_ptr<State> state_,
                                 std::string handle_, std::string groupHandle_,
                                 double k_,  py::object tetherFunc_, Vector multiplier_)
  : Fix(state_, handle_, groupHandle_, springStaticType, true, false, false, 1),
    k(k_), tetherFunc(tetherFunc_), multiplier(multiplier_)
{
    updateTethers();
}

void FixSpringStatic::updateTethers() {
    std::vector<float4> tethers_loc;
    PyObject *funcRaw = tetherFunc.ptr();
    if (PyCallable_Check(funcRaw)) {
        for (Atom &a : state->atoms) {
            if (a.groupTag & groupTag) {
                Vector res = boost::python::call<Vector>(funcRaw, a.id, a.pos);
                tethers_loc.push_back(make_float4(res[0], res[1], res[2], *(float *)&a.id));
            }
        }
    } else {
        for (Atom &a : state->atoms) {
            if (a.groupTag & groupTag) {
                tethers_loc.push_back(make_float4(a.pos[0], a.pos[1], a.pos[2], *(float *)&a.id));
            }
        }
    }
    tethers = tethers_loc;
}


bool FixSpringStatic::prepareForRun() {
    tethers.dataToDevice();
    return true;
}

void __global__ compute_cu(int nTethers, float4 *tethers, float4 *xs, float4 *fs,
                           cudaTextureObject_t idToIdxs, float k,
                           BoundsGPU bounds, float3 multiplier) {
    int idx = GETIDX();
    if (idx < nTethers) {
        float4 tether = tethers[idx];
        float3 tetherPos = make_float3(tether);
        int id = * (int *) &tether.w;
        int atomIdx = tex2D<int>(idToIdxs, XIDX(id, sizeof(int)), YIDX(id, sizeof(int)));
        float3 curPos = make_float3(xs[atomIdx]);
        //printf("cur is %f %f, tether is %f %f, mult is %f %f %f, k is %f \n", curPos.x, curPos.y, tetherPos.x, tetherPos.y, multiplier.x, multiplier.y, multiplier.z, k);
        float3 force = multiplier * harmonicForce(bounds, curPos, tetherPos, k, 0);
        //printf("forces %f %f %f\n", force.x, force.y, force.z);
        fs[atomIdx] += force;
    }
}

void FixSpringStatic::compute(bool computeVirials) {
    GPUData &gpd = state->gpd;
    int activeIdx = state->gpd.activeIdx();
    compute_cu<<<NBLOCK(tethers.h_data.size()), PERBLOCK>>>(
                    tethers.h_data.size(), tethers.getDevData(),
                    gpd.xs(activeIdx), gpd.fs(activeIdx), gpd.idToIdxs.getTex(),
                    k, state->boundsGPU, multiplier.asFloat3());
}


void export_FixSpringStatic() {
    py::class_<FixSpringStatic, boost::shared_ptr<FixSpringStatic>, py::bases<Fix> > (
            "FixSpringStatic",
            py::init<boost::shared_ptr<State>, std::string, std::string,
                     double, py::optional<py::object, Vector>
                    >(
                py::args("state", "handle", "groupHandle",
                         "k", "tetherFunc", "multiplier")
            )
    )
    .def("updateTethers", &FixSpringStatic::updateTethers)
    .def_readwrite("multiplier", &FixSpringStatic::multiplier)
    .def_readwrite("tetherFunc", &FixSpringStatic::tetherFunc)
    .def_readwrite("k", &FixSpringStatic::k)
    ;
}

