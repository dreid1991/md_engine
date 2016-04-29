#pragma once
#ifndef VARIANTPYLISTINTERFACE_H
#define VARIENTPYLISTINTERFACE_H

#include "Python.h"

#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include <boost/variant.hpp>
#include <vector>
//
/*! \brief Bond connecting atomsclass for exposing vectors of variants (bonds, angles, dihedrals, impropers) to the python api
 */

template <class CPUMember>
void deleter(CPUMember *ptr) {};

template <class CPUVariant, class CPUMember>
class VariantPyListInterface {
    private:
        std::vector<CPUVariant> *CPUMembers;
        boost::python::list *pyList;
        CPUVariant *CPUData;
        void refreshPyList() {
            int ii = boost::python::len(*pyList);
            for (int i=0; i<ii; i++) {
                CPUMember *member = boost::get<CPUMember>(&(*CPUMembers)[i]);
                boost::shared_ptr<CPUMember> shrptr (member, deleter<CPUMember>);
                (*pyList)[i] = shrptr;
            }
        }
        
    public:
        VariantPyListInterface(vector<CPUVariant> *CPUMembers_, boost::python::list *pyList_) : CPUMembers(CPUMembers_), pyList(pyList_), CPUData(CPUMembers->data()) {}
        void updateAppendedMember() {

            if (CPUMembers->data() != CPUData) {
                refreshPyList();
                CPUData = CPUMembers->data();
            }
            CPUMember *member = boost::get<CPUMember>(&(CPUMembers->back()));
            boost::shared_ptr<CPUMember> shrptr (member, deleter<CPUMember>);
            pyList->append(shrptr);
        }

};








#endif
