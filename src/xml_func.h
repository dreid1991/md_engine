#pragma once
#ifndef XML_FUNC_H
#define XML_FUNC_H
#include "Python.h"
#include <functional>
#include "base64.h"
#include <pugixml.hpp>
#include <vector>
#include <sstream>
using namespace std;
template 
<typename T, int NUM>
void xml_assignValuesBase64(pugi::xml_node &child, std::function<void (int, T *)> assign_loc) {
    istringstream ss(child.first_child().value());
    string x;
    while(ss >> x){};
    string decoded = base64_decode(x);
    T *raw = (T *) decoded.c_str();
    int numVals = decoded.size() / (sizeof(T)*NUM);
    for (int i=0; i<numVals; i++) {
        assign_loc(i, raw + i*NUM);
    }
}


template 
<typename T, int NUM>
void xml_assignValues(pugi::xml_node &child, std::function<void (int, T *)> assign_loc) {
    T buffer[NUM];
    int idx = 0;
    istringstream ss(child.first_child().value());
    string bit;
    int itemIdx = 0;
    while (ss >> bit) {
        T val = (T) atof(bit.c_str());
        buffer[idx] = val;
        idx++;
        if (idx == NUM) {
            assign_loc(itemIdx, buffer);
            idx = 0;
            itemIdx++;

        }

    }

}


template
<typename T, int NUM>
bool xml_assign(pugi::xml_node &config, string tag, std::function<void (int, T *) > assign_loc) {
    auto child = config.child(tag.c_str());
    if (child) {
        auto base64 = child.attribute("base64").value();
        if (strcmp(base64, "") == 0 and strcmp(base64, "1") != 0) {
            xml_assignValues<T, NUM>(child, assign_loc);
        } else {
            xml_assignValuesBase64<T, NUM>(child, assign_loc);
        }
    } else {
        return false;
    }
    return true;
}


template
<typename T>
vector<T> xml_readNums(pugi::xml_node &parent, string tag) {
    auto child = parent.child(tag.c_str());
    if (child) {
        vector<T> res;
        istringstream ss(child.first_child().value());
        string s;
        while (ss >> s) {
            res.push_back(atof(s.c_str()));
        }
        return res;
    }
    return vector<T>();
}


vector<string> xml_readStrings(pugi::xml_node &parent, string tag);
#endif
