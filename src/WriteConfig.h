#pragma once
#ifndef WRITECONFIG_H
#define WRITECONFIG_H

#include <fstream>
#include <sstream>

#include "State.h"
#include "base64.h"
#include "boost_for_export.h"

#define FN_LEN 150

void export_WriteConfig();

class WriteConfig {

private:
    void (*writeFormat)(State *, std::string, int64_t, bool, double);    

public:
    State *state;
    std::string fn;
    std::string handle;
    std::string format;
    int writeEvery;

    double unitLen;
    bool isXML;
    int orderPreference; //just there so I can use same functions as fix for adding/removing
    bool oneFilePerWrite;

    WriteConfig(boost::shared_ptr<State>,
                std::string fn, std::string handle, std::string format, int writeEvery);

    ~WriteConfig() {
        finish();
    }
    void finish();

    void write(int64_t turn);
    void writePy();
    std::string getCurrentFn(int64_t turn);

};

#endif
