#pragma once
#ifndef WRITECONFIG_H
#define WRITECONFIG_H
#include "State.h"
#include "base64.h"
#include <fstream>
#include <sstream>
#include "boost_for_export.h"
#define FN_LEN 150

void export_WriteConfig();

class WriteConfig {
	void (*writeFormat)(State *, string, int64_t, bool, double);	
	public:
		State *state;
		string fn;
		string handle;
		string format;
		int writeEvery;
        double unitLen;
        bool isXML;
		void write(int64_t turn);
		void finish();
        int orderPreference; //just there so I can use same functions as fix for adding/removing
        bool oneFilePerWrite;
        string getCurrentFn(int64_t turn);

		WriteConfig(SHARED(State), string fn, string handle, string format, int writeEvery);
        ~WriteConfig() {
            finish();
        }
};

#endif
