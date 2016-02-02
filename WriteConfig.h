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
	void (*writeFormat)(State *, char [FN_LEN]);	
	public:
		State *state;
		string fn;
		string handle;
		string format;
		int writeEvery;
		int turnInit;
		void write();
		char fnFinal[FN_LEN];
		void finish();
        int orderPreference; //just there so I can use same functions as fix for adding/removing
		WriteConfig(SHARED(State), string fn, string handle, string format, int writeEvery);
        ~WriteConfig() {
            finish();
        }
};

#endif
