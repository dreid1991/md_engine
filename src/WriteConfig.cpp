#include "WriteConfig.h"
#include "includeFixes.h"
#include <inttypes.h>
#define BUFFERLEN 700
template <typename T>
void writeXMLChunk(ofstream &outFile, vector<T> &vals, string tag, std::function<void (T &, char [BUFFERLEN])> getLine) {

	char buffer[BUFFERLEN];
	outFile << "<" << tag << ">\n";
	for (T &v : vals) {
		getLine(v, buffer);
		outFile << buffer;
	}
	outFile << "</" << tag << ">\n";
}

template <typename T, typename K>
void writeXMLChunkBase64(ofstream &outFile, vector<T> &vals, string tag, std::function<K (T &)> getItem) {
    vector<K> slicedVals;
    slicedVals.reserve(vals.size());
    for (T &v : vals) {
        slicedVals.push_back(getItem(v));
    }
    vector<unsigned char> copied = vector<unsigned char>((unsigned char *) slicedVals.data(), (unsigned char *) slicedVals.data() + (int) (sizeof(K)/sizeof(unsigned char) * slicedVals.size()));
    string base64 = base64_encode(copied.data(), copied.size());  
	outFile << "<" << tag << " base64=\"1\">\n";
    outFile << base64.c_str();
	outFile << "</" << tag << ">\n";
}


void writeAtomParams(ofstream &outFile, AtomParams &params) {
	
	outFile << "<atomParams numTypes=\"" << params.numTypes << "\">\n";
	outFile << "<handle>\n";
	for (string handle : params.handles) {
		outFile << handle << "\n";
	}
	outFile << "</handle>\n";

	outFile << "<mass>\n";
	for (num mass : params.masses) {
		outFile << mass << "\n";
	}
	outFile << "</mass>\n";


	outFile << "</atomParams>\n";

}

void writeXMLfileBase64(State *state, string fnFinal, int64_t turn, bool oneFilePerWrite, double unitLen) {
	vector<Atom> &atoms = state->atoms;
	ofstream outFile;
	Bounds b = state->bounds;
    if (oneFilePerWrite) {
        outFile.open(fnFinal.c_str(), ofstream::out);
        outFile << "<data>" << endl;
    } else {
        outFile.open(fnFinal.c_str(), ofstream::app);
    }
	char buffer[BUFFERLEN];
	int ndims = state->is2d ? 2 : 3;	

    double dims[12];
    * (Vector *)dims = b.lo;
    for (int i=1; i<4; i++) {
        * (((Vector *)dims)+i) = b.sides[i-1];
    }
    string b64[12];
    for (int i=0; i<12; i++) {
        b64[i] = base64_encode((const unsigned char *) (dims + i), 8);
    }
	sprintf(buffer, "<configuration turn=\"%" PRId64 "\" numAtoms=\"%d\" dimension=\"%d\" periodic=\"%d%d%d\">\n", turn, (int) atoms.size(), ndims, state->periodic[0], state->periodic[1], state->periodic[2]);
	outFile << buffer;
	sprintf(buffer, "<bounds base64=\"1\" xlo=\"%s\" ylo=\"%s\" zlo=\"%s\" sxx=\"%s\" sxy=\"%s\" sxz=\"%s\" syx=\"%s\" syy=\"%s\" syz=\"%s\" szx=\"%s\" szy=\"%s\" szz=\"%s\"/>\n", b64[0].c_str(), b64[1].c_str(), b64[2].c_str(), b64[3].c_str(), b64[4].c_str(), b64[5].c_str(), b64[6].c_str(), b64[7].c_str(), b64[8].c_str(), b64[9].c_str(), b64[10].c_str(), b64[11].c_str());
	outFile << buffer;
    writeAtomParams(outFile, state->atomParams);
    writeXMLChunkBase64<Atom, Vector>(outFile, atoms, "position", [] (Atom &a) {
            return a.pos;	
            }
            );

	writeXMLChunkBase64<Atom, Vector> (outFile, atoms, "velocity", [] (Atom &a) {
			return a.vel;
			}
			);
	writeXMLChunkBase64<Atom, Vector> (outFile, atoms, "force", [] (Atom &a) {
			return a.force;
            }
			);

	writeXMLChunkBase64<Atom, Vector >(outFile, atoms, "forceLast", [] (Atom &a) {
            return a.forceLast;
			}
			);
	writeXMLChunkBase64<Atom, uint>(outFile, atoms, "groupTag", [] (Atom &a) {
            return a.groupTag;
			}
			);
	writeXMLChunkBase64<Atom, int>(outFile, atoms, "type", [] (Atom &a) {
            return a.type;
			}
			);
	writeXMLChunkBase64<Atom, int>(outFile, atoms, "id", [] (Atom &a) {
            return a.id;
			}
			);

	sprintf(buffer, "</configuration>\n");
	outFile << buffer;
    if (oneFilePerWrite) {
        outFile << "/<data>" << endl;
    }
	outFile.close();


}


void writeXYZFile(State *state, string fn, int64_t turn, bool oneFilePerWrite, double unitLen) {
    vector<Atom> &atoms = state->atoms;
    AtomParams &params = state->atomParams;
    bool useAtomicNums = true;
    for (int atomicNum : params.atomicNums) {
        if (atomicNum == -1) {
            useAtomicNums = false;
        }
    }
    ofstream outFile;
    if (oneFilePerWrite) {
        outFile.open(fn.c_str(), ofstream::out);
    } else {
        outFile.open(fn.c_str(), ofstream::app);
    }
    outFile << atoms.size() << endl;
    for (Atom &a : atoms) {
        int atomicNum;
        if (useAtomicNums) {
            atomicNum = params.atomicNums[a.type];
        } else {
            atomicNum = a.type;
        }
        outFile << endl << atomicNum << " " << (a.pos[0] / unitLen) << " " << (a.pos[1] / unitLen) << " " << (a.pos[2] / unitLen);
    }
    outFile << endl;
    outFile.close();
}

void writeXMLfile(State *state, string fnFinal, int64_t turn, bool oneFilePerWrite, double unitLen) {
	vector<Atom> &atoms = state->atoms;
	ofstream outFile;
	Bounds b = state->bounds;
    if (oneFilePerWrite) {
        outFile.open(fnFinal.c_str(), ofstream::out);
        outFile << "<data>" << endl;
    } else {
        outFile.open(fnFinal.c_str(), ofstream::app);
    }
	char buffer[BUFFERLEN];
	int ndims = state->is2d ? 2 : 3;	
	double s[9];
	for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			s[i*3 + j] = (double) b.sides[i][j];
		}
	}
	sprintf(buffer, "<configuration turn=\"%" PRId64 "\" numAtoms=\"%d\" dimension=\"%d\" periodic=\"%d%d%d\">\n", turn, (int) atoms.size(), ndims, state->periodic[0], state->periodic[1], state->periodic[2]);
	outFile << buffer;
	sprintf(buffer, "<bounds base64=\"0\" xlo=\"%f\" ylo=\"%f\" zlo=\"%f\" sxx=\"%f\" sxy=\"%f\" sxz=\"%f\" syx=\"%f\" syy=\"%f\" syz=\"%f\" szx=\"%f\" szy=\"%f\" szz=\"%f\"/>\n", (double) b.lo[0], (double) b.lo[1], (double) b.lo[2], s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]);
	outFile << buffer;
    writeAtomParams(outFile, state->atomParams);
	//going to store atom params as 
	//<atom_params num_types="x" force_type="type">
	//<handle> ... </handle>
	//<sigma>2d array mapped to 1d </sigma>
	//<epsilon> ... </epsilon>
	//<mass> ... </mass>
	//</atom_params>

    //looping over fixes                                                                                                                                    
    outFile << "<fixes>\n";
    for(Fix *f : state->fixes) {
      char buffer[BUFFERLEN];
      sprintf(buffer, "<fix type=\"%s\" handle=\"%s\">", f->type.c_str(), f->handle.c_str());
      outFile << buffer << "\n";
      outFile << f->restartChunk("xml");
      outFile << "</fix>\n";
    }
    outFile << "</fixes>\n";


	writeXMLChunk<Atom>(outFile, atoms, "position", [] (Atom &a, char buffer[BUFFERLEN]) {
			Vector pos = a.pos; sprintf(buffer, "%f %f %f\n", (double) pos[0], (double) pos[1], (double) pos[2]);
			}
			);

	writeXMLChunk<Atom>(outFile, atoms, "velocity", [] (Atom &a, char buffer[BUFFERLEN]) {
			Vector vel = a.vel; sprintf(buffer, "%f %f %f\n", (double) vel[0], (double) vel[1], (double) vel[2]);
			}
			);
	writeXMLChunk<Atom>(outFile, atoms, "force", [] (Atom &a, char buffer[BUFFERLEN]) {
			Vector force = a.force; sprintf(buffer, "%f %f %f\n", (double) force[0], (double) force[1], (double) force[2]);
			}
			);

	writeXMLChunk<Atom>(outFile, atoms, "forceLast", [] (Atom &a, char buffer[BUFFERLEN]) {
			Vector forceLast = a.forceLast; sprintf(buffer, "%f %f %f\n", (double) forceLast[0], (double) forceLast[1], (double) forceLast[2]);
			}
			);
	writeXMLChunk<Atom>(outFile, atoms, "groupTag", [] (Atom &a, char buffer[BUFFERLEN]) {
			sprintf(buffer, "%u\n", a.groupTag);
			}
			);
	writeXMLChunk<Atom>(outFile, atoms, "type", [] (Atom &a, char buffer[BUFFERLEN]) {
			sprintf(buffer, "%d\n", a.type);
			}
			);
	writeXMLChunk<Atom>(outFile, atoms, "id", [] (Atom &a, char buffer[BUFFERLEN]) {
			sprintf(buffer, "%d\n", a.id);
			}
			);
	sprintf(buffer, "</configuration>\n");
	outFile << buffer;
    if (oneFilePerWrite) {
        outFile << "/<data>" << endl;
    }
	outFile.close();

	/*
	   Vector lo = state->bounds->lo;
	   Vector hi = state->bounds->hi;
	FILE *f = fopen(fnFinal, "a+");
	fprintf(f, "Turn %d\n", state->turn);
	fprintf(f, "Bounds %f %f %f %f %f %f\n", lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
	fprintf(f, "Atoms\n");
	for (Atom &a : state->atoms) {
		fprintf(f, "%d %d %f %f %f\n", a.id, a.type, a.pos[0], a.pos[1], a.pos[2]);
	}
	fprintf(f, "end Atoms\n");
	fprintf(f, "Bonds\n");
	for (Bond &b : state->bonds) {
		fprintf(f, "%d %d %f %f %f %f %f %f\n", b.atoms[0]->id, b.atoms[1]->id, b.atoms[0]->pos[0], b.atoms[0]->pos[1], b.atoms[0]->pos[2], b.atoms[1]->pos[0], b.atoms[1]->pos[1], b.atoms[1]->pos[2]);
	}
	fprintf(f, "end Bonds\n");
	fclose(f);
*/
}




string WriteConfig::getCurrentFn(int64_t turn) {
    char buffer[200];
    if (format == "base64") {
        sprintf(buffer, "%s.xml", fn.c_str());
    } else if (format == "xyz" ) {
        sprintf(buffer, "%s.xyz", fn.c_str());
    } else {
        sprintf(buffer, "%s.xml", fn.c_str());
    }

    if (oneFilePerWrite) {
        string asStr = string(buffer);
        string turnStr = to_string(turn);
        size_t pos = asStr.find("*");
        assert(pos != string::npos);
        string finalFn = asStr.substr(0, pos) + turnStr + asStr.substr(pos+1, asStr.size());
        return finalFn;



    }
    return string(buffer);
}

WriteConfig::WriteConfig(SHARED(State) state_, string fn_, string handle_, string format_, int writeEvery_) : state(state_.get()), fn(fn_), handle(handle_), format(format_), writeEvery(writeEvery_), unitLen(1) {
    if (format == "base64") {
        writeFormat = &writeXMLfileBase64;
        isXML = true;
    } else if (format == "xyz") {
        writeFormat = &writeXYZFile;
        isXML = false;
    } else {
        writeFormat = &writeXMLfile;
        isXML = true;
    }

    if (fn.find("*") != string::npos) {
        oneFilePerWrite = true;
    } else {
        oneFilePerWrite = false;
        string fn = getCurrentFn(0);
        unlink(fn.c_str());
        if (isXML) {
            ofstream outFile;
            outFile.open(fn.c_str(), ofstream::app);
            outFile << "<data>\n";
        }
    }
}

void WriteConfig::finish() {
    if (isXML and not oneFilePerWrite) {
        ofstream outFile;
        outFile.open(getCurrentFn(0), ofstream::app);
        outFile << "</data>";
    }
}
void WriteConfig::write(int64_t turn) {
	writeFormat(state, getCurrentFn(turn), turn, oneFilePerWrite, unitLen);
}


void export_WriteConfig() {
    boost::python::class_<WriteConfig,
                          SHARED(WriteConfig) >(
        "WriteConfig",
        boost::python::init<SHARED(State), string, string, string, int>(
            boost::python::args("fn", "handle", "format", "writeEvery"))
    )
    .def_readwrite("writeEvery", &WriteConfig::writeEvery)
    .def_readonly("handle", &WriteConfig::handle)
    .def_readwrite("unitLen", &WriteConfig::unitLen);
    ;
}
