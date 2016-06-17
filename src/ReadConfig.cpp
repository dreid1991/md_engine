#include "ReadConfig.h"
#include "State.h"
#include "xml_func.h"
#include "includeFixes.h"
#include <boost/lexical_cast.hpp> //for case string to int64 (turn)

using namespace std;

vector<vector<num> > mapTo2d(vector<num> &xs, const int dim) {
	vector<vector<num> > mapped;
	for (int i=0; i<dim; i++) {
		vector<num> bit;
		bit.reserve(dim);
		for (int j=i*dim; j<(i+1)*dim; j++) {
			bit.push_back(xs[j]);
		}
		mapped.push_back(bit);
	}
	return mapped;
}

void loadAtomParams(pugi::xml_node &config, State *state) {
    state->atomParams.clear();
	AtomParams &params = state->atomParams;
	auto params_xml = config.child("atomParams");

	int numTypes = atoi(params_xml.attribute("numTypes").value());
	string forceType = string(params_xml.attribute("forceType").value());

	params.numTypes = numTypes;
//	params->forceType = forceType;
	vector<double> mass = xml_readNums<double>(params_xml, "mass");
	assert((int) mass.size() == numTypes);
	params.masses = mass;

	vector<string> handle = xml_readStrings(params_xml, "handle");
	assert((int) handle.size() == numTypes);
	params.handles = handle;
/*
	vector<num> sigma = readNums(params_xml, "sigma");
	assert((int) sigma.size() == numTypes * numTypes);
	params->sigmas = mapTo2d(sigma, numTypes);

	vector<num> epsilon = readNums(params_xml, "epsilon");
	assert((int) epsilon.size() == numTypes * numTypes);
	params->epsilons = mapTo2d(epsilon, numTypes);

	params->setParams();
	
*/
	

}


void loadBounds(pugi::xml_node &config, State *state) {
    auto bounds_xml = config.child("bounds");

    if (bounds_xml) {
        auto base64Str = bounds_xml.attribute("base64").value();
        bool base64;


        auto processRaw = [&] (const char *value) { //could do multiple lambdas for each case, but whatever. logic so small
            if (base64) {
                string toDecode = string(value);
                string decoded = base64_decode(toDecode);
                double res = * (double *) decoded.data();
                return res;
            }
            return atof(value);
        };
        if (strcmp(base64Str, "1") == 0) {
            base64 = true;
        } else {
            base64 = false;
        }
        Vector lo, hi;
        lo[0] = processRaw(bounds_xml.attribute("xlo").value());
        lo[1] = processRaw(bounds_xml.attribute("ylo").value());
        lo[2] = processRaw(bounds_xml.attribute("zlo").value());
		auto xhi = bounds_xml.attribute("xhi").value();
		auto sxx = bounds_xml.attribute("sxx").value();
		if (strcmp(xhi, "") != 0) { //using square box
			hi[0] = processRaw(xhi);
			hi[1] = processRaw(bounds_xml.attribute("yhi").value());
			hi[2] = processRaw(bounds_xml.attribute("zhi").value());
            state->bounds = Bounds(state, lo, hi);
		} else if (strcmp(sxx, "") != 0) {
			Vector sides[3];
			sides[0][0] = processRaw(sxx);
			sides[0][1] = processRaw(bounds_xml.attribute("sxy").value());
			sides[0][2] = processRaw(bounds_xml.attribute("sxz").value());
			sides[1][0] = processRaw(bounds_xml.attribute("syx").value());
			sides[1][1] = processRaw(bounds_xml.attribute("syy").value());
			sides[1][2] = processRaw(bounds_xml.attribute("syz").value());
			sides[2][0] = processRaw(bounds_xml.attribute("szx").value());
			sides[2][1] = processRaw(bounds_xml.attribute("szy").value());
			sides[2][2] = processRaw(bounds_xml.attribute("szz").value());
			state->bounds = Bounds(state, lo, sides);
           
		} else {
			assert(strcmp("Tried to load bad bounds data", ""));
		}


	} else {
		cout << "Failed to load bounds from file" << endl;
	}

}
/*
vector<Bond> buildBonds(pugi::xml_node &config, State *state, string tag, int numBonds) {
	vector<Bond> bonds;
	auto bonds_xml = config.child("bond");
	if (bonds_xml) {
		istringstream ss(bonds_xml.first_child().value());
		string line;
		while (ss >> line) {
			Atom *atoms[2];
			atoms[0] = state->atomFromId(atoi(line.c_str()));
			ss >> line;
			atoms[1] = state->atomFromId(atoi(line.c_str()));
			ss >> line;
			num k = atof(line.c_str());
			ss >> line;
			num rEq = atof(line.c_str());
			assert(atoms[0] != (Atom *) NULL and atoms[1] != (Atom *) NULL);
			state->addBond(atoms[0], atoms[1], k, rEq);
		}
	}
	return bonds;
}
*/

bool ReadConfig::read() {
    cout << "READING A CONFIG" << endl;
	//state->deleteBonds();
	state->deleteAtoms();
	vector<Atom> readAtoms;
	int64_t readTurn = boost::lexical_cast<int64_t>(config->attribute("turn").value());
	int numAtoms = boost::lexical_cast<int>(config->attribute("numAtoms").value());
	bool readIs2d = !strcmp(config->attribute("dimension").value(), "2");
	const char *periodic = config->attribute("periodic").value();
    cout << "periodic is " << periodic << endl;
	for (int i=0; i<3; i++) {
		char bit[1];
		bit[0] = periodic[i];
		state->periodic[i] = atoi(bit);
	}
	state->turn = readTurn;
	state->is2d = readIs2d;
	readAtoms.reserve(numAtoms);
	for (int i=0; i<numAtoms; i++) {
		readAtoms.push_back(Atom());
	}
    assert(
            (xml_assign<double, 3>(*config, "position", [&] (int i, double *vals) {
                               readAtoms[i].pos = Vector(vals);	
                               }
                              ))
          ) ;
    assert(
            (xml_assign<double, 3>(*config, "velocity", [&] (int i, double *vals) {
                               readAtoms[i].vel= Vector(vals);	
                               }
                              ))
          ) ;
    assert(
            (xml_assign<double, 3>(*config, "force", [&] (int i, double *vals) {
                               readAtoms[i].force = Vector(vals);	
                               }
                              ))
          ) ;
    assert(
            (xml_assign<unsigned int, 1>(*config, "groupTag", [&] (int i, unsigned int *vals) {
                                     readAtoms[i].groupTag = *vals;
                                     }
                                    ))
          ) ;
    assert(
            (xml_assign<int, 1>(*config, "type", [&] (int i, int *vals) {
                            readAtoms[i].type = *vals;
                            }
                           ))
          ) ;

    assert(
            (xml_assign<int, 1>(*config, "id", [&] (int i, int *vals) {
                            readAtoms[i].id = *vals;
                            }
                           ))
          ) ;

	loadAtomParams(*config, state);
	loadBounds(*config, state);
	for (Atom &a : readAtoms) {
		state->addAtomDirect(a);
	}
//	buildBonds(*config, state.get(), "bond", numBonds);
	return true;

}

bool ReadConfig::next() {
	if (not *config or not haveReadYet) {
		*config = doc->first_child().first_child();
	} else {
		*config = config->next_sibling();
	}
	if (not *config) { //not sure if this works with shared pointer
		return false;
	}
	haveReadYet = true;
	return read();
}


bool ReadConfig::prev() {
	if (not *config or not haveReadYet) {
		*config = doc->first_child().last_child();
	} else {
		*config = config->previous_sibling();
	}
	if (not *config) {
		return false;
	}
	haveReadYet = true;
	return read();
}


bool ReadConfig::moveBy(int by) {
    if (not by) {
        return *config;
    }
    if (not *config or not haveReadYet) {
        *config = doc->first_child().first_child();
        if (by > 0) {
            by --;
        } else {
            return false;
        }

    }
    while (by) {
        if (by > 0) {
            *config = config->next_sibling();
            by --;
        }
        if (by < 0) {
            *config = config->next_sibling();
            by ++;
        }
        if (not *config) {
            return false;
        }
    }
    haveReadYet = true;
    return read();
}

ReadConfig::ReadConfig(State *state_) : state(state_) {};

void ReadConfig::loadFile(string fn_) {
    doc = SHARED(pugi::xml_document) (new pugi::xml_document());
	config = SHARED(pugi::xml_node) (new pugi::xml_node());
	fn = fn_;
	pugi::xml_parse_result result = doc->load_file(fn.c_str());
	if (result != pugi::status_ok) {
	  std::cout << "XML [" << fn << "] parsed with errors\n";
	  std::cout << "Error description: " << result.description() << "\n";
	  std::cout << "Error offset: " << result.offset << "\n\n";
        }
	assert(result.status == pugi::status_ok);
    fileOpen = true;
}

pugi::xml_node ReadConfig::readFix(string type, string handle) {
  if (config) {
    cout << "config exists" << endl;
    auto node = config->child("fixes").first_child();
    while (node) {
      string t = node.attribute("type").value();
      string h = node.attribute("handle").value();
      if (t == type && h == handle) {
        cout << "found the correct node: " << node.attribute("handle").value() << endl;
        return node;
      }
      node = node.next_sibling();
    }
  }
  return pugi::xml_node();
}

pugi::xml_node ReadConfig::readNode(string nodeTag) {
    if (config) {
        auto node = config->child(nodeTag.c_str());
        return node;
    }
    return pugi::xml_node();
}

void export_ReadConfig() {
    boost::python::class_<ReadConfig,
                          SHARED(ReadConfig) >(
        "ReadConfig"
    )
    .def("loadFile", &ReadConfig::loadFile)
    .def("next", &ReadConfig::next)
    .def("prev", &ReadConfig::prev)
    .def("moveBy", &ReadConfig::moveBy)
    ;
}
