#include "xml_func.h"



vector<string> xml_readStrings(pugi::xml_node &parent, string tag) {
	auto child = parent.child(tag.c_str());
	if (child) {
		vector<string> res;
		istringstream ss(child.first_child().value());
		string s;
		while (ss >> s) {
			res.push_back(s);
		}
		return res;
	}
	return vector<string>();
}
