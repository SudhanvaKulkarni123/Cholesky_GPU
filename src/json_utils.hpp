#include "json.hpp"
int set_matrix_params(int& n, float& cond, bool& is_symmetric, bool& diag_dom, float& work_factor, nlohmann::json& outer_settings)
{

    string tmp;
    auto settings = outer_settings["matrix settings"];
    tmp = settings["n"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    n = stoi(tmp);

    tmp = settings["condition number"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    cond = stof(tmp);

    tmp = settings["symmetric"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    is_symmetric = (tmp == "true");

    tmp = settings["diag_dom"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    diag_dom = (tmp == "true");

    tmp = settings["work_factor"].dump();
    tmp = tmp.substr(1, tmp.size() - 2);
    work_factor = stof(tmp);


    if(diag_dom) cout << "it is true\n";

    return 0;

}