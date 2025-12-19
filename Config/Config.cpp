#include "Config.h"
#include <fstream>
#include <sstream>

bool Config::Load(const std::string& filename)
{
    std::ifstream in(filename);
    if (!in.is_open())
        return false;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;

        auto pos = line.find('=');
        if (pos == std::string::npos)
            continue;

        std::string key = line.substr(0, pos);
        std::string val = line.substr(pos + 1);

        // trim
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        val.erase(0, val.find_first_not_of(" \t"));
        val.erase(val.find_last_not_of(" \t") + 1);

        if (key == "Pub_Key_x") pub_key_x = val;
        else if (key == "target_prefix") target_prefix = val;
        else if (key == "d_min") d_min = val;
        else if (key == "d_max") d_max = val;
        else if (key == "num_iters") num_iters = std::stoull(val);
        else if (key == "num_threads") num_threads = std::stoul(val);
        else if (key == "mode") mode = val;
    }

    return true;
}
