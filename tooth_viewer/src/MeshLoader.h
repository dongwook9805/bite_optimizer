#ifndef MESHLOADER_H
#define MESHLOADER_H

#include "Mesh.h"
#include <string>
#include <memory>

class MeshLoader {
public:
    static std::unique_ptr<Mesh> load(const std::string& filepath);
    static bool save(const Mesh& mesh, const std::string& filepath);

private:
    static std::unique_ptr<Mesh> loadOBJ(const std::string& filepath);
    static std::unique_ptr<Mesh> loadPLY(const std::string& filepath);
    static std::unique_ptr<Mesh> loadSTL(const std::string& filepath);

    static bool saveOBJ(const Mesh& mesh, const std::string& filepath);
    static bool savePLY(const Mesh& mesh, const std::string& filepath);

    static std::string getExtension(const std::string& filepath);
    static std::string toLower(const std::string& str);
};

#endif // MESHLOADER_H
