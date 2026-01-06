#include "MeshLoader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <cstring>

std::string MeshLoader::getExtension(const std::string& filepath)
{
    size_t pos = filepath.rfind('.');
    if (pos == std::string::npos) return "";
    return toLower(filepath.substr(pos + 1));
}

std::string MeshLoader::toLower(const std::string& str)
{
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::unique_ptr<Mesh> MeshLoader::load(const std::string& filepath)
{
    std::string ext = getExtension(filepath);

    if (ext == "obj") {
        return loadOBJ(filepath);
    } else if (ext == "ply") {
        return loadPLY(filepath);
    } else if (ext == "stl") {
        return loadSTL(filepath);
    }

    std::cerr << "Unsupported file format: " << ext << std::endl;
    return nullptr;
}

std::unique_ptr<Mesh> MeshLoader::loadOBJ(const std::string& filepath)
{
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << filepath << std::endl;
        return nullptr;
    }

    size_t fileSize = file.tellg();
    file.seekg(0);
    
    std::string content(fileSize, '\0');
    file.read(&content[0], fileSize);
    file.close();

    auto mesh = std::make_unique<Mesh>();
    
    size_t estimatedVerts = fileSize / 50;
    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Vector3f> colors;
    positions.reserve(estimatedVerts);
    colors.reserve(estimatedVerts);

    const char* ptr = content.data();
    const char* end = ptr + content.size();
    
    while (ptr < end) {
        while (ptr < end && (*ptr == ' ' || *ptr == '\t' || *ptr == '\r' || *ptr == '\n')) ptr++;
        if (ptr >= end) break;
        
        if (*ptr == 'v' && (ptr[1] == ' ' || ptr[1] == '\t')) {
            ptr += 2;
            float x, y, z, r = 0.8f, g = 0.8f, b = 0.8f;
            char* next;
            x = strtof(ptr, &next); ptr = next;
            y = strtof(ptr, &next); ptr = next;
            z = strtof(ptr, &next); ptr = next;
            
            while (ptr < end && (*ptr == ' ' || *ptr == '\t')) ptr++;
            if (ptr < end && *ptr != '\n' && *ptr != '\r' && *ptr != '#') {
                r = strtof(ptr, &next); 
                if (next != ptr) {
                    ptr = next;
                    g = strtof(ptr, &next); ptr = next;
                    b = strtof(ptr, &next); ptr = next;
                }
            }
            
            positions.emplace_back(x, y, z);
            colors.emplace_back(r, g, b);
        }
        else if (*ptr == 'f' && (ptr[1] == ' ' || ptr[1] == '\t')) {
            ptr += 2;
            unsigned int indices[16];
            int count = 0;
            
            while (ptr < end && *ptr != '\n' && *ptr != '\r' && count < 16) {
                while (ptr < end && (*ptr == ' ' || *ptr == '\t')) ptr++;
                if (ptr >= end || *ptr == '\n' || *ptr == '\r') break;
                
                char* next;
                long idx = strtol(ptr, &next, 10);
                if (next == ptr) break;
                ptr = next;
                
                indices[count++] = (idx > 0) ? (idx - 1) : (positions.size() + idx);
                
                while (ptr < end && *ptr != ' ' && *ptr != '\t' && *ptr != '\n' && *ptr != '\r') ptr++;
            }
            
            for (int i = 1; i + 1 < count; ++i) {
                mesh->addFace(indices[0], indices[i], indices[i + 1]);
            }
        }
        
        while (ptr < end && *ptr != '\n') ptr++;
        if (ptr < end) ptr++;
    }

    for (size_t i = 0; i < positions.size(); ++i) {
        mesh->addVertex(positions[i], Eigen::Vector3f::Zero(), colors[i]);
    }

    mesh->computeNormals();
    mesh->computeBoundingBox();

    std::cout << "Loaded OBJ: " << mesh->vertexCount() << " vertices, "
              << mesh->faceCount() << " faces" << std::endl;

    return mesh;
}

std::unique_ptr<Mesh> MeshLoader::loadPLY(const std::string& filepath)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << filepath << std::endl;
        return nullptr;
    }

    auto mesh = std::make_unique<Mesh>();
    std::string line;
    int vertexCount = 0;
    int faceCount = 0;
    bool isBinary = false;
    bool hasColors = false;
    bool hasLabels = false;

    // Track property order for proper parsing
    std::vector<std::string> vertexProperties;

    // Parse header
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "format") {
            std::string format;
            iss >> format;
            isBinary = (format != "ascii");
        }
        else if (token == "element") {
            std::string elemType;
            int count;
            iss >> elemType >> count;
            if (elemType == "vertex") vertexCount = count;
            else if (elemType == "face") faceCount = count;
        }
        else if (token == "property") {
            std::string propType, propName;
            iss >> propType >> propName;
            if (propName == "red" || propName == "r") hasColors = true;
            if (propName == "label") hasLabels = true;
            vertexProperties.push_back(propName);
        }
        else if (token == "end_header") {
            break;
        }
    }

    // Read vertices (ASCII only for now)
    mesh->reserveLabels(vertexCount);
    for (int i = 0; i < vertexCount; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);

        float x, y, z;
        iss >> x >> y >> z;

        Eigen::Vector3f color(0.8f, 0.8f, 0.8f);
        int label = 0;

        if (hasColors) {
            float r, g, b;
            iss >> r >> g >> b;
            color = Eigen::Vector3f(r / 255.0f, g / 255.0f, b / 255.0f);
        }

        if (hasLabels) {
            iss >> label;
        }

        mesh->addVertex(Eigen::Vector3f(x, y, z), Eigen::Vector3f::Zero(), color);
        if (hasLabels) {
            mesh->setLabel(i, label);
        }
    }

    // Read faces
    for (int i = 0; i < faceCount; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);

        int numVerts;
        iss >> numVerts;

        std::vector<unsigned int> indices(numVerts);
        for (int j = 0; j < numVerts; ++j) {
            iss >> indices[j];
        }

        // Triangulate
        for (int j = 1; j + 1 < numVerts; ++j) {
            mesh->addFace(indices[0], indices[j], indices[j + 1]);
        }
    }

    mesh->computeNormals();
    mesh->computeBoundingBox();

    std::cout << "Loaded PLY: " << mesh->vertexCount() << " vertices, "
              << mesh->faceCount() << " faces";
    if (hasLabels) {
        std::cout << ", with labels";
    }
    std::cout << std::endl;

    return mesh;
}

std::unique_ptr<Mesh> MeshLoader::loadSTL(const std::string& filepath)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << filepath << std::endl;
        return nullptr;
    }

    auto mesh = std::make_unique<Mesh>();

    // Check if ASCII or binary
    char header[80];
    file.read(header, 80);

    uint32_t numTriangles;
    file.read(reinterpret_cast<char*>(&numTriangles), 4);

    // Binary STL
    for (uint32_t i = 0; i < numTriangles; ++i) {
        float normal[3], v1[3], v2[3], v3[3];
        uint16_t attr;

        file.read(reinterpret_cast<char*>(normal), 12);
        file.read(reinterpret_cast<char*>(v1), 12);
        file.read(reinterpret_cast<char*>(v2), 12);
        file.read(reinterpret_cast<char*>(v3), 12);
        file.read(reinterpret_cast<char*>(&attr), 2);

        unsigned int baseIdx = mesh->vertexCount();

        Eigen::Vector3f n(normal[0], normal[1], normal[2]);
        mesh->addVertex(Eigen::Vector3f(v1[0], v1[1], v1[2]), n);
        mesh->addVertex(Eigen::Vector3f(v2[0], v2[1], v2[2]), n);
        mesh->addVertex(Eigen::Vector3f(v3[0], v3[1], v3[2]), n);

        mesh->addFace(baseIdx, baseIdx + 1, baseIdx + 2);
    }

    mesh->computeBoundingBox();

    std::cout << "Loaded STL: " << mesh->vertexCount() << " vertices, "
              << mesh->faceCount() << " faces" << std::endl;

    return mesh;
}

bool MeshLoader::save(const Mesh& mesh, const std::string& filepath)
{
    std::string ext = getExtension(filepath);

    if (ext == "obj") {
        return saveOBJ(mesh, filepath);
    } else if (ext == "ply") {
        return savePLY(mesh, filepath);
    }

    std::cerr << "Unsupported export format: " << ext << std::endl;
    return false;
}

bool MeshLoader::saveOBJ(const Mesh& mesh, const std::string& filepath)
{
    std::ofstream file(filepath);
    if (!file.is_open()) return false;

    file << "# ToothViewer OBJ Export\n";

    // Write vertices with colors
    for (const auto& v : mesh.vertices()) {
        file << "v " << v.position.x() << " " << v.position.y() << " " << v.position.z()
             << " " << v.color.x() << " " << v.color.y() << " " << v.color.z() << "\n";
    }

    // Write faces (1-indexed)
    for (const auto& f : mesh.faces()) {
        file << "f " << (f.v0 + 1) << " " << (f.v1 + 1) << " " << (f.v2 + 1) << "\n";
    }

    return true;
}

bool MeshLoader::savePLY(const Mesh& mesh, const std::string& filepath)
{
    std::ofstream file(filepath);
    if (!file.is_open()) return false;

    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << mesh.vertexCount() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "element face " << mesh.faceCount() << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";

    // Write vertices
    for (const auto& v : mesh.vertices()) {
        int r = static_cast<int>(v.color.x() * 255);
        int g = static_cast<int>(v.color.y() * 255);
        int b = static_cast<int>(v.color.z() * 255);
        file << v.position.x() << " " << v.position.y() << " " << v.position.z()
             << " " << r << " " << g << " " << b << "\n";
    }

    // Write faces
    for (const auto& f : mesh.faces()) {
        file << "3 " << f.v0 << " " << f.v1 << " " << f.v2 << "\n";
    }

    return true;
}
