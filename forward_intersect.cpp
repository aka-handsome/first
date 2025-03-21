#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <map>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <Eigen/Dense>

namespace fs = std::filesystem;
// 定义图像信息结构体
struct ImageInfo {
    std::string tifPath;
    std::string oriPath;
    cv::Mat cameraMatrix;    // 相机内参矩阵
    cv::Mat rotationMatrix;  // 旋转矩阵
    cv::Mat translationVector; // 平移向量
};

// 定义匹配点结构体
struct MatchPoint {
    int img1_id;
    double img1_x;
    double img1_y;
    int img2_id;
    double img2_x;
    double img2_y;
    double distance;
};

// 定义3D点结构体
struct Point3D {
    double x;
    double y;
    double z;
    std::vector<std::pair<int, cv::Point2f>> observations; // 图像ID和对应的2D点
};

// 读取ori文件中的相机参数
bool readOriFile(const std::string& oriPath, ImageInfo& imageInfo) {
    std::ifstream file(oriPath);
    if (!file.is_open()) {
        std::cerr << "无法打开ori文件: " << oriPath << std::endl;
        return false;
    }

    // 初始化相机矩阵
    imageInfo.cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    imageInfo.rotationMatrix = cv::Mat::eye(3, 3, CV_64F);
    imageInfo.translationVector = cv::Mat::zeros(3, 1, CV_64F);

    std::string line;
    int lineCount = 0;
    bool readingCameraMatrix = false;
    bool readingRotationMatrix = false;
    bool readingTranslationVector = false;
    int matrixRow = 0;

    while (std::getline(file, line)) {
        lineCount++;
        
        // 跳过空行
        if (line.empty()) continue;
        
        // 检查是否开始读取相机矩阵
        if (line.find("$IntOri_CameraMatrix") != std::string::npos) {
            readingCameraMatrix = true;
            matrixRow = 0;
            continue;
        }
        
        // 检查是否开始读取旋转矩阵
        if (line.find("$ExtOri_RotationMatrix") != std::string::npos) {
            readingCameraMatrix = false;
            readingRotationMatrix = true;
            matrixRow = 0;
            continue;
        }
        
        // 检查是否开始读取平移向量
        if (line.find("$ExtOri_TranslationVector") != std::string::npos) {
            readingRotationMatrix = false;
            readingTranslationVector = true;
            continue;
        }
        
        // 如果遇到新的参数段，停止之前的读取
        if (line[0] == '$') {
            readingCameraMatrix = false;
            readingRotationMatrix = false;
            readingTranslationVector = false;
            continue;
        }
        
        // 读取相机矩阵
        if (readingCameraMatrix && matrixRow < 3) {
            double v1, v2, v3;
            if (sscanf(line.c_str(), "%lf %lf %lf", &v1, &v2, &v3) == 3) {
                imageInfo.cameraMatrix.at<double>(matrixRow, 0) = v1;
                imageInfo.cameraMatrix.at<double>(matrixRow, 1) = v2;
                imageInfo.cameraMatrix.at<double>(matrixRow, 2) = v3;
                matrixRow++;
            }
        }
        
        // 读取旋转矩阵
        if (readingRotationMatrix && matrixRow < 3) {
            double v1, v2, v3;
            if (sscanf(line.c_str(), "%lf %lf %lf", &v1, &v2, &v3) == 3) {
                imageInfo.rotationMatrix.at<double>(matrixRow, 0) = v1;
                imageInfo.rotationMatrix.at<double>(matrixRow, 1) = v2;
                imageInfo.rotationMatrix.at<double>(matrixRow, 2) = v3;
                matrixRow++;
            }
        }
        
        // 读取平移向量
        if (readingTranslationVector) {
            double v1, v2, v3;
            if (sscanf(line.c_str(), "%lf %lf %lf", &v1, &v2, &v3) == 3) {
                imageInfo.translationVector.at<double>(0, 0) = v1;
                imageInfo.translationVector.at<double>(1, 0) = v2;
                imageInfo.translationVector.at<double>(2, 0) = v3;
                readingTranslationVector = false;
            }
        }
    }

    file.close();
    return true;
}

// 查找data目录中的所有ori文件
std::vector<ImageInfo> findImageFiles(const std::string& dataDir) {
    std::vector<ImageInfo> imageInfos;
    std::vector<std::string> tifFiles;
    std::vector<std::string> oriFiles;

    // 检查目录是否存在
    if (!fs::exists(dataDir)) {
        std::cerr << "数据目录不存在: " << dataDir << std::endl;
        return imageInfos;
    }

    // 收集所有tif和ori文件
    for (const auto& entry : fs::directory_iterator(dataDir)) {
        std::string path = entry.path().string();
        std::string ext = entry.path().extension().string();
        
        if (ext == ".tif") {
            tifFiles.push_back(path);
        } else if (ext == ".ori") {
            oriFiles.push_back(path);
        }
    }

    // 匹配tif和ori文件
    for (const auto& tifPath : tifFiles) {
        std::string baseName = fs::path(tifPath).stem().string();
        std::string oriPath;
        
        // 查找对应的ori文件
        for (const auto& oriFile : oriFiles) {
            if (oriFile.find(baseName) != std::string::npos) {
                oriPath = oriFile;
                break;
            }
        }
        
        if (!oriPath.empty()) {
            ImageInfo info;
            info.tifPath = tifPath;
            info.oriPath = oriPath;
            
            // 读取ori文件
            if (readOriFile(oriPath, info)) {
                imageInfos.push_back(info);
            }
        }
    }

    return imageInfos;
}

// 读取匹配点文件
std::vector<MatchPoint> readMatchesFile(const std::string& matchesFile) {
    std::vector<MatchPoint> matches;
    std::ifstream file(matchesFile);
    
    if (!file.is_open()) {
        std::cerr << "无法打开匹配点文件: " << matchesFile << std::endl;
        return matches;
    }
    
    std::string line;
    // 跳过CSV头
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        MatchPoint match;
        int match_id;
        char comma;
        std::istringstream iss(line);
        
        iss >> match_id >> comma
            >> match.img1_id >> comma
            >> match.img1_x >> comma
            >> match.img1_y >> comma
            >> match.img2_id >> comma
            >> match.img2_x >> comma
            >> match.img2_y >> comma
            >> match.distance;
            
        if (!iss.fail()) {
            matches.push_back(match);
        }
    }
    
    file.close();
    return matches;
}

// 查找所有匹配文件
std::vector<std::string> findMatchesFiles(const std::string& outputDir) {
    std::vector<std::string> matchesFiles;
    
    // 检查目录是否存在
    if (!fs::exists(outputDir)) {
        std::cerr << "匹配结果目录不存在: " << outputDir << std::endl;
        return matchesFiles;
    }
    
    for (const auto& entry : fs::directory_iterator(outputDir)) {
        std::string path = entry.path().string();
        if (path.find("_matches.csv") != std::string::npos) {
            matchesFiles.push_back(path);
        }
    }
    
    return matchesFiles;
}

// 使用线性三角测量法计算3D点
Point3D triangulatePoint(const ImageInfo& img1, const ImageInfo& img2, 
                         const cv::Point2f& pt1, const cv::Point2f& pt2) {
    // 构建投影矩阵
    cv::Mat P1(3, 4, CV_64F);
    cv::Mat P2(3, 4, CV_64F);
    
    // P = K[R|t]
    cv::Mat Rt1, Rt2;
    cv::hconcat(img1.rotationMatrix, img1.translationVector, Rt1);
    cv::hconcat(img2.rotationMatrix, img2.translationVector, Rt2);
    
    P1 = img1.cameraMatrix * Rt1;
    P2 = img2.cameraMatrix * Rt2;
    
    // 构建DLT方程
    cv::Mat A(4, 4, CV_64F);
    
    A.row(0) = pt1.x * P1.row(2) - P1.row(0);
    A.row(1) = pt1.y * P1.row(2) - P1.row(1);
    A.row(2) = pt2.x * P2.row(2) - P2.row(0);
    A.row(3) = pt2.y * P2.row(2) - P2.row(1);
    
    // 使用SVD求解
    cv::Mat u, w, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::FULL_UV);
    
    // 最小奇异值对应的右奇异向量就是解
    cv::Mat X = vt.row(3).t();
    
    // 转换为非齐次坐标
    X = X / X.at<double>(3, 0);
    
    Point3D point;
    point.x = X.at<double>(0, 0);
    point.y = X.at<double>(1, 0);
    point.z = X.at<double>(2, 0);
    
    return point;
}

// 计算重投影误差
double computeReprojectionError(const ImageInfo& img, const Point3D& point, const cv::Point2f& pt) {
    // 构建3D点
    cv::Mat X(4, 1, CV_64F);
    X.at<double>(0, 0) = point.x;
    X.at<double>(1, 0) = point.y;
    X.at<double>(2, 0) = point.z;
    X.at<double>(3, 0) = 1.0;
    
    // 构建投影矩阵
    cv::Mat P(3, 4, CV_64F);
    cv::Mat Rt;
    cv::hconcat(img.rotationMatrix, img.translationVector, Rt);
    P = img.cameraMatrix * Rt;
    
    // 投影
    cv::Mat x = P * X;
    x = x / x.at<double>(2, 0);
    
    // 计算误差
    double dx = x.at<double>(0, 0) - pt.x;
    double dy = x.at<double>(1, 0) - pt.y;
    
    return std::sqrt(dx*dx + dy*dy);
}

// 导出3D点到文件
void export3DPoints(const std::vector<Point3D>& points, const std::string& outputFile) {
    std::ofstream file(outputFile);
    if (!file.is_open()) {
        std::cerr << "无法创建输出文件: " << outputFile << std::endl;
        return;
    }
    
    // 写入CSV头
    file << "id,x,y,z,num_observations" << std::endl;
    
    // 写入3D点数据
    for (size_t i = 0; i < points.size(); i++) {
        const auto& point = points[i];
        file << i << ","
             << point.x << ","
             << point.y << ","
             << point.z << ","
             << point.observations.size() << std::endl;
    }
    
    file.close();
    std::cout << "3D点已导出到: " << outputFile << std::endl;
}

// 导出3D点和观测点的对应关系
void exportPointObservations(const std::vector<Point3D>& points, const std::string& outputFile) {
    std::ofstream file(outputFile);
    if (!file.is_open()) {
        std::cerr << "无法创建输出文件: " << outputFile << std::endl;
        return;
    }
    
    // 写入CSV头
    file << "point_id,image_id,x,y" << std::endl;
    
    // 写入观测数据
    for (size_t i = 0; i < points.size(); i++) {
        const auto& point = points[i];
        for (const auto& obs : point.observations) {
            file << i << ","
                 << obs.first << ","
                 << obs.second.x << ","
                 << obs.second.y << std::endl;
        }
    }
    
    file.close();
    std::cout << "观测点对应关系已导出到: " << outputFile << std::endl;
}

// 导出点云为PLY格式
void exportPointCloudPLY(const std::vector<Point3D>& points, const std::string& outputFile) {
    std::ofstream file(outputFile);
    if (!file.is_open()) {
        std::cerr << "无法创建PLY文件: " << outputFile << std::endl;
        return;
    }
    
    // 写入PLY头
    file << "ply" << std::endl;
    file << "format ascii 1.0" << std::endl;
    file << "element vertex " << points.size() << std::endl;
    file << "property float x" << std::endl;
    file << "property float y" << std::endl;
    file << "property float z" << std::endl;
    file << "property uchar red" << std::endl;
    file << "property uchar green" << std::endl;
    file << "property uchar blue" << std::endl;
    file << "end_header" << std::endl;
    
    // 写入点数据
    for (const auto& point : points) {
        // 根据观测点数量设置颜色
        int color = std::min(255, static_cast<int>(point.observations.size() * 50));
        file << point.x << " " << point.y << " " << point.z << " "
             << color << " " << 0 << " " << 255 - color << std::endl;
    }
    
    file.close();
    std::cout << "点云已导出为PLY格式: " << outputFile << std::endl;
}

int main(int argc, char** argv) {
    // 设置数据目录和输出目录
    std::string dataDir = "/home/handsome/code/ImageMatcher/data";
    std::string matchesDir = "/home/handsome/code/ImageMatcher/output";
    std::string outputDir = "./output";
    
    // 如果命令行提供了参数，则使用命令行参数
    if (argc > 1) {
        dataDir = argv[1];
    }
    if (argc > 2) {
        matchesDir = argv[2];
    }
    if (argc > 3) {
        outputDir = argv[3];
    }
    
    std::cout << "数据目录: " << dataDir << std::endl;
    std::cout << "匹配结果目录: " << matchesDir << std::endl;
    std::cout << "输出目录: " << outputDir << std::endl;
    
    // 确保输出目录存在
    fs::create_directories(outputDir);
    
    // 读取图像信息
    std::vector<ImageInfo> imageInfos = findImageFiles(dataDir);
    std::cout << "找到 " << imageInfos.size() << " 个图像和ori文件" << std::endl;
    
    if (imageInfos.empty()) {
        std::cerr << "未找到有效的图像文件，程序退出" << std::endl;
        return 1;
    }
    
    // 创建图像ID到索引的映射
    std::map<std::string, int> imageNameToIndex;
    for (size_t i = 0; i < imageInfos.size(); i++) {
        std::string baseName = fs::path(imageInfos[i].tifPath).stem().string();
        // 提取基本名称，去掉可能的后缀
        size_t pos = baseName.find("_");
        if (pos != std::string::npos) {
            baseName = baseName.substr(0, pos);
        }
        std::cout << "图像映射: " << baseName << " -> " << i << std::endl;
        imageNameToIndex[baseName] = i;
    }
    
    // 查找所有匹配文件
    std::vector<std::string> matchesFiles = findMatchesFiles(matchesDir);
    std::cout << "找到 " << matchesFiles.size() << " 个匹配文件" << std::endl;
    
    if (matchesFiles.empty()) {
        std::cerr << "未找到匹配文件，程序退出" << std::endl;
        return 1;
    }
    
    // 存储所有3D点
    std::vector<Point3D> points3D;
    
    // 处理每个匹配文件
    for (const auto& matchesFile : matchesFiles) {
        std::cout << "处理匹配文件: " << matchesFile << std::endl;
        
        // 从文件名中提取图像名称
        std::string fileName = fs::path(matchesFile).filename().string();
        
        // 解析文件名格式: 20130424121540_61005_20130424121525_61005_matches.csv
        size_t firstUnderscore = fileName.find("_");
        if (firstUnderscore == std::string::npos) continue;
        
        // 提取第一个图像名称 (例如: 20130424121540)
        std::string img1Name = fileName.substr(0, firstUnderscore);
        
        // 查找第三个下划线的位置（跳过 "_61005_"）
        size_t secondUnderscore = fileName.find("_", firstUnderscore + 1);
        if (secondUnderscore == std::string::npos) continue;
        
        size_t thirdUnderscore = fileName.find("_", secondUnderscore + 1);
        if (thirdUnderscore == std::string::npos) continue;
        
        // 提取第二个图像名称 (例如: 20130424121525)
        std::string img2Name = fileName.substr(secondUnderscore + 1, thirdUnderscore - secondUnderscore - 1);
        
        std::cout << "图像对: " << img1Name << " - " << img2Name << std::endl;
        
        // 查找对应的图像索引
        if (imageNameToIndex.find(img1Name) == imageNameToIndex.end() ||
            imageNameToIndex.find(img2Name) == imageNameToIndex.end()) {
            std::cerr << "无法找到对应的图像: " << img1Name << " 或 " << img2Name << std::endl;
            continue;
        }
        
        int img1Index = imageNameToIndex[img1Name];
        int img2Index = imageNameToIndex[img2Name];
        
        // 读取匹配点
        std::vector<MatchPoint> matches = readMatchesFile(matchesFile);
        std::cout << "读取到 " << matches.size() << " 对匹配点" << std::endl;
        
        // 对每对匹配点进行前方交会
        for (const auto& match : matches) {
            cv::Point2f pt1(match.img1_x, match.img1_y);
            cv::Point2f pt2(match.img2_x, match.img2_y);
            
            // 使用线性三角测量法计算3D点
            Point3D point = triangulatePoint(imageInfos[img1Index], imageInfos[img2Index], pt1, pt2);
            
            // 计算重投影误差
            double err1 = computeReprojectionError(imageInfos[img1Index], point, pt1);
            double err2 = computeReprojectionError(imageInfos[img2Index], point, pt2);
            double avgErr = (err1 + err2) / 2.0;
            
            // 如果重投影误差太大，跳过这个点
            if (avgErr > 5.0) {
                continue;
            }
            
            // 添加观测点
            point.observations.push_back(std::make_pair(img1Index, pt1));
            point.observations.push_back(std::make_pair(img2Index, pt2));
            
            // 添加到3D点集合
            points3D.push_back(point);
        }
    }
    
    std::cout << "共计算出 " << points3D.size() << " 个3D点" << std::endl;
    
    // 导出3D点
    export3DPoints(points3D, outputDir + "/points3D.csv");
    
    // 导出观测点对应关系
    exportPointObservations(points3D, outputDir + "/point_observations.csv");
    
    // 导出点云
    exportPointCloudPLY(points3D, outputDir + "/point_cloud.ply");
    
    std::cout << "前方交会完成！" << std::endl;
    return 0;
} 