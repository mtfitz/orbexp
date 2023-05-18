#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#include "orb.hpp"

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Please give a filename" << std::endl;
        return 1;
    }
    std::string filename(argv[1]);

    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    std::cout << "(" << img.cols << "," << img.rows << ")" << std::endl;
    //std::cout << img.type() << std::endl;

    cv::Mat imgf;
    img.convertTo(imgf, CV_32F, 1.0f / 255);
    // auto keypoints = fast_detector(imgf, 0.35f);

    Orb orb(imgf);
    orb.compute(0.35f);
    auto keypoints = orb.get_keypoints();

    std::cout << "Number of keypoints: " << keypoints.size() << std::endl;

    cv::Mat out;
    cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
    for (auto& entry : keypoints) {
        auto pt = entry.point;
        auto theta = entry.theta;
        std::cout << "(" << pt.x << "," << pt.y << ") theta=" << theta << std::endl;
        cv::circle(out, pt, 4, {255, 0, 0}, cv::FILLED);
    }
    cv::imwrite("fast_" + filename, out);

    return 0;
}