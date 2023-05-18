#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/mat.hpp>
#include <vector>
#include <cmath>
#include <bitset>

#define DEFAULT_THRESHOLD 0.1f

struct Marker {
    cv::Point point;
    float theta;

    Marker(cv::Point p, float t) : point(p), theta(t) {}
};

typedef std::bitset<128> Descriptor;

class Orb {
    cv::Mat img;
    std::vector<Marker> kpts;
    std::vector<Descriptor> descs;

    std::vector<Marker> fast_detector(cv::Mat& img, float threshold, int pixel_string_len = 12);
    void fast_generate_circle(cv::Mat& img, size_t i, size_t j, float out[16]);
    float fast_get_orientation(cv::Mat& img, size_t i, size_t j, int radius = 3);

    //Descriptor brief_generate(size_t i, size_t j, float theta);

public:
    Orb(cv::Mat& img, bool precompute = false);

    void compute(float threshold = DEFAULT_THRESHOLD);

    std::vector<cv::Point> match(Orb& other);

    std::vector<Marker> get_keypoints();
};