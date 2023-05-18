#include "orb.hpp"
#include <Eigen/Dense>
#include <Eigen/Geometry>

#define PADDING 8

Orb::Orb(cv::Mat& img, bool precompute)
{
    copyMakeBorder(img, this->img, PADDING, PADDING, PADDING, PADDING, cv::BORDER_REPLICATE, 0);
    std::cout << "(" << this->img.cols << "," << this->img.rows << ")" << std::endl;
    if (precompute) this->compute();
}

void Orb::fast_generate_circle(cv::Mat& img, size_t i, size_t j, float out[16])
{
    out[0] = img.at<float>(i-3,j);
    out[1] = img.at<float>(i-3,j+1);
    out[2] = img.at<float>(i-2,j+2);
    out[3] = img.at<float>(i-1,j+3);
    out[4] = img.at<float>(i,j+3);
    out[5] = img.at<float>(i+1,j+3);
    out[6] = img.at<float>(i+2,j+2);
    out[7] = img.at<float>(i+3,j+1);
    out[8] = img.at<float>(i+3,j);
    out[9] = img.at<float>(i+3,j-1);
    out[10] = img.at<float>(i+2,j-2);
    out[11] = img.at<float>(i+1,j-3);
    out[12] = img.at<float>(i,j-3);
    out[13] = img.at<float>(i-1,j-3);
    out[14] = img.at<float>(i-2,j-2);
    out[15] = img.at<float>(i-3,j-1);
}

float Orb::fast_get_orientation(cv::Mat& img, size_t i, size_t j, int radius)
{
    float m00 = 0.0f, m10 = 0.0f, m01 = 0.0f;
    
    // compute moments
    for (int r = -radius; r <= radius; r++) {
        for (int c = -radius; c <= radius; c++) {
            float lum = img.at<float>(i+r, j+c);
            m00 += lum;
            m10 += (j+c) * lum;
            m01 += (i+r) * lum;
        }
    }

    //cv::Point centroid(m10 / m00, m01 / m00);
    float theta = std::atan2(m01, m10);
    return theta;
}

std::vector<Marker> Orb::fast_detector(cv::Mat& img, float threshold, int pixel_string_len)
{
    std::vector<Marker> keypoints;
    size_t r_begin = PADDING, r_end = img.rows - PADDING, c_begin = PADDING, c_end = img.cols - PADDING;
    for (int i = r_begin; i < r_end; i++) {
        for (int j = c_begin; j < c_end; j++) {
            float lum = img.at<float>(i,j);
            float circle[16];
            fast_generate_circle(img, i, j, circle);
            // find pixel string of length pixel_string_len, each w/ intensities at least lum+threshold
            for (int base = 0; base < 16; base++) {
                bool fail = false;
                // only fail if there is one such pixel with too small/big intensity
                for (int k = 0; k < pixel_string_len; k++) {
                    auto this_lum = circle[(base+k)%16];
                    if ((this_lum <= lum+threshold) && (this_lum >= lum-threshold)) {
                        fail = true;
                        break;
                    }
                }
                // if the string is perfect, i.e. all pixels have viable intensity, then add this point
                if (!fail) {
                    //std::cout << j << "," << i << " -> " << lum << std::endl;
                    float theta = fast_get_orientation(img, i, j);
                    keypoints.emplace_back(cv::Point {j,i}, theta);
                    //descs.push_back(this->brief_generate(i, j, theta));
                    break;
                }
            }
        }
    }
    return keypoints;
}

/*Descriptor Orb::brief_generate(size_t i, size_t j, float theta)
{
    Descriptor desc;
    Eigen::Rotation2D rot(-theta);
    Eigen::Matrix<int, 2, 128> bitmat;
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            Eigen::Vector2d vec(x, y);
            auto newvec = rot * vec;
            bitmat(0, y*8 + x) = newvec(0);
            bitmat(1, y*8 + x) = newvec(1);
        }
    }
}*/

void Orb::compute(float threshold)
{
    this->kpts.clear();
    this->descs.clear();

    // oriented FAST
    this->kpts = this->fast_detector(this->img, threshold);
    
    // rotated BRIEF
    /*this->descs.reserve(this->kpts.size());
    for (auto& k : this->kpts) {
        this->descs.push_back(this->brief_generate(k.point.y, k.point.x, k.theta));
    }*/
}

std::vector<Marker> Orb::get_keypoints()
{
    return this->kpts;
}