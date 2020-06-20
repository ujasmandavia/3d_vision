//Lukas Kanade tracker

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

//Just for metioning
using namespace std;
using namespace cv;
using namespace Eigen;

string file_1 = "/home/ujasmandavia/dataset/LK1.png";
string file_2 = "/home/ujasmandavia/dataset/LK2.png";

//Optical flow tracker and interface
class OpticalFlowTracker{
public:
  OpticalFlowTracker(const cv::Mat &img1_, const cv::Mat &img2_,const std::vector<KeyPoint> &kp1_, std::vector<KeyPoint> &kp2_,
                     std::vector<bool> &success_, bool inverse_ = true, bool has_initial_ = false) : img1(img1_), img2(img2_), kp1(kp1_),
                     kp2(kp2_), success(success_), inverse(inverse_), has_initial(has_initial_) {}

  void calculateOpticalFlow(const Range &range);

private:
  const Mat &img1;
  const Mat &img2;
  const vector<KeyPoint> &kp1;
  vector<KeyPoint> &kp2;
  vector<bool> &success;
  bool inverse = true;
  bool has_initial = false;
}; //class

/*
single level optical flow

*/
void OpticalFLowSingleLevel(const Mat &img1, const Mat &img2, const std::vector<KeyPoint> &kp1, std::vector<KeyPoint> &kp2, vector<bool> &success,
                            bool inverse = false, bool has_initial = false);

/*
multi level optical flow using pyramid
*/
void OpticalFLowMultiLevel(const Mat &img1, const Mat &img2, const std::vector<KeyPoint> &kp1, std::vector<KeyPoint> &kp2,
                           vector<bool> &success, bool inverse = false);

/*
BILINEAR INTERPOLATION (Get the gray scale pixel values from reference image)
@param img
@param x
@param y
@return the interpolated values of this pixels
*/

inline float GetPixelValues(const cv::Mat &img, float x, float y){
  //boundary check
  if(x < 0) x = 0;
  if(y < 0) y = 0;
  if(x >= img.cols) x = img.cols - 1;
  if(y >= img.rows) y = img.rows - 1;
  uchar *data = &img.data[int(y) * img.step + int(x)];
  float xx = x - floor(x);
  float yy = y - floor(y);
  return float((1-xx) * (1-yy) * data[0] +
                xx * (1-yy) * data[1] +
                (1-xx) * yy * data[img.step] +
                xx * yy * data[img.step + 1]);
}

int main(int argc, char **argv){

  //images, note they are CV_8UC1 and not CV_8UC3
  Mat img1 = cv::imread(file_1,0);
  Mat img2 = cv::imread(file_2,0);

  //key point, using GFTT here
  std::vector<KeyPoint> kp1;
  Ptr<GFTTDetector> detector = GFTTDetector::create(500,0.01,20);  //maximum 500 keypoints
  detector->detect(img1,kp1);

  //now lets track these key points in the second image
  //first use single level LK in the validation picture
  vector<KeyPoint> kp2_single;
  vector<bool> success_single;
  OpticalFLowSingleLevel(img1, img2, kp1, kp2_single, success_single);

  //then tesy multilevel LK
  vector<KeyPoint> kp2_multi;
  vector<bool> success_multi;
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  OpticalFLowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
  std::cout << "optical flow by gauss-newton: " << time_used.count() << "\n";

  //use opencv's optical flow for validation
  vector<Point2f> pt1,pt2;
  for(auto &kp:kp1) pt1.push_back(kp.pt);
  vector<uchar> status;
  vector<float> error;
  t1 = std::chrono::steady_clock::now();
  cv::calcOpticalFlowPyrLK(img1,img2, pt1, pt2, status, error);
  t2 = std::chrono::steady_clock::now();

  time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
  cout <<"optical flow by opencv: " << time_used.count() << "\n";

  //plot the dfferences of those functions
  Mat img2_single;
  cv::cvtColor(img2, img2_single, CV_GRAY2BGR);
  for(int i=0; i<kp2_single.size(); i++){
    if(success_single[i]){
      cv::circle(img2_single,kp2_single[i].pt, 2, cv::Scalar(0,250,0), 2);
      cv::line(img2_single, kp1[i].pt,kp2_single[i].pt, cv::Scalar(0,250,0));
    }
  }

  Mat img2_multi;
  cv::cvtColor(img2, img2_multi, CV_GRAY2BGR);
  for(int i=0; i<kp2_multi.size(); i++){
    if(success_multi[i]){
      cv::circle(img2_multi,kp2_multi[i].pt, 2, cv::Scalar(0,250,0), 2);
      cv::line(img2_multi, kp1[i].pt,kp2_multi[i].pt, cv::Scalar(0,250,0));
    }
  }

  Mat img2_CV;
  cv::cvtColor(img2, img2_CV, CV_GRAY2BGR);
  for(int i=0; i<pt2.size(); i++){
    if(status[i]){
      cv::circle(img2_CV,pt2[i], 2, cv::Scalar(0,250,0), 2);
      cv::line(img2_CV, pt1[i],pt2[i], cv::Scalar(0,250,0));
    }
  }

  cv::imshow("tracked single level",img2_single);
  cv::imshow("tracked multi level",img2_multi);
  cv::imshow("tracked by opencv",img2_CV);
  cv::waitKey(0);

  return 0;
}

void OpticalFLowSingleLevel(const Mat &img1, const Mat &img2, const std::vector<KeyPoint> &kp1, vector<KeyPoint> &kp2,vector<bool> &success, bool inverse, bool has_initial){
  kp2.resize(kp1.size());
  success.resize(kp1.size());
  OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
  parallel_for_(Range(0,kp1.size()), std::bind(&OpticalFlowTracker::calculateOpticalFlow,&tracker,placeholders::_1));
}

void OpticalFlowTracker::calculateOpticalFlow(const Range &range){
  //parameters
  int half_patch_size = 4;
  int iterations = 10;
  for(size_t i=range.start; i<range.end; i++){
    auto kp = kp1[i];
    double dx = 0, dy = 0;   //dx and dy needs to be calculated (intensity change in x and y direction)
    if(has_initial){
      dx = kp2[i].pt.x - kp.pt.x;
      dy = kp2[i].pt.y - kp.pt.y;
    }

    double cost = 0, lastCost = 0;
    bool succ = true;  //indicate if this point succeded

    //Gauss-Newton iterations
    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();   //hessian matrix
    Eigen::Vector2d b = Eigen::Vector2d::Zero();   //bias
    Eigen::Vector2d J;  //jacobian
    for(int iter=0; iter<iterations; iter++){
      if(inverse == false){
        H = Eigen::Matrix2d::Zero();
        b = Eigen::Vector2d::Zero();
      }else{
        //only reset b
        b = Eigen::Vector2d::Zero();
      }

      cost = 0;

      //compute the cost and the jacobians
      for(int x = -half_patch_size; x< half_patch_size; x++){
        for(int y = -half_patch_size; y< half_patch_size; y++){
          double error = GetPixelValues(img1, kp.pt.x + x, kp.pt.y + y) - GetPixelValues(img2, kp.pt.x + x + dx, kp.pt.y +y +dy);  //jacobian
          if(inverse == false){
            J = -1.0 * Eigen::Vector2d(
                0.5 * (GetPixelValues(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) - GetPixelValues(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                0.5 * (GetPixelValues(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) - GetPixelValues(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
            );
          }else if(iter == 0){
            // in inverse mode, J keeps same for all iterations
            //Note this J doesnot change when dx,dy is updated, so we can store it and only compute error
            J = -1.0 * Eigen::Vector2d(
                0.5 * (GetPixelValues(img1,kp.pt.x + x + 1,kp.pt.y + y) - GetPixelValues(img2,kp.pt.x + x - 1,kp.pt.y + y)),
                0.5 * (GetPixelValues(img1,kp.pt.x + x,kp.pt.y + y + 1) - GetPixelValues(img2,kp.pt.x + x,kp.pt.y + y - 1))
            );
          }
          //compute H,b and set cost
          b += -error*J;
          cost += error*error;
          if(inverse == false || iter == 0){
            //also update H
            H += J*J.transpose();
          }
        }
      }
      //compute update
      Eigen:Vector2d update = H.ldlt().solve(b);

      if(std::isnan(update[0])){
        //sometimes it happens when we have a black or white patch and H is irreversible
        cout << "update is nan(not a number)!\n";
        succ = false;
        break;
      }

      if(iter > 0 && cost > lastCost)
        break;

      //update dx, dy
      dx += update[0];
      dy += update[1];
      lastCost = cost;
      succ = true;

      if(update.norm() < 1e-2)
        break;  //convergence
    }

    success[i] = succ;

    //set kp2
    kp2[i].pt = kp.pt + Point2f(dx,dy);
  }
}

void OpticalFLowMultiLevel(const Mat &img1, const Mat &img2, const vector<KeyPoint> &kp1, vector<KeyPoint> &kp2, vector<bool> &success, bool inverse){
  //parameters
  int pyramids = 4;
  double pyramid_scale = 0.5;
  double scales[] = {1.0,0.5,0.25,0.125};

  //create pyramids
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  vector<Mat> pyr1, pyr2;   //image pyramids
  for(int i=0; i<pyramids; i++){
    if(i==0){
      pyr1.push_back(img1);
      pyr2.push_back(img2);
    }else{
      cv::Mat img1_pyr, img2_pyr;
      cv::resize(pyr1[i-1],img1_pyr, cv::Size(pyr1[i-1].cols * pyramid_scale, pyr1[i-1].rows * pyramid_scale));
      cv::resize(pyr2[i-1],img2_pyr, cv::Size(pyr2[i-1].cols * pyramid_scale, pyr2[i-1].rows * pyramid_scale));
      pyr1.push_back(img1_pyr);
      pyr2.push_back(img2_pyr);
    }
  }
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
  std::cout << "build pyramid time: " << time_used.count() << "\n";

  //coarse-to-fine LK tracking in pyramids
  vector<KeyPoint> kp1_pyr, kp2_pyr;
  for(auto &kp:kp1){
    auto kp_top = kp;
    kp_top.pt *= scales[pyramids-1];
    kp1_pyr.push_back(kp_top);
    kp2_pyr.push_back(kp_top);
  }

  for(int level = pyramids - 1; level >= 0 ; level--){
    //from coarse to fine
    success.clear();
    t1 = std::chrono::steady_clock::now();
    OpticalFLowSingleLevel(pyr1[level],pyr2[level],kp1_pyr,kp2_pyr, success,inverse,true);
    t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
    std::cout << "track pyr " << level << "cost time: " << time_used.count() << "\n";

    if(level > 0){
      for(auto &kp:kp1_pyr)
        kp.pt /= pyramid_scale;
      for(auto &kp:kp2_pyr)
        kp.pt /= pyramid_scale;
    }
  }

  for(auto &kp:kp2_pyr)
    kp2.push_back(kp);
}
