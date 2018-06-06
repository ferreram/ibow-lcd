/**
* This file is part of ibow-lcd.
*
* Copyright (C) 2017 Emilio Garcia-Fidalgo <emilio.garcia@uib.es> (University of the Balearic Islands)
*
* ibow-lcd is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ibow-lcd is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ibow-lcd. If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>

#include <boost/filesystem.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
// #include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "ibow-lcd/lcdetector.h"

void getFilenames(const std::string& directory,
                  std::vector<std::string>* filenames) {
    using namespace boost::filesystem;

    filenames->clear();
    path dir(directory);

    // Retrieving, sorting and filtering filenames.
    std::vector<path> entries;
    copy(directory_iterator(dir), directory_iterator(), back_inserter(entries));
    sort(entries.begin(), entries.end());
    for (auto it = entries.begin(); it != entries.end(); it++) {
        std::string ext = it->extension().c_str();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".png" || ext == ".jpg" ||
            ext == ".ppm" || ext == ".jpeg") {
            filenames->push_back(it->string());
        }
    }
}

void gftt(cv::Mat &img, std::vector<cv::KeyPoint> &kps)
{
  std::vector<cv::Point2f> corners;

  cv::goodFeaturesToTrack(
                        img,
                        corners,
                        1000,
                        0.0001,
                        20.);

  /// Set the need parameters to find the refined corners
  cv::Size winSize = cv::Size( 5, 5 );
  cv::Size zeroZone = cv::Size( -1, -1 );
  cv::TermCriteria criteria = cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);

  cv::cornerSubPix(img, corners, winSize, zeroZone, criteria);
  
  cv::KeyPoint::convert(corners, kps);
}

void detectHomogeneously(cv::Ptr<cv::Feature2D> &detector, cv::Mat &img, std::vector<cv::KeyPoint> &kps)
{
  int box = 16;
  int half_box = box / 2;

  int width = img.cols;
  int height = img.rows;

  int wstep = width / box;
  int hstep = height / box;

  cv::Mat mask;

  std::vector<cv::KeyPoint> vkps;
  std::vector<std::vector<std::vector<cv::KeyPoint>>> vkps_ij;

  vkps_ij.resize(hstep);

  detector->detect(img, vkps);

  std::vector<int> vnb_oct(4, 0);

  for(int i = 0 ; i < hstep ; ++i)
  {
    vkps_ij[i].resize(wstep);

    for(int j = 0 ; j < wstep ; ++j)
    {
      int xmin = j*box;
      int ymin = i*box;

      int xmax = (j+1)*box;
      int ymax = (i+1)*box;

      for(int k = 0 ; k < vkps.size() ; ++k)
      {
        cv::KeyPoint kp = vkps[k];

        vnb_oct[kp.octave] += 1;
        
        if(xmin <= kp.pt.x && kp.pt.x < xmax && ymin <= kp.pt.y && kp.pt.y < ymax)
        {
          vkps_ij[i][j].push_back(kp);

          vkps.erase(vkps.begin() + k);

          k--;
        }
      }
    }
  }

  for(int i = 0 ; i < hstep ; ++i)
  {
    for(int j = 0 ; j < wstep ; ++j)
    {
      std::vector<cv::KeyPoint> vbestkps = vkps_ij[i][j];

      if(vbestkps.empty())
        continue;

      cv::KeyPointsFilter::retainBest(vbestkps,1);

      kps.push_back(vbestkps[0]);

      cv::circle(mask, vbestkps[0].pt, 20, 0, -1);
    }
  }

  while(kps.size() < 250)
  {
    for(int i = 0 ; i < hstep && kps.size() < 250 ; ++i)
    {
      for(int j = 0 ; j < wstep && kps.size() < 250 ; ++j)
      {
        std::vector<cv::KeyPoint> vbestkps = vkps_ij[i][j];

        cv::KeyPointsFilter::runByPixelsMask(vbestkps, mask);

        if(vbestkps.empty())
          continue;

        cv::KeyPointsFilter::retainBest(vbestkps,1);

        kps.push_back(vbestkps[0]);

        cv::circle(mask, vbestkps[0].pt, 20, 0, -1);
      }
    }
  }

  // cv::Mat outimg;
  // cv::drawKeypoints(img, kps, outimg, cv::Scalar::all(-1), 4);

  // cv::namedWindow("a");
  // cv::imshow("a", outimg);
  // cv::waitKey(0);

}

int main(int argc, char** argv) {
  // Creating feature detector and descriptor

  // cv::Ptr<cv::Feature2D> detector = cv::ORB::create(250);  // Default params
  cv::Ptr<cv::Feature2D> detector = cv::BRISK::create(15,3);

  // cv::Ptr<cv::Feature2D> descriptor = cv::ORB::create(250);
  cv::Ptr<cv::Feature2D> descriptor = cv::BRISK::create(20,3);
  // cv::Ptr<cv::Feature2D> descriptor = cv::xfeatures2d::BriefDescriptorExtractor::create();
  // cv::Ptr<cv::Feature2D> descriptor = cv::xfeatures2d::LATCH::create();

  // Loading image filenames
  std::vector<std::string> filenames;
  getFilenames(argv[1], &filenames);
  unsigned nimages = filenames.size();

  // Creating the loop closure detector object
  ibow_lcd::LCDetectorParams params;  // Assign desired parameters
  ibow_lcd::LCDetector lcdet(params);

  // Processing the sequence of images
  for (unsigned i = 0; i < nimages; i++) {
    // Processing image i
    // std::cout << "--- Processing image " << i << std::endl;

    // Loading and describing the image
    cv::Mat img = cv::imread(filenames[i]);
    cv::cvtColor(img, img, CV_RGB2GRAY);
    std::vector<cv::KeyPoint> kps;

    auto start = std::chrono::steady_clock::now();

    // detector->detect(img, kps);
    detectHomogeneously(detector, img, kps);
    // gftt(img, kps);

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;

    auto detect_time = std::chrono::duration<double, std::milli>(diff).count();

    // std::cout << "\n> Number of points described: #" << kps.size();
    // std::cout << "\n> Detection time: #" << detect_time << "ms";
    // std::cout << "\n> Detection time / feat: #" << detect_time / kps.size() << "ms\n";

    cv::Mat dscs;
    descriptor->compute(img, kps, dscs);

    // std::cout << "> Number of points described: #" << kps.size() << std::endl;

    ibow_lcd::LCDetectorResult result;
    lcdet.process(i, kps, dscs, &result);

    switch (result.status) {
      case ibow_lcd::LC_DETECTED:
        std::cout << "\n> Number of points described: #" << kps.size();
        std::cout << "\n> Detection time: #" << detect_time << "ms";
        std::cout << "\n> Detection time / feat: #" << detect_time / kps.size() << "ms\n\n";

        std::cout << "--- Processing image " << i << std::endl;
        std::cout << "--- Loop detected!!!: " << result.train_id <<
                     " with " << result.inliers << " inliers" << std::endl;
        break;
      // case ibow_lcd::LC_NOT_DETECTED:
      //   std::cout << "No loop found" << std::endl;
      //   break;
      // case ibow_lcd::LC_NOT_ENOUGH_IMAGES:
      //   std::cout << "Not enough images to found a loop" << std::endl;
      //   break;
      // case ibow_lcd::LC_NOT_ENOUGH_ISLANDS:
      //   std::cout << "Not enough islands to found a loop" << std::endl;
      //   break;
      // case ibow_lcd::LC_NOT_ENOUGH_INLIERS:
      //   std::cout << "Not enough inliers" << std::endl;
      //   break;
      // case ibow_lcd::LC_TRANSITION:
      //   std::cout << "Transitional loop closure" << std::endl;
      //   break;
      // default:
      //   std::cout << "No status information" << std::endl;
      //   break;
    }
  }

  return 0;
}
