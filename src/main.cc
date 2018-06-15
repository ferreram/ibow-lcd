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
                        250,
                        0.01,
                        20.);

  // /// Set the need parameters to find the refined corners
  // cv::Size winSize = cv::Size( 5, 5 );
  // cv::Size zeroZone = cv::Size( -1, -1 );
  // cv::TermCriteria criteria = cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);

  // cv::cornerSubPix(img, corners, winSize, zeroZone, criteria);
  
  if(corners.size() < 250)
  {
    cv::Mat mask;

    for(auto &pt : corners)
    {
      cv::circle(mask, pt, 20, 0, -1);
    }

    std::vector<cv::Point2f> more_corners;

    auto nb2detect = 250 - corners.size();

    cv::goodFeaturesToTrack(
                      img,
                      more_corners,
                      nb2detect,
                      0.01,
                      10.,
                      mask);
    
    for(auto &pt : more_corners)
    {
      corners.push_back(pt);
    }
  }

  cv::KeyPoint::convert(corners, kps, 12);

  // cv::Mat outimg;
  // cv::drawKeypoints(img, kps, outimg, cv::Scalar::all(-1), 4);

  // cv::namedWindow("a");
  // cv::imshow("a", outimg);
  // cv::waitKey(0);
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

      if(kps.size() == 250)
      {
        j = wstep;
        i = hstep;
      }

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

        if(kps.size() == 250)
        {
          j = wstep;
          i = hstep;
        }

        cv::circle(mask, vbestkps[0].pt, 20, 0, -1);
      }
    }
  }

  // std::cout << kps[0].size << std::endl;
  // std::cout << kps[0].response << std::endl;

  // cv::Mat outimg;
  // cv::drawKeypoints(img, kps, outimg, cv::Scalar::all(-1), 4);

  // cv::namedWindow("a");
  // cv::imshow("a", outimg);
  // cv::waitKey(0);

}

int main(int argc, char** argv) {
  // Creating feature detector and descriptor

  // cv::Ptr<cv::Feature2D> detector = cv::ORB::create(250);  // Default params
  cv::Ptr<cv::Feature2D> detector = cv::BRISK::create(10,3);

  cv::Ptr<cv::Feature2D> descriptor = cv::ORB::create(250);
  // cv::Ptr<cv::Feature2D> descriptor = cv::BRISK::create(20,3);
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
    // detectHomogeneously(detector, img, kps);
    gftt(img, kps);

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;

    auto detect_time = std::chrono::duration<double, std::milli>(diff).count();

    // std::cout << "\n> Number of points detected: #" << kps.size();
    // std::cout << "\n> Detection time: #" << detect_time << "ms";
    // std::cout << "\n> Detection time / feat: #" << detect_time / kps.size() << "ms\n";

    start = std::chrono::steady_clock::now();

    cv::Mat dscs;
    descriptor->compute(img, kps, dscs);

    end = std::chrono::steady_clock::now();
    diff = end - start;

    detect_time = std::chrono::duration<double, std::milli>(diff).count();

    // std::cout << "\n> Number of points described: #" << kps.size();
    // std::cout << "\n> Description time: #" << detect_time << "ms";
    // std::cout << "\n> Description time / feat: #" << detect_time / kps.size() << "ms\n";

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

        std::vector<cv::Point2f> query_pts = result.vquery_pts;
        std::vector<cv::Point2f> train_pts = result.vtrain_pts;
        
        std::vector<cv::KeyPoint> query_kps;
        std::vector<cv::KeyPoint> train_kps;
        
        cv::KeyPoint::convert(query_pts, query_kps, 6.);
        cv::KeyPoint::convert(train_pts, train_kps, 6.);

        std::vector<cv::DMatch> matches;

        for(int l = 0 ; l < query_kps.size() ; ++l)
        {
          cv::DMatch m(l,l,1.);

          matches.push_back(m);
        }

        cv::Mat img2 = cv::imread(filenames[result.train_id]);
        cv::cvtColor(img2, img2, CV_RGB2GRAY);
        
        cv::Mat outimg;

        cv::drawMatches(img,query_kps,img2,train_kps,matches,outimg);

        // cv::hconcat(img,img2,outimg);
        
        cv::namedWindow("match");
        cv::imshow("match",outimg);

        cv::waitKey(0);



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
