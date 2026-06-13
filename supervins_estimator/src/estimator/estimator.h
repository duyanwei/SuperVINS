/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <thread>
#include <mutex>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <fstream>
#include <iomanip>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "parameters.h"
#include "feature_manager.h"
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../initial/solve_5pts.h"
#include "../initial/initial_sfm.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../factor/imu_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/marginalization_factor.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionTwoFrameTwoCamFactor.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"
//#include "../featureTracker/feature_tracker.h"
#include "../featureTracker/feature_tracker_dpl.h"
#include "../featureTracker/feature_tracker.h"

class timeLog {
public:
  timeLog(const double &timeStamp_ = 0) {
    time_stamp = timeStamp_;
    time_feature = 0;
    time_poseTrack = 0;
    time_windowOpt = 0;
    time_total = 0;
    num_poses = 0;
    num_lmks = 0;
  };

  void setZero() {
    time_stamp = 0;
    time_feature = 0;
    time_poseTrack = 0;
    time_windowOpt = 0;
    time_total = 0;
    num_poses = 0;
    num_lmks = 0;
  };

  double time_stamp;
  double time_feature;
  double time_poseTrack;
  double time_windowOpt;
  double time_total;
  size_t num_poses;
  size_t num_lmks;
};

class PoseLog {
public:
  PoseLog(const double &timeStamp_, const double &Tx_, const double &Ty_, const double &Tz_,
       const double &Qx_, const double &Qy_, const double &Qz_, const double &Qw_) {
    time_stamp = timeStamp_;
    position(0) = Tx_;
    position(1) = Ty_;
    position(2) = Tz_;
    orientation(0) = Qx_;
    orientation(1) = Qy_;
    orientation(2) = Qz_;
    orientation(3) = Qw_;
  };

  double time_stamp;
  Eigen::Vector4d orientation;
  Eigen::Vector3d position;
};

class Estimator
{
  public:
    Estimator();
    ~Estimator();
    void setParameter();

    // interface
    void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);
    void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);
    void inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame);
    void inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header);
    void processMeasurements();
    void changeSensorType(int use_imu, int use_stereo);

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();
    bool getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                              vector<pair<double, Eigen::Vector3d>> &gyrVector);
    void getPoseInWorldFrame(Eigen::Matrix4d &T);
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);
    void predictPtsInNextFrame();
    void outliersRejection(set<int> &removeIndex);
    double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                     Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                     double depth, Vector3d &uvi, Vector3d &uvj);
    void updateLatestStates();
    void fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity);
    bool IMUAvailable(double t);
    void initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector);

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    std::mutex mProcess;
    std::mutex mBuf;
    std::mutex mPropagate;
    std::mutex mSuperPointDescriptors;
    queue<pair<double, Eigen::Vector3d>> accBuf;
    queue<pair<double, Eigen::Vector3d>> gyrBuf;
    queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > > featureBuf;

    // new code
    queue<pair<double, cv::Mat>> SuperPointDescriptorsBuf;

    double prevTime, curTime;
    bool openExEstimation;

    std::thread trackThread;
    std::thread processThread;

    //VINS-Fusion中将特征追踪器放入了estimator中，不再单独使用一个节点发布特征提取和追踪结果
    // FeatureTracker featureTracker;
    FeatureTrackerDPL featureTracker; // deep-learning based feature tracker, but still can use original functions

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;

    Matrix3d ric[2];
    Vector3d tic[2];

    Vector3d        Ps[(WINDOW_SIZE + 1)];
    Vector3d        Vs[(WINDOW_SIZE + 1)];
    Matrix3d        Rs[(WINDOW_SIZE + 1)];
    Vector3d        Bas[(WINDOW_SIZE + 1)];
    Vector3d        Bgs[(WINDOW_SIZE + 1)];
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    double Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
    int inputImageCnt;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[2][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;

    Eigen::Vector3d initP;
    Eigen::Matrix3d initR;

    double latest_time;
    Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
    Eigen::Quaterniond latest_Q;

    bool initFirstPoseFlag;
    bool initThreadFlag;

    std::vector<timeLog> logTracking;
    timeLog logCurFrame;

    void saveLogging(const std::string &filename) {
        std::cout << std::endl << "Saving " << logTracking.size() << " records to time log file " << filename << " ..." << std::endl;
        std::ofstream fFrameLog;
        fFrameLog.open(filename.c_str());
        fFrameLog << std::fixed;
        fFrameLog << "#frame_time_stamp time_feature time_poseTrack time_windowOpt time_total pose_num feature_num" << std::endl;
        for (size_t i = 0; i < logTracking.size(); i++) {
            fFrameLog << std::setprecision(6)
                      << logTracking[i].time_stamp << " "
                      << logTracking[i].time_feature << " "
                      << logTracking[i].time_poseTrack << " "
                      << logTracking[i].time_windowOpt << " "
                      << logTracking[i].time_total << " "
                      << std::setprecision(0)
                      << logTracking[i].num_poses << " "
                      << logTracking[i].num_lmks << std::endl;
        }
        fFrameLog.close();
        std::cout << "Finished saving log!" << std::endl;
    }

    std::vector<PoseLog> logFramePose;

    void saveAllFrameTrack(const std::string &filename) {
        std::cout << std::endl << "Saving " << logFramePose.size() << " records to track file " << filename << " ..." << std::endl;
        std::ofstream f_realTimeTrack;
        f_realTimeTrack.open(filename.c_str());
        f_realTimeTrack << std::fixed;
        f_realTimeTrack << "#TimeStamp Tx Ty Tz Qx Qy Qz Qw" << std::endl;
        for (size_t i = 0; i < logFramePose.size(); i++) {
            f_realTimeTrack << std::setprecision(6)
                            << logFramePose[i].time_stamp << " "
                            << std::setprecision(7)
                            << logFramePose[i].position.transpose() << " "
                            << logFramePose[i].orientation.transpose() << std::endl;
        }
        f_realTimeTrack.close();
        std::cout << "Finished saving track!" << std::endl;
    }
};
