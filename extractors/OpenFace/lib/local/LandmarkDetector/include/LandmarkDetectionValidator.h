///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt
//
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////

#ifndef __LANDMARK_DETECTION_VALIDATOR_h_
#define __LANDMARK_DETECTION_VALIDATOR_h_

// OpenCV includes
#include <opencv2/core/core.hpp>

// System includes
#include <vector>

// Local includes
#include "PAW.h"

using namespace std;

namespace LandmarkDetector
{
//===========================================================================
//
// Checking if landmark detection was successful using an SVR regressor
// Using multiple validators trained add different views
// The regressor outputs -1 for ideal alignment and 1 for worst alignment
//===========================================================================
class DetectionValidator
{
		
public:    
	
	// What type of validator we're using - 0 - linear svr, 1 - feed forward neural net, 2 - convolutional neural net
	int validator_type;

	// The orientations of each of the landmark detection validator
	vector<cv::Vec3d> orientations;

	// Piecewise affine warps to the reference shape (per orientation)
	vector<PAW>     paws;

	//==========================================
	// Linear SVR

	// SVR biases
	vector<double>  bs;

	// SVR weights
	vector<cv::Mat_<double> > ws;
	
	//==========================================
	// Neural Network

	// Neural net weights
	vector<vector<cv::Mat_<double> > > ws_nn;

	// What type of activation or output functions are used
	// 0 - sigmoid, 1 - tanh_opt, 2 - ReLU
	vector<int> activation_fun;
	vector<int> output_fun;

	//==========================================
	// Convolutional Neural Network

	// CNN layers for each view
	// view -> layer -> input maps -> kernels
	vector<vector<vector<vector<cv::Mat_<float> > > > > cnn_convolutional_layers;
	// Bit ugly with so much nesting, but oh well
	vector<vector<vector<vector<pair<int, cv::Mat_<double> > > > > > cnn_convolutional_layers_dft;
	vector<vector<vector<float > > > cnn_convolutional_layers_bias;
	vector< vector<int> > cnn_subsampling_layers;
	vector< vector<cv::Mat_<float> > > cnn_fully_connected_layers;
	vector< vector<float > > cnn_fully_connected_layers_bias;
	// 0 - convolutional, 1 - subsampling, 2 - fully connected
	vector<vector<int> > cnn_layer_types;
	
	//==========================================

	// Normalisation for face validation
	vector<cv::Mat_<double> > mean_images;
	vector<cv::Mat_<double> > standard_deviations;

	// Default constructor
	DetectionValidator(){;}

	// Copy constructor
	DetectionValidator(const DetectionValidator& other);

	// Given an image, orientation and detected landmarks output the result of the appropriate regressor
	double Check(const cv::Vec3d& orientation, const cv::Mat_<uchar>& intensity_img, cv::Mat_<double>& detected_landmarks);

	// Reading in the model
	void Read(string location);
			
	// Getting the closest view center based on orientation
	int GetViewId(const cv::Vec3d& orientation) const;

private:

	// The actual regressor application on the image

	// Support Vector Regression (linear kernel)
	double CheckSVR(const cv::Mat_<double>& warped_img, int view_id);

	// Feed-forward Neural Network
	double CheckNN(const cv::Mat_<double>& warped_img, int view_id);

	// Convolutional Neural Network
	double CheckCNN(const cv::Mat_<double>& warped_img, int view_id);

	// A normalisation helper
	void NormaliseWarpedToVector(const cv::Mat_<double>& warped_img, cv::Mat_<double>& feature_vec, int view_id);

};

}
#endif
