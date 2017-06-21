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

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltru�aitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltru�aitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltru�aitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltru�aitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////
// FaceLandmarkImg.cpp : Defines the entry point for the console application for detecting landmarks in images.

#include "LandmarkCoreIncludes.h"

// System includes
#include <fstream>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

#include <dlib/image_processing/frontal_face_detector.h>

#include <tbb/tbb.h>

#include <FaceAnalyser.h>
#include <GazeEstimation.h>

#ifndef CONFIG_DIR
#define CONFIG_DIR "~"
#endif

using namespace std;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for(int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

void convert_to_grayscale(const cv::Mat& in, cv::Mat& out)
{
	if(in.channels() == 3)
	{
		// Make sure it's in a correct format
		if(in.depth() != CV_8U)
		{
			if(in.depth() == CV_16U)
			{
				cv::Mat tmp = in / 256;
				tmp.convertTo(tmp, CV_8U);
				cv::cvtColor(tmp, out, CV_BGR2GRAY);
			}
		}
		else
		{
			cv::cvtColor(in, out, CV_BGR2GRAY);
		}
	}
	else if(in.channels() == 4)
	{
		cv::cvtColor(in, out, CV_BGRA2GRAY);
	}
	else
	{
		if(in.depth() == CV_16U)
		{
			cv::Mat tmp = in / 256;
			out = tmp.clone();
		}
		else if(in.depth() != CV_8U)
		{
			in.convertTo(out, CV_8U);
		}
		else
		{
			out = in.clone();
		}
	}
}

// Useful utility for creating directories for storing the output files
void create_directory_from_file(string output_path)
{

	// Creating the right directory structure

	// First get rid of the file
	auto p = boost::filesystem::path(boost::filesystem::path(output_path).parent_path());

	if (!p.empty() && !boost::filesystem::exists(p))
	{
		bool success = boost::filesystem::create_directories(p);
		if (!success)
		{
			cout << "Failed to create a directory... " << p.string() << endl;
		}
	}
}

// This will only be accurate when camera parameters are accurate, useful for work on 3D data
void write_out_pose_landmarks(const string& outfeatures, const cv::Mat_<double>& shape3D, const cv::Vec6d& pose, const cv::Point3f& gaze0, const cv::Point3f& gaze1)
{
	create_directory_from_file(outfeatures);
	std::ofstream featuresFile;
	featuresFile.open(outfeatures);

	if (featuresFile.is_open())
	{
		int n = shape3D.cols;
		featuresFile << "version: 1" << endl;
		featuresFile << "npoints: " << n << endl;
		featuresFile << "{" << endl;

		for (int i = 0; i < n; ++i)
		{
			// Use matlab format, so + 1
			featuresFile << shape3D.at<double>(i) << " " << shape3D.at<double>(i + n) << " " << shape3D.at<double>(i + 2*n) << endl;
		}
		featuresFile << "}" << endl;

		// Do the pose and eye gaze if present as well
		featuresFile << "pose: eul_x, eul_y, eul_z: " << endl;
		featuresFile << "{" << endl;
		featuresFile << pose[3] << " " << pose[4] << " " << pose[5] << endl;
		featuresFile << "}" << endl;

		// Do the pose and eye gaze if present as well
		featuresFile << "gaze: dir_x_1, dir_y_1, dir_z_1, dir_x_2, dir_y_2, dir_z_2: " << endl;
		featuresFile << "{" << endl;
		featuresFile << gaze0.x << " " << gaze0.y << " " << gaze0.z << " " << gaze1.x << " " << gaze1.y << " " << gaze1.z << endl;
		featuresFile << "}" << endl;
		featuresFile.close();
	}
}

void write_out_landmarks(const string& outfeatures, const LandmarkDetector::CLNF& clnf_model, const cv::Vec6d& pose, const cv::Point3f& gaze0, const cv::Point3f& gaze1, std::vector<std::pair<std::string, double>> au_intensities, std::vector<std::pair<std::string, double>> au_occurences)
{
	create_directory_from_file(outfeatures);
	std::ofstream featuresFile;
	featuresFile.open(outfeatures);

	if (featuresFile.is_open())
	{
		int n = clnf_model.patch_experts.visibilities[0][0].rows;
		featuresFile << "version: 1" << endl;
		featuresFile << "npoints: " << n << endl;
		featuresFile << "{" << endl;

		for (int i = 0; i < n; ++i)
		{
			// Use matlab format, so + 1
			featuresFile << clnf_model.detected_landmarks.at<double>(i) + 1 << " " << clnf_model.detected_landmarks.at<double>(i + n) + 1 << endl;
		}
		featuresFile << "}" << endl;

		// Do the pose and eye gaze if present as well
		featuresFile << "pose: eul_x, eul_y, eul_z: " << endl;
		featuresFile << "{" << endl;
		featuresFile << pose[3] << " " << pose[4] << " " << pose[5] << endl;
		featuresFile << "}" << endl;

		// Do the pose and eye gaze if present as well
		featuresFile << "gaze: dir_x_1, dir_y_1, dir_z_1, dir_x_2, dir_y_2, dir_z_2: " << endl;
		featuresFile << "{" << endl;
		featuresFile << gaze0.x << " " << gaze0.y << " " << gaze0.z << " " << gaze1.x << " " << gaze1.y << " " << gaze1.z << endl;
		featuresFile << "}" << endl;

		// Do the au intensities
		featuresFile << "au intensities: " << au_intensities.size() << endl;
		featuresFile << "{" << endl;

		for (size_t i = 0; i < au_intensities.size(); ++i)
		{
			// Use matlab format, so + 1
			featuresFile << au_intensities[i].first << " " << au_intensities[i].second << endl;
		}

		featuresFile << "}" << endl;

		// Do the au occurences
		featuresFile << "au occurences: " << au_occurences.size() << endl;
		featuresFile << "{" << endl;

		for (size_t i = 0; i < au_occurences.size(); ++i)
		{
			// Use matlab format, so + 1
			featuresFile << au_occurences[i].first << " " << au_occurences[i].second << endl;
		}

		featuresFile << "}" << endl;


		featuresFile.close();
	}
}

void create_display_image(const cv::Mat& orig, cv::Mat& display_image, LandmarkDetector::CLNF& clnf_model)
{
	
	// Draw head pose if present and draw eye gaze as well

	// preparing the visualisation image
	display_image = orig.clone();		

	// Creating a display image			
	cv::Mat xs = clnf_model.detected_landmarks(cv::Rect(0, 0, 1, clnf_model.detected_landmarks.rows/2));
	cv::Mat ys = clnf_model.detected_landmarks(cv::Rect(0, clnf_model.detected_landmarks.rows/2, 1, clnf_model.detected_landmarks.rows/2));
	double min_x, max_x, min_y, max_y;

	cv::minMaxLoc(xs, &min_x, &max_x);
	cv::minMaxLoc(ys, &min_y, &max_y);

	double width = max_x - min_x;
	double height = max_y - min_y;

	int minCropX = max((int)(min_x-width/3.0),0);
	int minCropY = max((int)(min_y-height/3.0),0);

	int widthCrop = min((int)(width*5.0/3.0), display_image.cols - minCropX - 1);
	int heightCrop = min((int)(height*5.0/3.0), display_image.rows - minCropY - 1);

	double scaling = 350.0/widthCrop;
	
	// first crop the image
	display_image = display_image(cv::Rect((int)(minCropX), (int)(minCropY), (int)(widthCrop), (int)(heightCrop)));
		
	// now scale it
	cv::resize(display_image.clone(), display_image, cv::Size(), scaling, scaling);

	// Make the adjustments to points
	xs = (xs - minCropX)*scaling;
	ys = (ys - minCropY)*scaling;

	cv::Mat shape = clnf_model.detected_landmarks.clone();

	xs.copyTo(shape(cv::Rect(0, 0, 1, clnf_model.detected_landmarks.rows/2)));
	ys.copyTo(shape(cv::Rect(0, clnf_model.detected_landmarks.rows/2, 1, clnf_model.detected_landmarks.rows/2)));

	// Do the shifting for the hierarchical models as well
	for (size_t part = 0; part < clnf_model.hierarchical_models.size(); ++part)
	{
		cv::Mat xs = clnf_model.hierarchical_models[part].detected_landmarks(cv::Rect(0, 0, 1, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2));
		cv::Mat ys = clnf_model.hierarchical_models[part].detected_landmarks(cv::Rect(0, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2, 1, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2));

		xs = (xs - minCropX)*scaling;
		ys = (ys - minCropY)*scaling;

		cv::Mat shape = clnf_model.hierarchical_models[part].detected_landmarks.clone();

		xs.copyTo(shape(cv::Rect(0, 0, 1, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2)));
		ys.copyTo(shape(cv::Rect(0, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2, 1, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2)));

	}

	LandmarkDetector::Draw(display_image, clnf_model);
						
}

int main (int argc, char **argv)
{
		
	//Convert arguments to more convenient vector form
	vector<string> arguments = get_arguments(argc, argv);

	// Search paths
	boost::filesystem::path config_path = boost::filesystem::path(CONFIG_DIR);
	boost::filesystem::path parent_path = boost::filesystem::path(arguments[0]).parent_path();

	// Some initial parameters that can be overriden from command line
	vector<string> files, depth_files, output_images, output_landmark_locations, output_pose_locations;

	// Bounding boxes for a face in each image (optional)
	vector<cv::Rect_<double> > bounding_boxes;
	
	LandmarkDetector::get_image_input_output_params(files, depth_files, output_landmark_locations, output_pose_locations, output_images, bounding_boxes, arguments);
	LandmarkDetector::FaceModelParameters det_parameters(arguments);	
	// No need to validate detections, as we're not doing tracking
	det_parameters.validate_detections = false;

	// Grab camera parameters if provided (only used for pose and eye gaze and are quite important for accurate estimates)
	float fx = 0, fy = 0, cx = 0, cy = 0;
	int device = -1;
	LandmarkDetector::get_camera_params(device, fx, fy, cx, cy, arguments);

	// If cx (optical axis centre) is undefined will use the image size/2 as an estimate
	bool cx_undefined = false;
	bool fx_undefined = false;
	if (cx == 0 || cy == 0)
	{
		cx_undefined = true;
	}
	if (fx == 0 || fy == 0)
	{
		fx_undefined = true;
	}

	// The modules that are being used for tracking
	cout << "Loading the model" << endl;
	LandmarkDetector::CLNF clnf_model(det_parameters.model_location);
	cout << "Model loaded" << endl;
	
	cv::CascadeClassifier classifier(det_parameters.face_detector_location);
	dlib::frontal_face_detector face_detector_hog = dlib::get_frontal_face_detector();

	// Loading the AU prediction models
	string au_loc = "AU_predictors/AU_all_static.txt";

	boost::filesystem::path au_loc_path = boost::filesystem::path(au_loc);
	if (boost::filesystem::exists(au_loc_path))
	{
		au_loc = au_loc_path.string();
	}
	else if (boost::filesystem::exists(parent_path/au_loc_path))
	{
		au_loc = (parent_path/au_loc_path).string();
	}
	else if (boost::filesystem::exists(config_path/au_loc_path))
	{
		au_loc = (config_path/au_loc_path).string();
	}
	else
	{
		cout << "Can't find AU prediction files, exiting" << endl;
		return 1;
	}

	// Used for image masking for AUs
	string tri_loc;
	boost::filesystem::path tri_loc_path = boost::filesystem::path("model/tris_68_full.txt");
	if (boost::filesystem::exists(tri_loc_path))
	{
		tri_loc = tri_loc_path.string();
	}
	else if (boost::filesystem::exists(parent_path/tri_loc_path))
	{
		tri_loc = (parent_path/tri_loc_path).string();
	}
	else if (boost::filesystem::exists(config_path/tri_loc_path))
	{
		tri_loc = (config_path/tri_loc_path).string();
	}
	else
	{
		cout << "Can't find triangulation files, exiting" << endl;
		return 1;
	}

	FaceAnalysis::FaceAnalyser face_analyser(vector<cv::Vec3d>(), 0.7, 112, 112, au_loc, tri_loc);

	bool visualise = !det_parameters.quiet_mode;

	// Do some image loading
	for(size_t i = 0; i < files.size(); i++)
	{
		string file = files.at(i);

		// Loading image
		cv::Mat read_image = cv::imread(file, -1);

		if (read_image.empty())
		{
			cout << "Could not read the input image" << endl;
			return 1;
		}

		// Loading depth file if exists (optional)
		cv::Mat_<float> depth_image;

		if(depth_files.size() > 0)
		{
			string dFile = depth_files.at(i);
			cv::Mat dTemp = cv::imread(dFile, -1);
			dTemp.convertTo(depth_image, CV_32F);
		}

		// Making sure the image is in uchar grayscale
		cv::Mat_<uchar> grayscale_image;
		convert_to_grayscale(read_image, grayscale_image);
		

		// If optical centers are not defined just use center of image
		if (cx_undefined)
		{
			cx = grayscale_image.cols / 2.0f;
			cy = grayscale_image.rows / 2.0f;
		}
		// Use a rough guess-timate of focal length
		if (fx_undefined)
		{
			fx = 500 * (grayscale_image.cols / 640.0);
			fy = 500 * (grayscale_image.rows / 480.0);

			fx = (fx + fy) / 2.0;
			fy = fx;
		}


		// if no pose defined we just use a face detector
		if(bounding_boxes.empty())
		{
			
			// Detect faces in an image
			vector<cv::Rect_<double> > face_detections;

			if(det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR)
			{
				vector<double> confidences;
				LandmarkDetector::DetectFacesHOG(face_detections, grayscale_image, face_detector_hog, confidences);
			}
			else
			{
				LandmarkDetector::DetectFaces(face_detections, grayscale_image, classifier);
			}

			// Detect landmarks around detected faces
			int face_det = 0;
			// perform landmark detection for every face detected
			for(size_t face=0; face < face_detections.size(); ++face)
			{
				// if there are multiple detections go through them
				bool success = LandmarkDetector::DetectLandmarksInImage(grayscale_image, depth_image, face_detections[face], clnf_model, det_parameters);

				// Estimate head pose and eye gaze				
				cv::Vec6d headPose = LandmarkDetector::GetCorrectedPoseWorld(clnf_model, fx, fy, cx, cy);

				// Gaze tracking, absolute gaze direction
				cv::Point3f gazeDirection0(0, 0, -1);
				cv::Point3f gazeDirection1(0, 0, -1);

				if (success && det_parameters.track_gaze)
				{
					FaceAnalysis::EstimateGaze(clnf_model, gazeDirection0, fx, fy, cx, cy, true);
					FaceAnalysis::EstimateGaze(clnf_model, gazeDirection1, fx, fy, cx, cy, false);

				}

				auto ActionUnits = face_analyser.PredictStaticAUs(read_image, clnf_model, false);

				// Writing out the detected landmarks (in an OS independent manner)
				if(!output_landmark_locations.empty())
				{
					char name[100];
					// append detection number (in case multiple faces are detected)
					sprintf(name, "_det_%d", face_det);

					// Construct the output filename
					boost::filesystem::path slash("/");
					std::string preferredSlash = slash.make_preferred().string();

					boost::filesystem::path out_feat_path(output_landmark_locations.at(i));
					boost::filesystem::path dir = out_feat_path.parent_path();
					boost::filesystem::path fname = out_feat_path.filename().replace_extension("");
					boost::filesystem::path ext = out_feat_path.extension();
					string outfeatures = dir.string() + preferredSlash + fname.string() + string(name) + ext.string();
					write_out_landmarks(outfeatures, clnf_model, headPose, gazeDirection0, gazeDirection1, ActionUnits.first, ActionUnits.second);
				}

				if (!output_pose_locations.empty())
				{
					char name[100];
					// append detection number (in case multiple faces are detected)
					sprintf(name, "_det_%d", face_det);

					// Construct the output filename
					boost::filesystem::path slash("/");
					std::string preferredSlash = slash.make_preferred().string();

					boost::filesystem::path out_pose_path(output_pose_locations.at(i));
					boost::filesystem::path dir = out_pose_path.parent_path();
					boost::filesystem::path fname = out_pose_path.filename().replace_extension("");
					boost::filesystem::path ext = out_pose_path.extension();
					string outfeatures = dir.string() + preferredSlash + fname.string() + string(name) + ext.string();
					write_out_pose_landmarks(outfeatures, clnf_model.GetShape(fx, fy, cx, cy), headPose, gazeDirection0, gazeDirection1);

				}

				if (det_parameters.track_gaze)
				{
					cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetCorrectedPoseWorld(clnf_model, fx, fy, cx, cy);

					// Draw it in reddish if uncertain, blueish if certain
					LandmarkDetector::DrawBox(read_image, pose_estimate_to_draw, cv::Scalar(255.0, 0, 0), 3, fx, fy, cx, cy);
					FaceAnalysis::DrawGaze(read_image, clnf_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
				}

				// displaying detected landmarks
				cv::Mat display_image;
				create_display_image(read_image, display_image, clnf_model);

				if(visualise && success)
				{
					imshow("colour", display_image);
					cv::waitKey(1);
				}

				// Saving the display images (in an OS independent manner)
				if(!output_images.empty() && success)
				{
					string outimage = output_images.at(i);
					if(!outimage.empty())
					{
						char name[100];
						sprintf(name, "_det_%d", face_det);

						boost::filesystem::path slash("/");
						std::string preferredSlash = slash.make_preferred().string();

						// append detection number
						boost::filesystem::path out_feat_path(outimage);
						boost::filesystem::path dir = out_feat_path.parent_path();
						boost::filesystem::path fname = out_feat_path.filename().replace_extension("");
						boost::filesystem::path ext = out_feat_path.extension();
						outimage = dir.string() + preferredSlash + fname.string() + string(name) + ext.string();
						create_directory_from_file(outimage);
						bool write_success = cv::imwrite(outimage, display_image);	
						
						if (!write_success)
						{
							cout << "Could not output a processed image" << endl;
							return 1;
						}

					}

				}

				if(success)
				{
					face_det++;
				}

			}
		}
		else
		{
			// Have provided bounding boxes
			LandmarkDetector::DetectLandmarksInImage(grayscale_image, bounding_boxes[i], clnf_model, det_parameters);

			// Estimate head pose and eye gaze				
			cv::Vec6d headPose = LandmarkDetector::GetCorrectedPoseWorld(clnf_model, fx, fy, cx, cy);

			// Gaze tracking, absolute gaze direction
			cv::Point3f gazeDirection0(0, 0, -1);
			cv::Point3f gazeDirection1(0, 0, -1);
			
			if (det_parameters.track_gaze)
			{
				FaceAnalysis::EstimateGaze(clnf_model, gazeDirection0, fx, fy, cx, cy, true);
				FaceAnalysis::EstimateGaze(clnf_model, gazeDirection1, fx, fy, cx, cy, false);
			}

			auto ActionUnits = face_analyser.PredictStaticAUs(read_image, clnf_model, false);

			// Writing out the detected landmarks
			if(!output_landmark_locations.empty())
			{
				string outfeatures = output_landmark_locations.at(i);
				write_out_landmarks(outfeatures, clnf_model, headPose, gazeDirection0, gazeDirection1, ActionUnits.first, ActionUnits.second);
			}

			// Writing out the detected landmarks
			if (!output_pose_locations.empty())
			{
				string outfeatures = output_pose_locations.at(i);
				write_out_pose_landmarks(outfeatures, clnf_model.GetShape(fx, fy, cx, cy), headPose, gazeDirection0, gazeDirection1);
			}

			// displaying detected stuff
			cv::Mat display_image;

			if (det_parameters.track_gaze)
			{
				cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetCorrectedPoseWorld(clnf_model, fx, fy, cx, cy);

				// Draw it in reddish if uncertain, blueish if certain
				LandmarkDetector::DrawBox(read_image, pose_estimate_to_draw, cv::Scalar(255.0, 0, 0), 3, fx, fy, cx, cy);
				FaceAnalysis::DrawGaze(read_image, clnf_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
			}

			create_display_image(read_image, display_image, clnf_model);

			if(visualise)
			{
				imshow("colour", display_image);
				cv::waitKey(1);
			}

			if(!output_images.empty())
			{
				string outimage = output_images.at(i);
				if(!outimage.empty())
				{
					create_directory_from_file(outimage);
					bool write_success = imwrite(outimage, display_image);	

					if (!write_success)
					{
						cout << "Could not output a processed image" << endl;
						return 1;
					}
				}
			}
		}				

	}
	
	return 0;
}

