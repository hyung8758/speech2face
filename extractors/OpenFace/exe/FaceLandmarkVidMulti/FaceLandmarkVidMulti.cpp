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


// FaceTrackingVidMulti.cpp : Defines the entry point for the multiple face tracking console application.
#include "LandmarkCoreIncludes.h"

#include <fstream>
#include <sstream>

// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort( const std::string & error )
{
    std::cout << error << std::endl;
    abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

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

void NonOverlapingDetections(const vector<LandmarkDetector::CLNF>& clnf_models, vector<cv::Rect_<double> >& face_detections)
{

	// Go over the model and eliminate detections that are not informative (there already is a tracker there)
	for(size_t model = 0; model < clnf_models.size(); ++model)
	{

		// See if the detections intersect
		cv::Rect_<double> model_rect = clnf_models[model].GetBoundingBox();
		
		for(int detection = face_detections.size()-1; detection >=0; --detection)
		{
			double intersection_area = (model_rect & face_detections[detection]).area();
			double union_area = model_rect.area() + face_detections[detection].area() - 2 * intersection_area;

			// If the model is already tracking what we're detecting ignore the detection, this is determined by amount of overlap
			if( intersection_area/union_area > 0.5)
			{
				face_detections.erase(face_detections.begin() + detection);
			}
		}
	}
}

int main (int argc, char **argv)
{

	vector<string> arguments = get_arguments(argc, argv);

	// Some initial parameters that can be overriden from command line	
	vector<string> files, depth_directories, tracked_videos_output, dummy_out;
	
	// By default try webcam 0
	int device = 0;

	// cx and cy aren't necessarilly in the image center, so need to be able to override it (start with unit vals and init them if none specified)
    float fx = 600, fy = 600, cx = 0, cy = 0;
			
	LandmarkDetector::FaceModelParameters det_params(arguments);
	det_params.use_face_template = true;
	// This is so that the model would not try re-initialising itself
	det_params.reinit_video_every = -1;

	det_params.curr_face_detector = LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR;

	vector<LandmarkDetector::FaceModelParameters> det_parameters;
	det_parameters.push_back(det_params);

	// Get the input output file parameters
	bool u;
	string output_codec;
	LandmarkDetector::get_video_input_output_params(files, depth_directories, dummy_out, tracked_videos_output, u, output_codec, arguments);
	// Get camera parameters
	LandmarkDetector::get_camera_params(device, fx, fy, cx, cy, arguments);
	
	// The modules that are being used for tracking
	vector<LandmarkDetector::CLNF> clnf_models;
	vector<bool> active_models;

	int num_faces_max = 4;

	LandmarkDetector::CLNF clnf_model(det_parameters[0].model_location);
	clnf_model.face_detector_HAAR.load(det_parameters[0].face_detector_location);
	clnf_model.face_detector_location = det_parameters[0].face_detector_location;
	
	clnf_models.reserve(num_faces_max);

	clnf_models.push_back(clnf_model);
	active_models.push_back(false);

	for (int i = 1; i < num_faces_max; ++i)
	{
		clnf_models.push_back(clnf_model);
		active_models.push_back(false);
		det_parameters.push_back(det_params);
	}
	
	// If multiple video files are tracked, use this to indicate if we are done
	bool done = false;	
	int f_n = -1;

	// If cx (optical axis centre) is undefined will use the image size/2 as an estimate
	bool cx_undefined = false;
	if(cx == 0 || cy == 0)
	{
		cx_undefined = true;
	}		
	
	while(!done) // this is not a for loop as we might also be reading from a webcam
	{
		
		string current_file;

		// We might specify multiple video files as arguments
		if(files.size() > 0)
		{
			f_n++;			
		    current_file = files[f_n];
		}

		bool use_depth = !depth_directories.empty();	

		// Do some grabbing
		cv::VideoCapture video_capture;
		if( current_file.size() > 0 )
		{
			INFO_STREAM( "Attempting to read from file: " << current_file );
			video_capture = cv::VideoCapture( current_file );
		}
		else
		{
			INFO_STREAM( "Attempting to capture from device: " << device );
			video_capture = cv::VideoCapture( device );

			// Read a first frame often empty in camera
			cv::Mat captured_image;
			video_capture >> captured_image;
		}

		if (!video_capture.isOpened())
		{
			FATAL_STREAM("Failed to open video source");
			return 1;
		}
		else INFO_STREAM( "Device or file opened");

		cv::Mat captured_image;
		video_capture >> captured_image;		
		

		// If optical centers are not defined just use center of image
		if(cx_undefined)
		{
			cx = captured_image.cols / 2.0f;
			cy = captured_image.rows / 2.0f;
		}
		
		int frame_count = 0;
		
		// saving the videos
		cv::VideoWriter writerFace;
		if(!tracked_videos_output.empty())
		{
			try
			{
				writerFace = cv::VideoWriter(tracked_videos_output[f_n], CV_FOURCC(output_codec[0],output_codec[1],output_codec[2],output_codec[3]), 30, captured_image.size(), true);
			}
			catch(cv::Exception e)
			{
				WARN_STREAM( "Could not open VideoWriter, OUTPUT FILE WILL NOT BE WRITTEN. Currently using codec " << output_codec << ", try using an other one (-oc option)");
			}
		}
		
		// For measuring the timings
		int64 t1,t0 = cv::getTickCount();
		double fps = 10;


		INFO_STREAM( "Starting tracking");
		while(!captured_image.empty())
		{		

			// Reading the images
			cv::Mat_<float> depth_image;
			cv::Mat_<uchar> grayscale_image;

			cv::Mat disp_image = captured_image.clone();

			if(captured_image.channels() == 3)
			{
				cv::cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);				
			}
			else
			{
				grayscale_image = captured_image.clone();				
			}
		
			// Get depth image
			if(use_depth)
			{
				char* dst = new char[100];
				std::stringstream sstream;

				sstream << depth_directories[f_n] << "\\depth%05d.png";
				sprintf(dst, sstream.str().c_str(), frame_count + 1);
				// Reading in 16-bit png image representing depth
				cv::Mat_<short> depth_image_16_bit = cv::imread(string(dst), -1);

				// Convert to a floating point depth image
				if(!depth_image_16_bit.empty())
				{
					depth_image_16_bit.convertTo(depth_image, CV_32F);
				}
				else
				{
					WARN_STREAM( "Can't find depth image" );
				}
			}

			vector<cv::Rect_<double> > face_detections;

			bool all_models_active = true;
			for(unsigned int model = 0; model < clnf_models.size(); ++model)
			{
				if(!active_models[model])
				{
					all_models_active = false;
				}
			}
						
			// Get the detections (every 8th frame and when there are free models available for tracking)
			if(frame_count % 8 == 0 && !all_models_active)
			{				
				if(det_parameters[0].curr_face_detector == LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR)
				{
					vector<double> confidences;
					LandmarkDetector::DetectFacesHOG(face_detections, grayscale_image, clnf_models[0].face_detector_HOG, confidences);
				}
				else
				{
					LandmarkDetector::DetectFaces(face_detections, grayscale_image, clnf_models[0].face_detector_HAAR);
				}

			}

			// Keep only non overlapping detections (also convert to a concurrent vector
			NonOverlapingDetections(clnf_models, face_detections);

			vector<tbb::atomic<bool> > face_detections_used(face_detections.size());

			// Go through every model and update the tracking
			tbb::parallel_for(0, (int)clnf_models.size(), [&](int model){
			//for(unsigned int model = 0; model < clnf_models.size(); ++model)
			//{

				bool detection_success = false;

				// If the current model has failed more than 4 times in a row, remove it
				if(clnf_models[model].failures_in_a_row > 4)
				{				
					active_models[model] = false;
					clnf_models[model].Reset();

				}

				// If the model is inactive reactivate it with new detections
				if(!active_models[model])
				{
					
					for(size_t detection_ind = 0; detection_ind < face_detections.size(); ++detection_ind)
					{
						// if it was not taken by another tracker take it (if it is false swap it to true and enter detection, this makes it parallel safe)
						if(face_detections_used[detection_ind].compare_and_swap(true, false) == false)
						{
					
							// Reinitialise the model
							clnf_models[model].Reset();

							// This ensures that a wider window is used for the initial landmark localisation
							clnf_models[model].detection_success = false;
							detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, depth_image, face_detections[detection_ind], clnf_models[model], det_parameters[model]);
													
							// This activates the model
							active_models[model] = true;

							// break out of the loop as the tracker has been reinitialised
							break;
						}

					}
				}
				else
				{
					// The actual facial landmark detection / tracking
					detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, depth_image, clnf_models[model], det_parameters[model]);
				}
			});
								
			// Go through every model and visualise the results
			for(size_t model = 0; model < clnf_models.size(); ++model)
			{
				// Visualising the results
				// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
				double detection_certainty = clnf_models[model].detection_certainty;

				double visualisation_boundary = -0.1;
			
				// Only draw if the reliability is reasonable, the value is slightly ad-hoc
				if(detection_certainty < visualisation_boundary)
				{
					LandmarkDetector::Draw(disp_image, clnf_models[model]);

					if(detection_certainty > 1)
						detection_certainty = 1;
					if(detection_certainty < -1)
						detection_certainty = -1;

					detection_certainty = (detection_certainty + 1)/(visualisation_boundary +1);

					// A rough heuristic for box around the face width
					int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);
					
					// Work out the pose of the head from the tracked model
					cv::Vec6d pose_estimate = LandmarkDetector::GetCorrectedPoseWorld(clnf_models[model], fx, fy, cx, cy);
					
					// Draw it in reddish if uncertain, blueish if certain
					LandmarkDetector::DrawBox(disp_image, pose_estimate, cv::Scalar((1-detection_certainty)*255.0,0, detection_certainty*255), thickness, fx, fy, cx, cy);
				}
			}

			// Work out the framerate
			if(frame_count % 10 == 0)
			{      
				t1 = cv::getTickCount();
				fps = 10.0 / (double(t1-t0)/cv::getTickFrequency()); 
				t0 = t1;
			}
			
			// Write out the framerate on the image before displaying it
			char fpsC[255];
			sprintf(fpsC, "%d", (int)fps);
			string fpsSt("FPS:");
			fpsSt += fpsC;
			cv::putText(disp_image, fpsSt, cv::Point(10,20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0), 1, CV_AA);
			
			int num_active_models = 0;

			for( size_t active_model = 0; active_model < active_models.size(); active_model++)
			{
				if(active_models[active_model])
				{
					num_active_models++;
				}
			}

			char active_m_C[255];
			sprintf(active_m_C, "%d", num_active_models);
			string active_models_st("Active models:");
			active_models_st += active_m_C;
			cv::putText(disp_image, active_models_st, cv::Point(10,60), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0), 1, CV_AA);
			
			if(!det_parameters[0].quiet_mode)
			{
				cv::namedWindow("tracking_result",1);
				cv::imshow("tracking_result", disp_image);

				if(!depth_image.empty())
				{
					// Division needed for visualisation purposes
					imshow("depth", depth_image/2000.0);
				}
			}

			// output the tracked video
			if(!tracked_videos_output.empty())
			{		
				writerFace << disp_image;
			}

			video_capture >> captured_image;
		
			// detect key presses
			char character_press = cv::waitKey(1);
			
			// restart the trackers
			if(character_press == 'r')
			{
				for(size_t i=0; i < clnf_models.size(); ++i)
				{
					clnf_models[i].Reset();
					active_models[i] = false;
				}
			}
			// quit the application
			else if(character_press=='q')
			{
				return(0);
			}

			// Update the frame count
			frame_count++;
		}
		
		frame_count = 0;

		// Reset the model, for the next video
		for(size_t model=0; model < clnf_models.size(); ++model)
		{
			clnf_models[model].Reset();
			active_models[model] = false;
		}

		// break out of the loop if done with all the files
		if(f_n == files.size() -1)
		{
			done = true;
		}
	}

	return 0;
}

