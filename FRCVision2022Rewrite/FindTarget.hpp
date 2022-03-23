#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/videoio.hpp>
#include <string>

#ifdef _WIN32
#define WINDOW
#endif

namespace TargetFinder {
	struct TargetData {
		bool targetFound;
		cv::Rect target;
		cv::Rect centerTarget;
		double targetWidth;
		double targetHeight;
		double centerTargetWidth;
		double centerTargetHeight;
	};
	class TargetFinder {
	public:
		/// <summary>
		/// Take a Black/White Image and Find a Segmented Target, Uses depth to target to find if target is valid
		/// </summary>
		/// <param name="img">Base Image to Find Target From</param>
		/// <param name="depth">Depth Image to Get Image Data From</param>
		/// <param name="cameraIntrinsics">Camera Intrinsics</param>
		/// <returns>2D Array of Points with Center of Target and Center of Center Target</returns>
		TargetData findTarget(cv::Mat img, rs2::depth_frame depth, rs2_intrinsics cameraIntrinsics);

		TargetData findTargetNoDepth(cv::Mat img);

		/// <summary>
		/// Make Sendable Data From Params
		/// </summary>
		/// <param name="targetFound">Is the Target/Packet Valid?</param>
		/// <param name="x">Radians to Turn Yaw</param>
		/// <param name="y">Motor Rotations to Turn Pitch</param>
		/// <param name="power">Power to Shoot At</param>
		/// <param name="groundDistance">Ground Distance to Target</param>
		/// <returns>String Ready to Send to Destination</returns>
		std::string makeSendableData(int targetFound, double x, double y, double power, double groundDistance);

		/// <summary>
		/// Send Data Through UDP
		/// </summary>
		/// <param name="in">Data to Send</param>
		/// <param name="ip">IPv4 of Destination</param>
		/// <param name="port">Port of Destination</param>
		void sendDataUDP(std::string in, std::string ip, int port);

		int VISION_AREA_OFFSET_X = 70;
		int VISION_AREA_OFFSET_Y = 20;
		std::string ROBOIPV4 = "10.36.95.2";
		int ROBOPORT = 7777;
	};
	
}