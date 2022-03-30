#include <iostream>
#include "librealsense2/rs.hpp"
#include "FindTarget.hpp"
#include "GripPipeline.h"
//#include "networktables/NetworkTable.h"
//#include "networktables/NetworkTableEntry.h"
//#include "networktables/NetworkTableInstance.h"
//#include "cameraserver/CameraServer.h"
#include <cmath>
#include <math.h>
#ifdef _WIN32
#define WINDOW
#define COUT
#endif
/// <summary>
/// Filter IR Image from color Image (Green)
/// </summary>
/// <param name="ir">IR/Grayscale Image</param>
/// <param name="roi">Region of Interest</param>
/// <returns>Filtered IR Image</returns>
cv::Mat filterIR(cv::Mat ir, cv::Rect roi);
static void rs2_deproject_pixel_to_point(float point[3], const struct rs2_intrinsics* intrin, const float pixel[2], float depth);

#define CAMERA_HFOV 69.4
#define CAMERA_VFOV 42.5
#define HFOV_PER_PX CAMERA_HFOV / 1280
#define HFOV_OFFSET CAMERA_HFOV / 2

#include <string>
#include <vector>
/*
class InputParser {
public:
	InputParser(int &argc, char** argv) {
		if (sizeof(argv) > 0)
			for (int i = 1; i < argc; ++i)
				this->tokens.push_back(std::string(argv[i]));
	}
	const std::string& getCmdOption(const std::string& option) const {
		std::vector<std::string>::const_iterator itr;
		itr = std::find(this->tokens.begin(), this->tokens.end(), option);
		if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
			return *itr;
		}
		static const std::string empty_string("");
		return empty_string;
	}
	bool cmdOptionExists(const std::string& option) const {
		return std::find(this->tokens.begin(), this->tokens.end(), option)
			!= this->tokens.end();
	}
private:
	std::vector <std::string> tokens;
};
*/
int main(int& argc, char** argv) {
	std::cout << "Starting Program!" << std::endl;

	rs2::pipeline p;
	rs2::config cfg;
	
	std::cout << "Enableing Streams & Settings..." << std::endl;

	cfg.enable_stream(RS2_STREAM_INFRARED, 1, 1280, 720, RS2_FORMAT_Y8, 30);
	cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_ANY, 30);
	cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);

	rs2::decimation_filter dec;
	dec.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);

	//rs2::disparity_transform depth2disparity;
	rs2::disparity_transform depth2disparity(false);

	rs2::spatial_filter spat;
	// 5 = fill all the zero pixels
	spat.set_option(RS2_OPTION_HOLES_FILL, 5);

	rs2::temporal_filter temp;
	std::cout << "Settings Set! Starting Camera..." << std::endl;
	auto profile = p.start(cfg);
	std::cout << "Camera Started! Setting Sensor Data..." << std::endl;
	auto sensor = profile.get_device().first<rs2::depth_sensor>();
	auto colorSensor = profile.get_device().first<rs2::color_sensor>();

	sensor.set_option(RS2_OPTION_VISUAL_PRESET, RS2_RS400_VISUAL_PRESET_HIGH_DENSITY);
	sensor.set_option(RS2_OPTION_LASER_POWER, 150);
	colorSensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0);

	rs2::align align_to_depth(RS2_STREAM_DEPTH);
	rs2::align align_to_ir(RS2_STREAM_INFRARED);
	rs2::align align_to_color(RS2_STREAM_COLOR);

	int frame_num = 0;

	using clock = std::chrono::system_clock;
	using sec = std::chrono::duration<double>;

	const rs2_intrinsics cameraIntrinsics = p.get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();

	std::cout << "Starting Network Tables..." << std::endl;

	
	/*
	nt::NetworkTableInstance NetworkTableInstance = nt::NetworkTableInstance::GetDefault();
	NetworkTableInstance.SetServerTeam(3695);
	std::shared_ptr<nt::NetworkTable> NetworkTable = NetworkTableInstance.GetTable("SmartDashboard");
	NetworkTableInstance.StartClient();

	nt::NetworkTableEntry YawTurnRadEntry = NetworkTable->GetEntry("YawTurnRad");
	nt::NetworkTableEntry YawTurnRadColorEntry = NetworkTable->GetEntry("YawTurnRadColor");
	nt::NetworkTableEntry PitchTickEntry = NetworkTable->GetEntry("PitchTick");
	nt::NetworkTableEntry PitchTickColorEntry = NetworkTable->GetEntry("PitchTickColor");
	nt::NetworkTableEntry PowerRPMEntry = NetworkTable->GetEntry("PowerRPM");
	nt::NetworkTableEntry PowerRPMColorEntry = NetworkTable->GetEntry("PowerRPMColor");
	nt::NetworkTableEntry DistanceToTargetEntry = NetworkTable->GetEntry("DistanceToTarget");
	nt::NetworkTableEntry DistanceToTargetGroundEntry = NetworkTable->GetEntry("DistanceToTargetGroundIR");
	nt::NetworkTableEntry DistanceToTargetGroundColorEntry = NetworkTable->GetEntry("DistanceToTargetGroundColor");
	nt::NetworkTableEntry FrameNumberEntry = NetworkTable->GetEntry("Frame Number Processed By Vision");
	nt::NetworkTableEntry FrameDurationEntry = NetworkTable->GetEntry("Frame Process Time By Vision");
	nt::NetworkTableEntry UseCameraServerEntry = NetworkTable->GetEntry("Use Vision Camera as Vision");
	UseCameraServerEntry.SetBoolean(true);
*/
	std::cout << "Starting Loop..." << std::endl;
	while (true) {

#ifdef WINDOW
		cv::waitKey(1);
#endif
		frame_num++;
		const auto before = clock::now();

		rs2::frameset frames = p.wait_for_frames();

		// align frames
		//frames = align_to_ir.process(frames);
		//align_to_depth.process(frames);
		// get depth frame
		rs2::depth_frame depth = frames.get_depth_frame();
		depth = dec.process(depth);
		depth = depth2disparity.process(depth);
		depth = spat.process(depth);
		depth = temp.process(depth);

		// get ir frame
		rs2::frame img = frames.get_infrared_frame(1);

		

		// get color frame
		rs2::frame color = frames.get_color_frame();

		//const int w = img.as<rs2::video_frame>().get_width();
		//const int h = img.as<rs2::video_frame>().get_height();
		//cv::Mat image = cv::Mat(cv::Size(w, h), CV_8UC1, (void*)img.get_data());

		const int wc = color.as<rs2::video_frame>().get_width();
		const int hc = color.as<rs2::video_frame>().get_height();
		cv::Mat imageColor = cv::Mat(cv::Size(wc, hc), CV_8UC3, (void*)color.get_data());

		
		//cs::CvSink cvSink = frc::CameraServer::GetInstance()->GetVideo();
		//cs::CvSource outputStream = frc::CameraServer::GetInstance()->PutVideo("Turret Camera", 1280, 720);
		// Check if we should cast the image to CameraServer
//		if (UseCameraServerEntry.GetBoolean(false)) {
			//outputStream.PutFrame(imageColor);
			
//		}


		cv::Mat channel[3];

		//image = channel[1];
		//cv::GaussianBlur(imageColor, imageColor, cv::Size(25, 25), 0, 0);

		cv::split(imageColor, channel);
#ifdef WINDOW
		//cv::imshow("RAW GREEN", channel[1]);
		//cv::imshow("RAW IR", image);
		cv::imshow("RAW COLOR", imageColor);
#endif
		//image = channel[1];
		//cv::GaussianBlur(image, image, cv::Size(11, 5), 0, 0);

		//threshold(image, image, 230, 255, cv::THRESH_BINARY);

		grip::GripPipeline gripPipeline;
		gripPipeline.Process(imageColor);
		cv::Mat* imageColorProcessed = gripPipeline.GetHsvThresholdOutput();

		cv::Mat binaryColor = cv::Mat::zeros(imageColor.size(), CV_8U);

		threshold(*imageColorProcessed, binaryColor, 230, 255, cv::THRESH_BINARY);

		

#ifdef WINDOW
		cv::imshow("GRIP Output", binaryColor);
#endif

		TargetFinder::TargetFinder targetFinder;

		TargetFinder::TargetData colorTargetData = targetFinder.findTargetNoDepth(binaryColor);

		if (colorTargetData.targetFound == false || colorTargetData.targetWidth == 0 || colorTargetData.targetHeight == 0 || colorTargetData.target.x == 0 || colorTargetData.target.y == 0) {
			const sec duration = clock::now() - before;
			std::string sendableData = targetFinder.makeSendableData(0, 0, 0, 0, 0);
			targetFinder.sendDataUDP(sendableData, targetFinder.ROBOIPV4, targetFinder.ROBOPORT);
#ifdef COUT
			std::cout << "----------------------------" << "\nGreen Target is at " << colorTargetData.centerTarget.y + (colorTargetData.centerTargetHeight / 2) << "\n";
			std::cout << "Frame Number	: " << std::to_string(frame_num) << " Took " << duration.count() << "s" << "\n";
			std::cout << "Sending: " << sendableData << std::endl;
			std::cout << "----------------------------" << std::endl;
#endif
			continue;
		}

		//image = filterIR(image, colorTargetData.target);
		// TEMP
		//cv::Mat image = imageColor;
		
#ifdef WINDOW
		//cv::Mat roi = image;
		//cv::rectangle(roi, colorTargetData.target, 255, 2);
		//cv::imshow("ROI",image);
#endif
		//TargetFinder::TargetData data = targetFinder.findTargetNoDepth(image);
		/*
		if (data.targetFound == false) {
			const sec duration = clock::now() - before;
			std::string sendableData = targetFinder.makeSendableData(0, 0, 0, 0, 0);
			targetFinder.sendDataUDP(sendableData, targetFinder.ROBOIPV4, targetFinder.ROBOPORT);
			std::cout << "Frame Number	: " << std::to_string(frame_num) << " Took " << duration.count() << "s" << "\n";
			std::cout << "Sending: " << sendableData << std::endl;
			continue;
		}
		*/

		//const float distance = depth.get_distance(((data.centerTargetWidth / 2) + data.centerTarget.x) / 2, ((data.centerTargetHeight / 2) + data.centerTarget.y) / 2);
		//const float distanceUnits = depth.as<rs2::depth_frame>().get_units();
		//const double targetDepth = distance;

		//const float targetPixel2D[2] = { (data.targetWidth / 2.0) + data.target.x, (data.targetHeight / 2.0) + data.target.y };
		//float targetPoint3D[3] = { 0, 0, 0 };
		//rs2_deproject_pixel_to_point(targetPoint3D, &cameraIntrinsics, targetPixel2D, targetDepth);

		//cv::Vec3f normal = { 0,0,1 };
		//cv::Vec3f targetVector = { targetPoint3D[0], 0, targetPoint3D[2] };
		//const float dot = targetVector.dot(normal);
		//const float targetVectorMag = sqrt(pow(targetVector[0], 2) + pow(targetVector[1], 2) + pow(targetVector[2], 2));
		//float xTurnRad = acos(dot / (targetVectorMag));

		int xTurnRad = (((colorTargetData.centerTarget.width + colorTargetData.centerTarget.x) * HFOV_PER_PX) - HFOV_OFFSET) * (3.14/180) ;

		// If target is to the right, should be negative
		if ((colorTargetData.target.x + (colorTargetData.target.width / 2)) >= imageColor.cols / 2) {
			xTurnRad *= -1;
		}

		/*
		const float distanceModifier = 1.31;
		float targetDepthAdjusted = targetDepth * (3.28084);
		const float targetHeightReal = 6.17;
		targetDepthAdjusted = sqrt(pow(targetDepthAdjusted, 2) - pow(targetHeightReal, 2));
		targetDepthAdjusted *= distanceModifier;
		targetDepthAdjusted = targetDepthAdjusted - 2.5;
		*/
		const float DISTANCE_PX = colorTargetData.centerTarget.y + (colorTargetData.centerTargetHeight / 2);
		const float distancePXA =  4.87057e-10;
		const float distancePXB = -5.17105e-07;
		const float distancePXC =  0.000222759;
		const float distancePXD = -0.027171;
		const float distancePXE =  3.37362;
		float distancePX  =  (distancePXA * pow(DISTANCE_PX,4)) + (distancePXB * pow(DISTANCE_PX,3)) + (distancePXC * pow(DISTANCE_PX,2)) + (distancePXD * pow(DISTANCE_PX, 1)) + distancePXE;
		// ADD OFFSET (FT)
		distancePX += 2;
		/*
		if (isnan(targetDepthAdjusted)) {
			const sec duration = clock::now() - before;
			std::string sendableData = targetFinder.makeSendableData(0, 0, 0, 0, 0);
			targetFinder.sendDataUDP(sendableData, targetFinder.ROBOIPV4, targetFinder.ROBOPORT);
			std::cout << "Frame Number	: " << std::to_string(frame_num) << " Took " << duration.count() << "s" << "\n";
			std::cout << "Sending: " << sendableData << std::endl;
			continue;
		}
		*/
		const float pitchA =  0.0431;
		const float pitchB = -1.4374;
		const float pitchC =  26.435;
		const float pitchD =  1.7124;
		//const float pitch = (pitchA * pow(targetDepthAdjusted, 3)) + (pitchB * pow(targetDepthAdjusted, 2)) + (pitchC * pow(targetDepthAdjusted, 1)) + pitchD;
		const float pitchPX = (pitchA * pow(distancePX, 3)) + (pitchB * pow(distancePX, 2)) + (pitchC * pow(distancePX, 1)) + pitchD;


		const float powerA = -0.4235;
		const float powerB =  12.502;
		const float powerC = -6.5796;
		const float powerD =  3024.6;
		//const float power = (powerA * pow(targetDepthAdjusted, 3)) + (powerB * pow(targetDepthAdjusted, 2)) + (powerC * pow(targetDepthAdjusted, 1)) + powerD;
		const float powerPX = (powerA * pow(distancePX, 3)) + (powerB * pow(distancePX, 2)) + (powerC * pow(distancePX, 1)) + powerD;

		const sec duration = clock::now() - before;
#ifdef COUT
//		std::cout << "Target Depth			: " << std::to_string(targetDepth) << " Meters" << "\n";
//		DistanceToTargetEntry.SetDouble(targetDepth);
//		std::cout << "Target Depth Adjusted		: " << std::to_string(targetDepthAdjusted) << " Ft" << "\n";
//		DistanceToTargetGroundEntry.SetDouble(targetDepthAdjusted);
		std::cout << "Target Depth Adjusted PX	: " << std::to_string(distancePX) << " Ft" << "\n";
//		DistanceToTargetGroundColorEntry.SetDouble(distancePX);
		std::cout << "X Turn Radians			: " << std::to_string(xTurnRad) << "\n";
//		YawTurnRadEntry.SetDouble(xTurnRad);
		//cout << "Vector to Target		: " << to_string(targetPoint3D[0]) << " " << to_string(targetPoint3D[1]) << " " << to_string(targetPoint3D[2]) << endl;
//		std::cout << "Y Turn Rotations			: " << std::to_string(pitch) << "\n";
//		PitchTickEntry.SetDouble(pitch);
		std::cout << "Y Turn Rotations Color		: " << std::to_string(pitchPX) << "\n";
//		PitchTickColorEntry.SetDouble(pitchPX);
//		std::cout << "Turret Power			: " << std::to_string(power) << "\n";
//		PowerRPMEntry.SetDouble(power);
		std::cout << "Turret Power Color		: " << std::to_string(powerPX) << "\n";
//		PowerRPMColorEntry.SetDouble(powerPX);
		std::cout << "----------------------------" << "\nGreen Target is at " << colorTargetData.centerTarget.y + (colorTargetData.centerTargetHeight / 2) << "\n";
//		DistanceToTargetGroundColorEntry.SetDouble(distancePX);
		std::cout << "Frame Number				: " << std::to_string(frame_num) << " Took " << duration.count() << "s" << "\n";
//		FrameNumberEntry.SetDouble(frame_num);
//		FrameDurationEntry.SetDouble(duration.count());
#endif
		std::string sendableData = targetFinder.makeSendableData(colorTargetData.targetFound, xTurnRad, pitchPX, powerPX, distancePX);
		targetFinder.sendDataUDP(sendableData, targetFinder.ROBOIPV4, targetFinder.ROBOPORT);
#ifdef COUT
		std::cout << "Sending: " << sendableData << std::endl;
		std::cout << "----------------------------" << std::endl;
#endif
#ifdef WINDOW
		cv::waitKey(1);
#endif
	}
	return 0;
}

cv::Mat filterIR(cv::Mat &ir, cv::Rect &roi)
{
	cv::Mat mask = cv::Mat::zeros(ir.size(), ir.type());
	cv::rectangle(mask, roi, 1, -1);
	//cv::bitwise_not(mask, mask);
	//cv::threshold(mask, mask, 240, 255, cv::ThresholdTypes::THRESH_BINARY);
	cv::Mat dst = cv::Mat::zeros(ir.size(), ir.type());
	ir.copyTo(dst, mask);
	return dst;
}
/*
static void VisionThread() {

	cs::CvSink cvSink = frc::CameraServer::GetInstance()->GetVideo();

	// Setup a CvSource. This will send images back to the Dashboard
	cs::CvSource outputStream = frc::CameraServer::GetInstance()->PutVideo("Rectangle", 640, 480);

	// Mats are very memory expensive. Lets reuse this Mat.
	cv::Mat mat;

	while (true) {
		// Tell the CvSink to grab a frame from the camera and
		// put it
		// in the source mat.  If there is an error notify the
		// output.
		if (cvSink.GrabFrame(mat) == 0) {
			// Send the output the error.
			outputStream.NotifyError(cvSink.GetError());
			// skip the rest of the current iteration
			continue;
		}
		// Put a rectangle on the image
		rectangle(mat, cv::Point(100, 100), cv::Point(400, 400),
			cv::Scalar(255, 255, 255), 5);
		// Give the output stream a new image to display
		outputStream.PutFrame(mat);
	}
}
*/
static void rs2_deproject_pixel_to_point(float point[3], const struct rs2_intrinsics* intrin, const float pixel[2], float depth)
{
	assert(intrin->model != RS2_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image
	assert(intrin->model != RS2_DISTORTION_FTHETA); // Cannot deproject to an ftheta image
	//assert(intrin->model != RS2_DISTORTION_BROWN_CONRADY); // Cannot deproject to an brown conrady model

	float x = (pixel[0] - intrin->ppx) / intrin->fx;
	float y = (pixel[1] - intrin->ppy) / intrin->fy;
	if (intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY)
	{
		float r2 = x * x + y * y;
		float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
		float ux = x * f + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
		float uy = y * f + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
		x = ux;
		y = uy;
	}
	point[0] = depth * x;
	point[1] = depth * y;
	point[2] = depth;
}
